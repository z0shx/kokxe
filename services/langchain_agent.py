"""
LangChain Agent 服务
核心功能：
- 使用 LangChain Agent + Tools
- 合成提示词
- 流式输出到 Gradio Chatbot
- 支持 Qwen think 模式
- 显示工具调用交互
- 支持持续对话
"""
import json
import asyncio
import traceback
from typing import Dict, List, AsyncGenerator, Optional, Any
from datetime import datetime

from database.models import (
    TradingPlan, AgentConversation, AgentMessage,
    LLMConfig, now_beijing
)
from database.db import get_db
from utils.logger import setup_logger
from services.trading_tools import OKXTradingTools

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = setup_logger(__name__, "langchain_agent.log")


class LangChainAgentService:
    """LangChain Agent 服务"""

    def __init__(self):
        self._trading_tools = None
        self._llm_clients = {}

    @property
    def trading_tools(self):
        """懒加载交易工具"""
        if self._trading_tools is None:
            from config import Config
            self._trading_tools = OKXTradingTools(
                api_key=Config.OKX_API_KEY,
                secret_key=Config.OKX_SECRET_KEY,
                passphrase=Config.OKX_PASSPHRASE
            )
        return self._trading_tools

    def _get_llm_client(self, llm_config: LLMConfig):
        """获取 LLM 客户端"""
        client_key = f"{llm_config.provider}_{llm_config.model_name}"

        if client_key not in self._llm_clients:
            if llm_config.provider == "openai":
                self._llm_clients[client_key] = ChatOpenAI(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature or 0.7,
                    max_tokens=llm_config.max_tokens or 2000,
                    openai_api_key=llm_config.api_key
                )
            elif llm_config.provider == "anthropic":
                self._llm_clients[client_key] = ChatAnthropic(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature or 0.7,
                    max_tokens=llm_config.max_tokens or 2000,
                    anthropic_api_key=llm_config.api_key
                )
            elif llm_config.provider == "qwen":
                # Qwen 使用 OpenAI 兼容接口
                base_url = llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

                # 获取额外参数
                extra_params = {}
                if hasattr(llm_config, 'extra_params') and llm_config.extra_params:
                    try:
                        extra_params = llm_config.extra_params if isinstance(llm_config.extra_params, dict) else json.loads(llm_config.extra_params)
                    except:
                        extra_params = {}

                model_kwargs = {}
                if extra_params.get('enable_thinking', False):
                    model_kwargs = {"enable_thinking": True}

                self._llm_clients[client_key] = ChatOpenAI(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature or 0.7,
                    max_tokens=llm_config.max_tokens or 2000,
                    openai_api_key=llm_config.api_key,
                    openai_api_base=base_url,
                    model_kwargs=model_kwargs
                )
            else:
                raise ValueError(f"不支持的 LLM 提供商: {llm_config.provider}")

        return self._llm_clients[client_key]

    def _create_langchain_tools(self, tools_config: Dict[str, bool], plan_id: int) -> List[Any]:
        """创建 LangChain 工具"""
        available_tools = {}
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        # 获取计划信息
        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

        # 1. 获取当前价格工具
        if "get_current_price" in enabled_tools:
            @tool
            def get_current_price(inst_id: str = None) -> str:
                """获取交易对当前价格"""
                try:
                    inst_id = inst_id or plan.inst_id
                    price = self.trading_tools.get_current_price(inst_id)
                    return json.dumps({
                        "success": True,
                        "inst_id": inst_id,
                        "current_price": price,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"获取价格失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_current_price"] = get_current_price

        # 2. 获取当前 UTC 时间
        if "get_current_utc_time" in enabled_tools:
            @tool
            def get_current_utc_time() -> str:
                """获取当前 UTC 时间"""
                return json.dumps({
                    "success": True,
                    "current_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    "timezone": "UTC+8"
                }, ensure_ascii=False)

            available_tools["get_current_utc_time"] = get_current_utc_time

        # 3. 查询持仓
        if "get_positions" in enabled_tools:
            @tool
            def get_positions() -> str:
                """查询当前持仓"""
                try:
                    positions = self.trading_tools.get_positions()
                    return json.dumps({
                        "success": True,
                        "positions": positions,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询持仓失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_positions"] = get_positions

        # 4. 下单工具
        if "place_order" in enabled_tools:
            @tool
            def place_order(inst_id: str, side: str, order_type: str, size: str, price: str = None) -> str:
                """下单交易

                Args:
                    inst_id: 交易对，如 ETH-USDT
                    side: 买卖方向，buy 或 sell
                    order_type: 订单类型，market 或 limit
                    size: 下单数量
                    price: 下单价格（限价单需要）
                """
                # 参数验证
                if not inst_id:
                    return json.dumps({"success": False, "error": "交易对不能为空"}, ensure_ascii=False)
                if side not in ["buy", "sell"]:
                    return json.dumps({"success": False, "error": "买卖方向必须是 buy 或 sell"}, ensure_ascii=False)
                if order_type not in ["market", "limit"]:
                    return json.dumps({"success": False, "error": "订单类型必须是 market 或 limit"}, ensure_ascii=False)
                if not size or float(size) <= 0:
                    return json.dumps({"success": False, "error": "下单数量必须大于0"}, ensure_ascii=False)
                if order_type == "limit" and (not price or float(price) <= 0):
                    return json.dumps({"success": False, "error": "限价单必须指定有效价格"}, ensure_ascii=False)
                try:
                    result = self.trading_tools.place_order(
                        inst_id=inst_id, side=side,
                        order_type=order_type, size=size, price=price
                    )
                    return json.dumps({
                        "success": True,
                        "result": result,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"下单失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["place_order"] = place_order

        # 5. 取消订单工具
        if "cancel_order" in enabled_tools:
            @tool
            def cancel_order(inst_id: str, order_id: str) -> str:
                """取消订单

                Args:
                    inst_id: 交易对
                    order_id: 订单ID
                """
                try:
                    result = self.trading_tools.cancel_order(inst_id, order_id)
                    return json.dumps({
                        "success": True,
                        "result": result,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"取消订单失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["cancel_order"] = cancel_order

        # 6. 查询交易限制
        if "get_trading_limits" in enabled_tools:
            @tool
            def get_trading_limits() -> str:
                """查询交易限制"""
                try:
                    limits = self.trading_tools.get_trading_limits()
                    return json.dumps({
                        "success": True,
                        "limits": limits,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询交易限制失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_trading_limits"] = get_trading_limits

        return list(available_tools.values())

    def _build_system_prompt(self, plan: TradingPlan, tools_config: Dict[str, bool]) -> str:
        """构建系统提示词"""
        # 动态部分
        dynamic_prompt = plan.agent_prompt or "你是一个专业的加密货币交易AI助手。"

        # 工具描述
        tools_desc = []
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        tool_descriptions = {
            "get_current_price": "获取交易对的当前价格",
            "get_current_utc_time": "获取当前UTC时间",
            "get_positions": "查询当前持仓信息",
            "place_order": "下单交易（买入或卖出）",
            "cancel_order": "取消订单",
            "get_trading_limits": "查询交易限制"
        }

        for tool_name in enabled_tools:
            if tool_name in tool_descriptions:
                tools_desc.append(f"- {tool_name}: {tool_descriptions[tool_name]}")

        # 交易限制
        limits_desc = ""
        if plan.trading_limits:
            try:
                limits = plan.trading_limits if isinstance(plan.trading_limits, dict) else json.loads(plan.trading_limits)
                if limits:
                    limits_desc = f"\n\n交易限制：{json.dumps(limits, ensure_ascii=False, indent=2)}"
            except:
                pass

        # 完整提示词
        system_prompt = f"""{dynamic_prompt}

可用工具：
{chr(10).join(tools_desc) if tools_desc else "无可用工具"}

交易计划信息：
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 初始本金: {plan.initial_capital} USDT
{limits_desc}

请根据当前市场情况、交易计划和技术分析，为用户提供专业的交易建议。如果需要执行交易操作，请使用相应的工具。所有交易操作都会在模拟环境中进行。"""

        return system_prompt

    async def stream_conversation(
        self,
        plan_id: int,
        user_message: str,
        conversation_type: str = "manual_chat"
    ) -> AsyncGenerator[List[Dict[str, str]], None]:
        """流式对话"""
        # 获取计划和配置
        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
            if not plan:
                yield [{"role": "assistant", "content": "❌ 计划不存在"}]
                return

            llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
            if not llm_config:
                yield [{"role": "assistant", "content": "❌ LLM配置不存在"}]
                return

            # 创建或获取对话
            conversation = db.query(AgentConversation).filter(
                AgentConversation.plan_id == plan_id,
                AgentConversation.status == 'active',
                AgentConversation.conversation_type == conversation_type
            ).first()

            if not conversation:
                conversation = AgentConversation(
                    plan_id=plan_id,
                    conversation_type=conversation_type,
                    status='active',
                    started_at=now_beijing(),
                    last_message_at=now_beijing()
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

        # 构建系统提示词
        tools_config = plan.agent_tools_config or {}
        system_prompt = self._build_system_prompt(plan, tools_config)

        # 输出系统消息
        yield [{"role": "system", "content": system_prompt}]

        # 保存系统消息到数据库
        with get_db() as db:
            await self._save_message(
                db, conversation.id, "system", system_prompt, "text"
            )

        # 输出用户消息
        yield [{"role": "user", "content": user_message}]
        with get_db() as db:
            await self._save_message(
                db, conversation.id, "user", user_message, "text"
            )

        try:
            # 获取 LLM 和工具
            llm = self._get_llm_client(llm_config)
            tools = self._create_langchain_tools(tools_config, plan_id)

            # 构建消息历史
            with get_db() as db:
                # 获取历史消息
                history = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation.id
                ).order_by(AgentMessage.created_at).all()

                messages = [SystemMessage(content=system_prompt)]

                # 添加历史对话（排除刚刚保存的系统和用户消息）
                for msg in history[:-2]:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))
                    elif msg.role == "tool":
                        messages.append(ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id or ""))

                # 添加当前用户消息
                messages.append(HumanMessage(content=user_message))

            # 创建 Agent
            if tools:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad")
                ])

                agent = create_openai_tools_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=False,
                    handle_parsing_errors=True,
                    return_intermediate_steps=True
                )

                # 流式执行 Agent
                response = ""
                async for chunk in agent_executor.astream({"input": user_message, "chat_history": messages[1:-1]}):
                    # 处理工具调用
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            tool_name = getattr(action, 'tool', 'unknown')
                            tool_input = getattr(action, 'tool_input', {})

                            # 输出工具调用
                            tool_call_data = {
                                "tool_name": tool_name,
                                "arguments": tool_input,
                                "status": "calling"
                            }
                            tool_call_str = f"🔧 **调用工具**: `{tool_name}`\n\n**参数**: \n```json\n{json.dumps(tool_input, ensure_ascii=False, indent=2)}\n```"
                            yield [{"role": "tool", "content": tool_call_str}]

                            # 保存工具调用到数据库
                            with get_db() as db:
                                await self._save_message(
                                    db, conversation.id, "tool",
                                    f"调用工具 {tool_name}", "tool_call",
                                    tool_name, tool_input
                                )

                    # 处理工具结果
                    if "steps" in chunk:
                        for step in chunk["steps"]:
                            if hasattr(step, 'observation') and step.observation:
                                obs = step.observation
                                tool_name = getattr(step.action, 'tool', 'unknown') if hasattr(step, 'action') else 'unknown'

                                # 格式化工具结果
                                try:
                                    if isinstance(obs, str) and obs.startswith('{'):
                                        result_data = json.loads(obs)
                                        result_str = f"**✅ 工具执行完成**: `{tool_name}`\n\n**参数**: \n```json\n{json.dumps(getattr(step.action, 'tool_input', {}), ensure_ascii=False, indent=2)}\n```\n\n**结果**:\n```json\n{obs}\n```"
                                    else:
                                        result_str = f"**✅ 工具执行完成**: `{tool_name}`\n\n**结果**:\n{obs}"
                                except:
                                    result_str = f"**✅ 工具执行完成**: `{tool_name}`\n\n**结果**:\n{obs}"

                                yield [{"role": "tool", "content": result_str}]

                                # 保存工具结果到数据库
                                with get_db() as db:
                                    await self._save_message(
                                        db, conversation.id, "tool",
                                        f"工具 {tool_name} 执行完成", "tool_result",
                                        tool_name, getattr(step.action, 'tool_input', {}), obs
                                    )

                    # 处理最终输出
                    if "output" in chunk:
                        output = chunk["output"]
                        if output and output.strip():
                            response = output
                            yield [{"role": "assistant", "content": output}]

                            # 保存助手回复到数据库
                            with get_db() as db:
                                await self._save_message(
                                    db, conversation.id, "assistant", output, "text"
                                )

            else:
                # 没有工具，直接使用 LLM
                response = ""
                async for chunk in llm.astream(messages):
                    content = self._extract_content_from_chunk(chunk)
                    if content and content.strip():
                        response += content
                        yield [{"role": "assistant", "content": content}]

                # 保存完整回复到数据库
                if response:
                    with get_db() as db:
                        await self._save_message(
                            db, conversation.id, "assistant", response, "text"
                        )

            # 更新对话状态
            with get_db() as db:
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation.id
                ).first()
                if conversation:
                    conversation.last_message_at = now_beijing()
                    db.commit()

        except Exception as e:
            logger.error(f"Agent 执行失败: {e}")
            logger.debug(f"Agent 执行失败详情: {traceback.format_exc()}")
            yield [{"role": "assistant", "content": f"❌ Agent 执行失败: {str(e)}"}]

            # 保存错误信息
            with get_db() as db:
                await self._save_message(
                    db, conversation.id, "assistant",
                    f"Agent 执行失败: {str(e)}", "text"
                )

    def _extract_content_from_chunk(self, chunk) -> Optional[str]:
        """从 chunk 中提取内容，支持多种格式"""
        if not chunk:
            return None

        # 方法1: 标准 content 属性
        if hasattr(chunk, 'content'):
            content = chunk.content
            if content and isinstance(content, str) and content.strip():
                return content

        # 方法2: text 属性
        if hasattr(chunk, 'text'):
            text = chunk.text
            if text and isinstance(text, str) and text.strip():
                return text

        # 方法3: 尝试转换为字符串，排除对象表示
        try:
            chunk_str = str(chunk)
            if (len(chunk_str) > 0 and
                not chunk_str.startswith('<') and
                not chunk_str.startswith('AIMessage') and
                not chunk_str.startswith('ChatMessage') and
                not 'content=' in chunk_str and
                not 'additional_kwargs=' in chunk_str and
                not 'response_metadata=' in chunk_str and
                chunk_str.strip()):
                return chunk_str
        except:
            pass

        return None

    async def _save_message(
        self,
        db,
        conversation_id: int,
        role: str,
        content: str,
        message_type: str,
        tool_name: str = None,
        tool_args: dict = None,
        tool_result: str = None
    ):
        """保存消息到数据库"""
        try:
            message = AgentMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                message_type=message_type,
                tool_name=tool_name,
                tool_args=json.dumps(tool_args) if tool_args else None,
                tool_result=json.dumps(tool_result) if tool_result else None,
                created_at=now_beijing()
            )
            db.add(message)
            db.commit()
        except Exception as e:
            logger.error(f"保存消息失败: {e}")

    async def test_connection(self, plan_id: int) -> bool:
        """测试连接"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return False

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    return False

            llm = self._get_llm_client(llm_config)
            messages = [
                SystemMessage(content="你是一个测试助手"),
                HumanMessage(content="简单回复：测试成功")
            ]

            result = await llm.ainvoke(messages)
            return hasattr(result, 'content') and result.content is not None

        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False


# 全局实例
agent_service = LangChainAgentService()