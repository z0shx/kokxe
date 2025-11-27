"""
基于LangChain的AI Agent服务
使用标准化的LangChain Agent实现
"""
import json
import asyncio
from typing import Dict, List, AsyncGenerator, Optional, Any
from datetime import datetime

from database.models import TradingPlan, PredictionData, AgentConversation, AgentMessage, LLMConfig
from database.db import get_db
from utils.logger import setup_logger
from services.trading_tools import OKXTradingTools
from services.agent_tools import get_all_tools
from enum import Enum

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

logger = setup_logger(__name__, "agent_service.log")


class ConversationType(Enum):
    """对话类型枚举"""
    MANUAL_CHAT = "manual_chat"
    AUTO_INFERENCE = "auto_inference"


class AgentService:
    """基于LangChain的AI Agent服务"""

    def __init__(self):
        self._trading_tools = None  # 懒加载
        self._llm_clients = {}     # LLM客户端缓存

    @property
    def trading_tools(self):
        """懒加载trading tools"""
        if self._trading_tools is None:
            from config import Config
            self._trading_tools = OKXTradingTools(
                api_key=Config.OKX_API_KEY,
                secret_key="your_secret_key",
                passphrase="your_passphrase"
            )
        return self._trading_tools

    def _get_llm_client(self, llm_config):
        """获取或创建LLM客户端"""
        cache_key = f"{llm_config.provider}_{llm_config.model_name}"

        if cache_key not in self._llm_clients:
            if llm_config.provider == 'qwen':
                import openai
                client = ChatOpenAI(
                    api_key=llm_config.api_key,
                    base_url=llm_config.api_base_url,
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif llm_config.provider == 'openai':
                import openai
                client = ChatOpenAI(
                    api_key=llm_config.api_key,
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            elif llm_config.provider == 'anthropic':
                from langchain_anthropic import ChatAnthropic
                client = ChatAnthropic(
                    api_key=llm_config.api_key,
                    model=llm_config.model_name,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
            else:
                logger.error(f"不支持的LLM提供商: {llm_config.provider}")
                return None

            self._llm_clients[cache_key] = client

        return self._llm_clients[cache_key]

    def _create_langchain_tools(self, enabled_tools_config: Dict[str, bool]) -> List[Any]:
        """创建LangChain工具"""
        tools = []
        tools_map = get_all_tools()

        @tool
        def get_current_price(inst_id: str) -> Dict[str, Any]:
            """获取当前市场价格"""
            return self.trading_tools.get_current_price(inst_id=inst_id)

        @tool
        def query_historical_kline_data(
            inst_id: str,
            interval: str = "1H",
            start_time: str = None,
            end_time: str = None,
            limit: int = 100
        ) -> Dict[str, Any]:
            """查询历史K线数据"""
            params = {
                "inst_id": inst_id,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time

            return self.trading_tools.query_historical_kline_data(**params)

        @tool
        async def place_order(
            inst_id: str,
            td_mode: str,
            side: str,
            order_type: str,
            size: str,
            price: str = None
        ) -> Dict[str, Any]:
            """下单交易"""
            params = {
                "inst_id": inst_id,
                "td_mode": td_mode,
                "side": side,
                "order_type": order_type,
                "size": size
            }
            if price:
                params["price"] = price

            return await self.trading_tools.place_order(**params)

        @tool
        async def cancel_order(inst_id: str, order_id: str) -> Dict[str, Any]:
            """取消订单"""
            return await self.trading_tools.cancel_order(inst_id=inst_id, order_id=order_id)

        @tool
        def get_positions(inst_id: str = None) -> Dict[str, Any]:
            """获取持仓信息"""
            params = {}
            if inst_id:
                params["inst_id"] = inst_id
            return self.trading_tools.get_positions(**params)

        @tool
        def get_trading_limits(inst_id: str) -> Dict[str, Any]:
            """获取交易限制"""
            return self.trading_tools.get_trading_limits(inst_id=inst_id)

        @tool
        def get_current_utc_time() -> Dict[str, Any]:
            """获取当前UTC时间"""
            return self.trading_tools.get_current_utc_time()

        @tool
        def get_account_balance() -> Dict[str, Any]:
            """获取账户余额"""
            return self.trading_tools.get_account_balance()

        @tool
        def get_order_info(inst_id: str, order_id: str) -> Dict[str, Any]:
            """获取订单信息"""
            return self.trading_tools.get_order_info(inst_id=inst_id, order_id=order_id)

        @tool
        def get_pending_orders(inst_id: str = None) -> Dict[str, Any]:
            """获取待成交订单"""
            params = {}
            if inst_id:
                params["inst_id"] = inst_id
            return self.trading_tools.get_pending_orders(**params)

        @tool
        def get_order_history(inst_id: str = None, limit: int = 100) -> Dict[str, Any]:
            """获取订单历史"""
            params = {"limit": limit}
            if inst_id:
                params["inst_id"] = inst_id
            return self.trading_tools.get_order_history(**params)

        @tool
        def get_fills(inst_id: str = None, limit: int = 100) -> Dict[str, Any]:
            """获取成交记录"""
            params = {"limit": limit}
            if inst_id:
                params["inst_id"] = inst_id
            return self.trading_tools.get_fills(**params)

        @tool
        def place_limit_order(
            inst_id: str,
            td_mode: str,
            side: str,
            order_type: str,
            size: str,
            price: str
        ) -> Dict[str, Any]:
            """下限价单"""
            return self.trading_tools.place_limit_order(
                inst_id=inst_id, td_mode=td_mode, side=side,
                order_type=order_type, size=size, price=price
            )

        @tool
        def amend_order(
            inst_id: str,
            order_id: str,
            new_sz: str = None,
            new_px: str = None
        ) -> Dict[str, Any]:
            """修改订单"""
            params = {"inst_id": inst_id, "order_id": order_id}
            if new_sz:
                params["new_sz"] = new_sz
            if new_px:
                params["new_px"] = new_px
            return self.trading_tools.amend_order(**params)

        @tool
        def get_prediction_history(plan_id: int, limit: int = 100) -> Dict[str, Any]:
            """获取预测历史"""
            return self.trading_tools.get_prediction_history(plan_id=plan_id, limit=limit)

        @tool
        def query_prediction_data(plan_id: int, limit: int = 100) -> Dict[str, Any]:
            """查询预测数据"""
            return self.trading_tools.query_prediction_data(plan_id=plan_id, limit=limit)

        @tool
        def run_latest_model_inference(plan_id: int) -> Dict[str, Any]:
            """运行最新模型推理"""
            return self.trading_tools.run_latest_model_inference(plan_id=plan_id)

        @tool
        def modify_order(
            inst_id: str,
            order_id: str,
            new_sz: str = None,
            new_px: str = None
        ) -> Dict[str, Any]:
            """修改订单 (amend_order的别名)"""
            params = {"inst_id": inst_id, "order_id": order_id}
            if new_sz:
                params["new_sz"] = new_sz
            if new_px:
                params["new_px"] = new_px
            return self.trading_tools.amend_order(**params)

        @tool
        def place_stop_loss_order(
            inst_id: str,
            td_mode: str,
            side: str,
            size: str,
            trigger_px: str,
            order_type: str = "conditional_market"
        ) -> Dict[str, Any]:
            """下止损单"""
            params = {
                "inst_id": inst_id,
                "td_mode": td_mode,
                "side": side,
                "size": size,
                "trigger_px": trigger_px,
                "order_type": order_type
            }
            return self.trading_tools.place_order(**params)

        @tool
        def delete_prediction_data_by_batch(
            batch_id: str,
            plan_id: int = None
        ) -> Dict[str, Any]:
            """删除指定批次的预测数据"""
            params = {"batch_id": batch_id}
            if plan_id:
                params["plan_id"] = plan_id
            return self.trading_tools.delete_prediction_data_by_batch(**params)

        # 根据配置启用工具
        available_tools = {
            "get_current_price": get_current_price,
            "query_historical_kline_data": query_historical_kline_data,
            "place_order": place_order,
            "cancel_order": cancel_order,
            "get_positions": get_positions,
            "get_trading_limits": get_trading_limits,
            "get_current_utc_time": get_current_utc_time,
            "get_account_balance": get_account_balance,
            "get_order_info": get_order_info,
            "get_pending_orders": get_pending_orders,
            "get_order_history": get_order_history,
            "get_fills": get_fills,
            "place_limit_order": place_limit_order,
            "amend_order": amend_order,
            "get_prediction_history": get_prediction_history,
            "query_prediction_data": query_prediction_data,
            "run_latest_model_inference": run_latest_model_inference,
            "modify_order": modify_order,
            "place_stop_loss_order": place_stop_loss_order,
            "delete_prediction_data_by_batch": delete_prediction_data_by_batch
        }

        for tool_name, tool_func in available_tools.items():
            if enabled_tools_config.get(tool_name, False):
                tools.append(tool_func)
                logger.info(f"启用LangChain工具: {tool_name}")

        return tools

    def _build_system_prompt(self, plan_id: int) -> str:
        """构建系统提示词"""
        try:
            from database.models import TradingPlan
            from services.agent_tools import get_all_tools

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "你是一个AI助手。"

                tools_config = plan.agent_tools_config or {}
                enabled_tools = [name for name, enabled in tools_config.items() if enabled]

                system_prompt = f"""你是一个专业的加密货币交易AI助手，负责分析市场数据并做出交易决策。

**交易计划信息**:
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 环境: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}
- 计划状态: {plan.status}

**推理模式**: ReAct (Reasoning + Acting)
1. **思考** (Thought): 分析市场状况和可用数据
2. **行动** (Action): 调用工具获取信息或执行交易
3. **观察** (Observation): 分析工具返回结果
4. **重复** 直到得出最终结论

**可用工具**: {', '.join(enabled_tools) if enabled_tools else '无可用工具'}"""

                if plan.agent_prompt:
                    system_prompt += f"""

**用户自定义指示**:
{plan.agent_prompt}"""

                system_prompt += """

**重要原则**:
- 始终谨慎决策，控制风险
- 在模拟盘环境中可以大胆尝试策略
- 所有交易操作都会被记录用于分析
- 使用限价单而非市价单以避免价格滑点
- 基于数据和事实进行分析，避免情绪化决策

请开始你的分析和推理过程。"""

                return system_prompt

        except Exception as e:
            logger.error(f"构建系统提示词失败: {e}")
            return "你是一个AI助手。"

    async def _create_langchain_agent(self, plan_id: int) -> Any:
        """创建LangChain Agent"""
        try:
            # 获取计划配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.llm_config_id:
                    raise ValueError(f"计划 {plan_id} 未配置LLM")

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    raise ValueError(f"LLM配置 {plan.llm_config_id} 不存在")

            # 创建LLM客户端
            llm = self._get_llm_client(llm_config)
            if not llm:
                raise ValueError("无法创建LLM客户端")

            # 创建工具
            tools = self._create_langchain_tools(plan.agent_tools_config or {})

            # 创建提示词模板
            system_prompt = self._build_system_prompt(plan_id)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # 创建Agent
            agent = create_openai_functions_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,  # 启用详细日志以便调试流式输出
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_iterations=5,  # 限制最大迭代次数
                early_stopping_method="generate"  # 早期停止策略
            )

            return agent_executor

        except Exception as e:
            logger.error(f"创建LangChain Agent失败: {e}")
            raise

    async def stream_manual_inference(self, plan_id: int) -> AsyncGenerator[List[Dict], None]:
        """流式手动推理"""
        try:
            # 创建对话会话
            conversation = await self._create_conversation(
                plan_id=plan_id,
                conversation_type=ConversationType.AUTO_INFERENCE
            )

            # 添加系统提示词
            system_message = self._build_system_prompt(plan_id)
            await self._add_message(conversation.id, "system", system_message)

            # 添加预测数据
            await self._add_prediction_data(conversation.id, plan_id)

            # 获取上下文
            context_messages = await self._get_conversation_context(conversation.id)
            yield context_messages

            # 创建LangChain Agent
            agent_executor = await self._create_langchain_agent(plan_id)

            # 准备输入消息
            chat_history = []
            for msg in context_messages:
                if msg["role"] == "system":
                    chat_history.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            # 执行推理
            input_message = "请基于预测数据进行分析和推理，考虑价格走势和概率指标。"

            try:
                # 添加思考消息
                thought_msg = "🤔 **开始思考**: 正在分析预测数据和市场情况..."
                await self._add_message(conversation.id, "assistant", thought_msg)
                current_messages = await self._get_conversation_context(conversation.id)
                yield current_messages

                # 使用异步流式推理
                logger.info("开始LangChain异步流式推理...")
                async for chunk in agent_executor.astream({
                    "input": input_message,
                    "chat_history": chat_history[:-1]  # 排除系统消息，避免重复
                }):
                    logger.info(f"收到流式chunk: {type(chunk)}, keys: {chunk.keys() if hasattr(chunk, 'keys') else 'no keys'}")

                    # 首先检查是否有中间步骤（工具调用）
                    if "intermediate_steps" in chunk and chunk["intermediate_steps"]:
                        steps = chunk["intermediate_steps"]
                        logger.info(f"检测到中间步骤: {len(steps)} 个工具调用")

                        for i, (action, observation) in enumerate(steps):
                            logger.info(f"处理工具调用 {i+1}: {action.tool}")

                            # 工具调用消息
                            action_msg = f"🛠️ **调用工具** ({i+1}/{len(steps)}): `{action.tool}`\n📝 **参数**: `{action.tool_input}`"
                            await self._add_message(conversation.id, "assistant", action_msg)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                            # 工具结果消息
                            obs_str = str(observation)[:500]  # 限制长度避免过长
                            result_msg = f"🔧 **工具结果** ({i+1}/{len(steps)}):\n```json\n{obs_str}\n```"
                            await self._add_message(conversation.id, "assistant", result_msg)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                            # 小延迟以显示流式效果
                            import asyncio
                            await asyncio.sleep(0.5)

                    # 然后处理最终输出
                    if "output" in chunk:
                        # 最终输出
                        final_output = chunk["output"]
                        logger.info(f"检测到最终输出: {final_output[:100]}...")
                        if final_output:
                            await self._add_message(conversation.id, "assistant", final_output)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                    # 添加小延迟以确保流式效果
                    import asyncio
                    await asyncio.sleep(0.1)

            except Exception as e:
                error_msg = f"推理执行失败: {str(e)}"
                logger.error(error_msg)
                await self._add_message(conversation.id, "assistant", error_msg)
                yield context_messages + [{"role": "assistant", "content": error_msg}]

        except Exception as e:
            logger.error(f"手动推理失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 推理失败: {str(e)}"}]

    async def stream_conversation(self, plan_id: int, user_message: str) -> AsyncGenerator[List[Dict], None]:
        """流式对话"""
        try:
            # 获取或创建对话会话
            conversation = await self._get_or_create_conversation(
                plan_id=plan_id,
                conversation_type=ConversationType.MANUAL_CHAT
            )

            # 添加用户消息
            await self._add_message(conversation.id, "user", user_message)

            # 获取上下文
            context_messages = await self._get_conversation_context(conversation.id)
            yield context_messages

            # 创建LangChain Agent
            agent_executor = await self._create_langchain_agent(plan_id)

            # 准备输入消息
            chat_history = []
            for msg in context_messages:
                if msg["role"] == "system":
                    chat_history.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            try:
                # 添加思考消息
                thought_msg = "🤔 **开始思考**: 正在分析您的问题..."
                await self._add_message(conversation.id, "assistant", thought_msg)
                current_messages = await self._get_conversation_context(conversation.id)
                yield current_messages

                # 使用异步流式对话
                logger.info("开始LangChain异步流式对话...")
                async for chunk in agent_executor.astream({
                    "input": user_message,
                    "chat_history": chat_history[:-1]  # 排除当前用户消息，避免重复
                }):
                    logger.info(f"收到对话chunk: {type(chunk)}, keys: {chunk.keys() if hasattr(chunk, 'keys') else 'no keys'}")

                    # 首先检查是否有中间步骤（工具调用）
                    if "intermediate_steps" in chunk and chunk["intermediate_steps"]:
                        steps = chunk["intermediate_steps"]
                        logger.info(f"检测到中间步骤: {len(steps)} 个工具调用")

                        for i, (action, observation) in enumerate(steps):
                            logger.info(f"处理工具调用 {i+1}: {action.tool}")

                            # 工具调用消息
                            action_msg = f"🛠️ **调用工具** ({i+1}/{len(steps)}): `{action.tool}`\n📝 **参数**: `{action.tool_input}`"
                            await self._add_message(conversation.id, "assistant", action_msg)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                            # 工具结果消息
                            obs_str = str(observation)[:500]  # 限制长度避免过长
                            result_msg = f"🔧 **工具结果** ({i+1}/{len(steps)}):\n```json\n{obs_str}\n```"
                            await self._add_message(conversation.id, "assistant", result_msg)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                            # 小延迟以显示流式效果
                            import asyncio
                            await asyncio.sleep(0.5)

                    # 然后处理最终输出
                    if "output" in chunk:
                        # 最终输出
                        final_output = chunk["output"]
                        logger.info(f"检测到最终输出: {final_output[:100]}...")
                        if final_output:
                            await self._add_message(conversation.id, "assistant", final_output)
                            current_messages = await self._get_conversation_context(conversation.id)
                            yield current_messages

                    # 添加小延迟以确保流式效果
                    import asyncio
                    await asyncio.sleep(0.1)

            except Exception as e:
                error_msg = f"对话执行失败: {str(e)}"
                logger.error(error_msg)
                await self._add_message(conversation.id, "assistant", error_msg)
                yield context_messages + [{"role": "assistant", "content": error_msg}]

        except Exception as e:
            logger.error(f"对话失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 对话失败: {str(e)}"}]

    # === 数据库操作方法 ===

    async def _create_conversation(self, plan_id: int, conversation_type: ConversationType) -> AgentConversation:
        """创建新对话会话"""
        with get_db() as db:
            conversation = AgentConversation(
                plan_id=plan_id,
                conversation_type=conversation_type.value,
                status='active',
                total_messages=0
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            return conversation

    async def _get_or_create_conversation(self, plan_id: int, conversation_type: ConversationType) -> AgentConversation:
        """获取或创建对话会话"""
        with get_db() as db:
            # 尝试获取最近的活跃对话
            conversation = db.query(AgentConversation).filter(
                AgentConversation.plan_id == plan_id,
                AgentConversation.conversation_type == conversation_type.value,
                AgentConversation.status == 'active'
            ).order_by(AgentConversation.last_message_at.desc()).first()

            if not conversation:
                # 创建新对话
                conversation = AgentConversation(
                    plan_id=plan_id,
                    conversation_type=conversation_type.value,
                    status='active',
                    total_messages=0
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                # 为新对话添加系统提示词
                system_message = self._build_system_prompt(plan_id)
                message = AgentMessage(
                    conversation_id=conversation.id,
                    role="system",
                    message_type="text",
                    content=system_message
                )
                db.add(message)

                # 更新对话消息计数
                conversation.total_messages = 1
                db.commit()

            return conversation

    async def _add_message(self, conversation_id: int, role: str, content: str):
        """添加消息到对话"""
        with get_db() as db:
            message = AgentMessage(
                conversation_id=conversation_id,
                role=role,
                message_type="text",
                content=content
            )
            db.add(message)

            # 更新对话状态
            db.query(AgentConversation).filter(
                AgentConversation.id == conversation_id
            ).update({
                "total_messages": AgentConversation.total_messages + 1,
                "last_message_at": message.created_at
            })

            db.commit()

    async def _get_conversation_context(self, conversation_id: int) -> List[Dict]:
        """获取对话上下文"""
        with get_db() as db:
            messages = db.query(AgentMessage).filter(
                AgentMessage.conversation_id == conversation_id
            ).order_by(AgentMessage.created_at.asc()).all()

            formatted_messages = []
            for msg in messages:
                if msg.role == "system":
                    formatted_messages.append({
                        "role": "system",
                        "content": msg.content
                    })
                else:
                    formatted_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            return formatted_messages

    async def _add_prediction_data(self, conversation_id: int, plan_id: int):
        """添加预测数据到对话"""
        try:
            with get_db() as db:
                # 获取最新一批预测数据
                latest_batch = db.query(PredictionData.inference_batch_id).filter(
                    PredictionData.plan_id == plan_id
                ).order_by(PredictionData.created_at.desc()).first()

                if not latest_batch:
                    await self._add_message(conversation_id, "system", "暂无预测数据")
                    return

                # 获取该批次的预测数据
                predictions = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id,
                    PredictionData.inference_batch_id == latest_batch[0]
                ).order_by(PredictionData.timestamp.asc()).all()

                if predictions:
                    # 构建CSV格式的预测数据
                    csv_lines = []
                    csv_lines.append("timestamp,open,high,low,close,close_mean,close_std,upward_probability,volatility_amplification_probability")

                    for pred in predictions:
                        timestamp = pred.timestamp.strftime("%Y-%m-%d %H:%M:%S") if pred.timestamp else ""
                        csv_lines.append(
                            f"{timestamp},{pred.open or 0},{pred.high or 0},{pred.low or 0},"
                            f"{pred.close or 0},{pred.close or 0},{pred.close_std or 0},"
                            f"{pred.upward_probability or 0:.3f},{pred.volatility_amplification_probability or 0:.3f}"
                        )

                    prediction_content = f"""**最新预测数据** (CSV格式，共{len(predictions)}条记录):

{chr(10).join(csv_lines)}

请基于以上预测数据进行分析，考虑价格走势和概率指标。"""

                    await self._add_message(conversation_id, "system", prediction_content)

        except Exception as e:
            logger.error(f"添加预测数据失败: {e}")


# 创建全局实例
agent_service = AgentService()