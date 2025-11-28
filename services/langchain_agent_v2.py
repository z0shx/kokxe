"""
统一的LangChain Agent服务 v2 - 使用现代API
整合了推理、对话、流式输出功能，适配Gradio Chatbot接口
"""
import json
import asyncio
from typing import Dict, List, AsyncGenerator, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from sqlalchemy import func

from database.models import (
    TradingPlan, PredictionData, AgentConversation,
    AgentMessage, LLMConfig, TrainingRecord
)
from database.db import get_db
from utils.logger import setup_logger
from services.trading_tools import OKXTradingTools

# Modern LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

logger = setup_logger(__name__, "langchain_agent_v2.log")


class ConversationType(Enum):
    """对话类型枚举"""
    MANUAL_CHAT = "manual_chat"
    AUTO_INFERENCE = "auto_inference"


class LangChainAgentV2Service:
    """统一的LangChain Agent服务 v2，使用现代API适配Gradio Chatbot接口"""

    def __init__(self):
        self._trading_tools = None
        self._llm_clients = {}

    @property
    def trading_tools(self):
        """懒加载trading tools"""
        if self._trading_tools is None:
            from config import Config
            self._trading_tools = OKXTradingTools(
                api_key=Config.OKX_API_KEY or "demo_key",
                secret_key="demo_secret",
                passphrase="demo_passphrase"
            )
        return self._trading_tools

    def _get_llm_client(self, llm_config: LLMConfig):
        """获取LLM客户端"""
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
                # 使用OpenAI接口兼容方式支持Qwen
                base_url = llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                self._llm_clients[client_key] = ChatOpenAI(
                    model=llm_config.model_name,
                    temperature=llm_config.temperature or 0.7,
                    max_tokens=llm_config.max_tokens or 2000,
                    openai_api_key=llm_config.api_key,
                    openai_api_base=base_url
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

        return self._llm_clients[client_key]

    def _create_langchain_tools(self, tools_config: Dict[str, bool]):
        """创建LangChain工具"""
        available_tools = {}

        # 只启用配置中启用的工具
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        if "get_current_price" in enabled_tools:
            @tool
            def get_current_price(inst_id: str) -> Dict[str, Any]:
                """获取当前市场价格"""
                return self.trading_tools.get_current_price(inst_id=inst_id)
            available_tools["get_current_price"] = get_current_price

        if "query_historical_kline_data" in enabled_tools:
            @tool
            def query_historical_kline_data(
                inst_id: str,
                interval: str = "1H",
                start_time: str = None,
                end_time: str = None,
                limit: int = 100
            ) -> Dict[str, Any]:
                """查询历史K线数据"""
                params = {"inst_id": inst_id, "interval": interval, "limit": limit}
                if start_time:
                    params["start_time"] = start_time
                if end_time:
                    params["end_time"] = end_time
                return self.trading_tools.query_historical_kline_data(**params)
            available_tools["query_historical_kline_data"] = query_historical_kline_data

        if "get_positions" in enabled_tools:
            @tool
            def get_positions(inst_id: str = None) -> Dict[str, Any]:
                """获取当前持仓"""
                return self.trading_tools.get_positions(inst_id=inst_id)
            available_tools["get_positions"] = get_positions

        if "get_trading_limits" in enabled_tools:
            @tool
            def get_trading_limits(inst_id: str) -> Dict[str, Any]:
                """获取交易限制"""
                return self.trading_tools.get_trading_limits(inst_id=inst_id)
            available_tools["get_trading_limits"] = get_trading_limits

        if "place_order" in enabled_tools:
            @tool
            def place_order(
                inst_id: str,
                side: str,
                order_type: str,
                size: float,
                price: Optional[float] = None
            ) -> Dict[str, Any]:
                """下单交易"""
                return self.trading_tools.place_order(
                    inst_id=inst_id,
                    side=side,
                    order_type=order_type,
                    size=size,
                    price=price
                )
            available_tools["place_order"] = place_order

        if "cancel_order" in enabled_tools:
            @tool
            def cancel_order(inst_id: str, order_id: str) -> Dict[str, Any]:
                """取消订单"""
                return self.trading_tools.cancel_order(inst_id=inst_id, order_id=order_id)
            available_tools["cancel_order"] = cancel_order

        if "get_account_balance" in enabled_tools:
            @tool
            def get_account_balance() -> Dict[str, Any]:
                """获取账户余额"""
                return self.trading_tools.get_account_balance()
            available_tools["get_account_balance"] = get_account_balance

        if "get_latest_predictions" in enabled_tools:
            @tool
            def get_latest_predictions(plan_id: int, limit: int = 10) -> Dict[str, Any]:
                """获取最新预测数据"""
                try:
                    with get_db() as db:
                        predictions = db.query(PredictionData).filter(
                            PredictionData.plan_id == plan_id
                        ).order_by(PredictionData.timestamp.desc()).limit(limit).all()

                        return {
                            "success": True,
                            "data": [
                                {
                                    "timestamp": pred.timestamp.isoformat(),
                                    "open": pred.open,
                                    "high": pred.high,
                                    "low": pred.low,
                                    "close": pred.close,
                                    "close_min": pred.close_min,
                                    "close_max": pred.close_max
                                }
                                for pred in predictions
                            ]
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            available_tools["get_latest_predictions"] = get_latest_predictions

        if "modify_order" in enabled_tools:
            @tool
            def modify_order(
                inst_id: str,
                order_id: str,
                new_size: Optional[float] = None,
                new_price: Optional[float] = None
            ) -> Dict[str, Any]:
                """修改订单"""
                return self.trading_tools.modify_order(
                    inst_id=inst_id,
                    order_id=order_id,
                    new_size=new_size,
                    new_price=new_price
                )
            available_tools["modify_order"] = modify_order

        if "get_pending_orders" in enabled_tools:
            @tool
            def get_pending_orders(inst_id: str = None) -> Dict[str, Any]:
                """获取挂单"""
                return self.trading_tools.get_pending_orders(inst_id=inst_id)
            available_tools["get_pending_orders"] = get_pending_orders

        if "get_current_utc_time" in enabled_tools:
            @tool
            def get_current_utc_time() -> Dict[str, Any]:
                """获取当前UTC时间"""
                from datetime import datetime, timezone
                return {
                    "success": True,
                    "data": {
                        "utc_time": datetime.now(timezone.utc).isoformat(),
                        "timestamp": int(datetime.now(timezone.utc).timestamp())
                    }
                }
            available_tools["get_current_utc_time"] = get_current_utc_time

        if "place_stop_loss_order" in enabled_tools:
            @tool
            def place_stop_loss_order(
                inst_id: str,
                side: str,
                size: float,
                stop_price: float
            ) -> Dict[str, Any]:
                """设置止损订单"""
                return self.trading_tools.place_stop_loss_order(
                    inst_id=inst_id,
                    side=side,
                    size=size,
                    stop_price=stop_price
                )
            available_tools["place_stop_loss_order"] = place_stop_loss_order

        if "query_prediction_data" in enabled_tools:
            @tool
            def query_prediction_data(
                plan_id: int,
                batch_id: str = None,
                start_time: str = None,
                end_time: str = None,
                limit: int = 100
            ) -> Dict[str, Any]:
                """查询预测数据"""
                try:
                    with get_db() as db:
                        query = db.query(PredictionData).filter(
                            PredictionData.plan_id == plan_id
                        )

                        if batch_id:
                            query = query.filter(PredictionData.batch_id == batch_id)
                        if start_time:
                            query = query.filter(PredictionData.timestamp >= start_time)
                        if end_time:
                            query = query.filter(PredictionData.timestamp <= end_time)

                        predictions = query.order_by(PredictionData.timestamp.desc()).limit(limit).all()

                        return {
                            "success": True,
                            "data": [
                                {
                                    "batch_id": pred.batch_id,
                                    "timestamp": pred.timestamp.isoformat(),
                                    "open": pred.open,
                                    "high": pred.high,
                                    "low": pred.low,
                                    "close": pred.close,
                                    "close_min": pred.close_min,
                                    "close_max": pred.close_max,
                                    "upward_probability": pred.upward_probability,
                                    "volatility_amplification_probability": pred.volatility_amplification_probability
                                }
                                for pred in predictions
                            ]
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            available_tools["query_prediction_data"] = query_prediction_data

        if "get_prediction_history" in enabled_tools:
            @tool
            def get_prediction_history(plan_id: int, days: int = 7) -> Dict[str, Any]:
                """获取预测历史"""
                try:
                    from datetime import datetime, timedelta
                    start_date = datetime.now() - timedelta(days=days)

                    with get_db() as db:
                        # 按批次分组
                        batches = db.query(
                            PredictionData.batch_id,
                            func.min(PredictionData.timestamp).label('start_time'),
                            func.max(PredictionData.timestamp).label('end_time'),
                            func.count(PredictionData.id).label('count')
                        ).filter(
                            PredictionData.plan_id == plan_id,
                            PredictionData.timestamp >= start_date
                        ).group_by(PredictionData.batch_id).order_by(
                            PredictionData.batch_id.desc()
                        ).all()

                        return {
                            "success": True,
                            "data": [
                                {
                                    "batch_id": batch.batch_id,
                                    "start_time": batch.start_time.isoformat(),
                                    "end_time": batch.end_time.isoformat(),
                                    "data_points": batch.count
                                }
                                for batch in batches
                            ]
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            available_tools["get_prediction_history"] = get_prediction_history

        if "run_latest_model_inference" in enabled_tools:
            @tool
            def run_latest_model_inference(plan_id: int) -> Dict[str, Any]:
                """运行最新模型推理"""
                try:
                    from services.schedule_service import ScheduleService
                    result = ScheduleService.trigger_inference(plan_id)
                    return result
                except Exception as e:
                    return {"success": False, "error": str(e)}
            available_tools["run_latest_model_inference"] = run_latest_model_inference

        if "delete_prediction_data_by_batch" in enabled_tools:
            @tool
            def delete_prediction_data_by_batch(batch_id: str) -> Dict[str, Any]:
                """按批次删除预测数据"""
                try:
                    with get_db() as db:
                        deleted_count = db.query(PredictionData).filter(
                            PredictionData.batch_id == batch_id
                        ).delete()
                        db.commit()

                        return {
                            "success": True,
                            "data": {
                                "batch_id": batch_id,
                                "deleted_count": deleted_count
                            }
                        }
                except Exception as e:
                    db.rollback()
                    return {"success": False, "error": str(e)}
            available_tools["delete_prediction_data_by_batch"] = delete_prediction_data_by_batch

        return list(available_tools.values())

    def _build_system_prompt(self, plan: TradingPlan, training_record: TrainingRecord) -> str:
        """构建系统提示词"""
        # 优先使用配置中的系统提示词
        if plan.agent_prompt and plan.agent_prompt.strip():
            system_prompt = plan.agent_prompt.strip()
        else:
            # 如果没有配置提示词，使用默认提示词
            system_prompt = """你是一个专业的加密货币交易分析师，基于AI预测模型提供交易建议。

你的任务是基于提供的预测数据进行市场分析，并在必要时执行交易操作。

**分析原则：**
1. 仔细分析预测数据的趋势和置信度
2. 考虑市场风险和资金管理
3. 提供清晰的交易建议和理由
4. 使用工具获取实时市场数据辅助决策

**风险管理：**
- 严格遵守交易限额
- 优先考虑资金安全
- 避免过度交易

现在开始分析..."""

        # 添加交易对信息
        if plan.inst_id:
            system_prompt += f"\n\n**交易对**: {plan.inst_id}"
            system_prompt += f"\n**时间周期**: {plan.interval}"

        # 添加训练模型信息
        if training_record:
            system_prompt += f"\n\n**使用模型**: v{training_record.version} (ID: {training_record.id})"
            if training_record.train_end_time:
                system_prompt += f"\n**训练完成时间**: {training_record.train_end_time}"

        return system_prompt

    def _get_prediction_data_for_context(self, plan_id: int) -> str:
        """获取预测数据用于上下文"""
        try:
            with get_db() as db:
                # 获取最新的预测数据批次
                latest_batch = db.query(PredictionData.inference_batch_id).filter(
                    PredictionData.plan_id == plan_id
                ).group_by(PredictionData.inference_batch_id).order_by(
                    func.max(PredictionData.created_at).desc()
                ).limit(1).first()

                if not latest_batch:
                    return ""

                batch_id = latest_batch[0]
                predictions = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id,
                    PredictionData.inference_batch_id == batch_id
                ).order_by(PredictionData.timestamp).limit(24).all()

                if not predictions:
                    return ""

                # 格式化为CSV格式
                csv_lines = ["timestamp,open,high,low,close,close_min,close_max"]
                for pred in predictions:
                    csv_lines.append(
                        f"{pred.timestamp.isoformat()},"
                        f"{pred.open},{pred.high},{pred.low},"
                        f"{pred.close},{pred.close_min or ''},{pred.close_max or ''}"
                    )

                return "\n".join(csv_lines)

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return ""

    async def stream_agent_response(
        self,
        plan_id: int,
        user_message: str = None,
        conversation_type: ConversationType = ConversationType.MANUAL_CHAT,
        append_mode: bool = True
    ) -> AsyncGenerator[List[Dict[str, str]], None]:
        """
        流式Agent响应，适配Gradio Chatbot接口
        使用现代LangChain API实现

        Args:
            plan_id: 计划ID
            user_message: 用户消息（可选）
            conversation_type: 对话类型
            append_mode: 是否追加模式（True=追加消息，False=替换最后一条）
        """
        try:
            # 获取计划配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield [{"role": "assistant", "content": "❌ 计划不存在"}]
                    return

                # 检查LLM配置
                if not plan.llm_config_id:
                    yield [{"role": "assistant", "content": "❌ 未配置LLM"}]
                    return

                # 获取最新训练记录
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not training_record:
                    yield [{"role": "assistant", "content": "❌ 没有可用的训练记录"}]
                    return

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    yield [{"role": "assistant", "content": "❌ LLM配置不存在"}]
                    return

            # 创建对话会话
            conversation = await self._create_conversation(
                plan_id=plan_id,
                conversation_type=conversation_type
            )

            # 发送系统提示（使用assistant角色以便在Gradio中显示）
            system_prompt = self._build_system_prompt(plan, training_record)
            yield [{"role": "assistant", "content": f"📋 **系统提示**:\n{system_prompt}"}]

            # 构建输入消息
            if conversation_type == ConversationType.AUTO_INFERENCE:
                # 自动推理：使用预测数据作为输入
                prediction_data = self._get_prediction_data_for_context(plan_id)
                if not prediction_data:
                    yield [{"role": "assistant", "content": "❌ 没有找到预测数据"}]
                    return

                input_message = f"最新一批预测数据（CSV格式）：\n{prediction_data}"

                # 保存系统消息
                await self._save_message(conversation.id, "system",
                    f"开始自动推理分析，预测数据批次包含时间序列分析")

                # 发送预测数据作为用户输入
                yield [{"role": "user", "content": f"📈 **预测数据输入**:\n```csv\n{prediction_data}\n```"}]

            else:
                # 手动对话：使用用户消息
                if not user_message:
                    yield [{"role": "assistant", "content": "❌ 请输入消息"}]
                    return

                input_message = user_message

                # 发送用户消息到chatbot
                yield [{"role": "user", "content": input_message}]

            # 保存用户消息（使用原始的input_message，不包含格式化）
            await self._save_message(conversation.id, "user", input_message)

            # 获取LLM客户端和工具
            with get_db() as db:
                tools = self._create_langchain_tools(plan.agent_tools_config or {})

            # 发送思考开始消息
            yield [{"role": "assistant", "content": "🤔 **开始思考**: 正在分析您的请求..."}]

            # 使用LLM生成响应
            llm = self._get_llm_client(llm_config)

            # 构建消息列表
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input_message)
            ]

            # 检查是否为Qwen供应商
            is_qwen = llm_config.provider == "qwen"

            # 发送分析开始消息
            analysis_start_msg = "🧠 **AI分析**: 基于以上信息开始生成交易建议..."
            yield [{"role": "assistant", "content": analysis_start_msg}]
            await self._save_message(conversation.id, "assistant", analysis_start_msg)

            # 如果是Qwen供应商，发送分析开始标识
            if is_qwen:
                qwen_start_msg = "🤖 **Qwen分析结果**:"
                yield [{"role": "assistant", "content": qwen_start_msg}]
                await self._save_message(conversation.id, "assistant", qwen_start_msg)

            # 使用自定义Agent执行（避免版本兼容性问题）
            try:
                # 先发送开始思考的消息
                thinking_msg = "🤔 **Agent思考**: 正在分析预测数据并准备工具调用..."
                yield [{"role": "assistant", "content": thinking_msg}]
                await self._save_message(conversation.id, "assistant", thinking_msg)
                await asyncio.sleep(0.5)

                logger.info(f"创建自定义Agent，供应商: {llm_config.provider}, 工具数量: {len(tools)}")

                # 构建包含工具信息的增强提示词
                tools_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
                enhanced_prompt = f"""{system_prompt}

可用工具:
{tools_info}

请根据需要调用这些工具来获取实时数据并执行交易操作。使用以下格式调用工具:

Tool_Name: 参数1=value1, 参数2=value2

分析步骤:
1. 分析预测数据的趋势和模式
2. 调用相关工具获取实时市场数据
3. 基于数据做出交易决策
4. 执行相应的交易操作

当前输入: {input_message}

请开始分析并调用必要的工具:"""

                # 保存增强提示词
                prompt_msg = f"📋 **增强系统提示**: 已加载 {len(tools)} 个可用工具"
                yield [{"role": "assistant", "content": prompt_msg}]
                await self._save_message(conversation.id, "assistant", prompt_msg)

                # 使用LLM进行Agent式推理
                messages = [SystemMessage(content=enhanced_prompt)]
                current_content = ""
                tool_call_count = 0
                chunk_count = 0

                logger.info("开始Agent式LLM推理...")

                async for chunk in llm.astream(messages):
                    if chunk.content:
                        chunk_content = chunk.content
                        current_content += chunk_content
                        chunk_count += 1

                        logger.debug(f"Agent chunk {chunk_count}: {chunk_content[:100]}...")

                        # 检查是否包含工具调用
                        lines = chunk_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if ':' in line and any(tool.name in line for tool in tools):
                                # 检测到工具调用
                                tool_call_count += 1

                                # 解析工具调用
                                tool_name = None
                                tool_params = {}

                                for tool in tools:
                                    if tool.name in line:
                                        tool_name = tool.name
                                        if ':' in line:
                                            params_str = line.split(':', 1)[1].strip()
                                            # 简单参数解析
                                            if '=' in params_str:
                                                for param in params_str.split(','):
                                                    if '=' in param:
                                                        key, value = param.split('=', 1)
                                                        tool_params[key.strip()] = value.strip()
                                        break

                                if tool_name:
                                    tool_call_msg = f"🛠️ **调用工具** ({tool_call_count}): `{tool_name}`\n📝 **参数**: `{tool_params}`"
                                    yield [{"role": "assistant", "content": tool_call_msg}]
                                    await self._save_message(conversation.id, "assistant", tool_call_msg)

                                    # 执行工具
                                    try:
                                        tool_func = next((t for t in tools if t.name == tool_name), None)
                                        if tool_func:
                                            # 等待工具执行
                                            await asyncio.sleep(0.5)

                                            # 执行工具（同步调用）
                                            result = tool_func.invoke(tool_params)

                                            # 格式化结果
                                            if isinstance(result, dict):
                                                result_str = f"```json\n{result}\n```"
                                            else:
                                                result_str = str(result)
                                                if len(result_str) > 300:
                                                    result_str = result_str[:300] + "..."

                                            tool_result_msg = f"✅ **{tool_name} 执行完成**:\n{result_str}"
                                            yield [{"role": "assistant", "content": tool_result_msg}]
                                            await self._save_message(conversation.id, "assistant", tool_result_msg)

                                            # 将工具结果添加到当前内容中
                                            current_content += f"\n\n工具调用结果: {result_str}"
                                        else:
                                            error_msg = f"❌ 工具 {tool_name} 未找到"
                                            yield [{"role": "assistant", "content": error_msg}]
                                            await self._save_message(conversation.id, "assistant", error_msg)
                                    except Exception as tool_error:
                                        error_msg = f"❌ {tool_name} 执行失败: {str(tool_error)}"
                                        yield [{"role": "assistant", "content": error_msg}]
                                        await self._save_message(conversation.id, "assistant", error_msg)

                        # 流式输出当前内容
                        if len(chunk_content.strip()) > 0:
                            yield [{"role": "assistant", "content": chunk_content}]

                        # 控制输出速度
                        await asyncio.sleep(0.02)

                logger.info(f"Agent推理完成，工具调用次数: {tool_call_count}, 总输出长度: {len(current_content)}")

                # 保存完成消息
                completion_msg = f"✅ **Agent执行完成**: 共调用 {tool_call_count} 次工具，生成完整交易决策。"
                yield [{"role": "assistant", "content": completion_msg}]
                await self._save_message(conversation.id, "assistant", completion_msg)

            except Exception as agent_error:
                logger.error(f"Agent执行错误: {agent_error}")
                import traceback
                traceback.print_exc()
                error_msg = f"❌ Agent执行失败: {str(agent_error)}"
                yield [{"role": "assistant", "content": error_msg}]
                await self._save_message(conversation.id, "assistant", error_msg)

            # 保存完整的最终响应
            if current_content:
                final_content = current_content if not is_qwen else f"Qwen分析: {current_content}"
                await self._save_message(conversation.id, "assistant", final_content)

        except Exception as e:
            logger.error(f"Agent流式响应失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 处理失败: {str(e)}"}]

    async def _create_conversation(self, plan_id: int, conversation_type: ConversationType):
        """创建对话会话"""
        with get_db() as db:
            conversation = AgentConversation(
                plan_id=plan_id,
                conversation_type=conversation_type.value,
                status="active"
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            return conversation

    async def _save_message(self, conversation_id: int, role: str, content: str):
        """保存消息"""
        try:
            with get_db() as db:
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=role,
                    content=content
                )
                db.add(message)
                db.commit()
        except Exception as e:
            logger.error(f"保存消息失败: {e}")

    async def get_conversation_history(self, plan_id: int) -> List[Dict[str, str]]:
        """获取对话历史，返回Chatbot格式"""
        try:
            with get_db() as db:
                # 获取最新的对话
                latest_conversation = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(AgentConversation.created_at.desc()).first()

                if not latest_conversation:
                    return []

                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == latest_conversation.id
                ).order_by(AgentMessage.created_at).all()

                return [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]

        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []

    # 兼容性方法 - 为了保持向后兼容
    async def stream_manual_inference(self, plan_id: int):
        """手动推理流式响应（兼容性方法）"""
        async for message_batch in self.stream_agent_response(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            yield message_batch

    async def stream_conversation(self, plan_id: int, user_message: str):
        """流式对话（兼容性方法）"""
        async for message_batch in self.stream_agent_response(
            plan_id=plan_id,
            user_message=user_message,
            conversation_type=ConversationType.MANUAL_CHAT
        ):
            yield message_batch


# 全局实例
langchain_agent_v2_service = LangChainAgentV2Service()