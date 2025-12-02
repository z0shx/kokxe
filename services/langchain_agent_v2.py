"""
统一的LangChain Agent服务 v2 - 使用现代API
整合了推理、对话、流式输出功能，适配Gradio Chatbot接口
"""
import json
import asyncio
from typing import Dict, List, AsyncGenerator, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re

from sqlalchemy import func

from database.models import (
    TradingPlan, PredictionData, AgentConversation,
    AgentMessage, LLMConfig, TrainingRecord
)
from database.db import get_db
from utils.logger import setup_logger
from services.trading_tools import OKXTradingTools
from database.models import now_beijing

# Modern LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# 使用更现代的Agent API，避免依赖冲突
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("警告: LangChain Agents API不可用，将使用手动工具调用")

logger = setup_logger(__name__, "langchain_agent_v2.log")


class ConversationType(Enum):
    """对话类型枚举"""
    MANUAL_CHAT = "manual_chat"
    AUTO_INFERENCE = "auto_inference"


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    PLAY = "play"


class MessageType(Enum):
    """消息类型枚举"""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PLAY_RESULT = "play_result"


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
                api_key=Config.OKX_API_KEY,
                secret_key=Config.OKX_SECRET_KEY,
                passphrase=Config.OKX_PASSPHRASE
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

                # 获取额外参数
                extra_params = {}
                if hasattr(llm_config, 'extra_params') and llm_config.extra_params:
                    try:
                        extra_params = llm_config.extra_params if isinstance(llm_config.extra_params, dict) else json.loads(llm_config.extra_params)
                    except:
                        extra_params = {}

                # 配置思考模式
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
                raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

        return self._llm_clients[client_key]

    def _create_langchain_tools(self, tools_config: Dict[str, bool], plan_id: int = None):
        """创建LangChain工具 - 重构版本，专注于10个核心工具"""
        from database.db import get_db
        from database.models import TradingPlan, PredictionData

        available_tools = {}

        # 只启用配置中启用的工具
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        # 获取计划信息用于工具调用
        plan_info = None
        if plan_id:
            with get_db() as db:
                plan_info = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

        # 1. 查询预测数据工具
        if "query_prediction_data" in enabled_tools:
            @tool
            def query_prediction_data(
                plan_id: int,
                start_time: str = None,
                end_time: str = None,
                inference_batch_id: str = None,
                limit: int = 50
            ) -> Dict[str, Any]:
                """查询数据库中的预测数据,按时间范围、批次ID等条件查询

                Args:
                    plan_id: 计划ID
                    start_time: 开始时间(UTC+8), 格式: '2025-01-01 00:00:00'
                    end_time: 结束时间(UTC+8), 格式: '2025-01-01 23:59:59'
                    inference_batch_id: 批次ID
                    limit: 返回数量限制，默认50
                """
                try:
                    with get_db() as db:
                        query = db.query(PredictionData).filter(PredictionData.plan_id == plan_id)

                        # 时间范围查询
                        if start_time:
                            from datetime import datetime
                            start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                            query = query.filter(PredictionData.timestamp >= start_dt)
                        if end_time:
                            from datetime import datetime
                            end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                            query = query.filter(PredictionData.timestamp <= end_dt)

                        # 批次ID查询
                        if inference_batch_id:
                            query = query.filter(PredictionData.inference_batch_id == inference_batch_id)

                        # 限制数量并按时间倒序
                        predictions = query.order_by(PredictionData.timestamp.desc()).limit(limit).all()

                        return {
                            "success": True,
                            "count": len(predictions),
                            "data": [
                                {
                                    "timestamp": pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    "inference_batch_id": pred.inference_batch_id,
                                    "open": pred.open,
                                    "high": pred.high,
                                    "low": pred.low,
                                    "close": pred.close,
                                    "close_min": pred.close_min,
                                    "close_max": pred.close_max,
                                    "upward_probability": pred.upward_probability,
                                    "volatility_amplification_probability": pred.volatility_amplification_probability
                                } for pred in predictions
                            ]
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["query_prediction_data"] = query_prediction_data

        # 2. 查询历史预测批次工具
        if "get_prediction_history" in enabled_tools:
            @tool
            def get_prediction_history(plan_id: int, limit: int = 30) -> Dict[str, Any]:
                """查询历史预测数据,返回推理批次列表

                Args:
                    plan_id: 计划ID
                    limit: 返回批次数量，最多30个
                """
                try:
                    with get_db() as db:
                        # 获取不同的inference_batch_id
                        batches = db.query(PredictionData.inference_batch_id).filter(
                            PredictionData.plan_id == plan_id
                        ).distinct().order_by(PredictionData.inference_batch_id.desc()).limit(limit).all()

                        batch_ids = [batch[0] for batch in batches if batch[0]]

                        # 获取每个批次的详细信息
                        batch_info = []
                        for batch_id in batch_ids:
                            first_pred = db.query(PredictionData).filter(
                                PredictionData.plan_id == plan_id,
                                PredictionData.inference_batch_id == batch_id
                            ).first()

                            if first_pred:
                                batch_info.append({
                                    "inference_batch_id": batch_id,
                                    "created_at": first_pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                                    "prediction_count": db.query(PredictionData).filter(
                                        PredictionData.plan_id == plan_id,
                                        PredictionData.inference_batch_id == batch_id
                                    ).count()
                                })

                        return {
                            "success": True,
                            "total_batches": len(batch_info),
                            "data": batch_info
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["get_prediction_history"] = get_prediction_history

        # 3. 查询历史K线数据工具
        if "query_historical_kline_data" in enabled_tools:
            @tool
            def query_historical_kline_data(
                inst_id: str,
                interval: str = "1H",
                start_time: str = None,
                end_time: str = None,
                limit: int = 100
            ) -> Dict[str, Any]:
                """查询历史K线实际交易数据,使用UTC+8时间戳作为查询条件

                Args:
                    inst_id: 交易对，如 'ETH-USDT'
                    interval: 时间间隔，默认 '1H'
                    start_time: 开始时间(UTC+8), 格式: '2025-01-01 00:00:00'
                    end_time: 结束时间(UTC+8), 格式: '2025-01-01 23:59:59'
                    limit: 返回数量，默认100
                """
                try:
                    with get_db() as db:
                        from database.models import KlineData

                        query = db.query(KlineData).filter(
                            KlineData.inst_id == inst_id,
                            KlineData.interval == interval
                        )

                        # 时间范围查询
                        if start_time:
                            from datetime import datetime
                            start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                            query = query.filter(KlineData.timestamp >= start_dt)
                        if end_time:
                            from datetime import datetime
                            end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                            query = query.filter(KlineData.timestamp <= end_dt)

                        # 限制数量并按时间倒序
                        klines = query.order_by(KlineData.timestamp.desc()).limit(limit).all()

                        return {
                            "success": True,
                            "count": len(klines),
                            "data": [
                                {
                                    "timestamp": kline.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    "open": kline.open,
                                    "high": kline.high,
                                    "low": kline.low,
                                    "close": kline.close,
                                    "volume": kline.volume,
                                    "amount": kline.amount
                                } for kline in klines
                            ]
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["query_historical_kline_data"] = query_historical_kline_data

        # 4. 获取当前UTC+8时间工具
        if "get_current_utc_time" in enabled_tools:
            @tool
            def get_current_utc_time() -> Dict[str, Any]:
                """读取当前日期与时间(UTC+8),用于时间相关操作"""
                from datetime import datetime
                import pytz

                # 获取北京时区当前时间
                beijing_tz = pytz.timezone('Asia/Shanghai')
                current_time = datetime.now(beijing_tz)

                return {
                    "success": True,
                    "data": {
                        "current_time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "timestamp": current_time.timestamp(),
                        "timezone": "UTC+8"
                    }
                }

            available_tools["get_current_utc_time"] = get_current_utc_time

        # 5. 执行模型推理工具
        if "run_latest_model_inference" in enabled_tools:
            @tool
            def run_latest_model_inference(plan_id: int) -> Dict[str, Any]:
                """执行最新微调版本模型的预测推理"""
                try:
                    # 导入推理服务
                    from services.inference_service import inference_service

                    # 检查自动推理配置
                    if plan_info:
                        auto_inference_enabled = plan_info.auto_inference_enabled
                        auto_inference_interval = plan_info.auto_inference_interval_hours or 4

                        if not auto_inference_enabled:
                            return {
                                "success": False,
                                "error": "此计划未启用自动推理，请在计划配置中启用"
                            }

                        # 检查距离上次推理的时间间隔
                        if plan_info.last_finetune_time:
                            from datetime import datetime, timedelta
                            time_diff = datetime.now() - plan_info.last_finetune_time
                            hours_diff = time_diff.total_seconds() / 3600

                            if hours_diff < auto_inference_interval:
                                return {
                                    "success": False,
                                    "error": f"距离上次推理不足{auto_inference_interval}小时，请稍后再试"
                                }

                    # 执行推理
                    result = inference_service.run_inference(plan_id)

                    return {
                        "success": True,
                        "message": "预测推理已启动（正在后台执行）",
                        "inference_params": result.get("inference_params", {})
                    }

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["run_latest_model_inference"] = run_latest_model_inference

        # 6. 查询账户余额工具
        if "get_account_balance" in enabled_tools:
            @tool
            def get_account_balance(ccy: str = "USDT") -> Dict[str, Any]:
                """查询账户余额,返回可用余额、冻结余额等信息

                Args:
                    ccy: 币种，默认查询USDT
                """
                try:
                    if not plan_info or not all([plan_info.okx_api_key, plan_info.okx_secret_key, plan_info.okx_passphrase]):
                        return {"success": False, "error": "计划未配置OKX API密钥"}

                    # 创建OKX交易工具实例
                    trading_tools = OKXTradingTools(
                        api_key=plan_info.okx_api_key,
                        secret_key=plan_info.okx_secret_key,
                        passphrase=plan_info.okx_passphrase,
                        is_demo=plan_info.is_demo
                    )

                    # 调用OKX API
                    result = trading_tools.get_account_balance(ccy=ccy)

                    return result

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["get_account_balance"] = get_account_balance

        # 7. 查询未成交订单工具
        if "get_pending_orders" in enabled_tools:
            @tool
            def get_pending_orders(
                inst_id: str,
                state: str = "live",
                limit: int = 300
            ) -> Dict[str, Any]:
                """查询当前所有OKX未成交订单(挂单)信息

                Args:
                    inst_id: 交易对ID，如 'ETH-USDT'
                    state: 订单状态，'live': 等待成交, 'partially_filled': 部分成交
                    limit: 返回数量限制，默认300
                """
                try:
                    if not plan_info or not all([plan_info.okx_api_key, plan_info.okx_secret_key, plan_info.okx_passphrase]):
                        return {"success": False, "error": "计划未配置OKX API密钥"}

                    # 创建OKX交易工具实例
                    trading_tools = OKXTradingTools(
                        api_key=plan_info.okx_api_key,
                        secret_key=plan_info.okx_secret_key,
                        passphrase=plan_info.okx_passphrase,
                        is_demo=plan_info.is_demo
                    )

                    # 调用OKX API
                    result = trading_tools.get_pending_orders(
                        inst_id=inst_id,
                        state=state,
                        limit=limit
                    )

                    return result

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["get_pending_orders"] = get_pending_orders

        # 8. 下限价单工具
        if "place_order" in enabled_tools:
            @tool
            def place_order(
                inst_id: str,
                side: str,
                sz: str,
                px: str,
                cl_ord_id: str = None
            ) -> Dict[str, Any]:
                """下限价单,以指定价格买入或卖出

                Args:
                    inst_id: 交易对ID，如 'ETH-USDT'
                    side: 订单方向，'buy': 买, 'sell': 卖
                    sz: 委托数量
                    px: 委托价格
                    cl_ord_id: 客户端订单ID，如不提供将自动生成
                """
                try:
                    if not plan_info or not all([plan_info.okx_api_key, plan_info.okx_secret_key, plan_info.okx_passphrase]):
                        return {"success": False, "error": "计划未配置OKX API密钥"}

                    # 创建OKX交易工具实例
                    trading_tools = OKXTradingTools(
                        api_key=plan_info.okx_api_key,
                        secret_key=plan_info.okx_secret_key,
                        passphrase=plan_info.okx_passphrase,
                        is_demo=plan_info.is_demo
                    )

                    # 调用OKX API
                    result = trading_tools.place_order(
                        inst_id=inst_id,
                        side=side,
                        td_mode="isolated",
                        ord_type="limit",
                        sz=sz,
                        px=px,
                        cl_ord_id=cl_ord_id,
                        tag="kokexAgent"
                    )

                    return result

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["place_order"] = place_order

        # 9. 撤销订单工具
        if "cancel_order" in enabled_tools:
            @tool
            def cancel_order(inst_id: str, cl_ord_id: str) -> Dict[str, Any]:
                """撤销未成交的订单,冻结资金将立即释放

                Args:
                    inst_id: 交易对ID，如 'ETH-USDT'
                    cl_ord_id: 客户端订单ID
                """
                try:
                    if not plan_info or not all([plan_info.okx_api_key, plan_info.okx_secret_key, plan_info.okx_passphrase]):
                        return {"success": False, "error": "计划未配置OKX API密钥"}

                    # 创建OKX交易工具实例
                    trading_tools = OKXTradingTools(
                        api_key=plan_info.okx_api_key,
                        secret_key=plan_info.okx_secret_key,
                        passphrase=plan_info.okx_passphrase,
                        is_demo=plan_info.is_demo
                    )

                    # 调用OKX API
                    result = trading_tools.cancel_order(
                        inst_id=inst_id,
                        cl_ord_id=cl_ord_id
                    )

                    return result

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["cancel_order"] = cancel_order

        # 10. 修改订单工具
        if "amend_order" in enabled_tools:
            @tool
            def amend_order(
                inst_id: str,
                cl_ord_id: str,
                new_sz: str = None,
                new_px: str = None,
                req_id: str = None
            ) -> Dict[str, Any]:
                """修改未成交订单的价格或数量

                Args:
                    inst_id: 交易对ID，如 'ETH-USDT'
                    cl_ord_id: 客户端订单ID
                    new_sz: 修改的新数量，必须大于0
                    new_px: 修改后的新价格
                    req_id: 用户自定义修改事件ID
                """
                try:
                    if not plan_info or not all([plan_info.okx_api_key, plan_info.okx_secret_key, plan_info.okx_passphrase]):
                        return {"success": False, "error": "计划未配置OKX API密钥"}

                    # 创建OKX交易工具实例
                    trading_tools = OKXTradingTools(
                        api_key=plan_info.okx_api_key,
                        secret_key=plan_info.okx_secret_key,
                        passphrase=plan_info.okx_passphrase,
                        is_demo=plan_info.is_demo
                    )

                    # 调用OKX API
                    result = trading_tools.amend_order(
                        inst_id=inst_id,
                        cl_ord_id=cl_ord_id,
                        new_sz=new_sz,
                        new_px=new_px,
                        req_id=req_id
                    )

                    return result

                except Exception as e:
                    return {"success": False, "error": str(e)}

            available_tools["amend_order"] = amend_order

        return available_tools

    async def _save_message(self, conversation_id: int, role: str, content: str,
                           message_type: str = "text", tool_name: str = None,
                           tool_arguments: Dict = None, tool_result: Dict = None):
        """保存消息到数据库"""
        try:
            with get_db() as db:
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    message_type=message_type,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    tool_result=tool_result,
                    timestamp=now_beijing()
                )
                db.add(message)

                # 更新会话统计
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).first()
                if conversation:
                    conversation.total_messages += 1
                    if message_type in ["tool_call", "tool_result"]:
                        conversation.total_tool_calls += 1
                    conversation.last_message_at = now_beijing()

                db.commit()
        except Exception as e:
            logger.error(f"保存消息失败: {e}")

    async def stream_agent_response_real(
        self,
        plan_id: int,
        user_message: str = None,
        conversation_type: str = "manual_chat"
    ):
        """真正的Agent响应流，支持新的消息格式（think模式、tool调用、play结果）"""
        try:
            # 获取计划配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield [{"role": "assistant", "content": "❌ 未找到指定计划"}]
                    return

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    yield [{"role": "assistant", "content": "❌ 未找到LLM配置"}]
                    return

            # 创建或获取对话
            with get_db() as db:
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

            # 获取LLM客户端和工具
            llm = self._get_llm_client(llm_config)
            tools_config = plan.agent_tools_config if isinstance(plan.agent_tools_config, dict) else json.loads(plan.agent_tools_config) if plan.agent_tools_config else {}
            tools = self._create_langchain_tools(tools_config, plan_id)

            # 绑定工具到LLM
            tools_list = list(tools.values())
            if tools_list:
                llm_with_tools = llm.bind_tools(tools_list)
            else:
                llm_with_tools = llm

            # 构建完整提示词（动态部分 + 工具部分 + 交易限制）
            system_prompt = self._build_system_prompt(plan, tools_config, plan_id)

            # 输出系统消息
            yield [{"role": "system", "content": system_prompt}]
            await self._save_message(conversation.id, MessageRole.SYSTEM.value, system_prompt, MessageType.TEXT.value)

            # 获取用户输入消息
            if conversation_type == "auto_inference":
                # 自动推理模式：获取最新预测数据
                prediction_data = self._get_prediction_data_for_context(plan_id)
                input_message = prediction_data
            else:
                # 手动聊天模式：使用用户消息
                input_message = user_message

            # 输出用户消息
            yield [{"role": "user", "content": input_message}]
            await self._save_message(conversation.id, MessageRole.USER.value, input_message, MessageType.TEXT.value)

            # 构建LangChain消息序列
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input_message)
            ]

            # 检查是否启用Qwen think模式
            enable_thinking = False
            if llm_config.provider == "qwen":
                # 检查 extra_params 中是否有 enable_thinking 配置
                if hasattr(llm_config, 'extra_params') and llm_config.extra_params:
                    try:
                        extra_params = llm_config.extra_params if isinstance(llm_config.extra_params, dict) else json.loads(llm_config.extra_params)
                        enable_thinking = extra_params.get('enable_thinking', False)
                    except:
                        pass

            # 流式调用LLM
            current_content = ""
            thinking_content = ""
            in_thinking_mode = False
            tool_calls_count = 0
            play_results = []

            async for chunk in llm_with_tools.astream(messages):
                # 处理思考模式（Qwen）
                if enable_thinking and hasattr(chunk, 'content'):
                    content = chunk.content or ""

                    # 检测思考标签
                    if "<think>" in content and not in_thinking_mode:
                        in_thinking_mode = True
                        thinking_content = content.split("<think>", 1)[1]
                        continue
                    elif "</think>" in content and in_thinking_mode:
                        # 结束思考模式
                        remaining_thinking = thinking_content + content.split("</think>", 1)[0]
                        if remaining_thinking.strip():
                            # 输出完整思考内容
                            yield [{"role": "assistant", "content": f"💭 **思考过程**:\n{remaining_thinking}"}]
                            await self._save_message(conversation.id, MessageRole.ASSISTANT.value,
                                                   remaining_thinking, MessageType.THINKING.value)
                        thinking_content = ""
                        in_thinking_mode = False
                        # 处理思考后的内容
                        after_think = content.split("</think>", 1)[1]
                        if after_think.strip():
                            yield [{"role": "assistant", "content": after_think}]
                            await self._save_message(conversation.id, MessageRole.ASSISTANT.value, after_think)
                        continue
                    elif in_thinking_mode:
                        # 在思考模式中，累积内容
                        thinking_content += content
                        continue

                # 处理工具调用
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        tool_calls_count += 1
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})

                        # 输出工具调用（独立消息气泡）
                        tool_call_data = {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "status": "calling"
                        }
                        yield [{"role": "tool", "content": json.dumps(tool_call_data, ensure_ascii=False)}]
                        await self._save_message(conversation.id, MessageRole.TOOL.value,
                                               f"调用工具 {tool_name}", MessageType.TOOL_CALL.value,
                                               tool_name, tool_args)

                        # 执行工具
                        try:
                            tool_func = next((t for t in tools_list if t.name == tool_name), None)
                            if tool_func:
                                result = tool_func.invoke(tool_args)

                                tool_result_data = {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": result,
                                    "status": "success" if not (isinstance(result, dict) and "error" in result) else "error"
                                }

                                # 输出工具结果（独立消息气泡）
                                yield [{"role": "tool", "content": json.dumps(tool_result_data, ensure_ascii=False)}]
                                await self._save_message(conversation.id, MessageRole.TOOL.value,
                                                       f"工具 {tool_name} 执行完成", MessageType.TOOL_RESULT.value,
                                                       tool_name, tool_args, result)

                                # 检查是否是投资决策结果
                                if tool_name in ["place_order", "cancel_order", "amend_order"]:
                                    play_results.append({
                                        "action": tool_name,
                                        "parameters": tool_args,
                                        "result": result,
                                        "timestamp": now_beijing().isoformat()
                                    })

                                # 将工具结果添加到消息中
                                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call.get("id", "")))

                        except Exception as tool_error:
                            error_data = {
                                "tool_name": tool_name,
                                "arguments": tool_args,
                                "error": str(tool_error),
                                "status": "error"
                            }
                            yield [{"role": "tool", "content": json.dumps(error_data, ensure_ascii=False)}]
                            await self._save_message(conversation.id, MessageRole.TOOL.value,
                                                   f"工具 {tool_name} 执行失败", MessageType.TOOL_RESULT.value,
                                                   tool_name, tool_args, {"error": str(tool_error)})

                # 处理普通文本内容
                if hasattr(chunk, 'content') and chunk.content and not in_thinking_mode:
                    current_content += chunk.content
                    if chunk.content.strip():
                        yield [{"role": "assistant", "content": chunk.content}]
                        await self._save_message(conversation.id, MessageRole.ASSISTANT.value, chunk.content)

            # 输出投资决策结果（如果有）
            if play_results:
                play_data = {
                    "investment_decisions": play_results,
                    "total_decisions": len(play_results),
                    "session_id": conversation.id,
                    "timestamp": now_beijing().isoformat()
                }
                yield [{"role": "play", "content": json.dumps(play_data, ensure_ascii=False)}]
                await self._save_message(conversation.id, MessageRole.PLAY.value,
                                       json.dumps(play_data, ensure_ascii=False), MessageType.PLAY_RESULT.value)

            # 完成对话
            completion_msg = f"✅ 对话完成，共调用 {tool_calls_count} 个工具"
            yield [{"role": "assistant", "content": completion_msg}]
            await self._save_message(conversation.id, MessageRole.ASSISTANT.value, completion_msg)

            # 更新对话状态
            with get_db() as db:
                conversation = db.query(AgentConversation).filter(AgentConversation.id == conversation.id).first()
                if conversation:
                    conversation.status = 'completed'
                    conversation.completed_at = now_beijing()
                    db.commit()

        except Exception as e:
            logger.error(f"Agent流式响应失败: {e}")
            import traceback
            traceback.print_exc()
            yield [{"role": "assistant", "content": f"❌ Agent执行失败: {str(e)}"}]

    def _build_system_prompt(self, plan: TradingPlan, tools_config: Dict[str, bool], plan_id: int) -> str:
        """构建完整的系统提示词（动态部分 + 工具部分 + 交易限制）"""

        # 1. 动态部分（来自 agent 配置）
        dynamic_prompt = plan.agent_prompt or "你是一个专业的加密货币交易AI助手。"

        # 2. 工具部分（根据可用工具动态生成）
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]
        tools_description = []

        if "query_prediction_data" in enabled_tools:
            tools_description.append("- **query_prediction_data**: 查询预测数据，支持时间范围和批次筛选")
        if "get_prediction_history" in enabled_tools:
            tools_description.append("- **get_prediction_history**: 查询历史预测批次信息")
        if "query_historical_kline_data" in enabled_tools:
            tools_description.append("- **query_historical_kline_data**: 查询历史K线实际数据")
        if "get_current_utc_time" in enabled_tools:
            tools_description.append("- **get_current_utc_time**: 获取当前UTC+8时间")
        if "run_latest_model_inference" in enabled_tools:
            tools_description.append("- **run_latest_model_inference**: 执行最新模型推理")
        if "get_account_balance" in enabled_tools:
            tools_description.append("- **get_account_balance**: 查询账户余额")
        if "get_pending_orders" in enabled_tools:
            tools_description.append("- **get_pending_orders**: 查询未成交订单")
        if "place_order" in enabled_tools:
            tools_description.append("- **place_order**: 下限价单（买入/卖出）")
        if "cancel_order" in enabled_tools:
            tools_description.append("- **cancel_order**: 撤销订单")
        if "amend_order" in enabled_tools:
            tools_description.append("- **amend_order**: 修改订单价格或数量")

        tools_section = f"## 可用工具\n{chr(10).join(tools_description)}" if tools_description else "## 可用工具\n当前未启用任何工具"

        # 3. 交易限制（从计划配置中获取）
        trading_limits = []
        if hasattr(plan, 'trading_limits') and plan.trading_limits:
            try:
                limits = plan.trading_limits if isinstance(plan.trading_limits, dict) else json.loads(plan.trading_limits)
                if limits:
                    trading_limits.append("### 交易限制")
                    if 'max_position_size' in limits:
                        trading_limits.append(f"- 最大持仓规模: {limits['max_position_size']}")
                    if 'max_order_value' in limits:
                        trading_limits.append(f"- 最大订单价值: {limits['max_order_value']}")
                    if 'max_daily_trades' in limits:
                        trading_limits.append(f"- 每日最大交易次数: {limits['max_daily_trades']}")
                    if 'risk_percentage' in limits:
                        trading_limits.append(f"- 风险控制比例: {limits['risk_percentage']}%")
            except:
                pass

        trading_limits_section = chr(10).join(trading_limits) if trading_limits else ""

        # 构建完整提示词
        full_prompt = f"""{dynamic_prompt}

## 计划信息
- 当前计划ID: {plan_id}
- 交易对: {plan.inst_id}
- 时间颗粒度: {plan.time_granularity}
- 环境: {'模拟盘' if plan.is_demo else '实盘'}

{tools_section}

{trading_limits_section}

## 重要提示
- 所有时间都使用UTC+8时区（北京时间）
- 使用需要plan_id参数的工具时，请使用: {plan_id}
- 严格遵守交易限制和风险控制要求
- 每次决策前请先分析当前市场状况
- 交易前请确认账户余额和风险承受能力

## 决策流程
1. 分析当前市场状况和最新数据
2. 获取必要的历史数据和预测信息
3. 评估风险和交易机会
4. 如需交易，调用相应的交易工具
5. 记录决策理由和结果

请基于以上信息提供专业的交易分析和建议。"""

        return full_prompt

    def _get_prediction_data_for_context(self, plan_id: int, limit: int = 20) -> str:
        """获取预测数据作为上下文"""
        try:
            with get_db() as db:
                predictions = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id
                ).order_by(PredictionData.timestamp.desc()).limit(limit).all()

                if not predictions:
                    return "暂无预测数据"

                # 格式化为CSV字符串
                csv_lines = ["timestamp,open,high,low,close,upward_probability,volatility_amplification_probability"]
                for pred in predictions:
                    csv_lines.append(
                        f"{pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
                        f"{pred.open},{pred.high},{pred.low},{pred.close},"
                        f"{pred.upward_probability or 0},{pred.volatility_amplification_probability or 0}"
                    )

                return "\\n".join(csv_lines)

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return f"获取预测数据失败: {str(e)}"

    async def stream_conversation(self, plan_id: int, user_message: str):
        """流式对话接口"""
        async for message_batch in self.stream_agent_response_real(
            plan_id=plan_id,
            user_message=user_message,
            conversation_type=ConversationType.MANUAL_CHAT.value
        ):
            yield message_batch

    async def stream_auto_inference(self, plan_id: int):
        """自动推理流式接口"""
        async for message_batch in self.stream_agent_response_real(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE.value
        ):
            yield message_batch


# 全局实例
langchain_agent_v2_service = LangChainAgentV2Service()
