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
import pytz
from typing import Dict, List, AsyncGenerator, Optional, Any
from datetime import datetime
from sqlalchemy import and_, desc

from database.models import (
    TradingPlan, AgentConversation, AgentMessage,
    LLMConfig, TrainingRecord, PredictionData, now_beijing
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

    @staticmethod
    def _parse_extra_params(extra_params):
        """解析额外参数"""
        if not extra_params:
            return {}
        try:
            return extra_params if isinstance(extra_params, dict) else json.loads(extra_params)
        except:
            return {}

    @staticmethod
    def _get_llm_base_params(llm_config):
        """获取LLM基础参数"""
        return {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature or 0.7,
            "max_tokens": llm_config.max_tokens or 2000
        }

    @staticmethod
    def _format_tool_response(success: bool, data=None, error=None, **kwargs):
        """统一的工具响应格式"""
        response = {"success": success}
        if success:
            response.update(data or {})
            response["timestamp"] = now_beijing().isoformat()
        else:
            response["error"] = error
        return json.dumps(response, ensure_ascii=False)

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

    def get_plan_trading_tools(self, plan_id: int):
        """获取计划特定的交易工具"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return None

                # 使用计划特定的API密钥
                return OKXTradingTools(
                    api_key=plan.okx_api_key,
                    secret_key=plan.okx_secret_key,
                    passphrase=plan.okx_passphrase,
                    is_demo=plan.is_demo,
                    trading_limits=plan.trading_limits,
                    plan_id=plan_id  # 传递计划ID用于订单存储
                )

        except Exception as e:
            logger.error(f"获取计划特定交易工具失败: {e}")
            return None

    def _get_llm_client(self, llm_config: LLMConfig):
        """获取 LLM 客户端"""
        client_key = f"{llm_config.provider}_{llm_config.model_name}"

        if client_key not in self._llm_clients:
            base_params = self._get_llm_base_params(llm_config)

            if llm_config.provider == "openai":
                self._llm_clients[client_key] = ChatOpenAI(
                    **base_params,
                    openai_api_key=llm_config.api_key
                )
            elif llm_config.provider == "anthropic":
                self._llm_clients[client_key] = ChatAnthropic(
                    **base_params,
                    anthropic_api_key=llm_config.api_key
                )
            elif llm_config.provider == "qwen":
                # Qwen 使用 OpenAI 兼容接口
                base_url = llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                extra_params = self._parse_extra_params(getattr(llm_config, 'extra_params', None))

                model_kwargs = {}
                if extra_params.get('enable_thinking', False):
                    model_kwargs = {"enable_thinking": True}

                self._llm_clients[client_key] = ChatOpenAI(
                    **base_params,
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

        # 获取计划特定的交易工具
        plan_trading_tools = self.get_plan_trading_tools(plan_id)
        if not plan_trading_tools:
            logger.error(f"无法获取计划 {plan_id} 的交易工具")
            return list(available_tools.values())

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
                    price = plan_trading_tools.get_current_price(inst_id)
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

        # 2. 获取当前北京时间 (UTC+8)
        if "get_current_utc_time" in enabled_tools:
            @tool
            def get_current_utc_time() -> str:
                """获取当前北京时间 (UTC+8)"""
                # 获取北京时间
                beijing_tz = pytz.timezone('Asia/Shanghai')
                beijing_time = datetime.now(beijing_tz)
                return json.dumps({
                    "success": True,
                    "current_time": beijing_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "timezone": "UTC+8"
                }, ensure_ascii=False)

            available_tools["get_current_utc_time"] = get_current_utc_time

        # 3. 查询持仓
        if "get_positions" in enabled_tools:
            @tool
            def get_positions() -> str:
                """查询当前持仓"""
                try:
                    positions = plan_trading_tools.get_positions()
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
                    result = plan_trading_tools.place_order(
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
                    result = plan_trading_tools.cancel_order(inst_id, order_id)
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
                    limits = plan_trading_tools.get_trading_limits()
                    return json.dumps({
                        "success": True,
                        "limits": limits,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询交易限制失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_trading_limits"] = get_trading_limits

        # 7. 查询账户余额
        if "get_account_balance" in enabled_tools:
            @tool
            def get_account_balance(ccy: str = None) -> str:
                """查询账户余额"""
                try:
                    balance = plan_trading_tools.get_account_balance(ccy)
                    return json.dumps({
                        "success": True,
                        "balance": balance,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询账户余额失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_account_balance"] = get_account_balance

        # 8. 查询待成交订单
        if "get_pending_orders" in enabled_tools:
            @tool
            def get_pending_orders(inst_id: str = None, state: str = "live") -> str:
                """查询待成交订单"""
                try:
                    orders = plan_trading_tools.get_pending_orders(inst_id, state)
                    return json.dumps({
                        "success": True,
                        "orders": orders,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询待成交订单失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_pending_orders"] = get_pending_orders

        # 9. 修改订单
        if "amend_order" in enabled_tools:
            @tool
            def amend_order(inst_id: str, order_id: str, new_size: str = None, new_price: str = None) -> str:
                """修改订单"""
                try:
                    result = plan_trading_tools.amend_order_with_db_save(
                        inst_id=inst_id,
                        order_id=order_id,
                        new_size=new_size,
                        new_price=new_price
                    )
                    return json.dumps({
                        "success": True,
                        "result": result,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"修改订单失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["amend_order"] = amend_order

        # 10. 查询历史K线数据
        if "query_historical_kline_data" in enabled_tools:
            @tool
            def query_historical_kline_data(
                inst_id: str,
                interval: str = '1H',
                start_time: str = None,
                end_time: str = None,
                limit: int = 100
            ) -> str:
                """查询历史K线数据

                Args:
                    inst_id: 交易对标识符，如 'ETH-USDT'
                    interval: K线周期，支持 '1m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M', '3M', '6M', '1Y'
                    start_time: 开始时间，ISO格式字符串，如 '2024-01-01T00:00:00Z'
                    end_time: 结束时间，ISO格式字符串，如 '2024-12-31T23:59:59Z'
                    limit: 返回数据条数，默认100，最大1000
                """
                try:
                    # 参数验证
                    if not inst_id or not isinstance(inst_id, str):
                        return json.dumps({"success": False, "error": "inst_id必须是非空字符串"}, ensure_ascii=False)

                    valid_intervals = ['1m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M', '3M', '6M', '1Y']
                    if interval not in valid_intervals:
                        return json.dumps({"success": False, "error": f"interval必须是以下值之一: {valid_intervals}"}, ensure_ascii=False)

                    if limit and (not isinstance(limit, int) or limit <= 0 or limit > 1000):
                        return json.dumps({"success": False, "error": "limit必须是1-1000之间的整数"}, ensure_ascii=False)

                    from services.trading_tools import OKXTradingTools
                    kline_data = plan_trading_tools.query_historical_kline_data(
                        inst_id=inst_id,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time,
                        limit=limit
                    )
                    return json.dumps({
                        "success": True,
                        "data": kline_data,
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询历史K线数据失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["query_historical_kline_data"] = query_historical_kline_data

        # 11. 执行模型推理
        if "run_latest_model_inference" in enabled_tools:
            @tool
            def run_latest_model_inference() -> str:
                """执行最新的AI模型推理"""
                try:
                    from services.inference_service import InferenceService
                    inference_service = InferenceService()
                    # 使用同步方法避免异步调用问题
                    result = inference_service.run_inference_async(plan_id)
                    import asyncio
                    # 如果返回的是协程，需要等待
                    if asyncio.iscoroutine(result):
                        result = asyncio.run(result)
                    return json.dumps({
                        "success": True,
                        "message": "模型推理已启动",
                        "result": str(result),
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"执行模型推理失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["run_latest_model_inference"] = run_latest_model_inference

        # 12. 查询预测数据
        if "query_prediction_data" in enabled_tools:
            @tool
            def query_prediction_data(limit: int = 50) -> str:
                """查询AI模型预测数据"""
                try:
                    # 参数验证
                    if not isinstance(limit, int) or limit <= 0 or limit > 500:
                        return json.dumps({"success": False, "error": "limit必须是1-500之间的整数"}, ensure_ascii=False)

                    from database.models import PredictionData
                    with get_db() as db:
                        predictions = db.query(PredictionData).filter(
                            PredictionData.plan_id == plan_id
                        ).order_by(PredictionData.timestamp.desc()).limit(limit).all()

                        data = []
                        for pred in predictions:
                            data.append({
                                "timestamp": pred.timestamp.isoformat(),
                                "close": pred.close,
                                "close_min": pred.close_min,
                                "close_max": pred.close_max,
                                "upward_probability": pred.upward_probability
                            })

                    return json.dumps({
                        "success": True,
                        "data": data,
                        "count": len(data),
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"查询预测数据失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["query_prediction_data"] = query_prediction_data

        # 13. 获取预测历史
        if "get_prediction_history" in enabled_tools:
            @tool
            def get_prediction_history(limit: int = 20) -> str:
                """获取历史预测记录"""
                try:
                    # 参数验证
                    if not isinstance(limit, int) or limit <= 0 or limit > 100:
                        return json.dumps({"success": False, "error": "limit必须是1-100之间的整数"}, ensure_ascii=False)

                    from database.models import TrainingRecord, PredictionData
                    with get_db() as db:
                        records = db.query(TrainingRecord).filter(
                            TrainingRecord.plan_id == plan_id,
                            TrainingRecord.status == 'completed'
                        ).order_by(TrainingRecord.created_at.desc()).limit(limit).all()

                        data = []
                        for record in records:
                            # 获取该训练记录的预测数据数量
                            pred_count = db.query(PredictionData).filter(
                                PredictionData.training_record_id == record.id
                            ).count()

                            data.append({
                                "training_id": record.id,
                                "model_version": record.version,
                                "created_at": record.created_at.isoformat(),
                                "prediction_count": pred_count,
                                "status": record.status
                            })

                    return json.dumps({
                        "success": True,
                        "data": data,
                        "count": len(data),
                        "timestamp": now_beijing().isoformat()
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"获取预测历史失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_prediction_history"] = get_prediction_history

        return list(available_tools.values())

    def _detect_qwen_thinking(self, content: str, llm_config: LLMConfig = None) -> bool:
        """检测 Qwen 思考模式的双重策略"""
        if not content or not content.strip():
            return False

        # 策略1: 内容检测
        thinking_indicators = [
            "思考:", "让我分析", "首先", "接下来", "综合考虑",
            "分析结果", "判断", "决策", "建议", "根据",
            "思考：", "考虑到", "从市场角度看", "技术分析"
        ]

        if any(indicator in content for indicator in thinking_indicators):
            return True

        # 策略2: Agent层级配置检测
        if llm_config and llm_config.provider == "qwen":
            if hasattr(llm_config, 'extra_params') and llm_config.extra_params:
                try:
                    extra_params = llm_config.extra_params if isinstance(llm_config.extra_params, dict) else json.loads(llm_config.extra_params)
                    if extra_params.get('enable_thinking', False):
                        return True
                except:
                    pass

        return False

    def _build_system_prompt(self, plan: TradingPlan, tools_config: Dict[str, bool]) -> str:
        """构建系统提示词 - 三部分结构"""
        # 第一部分：动态用户提示词
        dynamic_prompt = plan.agent_prompt or "你是一个专业的加密货币交易AI助手。"

        # 第二部分：可用工具描述
        tools_desc = []
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        tool_descriptions = {
            "get_current_price": "获取交易对的当前价格",
            "get_current_utc_time": "获取当前北京时间 (UTC+8)",
            "get_positions": "查询当前持仓信息",
            "place_order": "下单交易（买入或卖出）",
            "cancel_order": "取消订单",
            "get_trading_limits": "查询交易限制",
            "amend_order": "修改订单（调整价格或数量）",
            "query_historical_kline_data": "查询历史K线数据（支持多种时间周期和自定义时间范围）",
            "get_pending_orders": "查询待成交订单列表",
            "get_account_balance": "查询账户余额信息",
            "query_prediction_data": "查询AI模型预测数据",
            "get_prediction_history": "获取历史预测记录",
            "run_latest_model_inference": "执行最新的AI模型推理",
            "query_historical_kline_data": "查询历史K线数据"
        }

        for tool_name in enabled_tools:
            if tool_name in tool_descriptions:
                tools_desc.append(f"- {tool_name}: {tool_descriptions[tool_name]}")

        # 第三部分：交易限制和计划信息
        limits_desc = ""
        if plan.trading_limits:
            try:
                limits = plan.trading_limits if isinstance(plan.trading_limits, dict) else json.loads(plan.trading_limits)
                if limits:
                    # 生成友好的交易限制提示词文本
                    limits_text = self._build_trading_limits_prompt(limits)
                    limits_desc = f"\n\n交易限制配置：\n{limits_text}"
            except:
                pass

        # 构建完整的系统提示词
        system_prompt = f"""{dynamic_prompt}

可用工具：
{chr(10).join(tools_desc) if tools_desc else "无可用工具"}

交易计划信息：
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 初始本金: {plan.initial_capital} USDT
{limits_desc}

请根据以上要求，并使用相应的工具来完成交易决策操作。"""

        return system_prompt

    def _build_trading_limits_prompt(self, limits: Dict) -> str:
        """构建交易限制的提示词文本"""
        limits_parts = []

        # 可用资金 (USDT)
        available_usdt = limits.get('available_usdt_amount', 0)
        if available_usdt > 0:
            limits_parts.append(f"- 可用资金: {available_usdt} USDT (下单买入时使用)")

        # 资金比例 (%)
        usdt_percentage = limits.get('available_usdt_percentage', 0)
        if usdt_percentage > 0:
            limits_parts.append(f"- 资金比例: {usdt_percentage}% (如果可用资金不足，则使用账户可用资金百分比计算)")

        # 平摊单量
        avg_orders = limits.get('avg_order_count', 1)
        if avg_orders > 0:
            limits_parts.append(f"- 平摊单量: {avg_orders} 笔 (挂单限制，最多未成交订单数)")

        # 止损比例 (%)
        stop_loss = limits.get('stop_loss_percentage', 0)
        if stop_loss > 0:
            limits_parts.append(f"- 止损比例: {stop_loss}% (如果买入后价格低于预期，触发挂单调价)")

        if not limits_parts:
            return "- 未设置特殊交易限制"

        return "\n".join(limits_parts)

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
            # 对于不同类型采用不同策略
            if conversation_type == 'auto_inference':
                # 自动推理总是创建新对话，不复用
                conversation = None
            elif conversation_type == "inference_session":
                # 推理会话每次都创建新会话（重置上下文）
                conversation = None
            else:
                # 其他类型尝试复用现有对话
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id,
                    AgentConversation.status == 'active',
                    AgentConversation.conversation_type == conversation_type
                ).first()

            # 如果没有现有对话，创建新对话
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

        # 检查是否为新创建的对话
        is_new_conversation = (conversation.created_at == conversation.last_message_at)

        # 构建系统提示词
        tools_config = plan.agent_tools_config or {}
        system_prompt = self._build_system_prompt(plan, tools_config)

        # 如果是新对话，输出系统提示词
        if is_new_conversation:
            # 输出系统消息 - 使用用户要求的 "System:" 格式
            yield [{"role": "system", "content": system_prompt}]

            # 保存系统消息到数据库
            try:
                with get_db() as db:
                    await self._save_message(
                        db, conversation.id, "system", system_prompt, "text"
                    )
            except Exception as e:
                logger.error(f"保存系统消息失败: {e}")
        else:
            # 加载历史消息
            yield await self._load_conversation_history(conversation.id)

        # 输出用户消息
        yield [{"role": "user", "content": user_message}]
        try:
            with get_db() as db:
                await self._save_message(
                    db, conversation.id, "user", user_message, "text"
                )
        except Exception as e:
            logger.error(f"保存用户消息失败: {e}")

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
                logger.info(f"PLAN {plan_id} - 开始Agent流式执行，LLM: {llm_config.model_name}")
                logger.debug(f"PLAN {plan_id} - 输入消息长度: {len(user_message)} 字符")

                chunk_count = 0
                async for chunk in agent_executor.astream({"input": user_message, "chat_history": messages[1:-1]}):
                    chunk_count += 1
                    logger.debug(f"PLAN {plan_id} - Agent chunk #{chunk_count}: {type(chunk)} - {list(chunk.keys()) if isinstance(chunk, dict) else str(chunk)[:100]}")
                    # 处理工具调用
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            tool_name = getattr(action, 'tool', 'unknown')
                            tool_input = getattr(action, 'tool_input', {})

                            # 生成工具调用ID并记录开始时间
                            import uuid
                            import time
                            tool_call_id = str(uuid.uuid4())[:8]
                            tool_start_time = time.time()

                            # 为交易工具设置上下文信息
                            plan_trading_tools = self.get_plan_trading_tools(plan_id)
                            if plan_trading_tools:
                                plan_trading_tools.set_tool_context(
                                    conversation_id=conversation.id,
                                    tool_call_id=tool_call_id
                                )

                            # 输出工具调用 - 使用新的 role:tool_call
                            tool_call_data = {
                                "tool_name": tool_name,
                                "arguments": tool_input,
                                "status": "calling",
                                "tool_call_id": tool_call_id
                            }
                            tool_call_content = json.dumps(tool_call_data, ensure_ascii=False)
                            logger.info(f"PLAN {plan_id} - 工具调用: {tool_name}, ID: {tool_call_id}")
                            logger.debug(f"PLAN {plan_id} - 工具调用参数: {tool_input}")
                            yield [{"role": "tool_call", "content": tool_call_content}]

                            # 保存工具调用到数据库
                            with get_db() as db:
                                await self._save_message(
                                    db, conversation.id, "tool",
                                    f"调用工具 {tool_name}", "tool_call",
                                    tool_name=tool_name,
                                    tool_args=tool_input,
                                    tool_call_id=tool_call_id,
                                    tool_execution_time=None  # 调用时暂不记录时间
                                )

                    # 处理工具结果
                    if "steps" in chunk:
                        for step in chunk["steps"]:
                            if hasattr(step, 'observation') and step.observation:
                                obs = step.observation
                                tool_name = getattr(step.action, 'tool', 'unknown') if hasattr(step, 'action') else 'unknown'

                                # 计算工具执行时间
                                tool_execution_time = time.time() - tool_start_time

                                # 格式化工具结果 - 使用新的 role:tool_result
                                try:
                                    tool_params = getattr(step.action, 'tool_input', {})

                                    # 尝试解析结果
                                    if isinstance(obs, str) and obs.startswith('{'):
                                        try:
                                            result_data = json.loads(obs)
                                            result = result_data
                                        except:
                                            result = {"raw_result": obs}
                                    else:
                                        result = {"raw_result": obs}

                                    # 创建工具结果数据
                                    tool_result_data = {
                                        "tool_name": tool_name,
                                        "arguments": tool_params,
                                        "result": result,
                                        "status": "success" if not obs.startswith("ERROR") else "error"
                                    }

                                    tool_result_content = json.dumps(tool_result_data, ensure_ascii=False)
                                    logger.info(f"PLAN {plan_id} - 工具结果: {tool_name}, 状态: {tool_result_data['status']}")
                                    logger.debug(f"PLAN {plan_id} - 工具结果长度: {len(tool_result_content)} 字符")
                                    yield [{"role": "tool_result", "content": tool_result_content}]

                                except Exception as e:
                                    # 错误情况下也返回结构化数据
                                    error_data = {
                                        "tool_name": tool_name,
                                        "arguments": getattr(step.action, 'tool_input', {}),
                                        "result": {"error": str(e)},
                                        "status": "error"
                                    }
                                    tool_error_content = json.dumps(error_data, ensure_ascii=False)
                                    yield [{"role": "tool_result", "content": tool_error_content}]

                                # 保存工具结果到数据库
                                related_order_id = None
                                if tool_name in ['place_order', 'amend_order', 'cancel_order']:
                                    # 尝试从工具结果中提取订单ID
                                    try:
                                        if isinstance(obs, str) and obs.startswith('{'):
                                            result_data = json.loads(obs)
                                            if result_data.get('success') and result_data.get('order_id'):
                                                related_order_id = result_data['order_id']
                                    except:
                                        pass

                                with get_db() as db:
                                    await self._save_message(
                                        db, conversation.id, "tool",
                                        f"工具 {tool_name} 执行完成", "tool_result",
                                        tool_name=tool_name,
                                    tool_args=getattr(step.action, 'tool_input', {}),
                                    tool_result=obs,
                                        tool_call_id=tool_call_id,
                                        tool_execution_time=tool_execution_time,
                                        related_order_id=related_order_id
                                    )

                    # 处理最终输出
                    if "output" in chunk:
                        output = chunk["output"]
                        if output and output.strip():
                            response = output
                            # 检查是否是思考过程（某些模型如Qwen会输出思考过程）
                            if output.startswith("<think>") or output.startswith("思考:"):
                                formatted_output = f"🧠 **思考过程**:\n\n{output}"
                            else:
                                formatted_output = f"🤖 **AI助手回复**:\n\n{output}"
                            # 实现流式输出
                            chunk_size = 15
                            for i in range(0, len(output), chunk_size):
                                chunk_text = output[i:i+chunk_size]
                                if i == 0:
                                    if output.startswith("思考:") or "思考过程" in output:
                                        prefix = "🧠 **思考过程**:\n\n"
                                    else:
                                        prefix = "🤖 **AI助手回复**:\n\n"
                                    formatted_chunk = prefix + chunk_text
                                else:
                                    formatted_chunk = chunk_text

                                yield [{"role": "assistant", "content": formatted_chunk}]
                                import asyncio
                                await asyncio.sleep(0.03)

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
        tool_result: str = None,
        tool_call_id: str = None,
        tool_execution_time: float = None,
        related_order_id: str = None
    ):
        """保存消息到数据库"""
        try:
            message = AgentMessage(
                conversation_id=conversation_id,
                role=role,
                content=content,
                message_type=message_type,
                tool_name=tool_name,
                tool_arguments=json.dumps(tool_args) if tool_args else None,
                tool_result=json.dumps(tool_result) if tool_result else None,
                tool_call_id=tool_call_id,
                tool_execution_time=tool_execution_time,
                related_order_id=related_order_id,
                created_at=now_beijing()
            )
            db.add(message)

            # 更新对话的最后消息时间
            conversation = db.query(AgentConversation).filter(AgentConversation.id == conversation_id).first()
            if conversation:
                conversation.last_message_at = now_beijing()

            db.commit()
            logger.debug(f"成功保存消息: role={role}, conversation_id={conversation_id}")
        except Exception as e:
            logger.error(f"保存消息失败: conversation_id={conversation_id}, role={role}, error={e}")
            # 不重新抛出异常，避免中断agent执行

    async def _load_conversation_history(self, conversation_id: int) -> List[Dict[str, str]]:
        """加载对话历史消息"""
        try:
            from database.models import AgentMessage
            with get_db() as db:
                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id
                ).order_by(AgentMessage.created_at.asc()).all()

                # 转换为流式消息格式
                history_messages = []
                for message in messages:
                    role = message.role
                    content = message.content

                    # 根据消息类型转换格式
                    if message.message_type == "thinking":
                        formatted_content = f"💭 **思考过程**:\\n{content}"
                        history_messages.append({"role": "assistant", "content": formatted_content})
                    elif message.message_type in ["tool_call", "tool_result"]:
                        # 工具消息 - 构造JSON格式
                        tool_data = {
                            "tool_name": message.tool_name or "",
                            "arguments": message.tool_arguments or {},
                            "result": message.tool_result or {},
                            "status": "success" if message.message_type == "tool_result" else "calling",
                            "tool_call_id": message.tool_call_id or ""
                        }
                        tool_content = json.dumps(tool_data, ensure_ascii=False)

                        if message.message_type == "tool_call":
                            formatted_content = f"🔧 **工具调用**: `{tool_data['tool_name']}`\\n\\n参数: {json.dumps(tool_data.get('arguments', {}), indent=2, ensure_ascii=False)}"
                        else:
                            formatted_content = f"✅ **工具完成**: `{tool_data['tool_name']}`\\n\\n结果: {json.dumps(tool_data.get('result', {}), indent=2, ensure_ascii=False)}"

                        history_messages.append({"role": "assistant", "content": formatted_content})
                    elif message.message_type == "play_result":
                        # 投资结果
                        history_messages.append({"role": "assistant", "content": content})
                    else:
                        # 普通消息
                        history_messages.append({"role": role, "content": content})

                return history_messages

        except Exception as e:
            logger.error(f"加载对话历史失败: {e}")
            return []

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


    def extract_order_ids_from_tool_results(self, tool_results: List[Dict]) -> List[str]:
        """
        从工具结果中提取所有订单ID

        Args:
            tool_results: 工具执行结果列表

        Returns:
            List[str]: 提取到的订单ID列表（去重）
        """
        order_ids = []

        for result in tool_results:
            if not isinstance(result, dict):
                continue

            # 检查不同工具的订单ID位置
            if result.get('success'):
                result_data = result.get('result', result)

                # place_order, cancel_order, amend_order 的订单ID
                if 'order_id' in result_data:
                    order_ids.append(str(result_data['order_id']))

                # 批量操作的多个订单ID
                if 'order_ids' in result_data:
                    order_ids.extend([str(oid) for oid in result_data['order_ids']])

                # OKX API 响应格式
                if 'data' in result_data and isinstance(result_data['data'], list):
                    for item in result_data['data']:
                        if 'ordId' in item:
                            order_ids.append(str(item['ordId']))
                        if 'order_id' in item:
                            order_ids.append(str(item['order_id']))

        return list(set(order_ids))  # 去重

    async def auto_decision(
        self,
        plan_id: int,
        training_id: int = None,
        prediction_data: List[Dict] = None
    ) -> AsyncGenerator[List[Dict[str, str]], None]:
        """
        基于预测数据的自动决策（后台推理，不展示到chatbot）

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID（可选）
            prediction_data: 预测数据（可选，如果不提供则从数据库获取）

        Yields:
            流式消息列表
        """
        try:
            logger.info(f"开始自动决策: plan_id={plan_id}, training_id={training_id}")

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

            # 获取预测数据
            if not prediction_data:
                predictions = self._get_latest_predictions(plan_id, training_id)
            else:
                # 将字典格式转换为PredictionData对象（模拟）
                predictions = prediction_data

            if not predictions:
                yield [{"role": "assistant", "content": "❌ 没有可用的预测数据"}]
                return

            logger.info(f"获取到 {len(predictions)} 条预测数据")

            # 构建决策提示词
            decision_prompt = self._build_decision_prompt(plan, predictions)

            logger.info("开始流式自动决策...")

            # 使用统一的流式对话接口进行自动推理
            async for chunk in self.stream_conversation(
                plan_id=plan_id,
                user_message=decision_prompt,
                conversation_type="auto_inference"
            ):
                yield chunk

            logger.info("自动决策完成")

        except Exception as e:
            logger.error(f"自动决策失败: {e}")
            import traceback
            traceback.print_exc()
            yield [{"role": "assistant", "content": f"❌ 自动决策失败: {str(e)}"}]

    async def manual_inference(self, plan_id: int) -> AsyncGenerator[List[Dict[str, str]], None]:
        """
        统一的手动推理入口（流式）

        Args:
            plan_id: 计划ID

        Yields:
            流式消息列表
        """
        try:
            logger.info(f"开始手动推理: plan_id={plan_id}")

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

            # 获取最新的训练记录
            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    yield [{"role": "assistant", "content": "❌ 没有可用的训练记录，请先完成模型训练"}]
                    return

            # 使用自动决策功能，但指定为手动推理类型
            async for chunk in self.auto_decision(plan_id, latest_training.id):
                yield chunk

        except Exception as e:
            logger.error(f"手动推理失败: {e}")
            import traceback
            traceback.print_exc()
            yield [{"role": "assistant", "content": f"❌ 手动推理失败: {str(e)}"}]

    async def scheduled_decision(self, plan_id: int, training_id: int) -> AsyncGenerator[List[Dict[str, str]], None]:
        """
        定时任务决策入口

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID

        Yields:
            流式消息列表
        """
        try:
            logger.info(f"开始定时决策: plan_id={plan_id}, training_id={training_id}")

            # 使用自动决策功能，但指定为定时决策类型
            async for chunk in self.stream_conversation(
                plan_id=plan_id,
                user_message=f"请基于训练记录 v{training_id} 的预测数据进行定时交易决策分析",
                conversation_type="scheduled_decision"
            ):
                yield chunk

        except Exception as e:
            logger.error(f"定时决策失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 定时决策失败: {str(e)}"}]

    def _get_latest_predictions(self, plan_id: int, training_id: int = None) -> List[PredictionData]:
        """获取最新的预测数据（从agent_decision_service迁移逻辑）"""
        try:
            with get_db() as db:
                if training_id:
                    # 使用指定的训练记录
                    predictions = db.query(PredictionData).filter(
                        PredictionData.training_record_id == training_id
                    ).order_by(PredictionData.timestamp).all()
                    logger.info(f"使用指定训练记录 {training_id}，获取到 {len(predictions)} 条预测数据")
                else:
                    # 获取最新的预测数据
                    latest_training = db.query(TrainingRecord).filter(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed'
                    ).order_by(TrainingRecord.completed_at.desc()).first()

                    if latest_training:
                        predictions = db.query(PredictionData).filter(
                            PredictionData.training_record_id == latest_training.id
                        ).order_by(PredictionData.timestamp).all()
                        logger.info(f"使用最新训练记录 {latest_training.id}，获取到 {len(predictions)} 条预测数据")
                    else:
                        predictions = []
                        logger.warning(f"计划 {plan_id} 没有找到完成的训练记录")

                return predictions

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return []

    def _build_decision_prompt(self, plan: TradingPlan, predictions: List[PredictionData]) -> str:
        """构建决策提示词"""
        try:
            # 格式化预测数据
            pred_text = []
            for pred in predictions[:20]:  # 限制显示最新的20条预测数据
                pred_text.append(
                    f"时间: {pred.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                    f"预测价格: {pred.predicted_price:.6f}, "
                    f"置信度: {pred.confidence:.2f}, "
                    f"上涨概率: {pred.upward_prob:.2%}, "
                    f"波动率: {pred.volatility:.2%}"
                )

            # 获取当前价格信息
            current_price_info = ""
            try:
                trading_tools = self.get_plan_trading_tools(plan.id)
                if trading_tools:
                    current_price = trading_tools.get_current_price(plan.inst_id)
                    current_price_info = f"\n当前价格: {current_price}"
            except Exception as e:
                logger.warning(f"获取当前价格失败: {e}")
                current_price_info = "\n当前价格: 无法获取"

            # 构建决策提示词
            prompt = f"""基于以下预测数据进行交易决策分析：

交易计划：{plan.inst_id} ({plan.interval})
当前时间：{now_beijing().strftime('%Y-%m-%d %H:%M:%S')}{current_price_info}

预测数据：
{chr(10).join(pred_text)}

请分析这些预测数据，给出具体的交易建议：
1. 当前市场趋势分析
2. 建议的交易操作（买入/卖出/持有）
3. 具体的下单策略（价格、数量等）
4. 风险控制建议

如果决定执行交易，请使用相应的工具进行操作。请基于量化分析结果做出决策，而不是主观猜测。"""

            return prompt

        except Exception as e:
            logger.error(f"构建决策提示词失败: {e}")
            return "请基于可用的预测数据进行交易决策分析。"

    def get_unified_decisions(self, plan_id: int, limit: int = 50):
        """
        统一的决策查询接口，兼容历史数据

        Args:
            plan_id: 计划ID
            limit: 返回记录数限制

        Returns:
            List[Dict]: 统一格式的决策记录列表
        """
        decisions = []

        try:
            with get_db() as db:
                # 1. 优先从AgentMessage获取新数据
                auto_messages = db.query(AgentMessage).join(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id,
                    AgentConversation.conversation_type.in_(['auto_inference', 'scheduled_decision']),
                    AgentMessage.role == 'assistant'
                ).order_by(AgentMessage.created_at.desc()).limit(limit).all()

                for msg in auto_messages:
                    decisions.append({
                        'id': msg.id,
                        'created_at': msg.created_at,
                        'content': msg.content,
                        'source': 'agent_message',
                        'conversation_id': msg.conversation_id,
                        'conversation_type': db.query(AgentConversation).filter(
                            AgentConversation.id == msg.conversation_id
                        ).first().conversation_type if msg.conversation_id else 'unknown'
                    })

                # 2. 从旧的AgentDecision获取历史数据（只读，兼容性）
                try:
                    from database.models import AgentDecision
                    old_decisions = db.query(AgentDecision).filter(
                        AgentDecision.plan_id == plan_id
                    ).order_by(AgentDecision.decision_time.desc()).limit(limit).all()

                    for decision in old_decisions:
                        decisions.append({
                            'id': decision.id,
                            'created_at': decision.decision_time,
                            'content': f"决策类型: {decision.decision_type}\n推理: {decision.reasoning}\n状态: {decision.status}",
                            'source': 'agent_decision',
                            'training_id': decision.training_record_id,
                            'llm_model': decision.llm_model
                        })
                except ImportError:
                    # AgentDecision模型可能已被删除，跳过
                    pass

            # 按时间排序
            decisions.sort(key=lambda x: x['created_at'], reverse=True)
            return decisions[:limit]

        except Exception as e:
            logger.error(f"获取统一决策记录失败: {e}")
            return []


# 全局实例
agent_service = LangChainAgentService()