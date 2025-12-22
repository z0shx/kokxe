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
    LLMConfig, TrainingRecord, PredictionData, OrderEventLog, now_beijing
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
        # 基础参数
        base_params = {
            "model": llm_config.model_name,
            "temperature": llm_config.temperature or 0.7,
        }

        # 根据 LLM 提供商设置不同的参数
        if llm_config.provider == "qwen":
            # Qwen API 可能对 max_tokens 有特殊要求或使用不同参数名
            # 这里设置一个合理的默认值
            base_params["max_tokens"] = min(llm_config.max_tokens or 2000, 8000)  # 限制最大 tokens
        else:
            # 其他 LLM 提供商使用标准参数
            base_params["max_tokens"] = llm_config.max_tokens or 2000

        return base_params

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

                # 确保不会传递任何不支持的参数
                # 移除 enable_thinking 等不支持的参数
                if extra_params and 'enable_thinking' in extra_params:
                    logger.info(f"移除不支持的 enable_thinking 参数，保持思考功能通过内容处理实现")
                    extra_params = {k: v for k, v in extra_params.items() if k != 'enable_thinking'}

                model_kwargs = {}
                # 不设置任何 Qwen API 不支持的参数

                # 为 Qwen 启用流式输出，支持 think 模式
                self._llm_clients[client_key] = ChatOpenAI(
                    **base_params,
                    openai_api_key=llm_config.api_key,
                    openai_api_base=base_url,
                    model_kwargs=model_kwargs,
                    streaming=True
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

        # 获取计划信息（缓存到闭包中）
        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

        if not plan:
            logger.error(f"计划不存在: plan_id={plan_id}")
            return list(available_tools.values())

        # 缓存计划信息到闭包变量中
        plan_inst_id = plan.inst_id
        plan_trading_limits = plan.trading_limits
        plan_is_demo = plan.is_demo

        # 1. 获取当前价格工具
        if "get_current_price" in enabled_tools:
            @tool
            def get_current_price(inst_id: str = None) -> str:
                """获取交易对当前价格"""
                try:
                    inst_id = inst_id or plan_inst_id
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
                    # 检查订单是否存在于本地数据表
                    from database.models import TradeOrder
                    # 使用预获取的 plan_id 变量，避免嵌套数据库连接
                    with get_db() as db:
                        local_order = db.query(TradeOrder).filter(
                            TradeOrder.plan_id == plan_id,
                            TradeOrder.order_id == order_id
                        ).first()

                        if not local_order:
                            return json.dumps({
                                "success": False,
                                "error": f"订单 {order_id} 不存在于本地记录中，只能取消本系统创建的订单"
                            }, ensure_ascii=False)

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
                    # 检查订单是否存在于本地数据表
                    from database.models import TradeOrder
                    with get_db() as db:
                        local_order = db.query(TradeOrder).filter(
                            TradeOrder.plan_id == plan_id,
                            TradeOrder.order_id == order_id
                        ).first()

                        if not local_order:
                            return json.dumps({
                                "success": False,
                                "error": f"订单 {order_id} 不存在于本地记录中，只能修改本系统创建的订单"
                            }, ensure_ascii=False)

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
            def query_prediction_data(
                limit: int = 500,
                include_monte_carlo: bool = True,
                monte_carlo_sample_count: int = 5,
                include_path_statistics: bool = True
            ) -> str:
                """查询AI模型预测数据（获取最新批次），包含蒙特卡罗路径信息和统计数据

                Args:
                    limit: 返回预测数据条数限制（默认500条，确保包含最新批次）
                    include_monte_carlo: 是否包含蒙特卡罗路径样本
                    monte_carlo_sample_count: 显示蒙特卡罗路径样本数量（最多10条）
                    include_path_statistics: 是否包含路径统计分析

                Returns:
                    JSON格式的预测数据，包含以下字段：
                    - success: 查询是否成功
                    - total_batches: 批次数量
                    - total_predictions: 预测数据总数
                    - batch_data: 按批次组织的预测数据
                    - metadata: 查询元数据
                """
                try:
                    # 参数验证
                    if not isinstance(limit, int) or limit <= 0 or limit > 1000:
                        return json.dumps({"success": False, "error": "limit必须是1-1000之间的整数"}, ensure_ascii=False)
                    if not isinstance(monte_carlo_sample_count, int) or monte_carlo_sample_count <= 0 or monte_carlo_sample_count > 10:
                        return json.dumps({"success": False, "error": "monte_carlo_sample_count必须是1-10之间的整数"}, ensure_ascii=False)

                    from database.models import PredictionData
                    from database.models import now_beijing
                    import numpy as np

                    with get_db() as db:
                        # 获取最新的批次ID，确保返回完整批次数据
                        latest_batch_id = db.query(PredictionData.inference_batch_id).filter(
                            PredictionData.plan_id == plan_id
                        ).order_by(PredictionData.prediction_time.desc()).limit(1).scalar()

                        if not latest_batch_id:
                            return json.dumps({
                                "success": False,
                                "error": "没有找到预测数据"
                            }, ensure_ascii=False)

                        # 获取该完整批次的所有预测数据
                        predictions = db.query(PredictionData).filter(
                            PredictionData.plan_id == plan_id,
                            PredictionData.inference_batch_id == latest_batch_id
                        ).order_by(PredictionData.timestamp).all()

                        # 组织批次数据（现在只有一个批次）
                        batch_data = {
                            latest_batch_id: {
                                "batch_id": latest_batch_id,
                                "training_record_id": predictions[0].training_record_id if predictions else None,
                                "prediction_time": predictions[0].prediction_time.isoformat() if predictions and predictions[0].prediction_time else None,
                                "inference_params": predictions[0].inference_params if predictions else None,
                                "predictions": []
                            }
                        }

                        # 遍历该批次的所有预测数据
                        for pred in predictions:
                            prediction_info = {
                                "id": pred.id,
                                "timestamp": pred.timestamp.isoformat(),
                                "target_time": pred.timestamp.isoformat(),
                                "ohlc": {
                                    "open": pred.open,
                                    "high": pred.high,
                                    "low": pred.low,
                                    "close": pred.close
                                },
                                "uncertainty_range": {
                                    "close_min": pred.close_min,
                                    "close_max": pred.close_max,
                                    "close_std": pred.close_std
                                },
                                "probabilities": {
                                    "upward_probability": pred.upward_probability,
                                    "volatility_amplification_probability": pred.volatility_amplification_probability
                                },
                                "predicted_price": float(pred.close) if pred.close else None,
                                "current_price": None,  # 需要从实时数据获取
                                "price_change": None,  # 需要计算
                                "price_change_pct": None,  # 需要计算
                                "upward_prob": float(pred.upward_probability) if pred.upward_probability else None,
                                "volatility_prob": float(pred.volatility_amplification_probability) if pred.volatility_amplification_probability else None,
                                "confidence": None,  # confidence 字段不存在
                                "samples_used": None,  # samples_used 字段不存在
                                "temperature": None,  # temperature 字段在 inference_params 中
                            }

                            # 处理蒙特卡罗数据 - prediction_data字段不存在，跳过此部分
                            # 直接使用下面的inference_params逻辑
                            if False and include_monte_carlo:  # 暂时禁用此分支
                                try:
                                    pred_json = json.loads(pred.prediction_data)

                                    # 如果是新的多路径格式
                                    if "monte_carlo_paths" in pred_json and len(pred_json["monte_carlo_paths"]) > 0:
                                        paths = pred_json["monte_carlo_paths"]
                                        prediction_steps = pred_json.get("prediction_steps", len(paths[0]) if paths else 0)

                                        # 限制显示的路径数量
                                        if len(paths) > monte_carlo_sample_count:
                                            selected_paths = paths[:monte_carlo_sample_count]
                                        else:
                                            selected_paths = paths

                                        prediction_info["monte_carlo_paths"] = selected_paths
                                        prediction_info["monte_carlo_metadata"] = {
                                            "total_paths": len(paths),
                                            "prediction_steps": prediction_steps,
                                            "selected_paths": len(selected_paths)
                                        }

                                        # 如果需要路径统计分析
                                        if include_path_statistics:
                                            path_stats = []
                                            for step_idx in range(prediction_steps):
                                                step_prices = []
                                                for path in paths:
                                                    if step_idx < len(path):
                                                        if isinstance(path[step_idx], dict):
                                                            step_prices.append(float(path[step_idx]["price"]))
                                                        else:
                                                            step_prices.append(float(path[step_idx]))

                                                if step_prices:
                                                    step_array = np.array(step_prices)
                                                    stats = {
                                                        "step": step_idx,
                                                        "mean": float(np.mean(step_array)),
                                                        "std": float(np.std(step_array)),
                                                        "min": float(np.min(step_array)),
                                                        "max": float(np.max(step_array)),
                                                        "median": float(np.median(step_array)),
                                                        "q25": float(np.percentile(step_array, 25)),
                                                        "q75": float(np.percentile(step_array, 75)),
                                                        "count": len(step_prices)
                                                    }
                                                    path_stats.append(stats)

                                            prediction_info["path_statistics"] = path_stats

                                            # 添加最终步骤统计摘要
                                            if path_stats:
                                                final_stats = path_stats[-1]
                                                prediction_info["final_step_summary"] = {
                                                    "mean_price": final_stats["mean"],
                                                    "price_std": final_stats["std"],
                                                    "price_range": final_stats["max"] - final_stats["min"],
                                                    "confidence_interval_90": [
                                                        final_stats["q25"],
                                                        final_stats["q75"]
                                                    ]
                                                }

                                    # 兼容旧格式
                                    elif "sample_trajectories" in pred_json:
                                        trajectories = pred_json["sample_trajectories"]
                                        if len(trajectories) > monte_carlo_sample_count:
                                            selected_trajectories = trajectories[:monte_carlo_sample_count]
                                        else:
                                            selected_trajectories = trajectories

                                        prediction_info["sample_trajectories"] = selected_trajectories
                                        prediction_info["trajectory_metadata"] = {
                                            "total_trajectories": len(trajectories),
                                            "selected_trajectories": len(selected_trajectories)
                                        }

                                except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
                                    logger.debug(f"解析预测数据失败: {e}")
                                    prediction_info["parse_error"] = str(e)

                            # 保留原有的推理参数解析作为备用
                            elif include_monte_carlo and pred.inference_params:
                                try:
                                    inference_params = pred.inference_params if isinstance(pred.inference_params, dict) else json.loads(pred.inference_params)

                                    # 查找蒙特卡罗路径数据
                                    if "monte_carlo_paths" in inference_params:
                                        mc_paths = inference_params["monte_carlo_paths"]
                                        if len(mc_paths) > monte_carlo_sample_count:
                                            selected_paths = mc_paths[:monte_carlo_sample_count]
                                        else:
                                            selected_paths = mc_paths

                                        prediction_info["legacy_monte_carlo_paths"] = selected_paths
                                        prediction_info["legacy_total_count"] = len(mc_paths)

                                    # 添加推理配置信息
                                    prediction_info["inference_config"] = {
                                        "temperature": inference_params.get("temperature"),
                                        "sample_count": inference_params.get("sample_count"),
                                        "top_p": inference_params.get("top_p")
                                    }

                                except Exception as parse_error:
                                    logger.debug(f"解析推理参数失败: {parse_error}")
                                    prediction_info["inference_config_error"] = str(parse_error)

                            # 添加到最新批次中
                            batch_data[latest_batch_id]["predictions"].append(prediction_info)

                        # 转换为列表格式
                        result_data = list(batch_data.values())

                        # 统计信息
                        total_predictions_with_paths = sum(
                            1 for batch in result_data
                            for pred in batch["predictions"]
                            if "monte_carlo_paths" in pred or "legacy_monte_carlo_paths" in pred
                        )

                        return json.dumps({
                            "success": True,
                            "data": result_data,
                            "summary": {
                                "total_batches": len(result_data),
                                "total_predictions": len(predictions),
                                "predictions_with_paths": total_predictions_with_paths,
                                "include_monte_carlo": include_monte_carlo,
                                "monte_carlo_sample_count": monte_carlo_sample_count,
                                "include_path_statistics": include_path_statistics
                            },
                            "timestamp": now_beijing().isoformat()
                        }, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"查询预测数据失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["query_prediction_data"] = query_prediction_data

        # 13. 获取预测历史
        if "get_prediction_history" in enabled_tools:
            @tool
            def get_prediction_history(limit: int = 20) -> str:
                """获取历史预测记录 - 支持多路径Monte Carlo格式"""
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
                            # 获取该训练记录的预测数据
                            predictions = db.query(PredictionData).filter(
                                PredictionData.training_record_id == record.id
                            ).order_by(PredictionData.created_at.desc()).all()

                            # 统计Monte Carlo路径信息
                            total_paths = 0
                            path_stats_available = 0
                            latest_with_paths = None

                            for pred in predictions:
                                if False and pred.prediction_data:  # 暂时禁用，字段不存在
                                    try:
                                        pred_json = json.loads(pred.prediction_data)
                                        if "monte_carlo_paths" in pred_json and len(pred_json["monte_carlo_paths"]) > 0:
                                            total_paths = max(total_paths, len(pred_json["monte_carlo_paths"]))
                                            path_stats_available += 1
                                            if not latest_with_paths:
                                                latest_with_paths = {
                                                    "prediction_id": pred.id,
                                                    "created_at": pred.created_at.isoformat(),
                                                    "num_paths": len(pred_json["monte_carlo_paths"]),
                                                    "prediction_steps": pred_json.get("prediction_steps", len(pred_json["monte_carlo_paths"][0]) if pred_json["monte_carlo_paths"] else 0)
                                                }
                                    except (json.JSONDecodeError, KeyError):
                                        pass

                            record_info = {
                                "training_id": record.id,
                                "model_version": record.version,
                                "created_at": record.created_at.isoformat(),
                                "prediction_count": len(predictions),
                                "status": record.status,
                                "monte_carlo_info": {
                                    "total_paths": total_paths,
                                    "predictions_with_paths": path_stats_available,
                                    "latest_with_paths": latest_with_paths
                                }
                            }

                            data.append(record_info)

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

        # 14. 获取最新批次预测均值数据
        if "get_latest_prediction_analysis" in enabled_tools:
            @tool
            def get_latest_prediction_analysis(plan_id: int = 3) -> str:
                """获取最新批次预测均值数据"""
                try:
                    from services.trading_tools import get_latest_prediction_analysis

                    # 调用预测分析工具
                    result = get_latest_prediction_analysis(plan_id)

                    # 处理datetime对象的序列化
                    def datetime_handler(obj):
                        if hasattr(obj, 'isoformat'):
                            return obj.isoformat()
                        raise TypeError(repr(obj) + " is not JSON serializable")

                    return json.dumps({
                        "success": True,
                        "data": {
                            "training_version": result.get("training_version"),
                            "training_id": result.get("training_id"),
                            "plan_id": result.get("plan_id"),
                            "data_points_count": result.get("data_points_count"),
                            "time_points_count": result.get("time_points_count"),
                            "extremes": result.get("extremes"),
                            "analysis_summary": result.get("analysis_summary"),
                            "raw_data": result.get("raw_data", [])
                        },
                        "message": result.get("message", "预测分析完成")
                    }, ensure_ascii=False, default=datetime_handler)
                except Exception as e:
                    logger.error(f"获取最新预测分析失败: {e}")
                    return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

            available_tools["get_latest_prediction_analysis"] = get_latest_prediction_analysis

        return list(available_tools.values())

    def _detect_qwen_thinking(self, content: str, llm_config: LLMConfig = None) -> bool:
        """检测 Qwen 思考模式的多重策略"""
        if not content or not content.strip():
            return False

        # 策略1: 强力的思考内容检测
        thinking_patterns = [
            r'思考[:：]',  # 思考: 或 思考：
            r'让我分析',   # 让我分析
            r'首先',       # 首先
            r'接下来',     # 接下来
            r'综合考虑',   # 综合考虑
            r'分析结果',   # 分析结果
            r'判断',       # 判断
            r'决策',       # 决策
            r'建议',       # 建议
            r'根据',       # 根据
            r'考虑到',     # 考虑到
            r'从市场角度看', # 从市场角度看
            r'技术分析',   # 技术分析
            r'推理过程',   # 推理过程
            r'逻辑分析',   # 逻辑分析
            r'评估',       # 评估
            r'权衡',       # 权衡
            r'检查',       # 检查
            r'观察',       # 观察
            r'推断',       # 推断
            r'结论',       # 结论
            r'步骤',       # 步骤
            r'第一步',     # 第一步
            r'第二',       # 第二
            r'第三',       # 第三
        ]

        import re
        for pattern in thinking_patterns:
            if re.search(pattern, content):
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
        """构建系统提示词 - 强制使用计划配置的提示词"""
        # 强制要求必须配置计划提示词，否则拒绝执行
        if not plan.agent_prompt:
            raise ValueError("计划未配置 Agent 提示词，请先在计划设置中配置自定义提示词内容")

        # 直接使用计划配置的提示词，不做任何简化
        dynamic_prompt = plan.agent_prompt

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
            "query_prediction_data": "查询AI模型预测数据（包含蒙特卡罗路径和不确定性范围）",
            "get_prediction_history": "获取历史预测记录",
            "run_latest_model_inference": "执行最新的AI模型推理",
            "get_latest_prediction_analysis": "获取最新批次预测均值数据，基于多批次蒙特卡罗路径计算最高价、最低价、时间范围等关键指标",
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

重要执行原则：
1. **避免重复操作**: 在执行交易操作前，先查询当前状态，避免重复下单或取消不存在的订单
2. **单次完成目标**: 每个工具调用应该是一个完整的操作，避免为同一目标多次调用相同工具
3. **明确决策逻辑**: 每次交易操作都要有明确的理由和目标
4. **及时停止**: 达到交易目标后立即停止，不要继续无意义的操作
5. **工具调用策略**: 一次调用一个工具，确认结果后再决定下一步

执行步骤建议：
1. 首先获取当前市场信息和持仓状态
2. 分析数据并做出明确的交易决策
3. 如果决定交易，一次性执行完整的下单操作
4. 确认操作结果，完成交易流程
5. 避免反复修改订单或重复下单

请根据以上原则和可用的工具来完成交易决策操作。"""

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
                current_time = now_beijing()
                conversation = AgentConversation(
                    plan_id=plan_id,
                    conversation_type=conversation_type,
                    status='active',
                    started_at=current_time,
                    last_message_at=current_time
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

        # 检查是否为新创建的对话
        # 方法1: 使用时间差判断
        time_diff = abs((conversation.last_message_at - conversation.created_at).total_seconds())
        is_new_conversation_by_time = (time_diff < 1.0)

        # 方法2: 检查是否已有系统消息
        with get_db() as db:
            system_message_count = db.query(AgentMessage).filter(
                AgentMessage.conversation_id == conversation.id,
                AgentMessage.role == "system"
            ).count()
            is_new_conversation_by_msg = (system_message_count == 0)

        # 综合判断：如果任一条件满足，认为是新对话
        is_new_conversation = is_new_conversation_by_time or is_new_conversation_by_msg

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
            # 加载历史消息，但要确保包含系统消息
            history = await self._load_conversation_history(conversation.id)

            # 检查历史中是否包含系统消息，如果没有，则添加
            if not any(msg.get("role") == "system" for msg in history):
                logger.info(f"对话 {conversation.id} 缺少系统消息，添加系统提示词")
                # 保存系统消息到数据库
                try:
                    with get_db() as db:
                        await self._save_message(
                            db, conversation.id, "system", system_prompt, "text"
                        )
                except Exception as e:
                    logger.error(f"保存系统消息失败: {e}")

                # 在历史开头添加系统消息
                yield [{"role": "system", "content": system_prompt}]
            else:
                yield history

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
                # 根据 LLM 类型设置不同的参数
                agent_kwargs = {
                    "agent": agent,
                    "tools": tools,
                    "verbose": False,
                    "handle_parsing_errors": True,
                    "return_intermediate_steps": True,
                    "max_iterations": 15,  # 减少最大迭代次数以防止循环，15次足够完成正常交易流程
                    "max_execution_time": 300,  # 减少最大执行时间为5分钟，避免长时间卡住
                    "early_stopping_method": "force",  # 强制早停，防止无限循环
                }

                # 只有 Qwen 模型需要特殊处理
                if llm_config.provider == "qwen":
                    # Qwen 不支持 early_stopping_method，移除该参数
                    agent_kwargs.pop("early_stopping_method", None)
                # 其他模型保留 "force" 设置

                # 创建 AgentExecutor
                agent_executor = AgentExecutor(**agent_kwargs)

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
                                                related_order_id = str(result_data['order_id'])
                                            elif result_data.get('success') and result_data.get('result') and isinstance(result_data['result'], dict):
                                                # 检查 result 字段中是否有 order_id
                                                if result_data['result'].get('order_id'):
                                                    related_order_id = str(result_data['result']['order_id'])
                                    except:
                                        # 如果 JSON 解析失败，尝试使用正则表达式提取订单ID
                                        try:
                                            import re
                                            # 匹配常见的订单ID格式（数字或字母数字组合）
                                            order_id_patterns = [
                                                r'"order_id":\s*["\']?([a-zA-Z0-9]+)["\']?',
                                                r'"ordId":\s*["\']?([a-zA-Z0-9]+)["\']?',
                                                r'order_id["\']?\s*:\s*["\']?([a-zA-Z0-9]+)["\']?'
                                            ]
                                            for pattern in order_id_patterns:
                                                match = re.search(pattern, obs)
                                                if match:
                                                    related_order_id = match.group(1)
                                                    break
                                        except:
                                            pass

                                with get_db() as db:
                                    await self._save_message(
                                        db, conversation.id, "tool",
                                        f"工具 {tool_name} 执行完成", "tool_result",
                                        tool_name=tool_name,
                                    tool_args=json.dumps(getattr(step.action, 'tool_input', {})),
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
                            # 优化流式输出 - 使用更大的chunk减少频繁调用
                            chunk_size = 100
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
                                await asyncio.sleep(0.01)  # 减少延迟提高响应性

                            # 保存助手回复到数据库
                            with get_db() as db:
                                await self._save_message(
                                    db, conversation.id, "assistant", output, "text"
                                )

            else:
                # 没有工具，直接使用 LLM
                response = ""
                async for chunk in llm.astream(messages):
                    # 传递 llm_config 以便处理 Qwen think 模式
                    content = self._extract_content_from_chunk(chunk, llm_config)
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

            # 分析异常类型并提供相应的用户友好消息
            error_message = str(e)
            user_message = ""

            if "Agent stopped due to max iterations" in error_message:
                logger.warning(f"PLAN {plan_id} - Agent 达到最大迭代次数限制")
                user_message = "⚠️ Agent 达到最大迭代次数限制（已提升至25次）。建议：\n1. 将复杂任务分解为多个简单步骤\n2. 一次只使用一个工具进行单一操作\n3. 避免一次性查询过多数据\n4. 重新提交更简洁的请求"
            elif "Agent stopped due to max execution time" in error_message or "timeout" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - Agent 执行超时")
                user_message = "⏱️ Agent 执行超时（已提升至10分钟）。建议：\n1. 检查网络连接状况\n2. 简化查询范围（如减少时间范围或数据条数）\n3. 分步骤执行复杂操作\n4. 稍后重试"
            elif "Rate limit" in error_message or "rate limit" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - API 速率限制")
                user_message = "🚦 API 调用频率超限，请稍后重试。"
            elif "Connection" in error_message or "network" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - 网络连接错误")
                user_message = "🌐 网络连接问题，请检查网络连接后重试。"
            else:
                user_message = f"❌ Agent 执行出现异常: {error_message}"

            # 向用户输出错误信息
            yield [{"role": "assistant", "content": user_message}]

            # 保存详细的错误信息到数据库
            with get_db() as db:
                await self._save_message(
                    db, conversation.id, "assistant",
                    f"Agent 执行失败: {error_message}", "text"
                )

                # 如果是严重的异常，同时记录到系统日志表
                if "max iterations" in error_message or "timeout" in error_message.lower():
                    try:
                        from database.models import SystemLog
                        error_log = SystemLog(
                            level="WARNING",
                            category="agent_error",
                            message=f"Agent执行异常 - Plan {plan_id}",
                            details={
                                "error_type": "max_iterations_or_timeout",
                                "error_message": error_message,
                                "conversation_id": conversation.id,
                                "plan_id": plan_id,
                                "llm_model": llm_config.model_name if llm_config else None
                            }
                        )
                        db.add(error_log)
                        db.commit()
                        logger.info(f"PLAN {plan_id} - Agent 异常已记录到系统日志")
                    except Exception as log_error:
                        logger.error(f"保存系统日志失败: {log_error}")

    def _extract_content_from_chunk(self, chunk, llm_config=None) -> Optional[str]:
        """从 chunk 中提取内容，支持多种格式，特别处理 Qwen think 模式"""
        if not chunk:
            return None

        # 方法1: 标准 content 属性
        if hasattr(chunk, 'content'):
            content = chunk.content
            if content and isinstance(content, str) and content.strip():
                # 对于 Qwen think 模式的特殊处理
                if llm_config and llm_config.provider == "qwen":
                    content = self._process_qwen_thinking_content(content)
                return content

        # 方法2: text 属性
        if hasattr(chunk, 'text'):
            text = chunk.text
            if text and isinstance(text, str) and text.strip():
                # 对于 Qwen think 模式的特殊处理
                if llm_config and llm_config.provider == "qwen":
                    text = self._process_qwen_thinking_content(text)
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

                # 对于 Qwen think 模式的特殊处理
                if llm_config and llm_config.provider == "qwen":
                    chunk_str = self._process_qwen_thinking_content(chunk_str)
                return chunk_str
        except:
            pass

        return None

    def _process_qwen_thinking_content(self, content: str) -> str:
        """
        处理 Qwen think 模式的内容，确保思考过程正确格式化

        Args:
            content: 原始内容

        Returns:
            str: 处理后的内容
        """
        if not content or not content.strip():
            return content

        # 检查是否包含思考标记
        if '<think>' in content or '</think>' in content:
            # 移除 HTML 标记，保留思考内容
            import re
            # 匹配 <think>...</think> 标签内的内容
            think_pattern = r'<think>(.*?)</think>'
            think_matches = re.findall(think_pattern, content, re.DOTALL)

            if think_matches:
                # 提取思考内容
                thinking_content = think_matches[0].strip()
                # 移除 <think> 标签，保留其他内容
                cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                # 格式化输出
                if thinking_content and cleaned_content:
                    # 如果既有思考又有回答
                    return f"🧠 **思考过程**:\n{thinking_content}\n\n🤖 **AI助手回复**:\n{cleaned_content}"
                elif thinking_content:
                    # 只有思考内容
                    return f"🧠 **思考过程**:\n{thinking_content}"
                else:
                    # 只有回答内容
                    return f"🤖 **AI助手回复**:\n{cleaned_content}"

        # 检查是否是中文思考标记
        if content.startswith('思考:') or content.startswith('思考：') or '思考过程' in content:
            # 处理中文思考标记
            if content.startswith(('思考:', '思考：')):
                parts = content.split('\n', 1)
                if len(parts) > 1:
                    thinking_part = parts[0].replace('思考:', '').replace('思考：', '').strip()
                    answer_part = parts[1].strip()
                    return f"🧠 **思考过程**:\n{thinking_part}\n\n🤖 **AI助手回复**:\n{answer_part}"
                else:
                    thinking_content = content.replace('思考:', '').replace('思考：', '').strip()
                    return f"🧠 **思考过程**:\n{thinking_content}"

        return content

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
            from database.models import AgentMessage, TradeOrder
            with get_db() as db:
                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id
                ).order_by(AgentMessage.created_at.asc()).all()

                # 分离系统消息和其他消息
                system_messages = []
                other_messages = []
                for message in messages:
                    if message.role == "system":
                        system_messages.append(message)
                    else:
                        other_messages.append(message)

                # 系统消息总是放在最前面，按创建时间排序
                system_messages.sort(key=lambda x: x.created_at)
                # 其他消息按创建时间排序
                other_messages.sort(key=lambda x: x.created_at)

                # 合并消息：系统消息在前，其他消息在后
                ordered_messages = system_messages + other_messages

                # 转换为流式消息格式
                history_messages = []
                for message in ordered_messages:
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
                            "arguments": json.loads(message.tool_arguments) if message.tool_arguments else {},
                            "result": json.loads(message.tool_result) if message.tool_result else {},
                            "status": "success" if message.message_type == "tool_result" else "calling",
                            "tool_call_id": message.tool_call_id or ""
                        }
                        tool_content = json.dumps(tool_data, ensure_ascii=False)

                        if message.message_type == "tool_call":
                            formatted_content = f"🔧 **工具调用**: `{tool_data['tool_name']}`\\n\\n参数: {json.dumps(tool_data.get('arguments', {}), indent=2, ensure_ascii=False)}"
                        else:
                            # 对于工具结果，如果有相关的订单ID，显示订单详情
                            if message.related_order_id and tool_data['tool_name'] in ['place_order', 'amend_order', 'cancel_order']:
                                try:
                                    order = db.query(TradeOrder).filter(
                                        TradeOrder.order_id == message.related_order_id
                                    ).first()
                                    if order:
                                        order_info = {
                                            "order_id": order.order_id,
                                            "inst_id": order.inst_id,
                                            "side": order.side,
                                            "order_type": order.order_type,
                                            "size": order.size,
                                            "price": order.price,
                                            "status": order.status,
                                            "created_at": order.created_at.isoformat() if order.created_at else None
                                        }
                                        formatted_content = f"✅ **工具完成**: `{tool_data['tool_name']}`\\n\\n订单详情: {json.dumps(order_info, indent=2, ensure_ascii=False)}\\n\\n操作结果: {json.dumps(tool_data.get('result', {}), indent=2, ensure_ascii=False)}"
                                    else:
                                        formatted_content = f"✅ **工具完成**: `{tool_data['tool_name']}`\\n\\n结果: {json.dumps(tool_data.get('result', {}), indent=2, ensure_ascii=False)}"
                                except:
                                    formatted_content = f"✅ **工具完成**: `{tool_data['tool_name']}`\\n\\n结果: {json.dumps(tool_data.get('result', {}), indent=2, ensure_ascii=False)}"
                            else:
                                formatted_content = f"✅ **工具完成**: `{tool_data['tool_name']}`\\n\\n结果: {json.dumps(tool_data.get('result', {}), indent=2, ensure_ascii=False)}"

                        history_messages.append({"role": "assistant", "content": formatted_content})
                    elif message.message_type == "play_result":
                        # 投资结果
                        history_messages.append({"role": "assistant", "content": content})
                    else:
                        # 普通消息 - 包括系统消息
                        if role == "system":
                            # 系统消息需要特殊格式化显示
                            formatted_content = f"💻 **系统提示词**:\n\n{content}"
                            history_messages.append({"role": "system", "content": formatted_content})
                        else:
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
            logger.debug(f"自动决策失败详情: {traceback.format_exc()}")

            # 分析异常类型并提供相应的用户友好消息
            error_message = str(e)
            user_message = ""

            if "Agent stopped due to max iterations" in error_message:
                logger.warning(f"PLAN {plan_id} - 自动决策达到最大迭代次数限制")
                user_message = "⚠️ 自动决策达到最大迭代次数限制（已提升至25次）。建议：\n1. 简化分析请求\n2. 分步骤进行决策\n3. 减少同时使用的工具数量\n4. 重新提交更具体的分析请求"
            elif "Agent stopped due to max execution time" in error_message or "timeout" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - 自动决策执行超时")
                user_message = "⏱️ 自动决策执行超时（已提升至10分钟）。建议：\n1. 检查网络连接\n2. 减少分析数据范围\n3. 分步骤执行复杂分析\n4. 稍后重试"
            elif "Rate limit" in error_message or "rate limit" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - API 速率限制")
                user_message = "🚦 API 调用频率超限，请稍后重试。"
            elif "Connection" in error_message or "network" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - 网络连接错误")
                user_message = "🌐 网络连接问题，请检查网络连接后重试。"
            else:
                user_message = f"❌ 自动决策出现异常: {error_message}"

            yield [{"role": "assistant", "content": user_message}]

            # 记录到系统日志
            try:
                from database.models import SystemLog
                with get_db() as db:
                    error_log = SystemLog(
                        level="WARNING",
                        category="agent_auto_decision_error",
                        message=f"自动决策异常 - Plan {plan_id}",
                        details={
                            "error_type": "auto_decision_failure",
                            "error_message": error_message,
                            "training_id": training_id,
                            "plan_id": plan_id
                        }
                    )
                    db.add(error_log)
                    db.commit()
                    logger.info(f"PLAN {plan_id} - 自动决策异常已记录到系统日志")
            except Exception as log_error:
                logger.error(f"保存自动决策系统日志失败: {log_error}")

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
            logger.debug(f"手动推理失败详情: {traceback.format_exc()}")

            # 分析异常类型并提供相应的用户友好消息
            error_message = str(e)
            user_message = ""

            if "Agent stopped due to max iterations" in error_message:
                logger.warning(f"PLAN {plan_id} - 手动推理达到最大迭代次数限制")
                user_message = "⚠️ 手动推理达到最大迭代次数限制（已提升至25次）。建议：\n1. 简化推理请求\n2. 分步骤进行分析\n3. 一次专注于单一问题\n4. 重新提交更具体的推理请求"
            elif "Agent stopped due to max execution time" in error_message or "timeout" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - 手动推理执行超时")
                user_message = "⏱️ 手动推理执行超时（已提升至10分钟）。建议：\n1. 检查网络连接状态\n2. 减少数据查询范围\n3. 避免同时执行多个复杂操作\n4. 稍后重试或简化问题"
            elif "Rate limit" in error_message or "rate limit" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - API 速率限制")
                user_message = "🚦 API 调用频率超限，请稍后重试。"
            elif "Connection" in error_message or "network" in error_message.lower():
                logger.warning(f"PLAN {plan_id} - 网络连接错误")
                user_message = "🌐 网络连接问题，请检查网络连接后重试。"
            else:
                user_message = f"❌ 手动推理出现异常: {error_message}"

            yield [{"role": "assistant", "content": user_message}]

            # 记录到系统日志
            try:
                from database.models import SystemLog
                with get_db() as db:
                    error_log = SystemLog(
                        level="WARNING",
                        category="agent_manual_inference_error",
                        message=f"手动推理异常 - Plan {plan_id}",
                        details={
                            "error_type": "manual_inference_failure",
                            "error_message": error_message,
                            "plan_id": plan_id
                        }
                    )
                    db.add(error_log)
                    db.commit()
                    logger.info(f"PLAN {plan_id} - 手动推理异常已记录到系统日志")
            except Exception as log_error:
                logger.error(f"保存手动推理系统日志失败: {log_error}")

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
        """获取最新完整批次的预测数据"""
        try:
            with get_db() as db:
                if training_id:
                    # 指定训练记录：获取该训练记录的最新批次ID
                    latest_batch_id = db.query(PredictionData.inference_batch_id).filter(
                        PredictionData.training_record_id == training_id
                    ).order_by(PredictionData.prediction_time.desc()).limit(1).scalar()

                    if not latest_batch_id:
                        logger.warning(f"训练记录 {training_id} 没有找到预测数据")
                        return []

                    # 获取该批次的所有预测数据
                    predictions = db.query(PredictionData).filter(
                        PredictionData.training_record_id == training_id,
                        PredictionData.inference_batch_id == latest_batch_id
                    ).order_by(PredictionData.timestamp).all()

                    logger.info(f"获取训练记录 {training_id} 的最新批次 {latest_batch_id}，共 {len(predictions)} 条预测数据")
                else:
                    # 获取计划的最新批次ID
                    latest_batch_id = db.query(PredictionData.inference_batch_id).filter(
                        PredictionData.plan_id == plan_id
                    ).order_by(PredictionData.prediction_time.desc()).limit(1).scalar()

                    if not latest_batch_id:
                        logger.warning(f"计划 {plan_id} 没有找到预测数据")
                        return []

                    # 获取该批次的所有预测数据
                    predictions = db.query(PredictionData).filter(
                        PredictionData.plan_id == plan_id,
                        PredictionData.inference_batch_id == latest_batch_id
                    ).order_by(PredictionData.timestamp).all()

                    if predictions:
                        latest_training_id = predictions[0].training_record_id
                        logger.info(f"获取计划 {plan_id} 的最新批次 {latest_batch_id}，训练记录 {latest_training_id}，共 {len(predictions)} 条预测数据")
                    else:
                        logger.warning(f"计划 {plan_id} 的最新批次 {latest_batch_id} 没有预测数据")

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
                upward_prob = f"{float(pred.upward_probability):.2%}" if pred.upward_probability else "N/A"
                volatility_prob = f"{float(pred.volatility_amplification_probability):.2%}" if pred.volatility_amplification_probability else "N/A"

                pred_text.append(
                    f"时间: {pred.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                    f"预测价格: {float(pred.close) if pred.close else 'N/A'}, "
                    f"上涨概率: {upward_prob}, "
                    f"波动率: {volatility_prob}"
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
请基于量化分析结果做出决策，而不是主观猜测。
请分析这些预测数据，然后决定执行交易相应的工具进行操作。"""

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

    async def handle_new_kline_data(self, plan_id: int, inst_id: str, kline_data: dict) -> bool:
        """
        处理新的K线数据，触发自动Agent决策

        Args:
            plan_id: 计划ID
            inst_id: 交易对ID
            kline_data: K线数据

        Returns:
            bool: 是否成功触发
        """
        try:
            logger.info(f"处理新K线数据: plan_id={plan_id}, inst_id={inst_id}")

            # 检查计划是否存在且启用自动决策
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在")
                    return False

                if not plan.auto_agent_enabled:
                    logger.info(f"计划 {plan_id} 未启用自动Agent决策")
                    return False

                # 获取LLM配置
                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    logger.warning(f"计划 {plan_id} 的LLM配置不存在")
                    return False

            # 获取最新的对话会话
            with get_db() as db:
                latest_conversation = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(AgentConversation.created_at.desc()).first()

                if not latest_conversation:
                    logger.info(f"计划 {plan_id} 没有找到对话会话，创建新会话")
                    # 创建新的对话会话
                    latest_conversation = AgentConversation(
                        plan_id=plan_id,
                        title=f"K线事件触发 - {inst_id}",
                        status="active"
                    )
                    db.add(latest_conversation)
                    db.commit()
                    db.refresh(latest_conversation)

                # 添加K线数据消息到对话
                # 处理datetime序列化问题
                safe_kline_data = {}
                for key, value in kline_data.items():
                    if isinstance(value, datetime):
                        safe_kline_data[key] = value.isoformat()
                    elif hasattr(value, '__dict__'):  # 处理复杂对象
                        safe_kline_data[key] = str(value)
                    else:
                        safe_kline_data[key] = value

                kline_message = AgentMessage(
                    conversation_id=latest_conversation.id,
                    role="user",
                    message_type="user_message",
                    content="new_k_line_data",
                    tool_arguments=safe_kline_data
                )
                db.add(kline_message)

                # 添加系统消息说明
                system_message = AgentMessage(
                    conversation_id=latest_conversation.id,
                    role="system",
                    message_type="system_message",
                    content=f"收到新的K线数据 - 交易对: {inst_id}, 时间: {kline_data.get('timestamp', 'unknown')}",
                )
                db.add(system_message)
                db.commit()

            logger.info(f"已为计划 {plan_id} 的新K线数据创建消息，会话ID: {latest_conversation.id}")
            return True

        except Exception as e:
            logger.error(f"处理新K线数据失败: plan_id={plan_id}, error={e}")
            logger.debug(f"处理新K线数据失败详情: {traceback.format_exc()}")
            return False

    async def handle_order_event(self, plan_id: int, event_type: str, order_data: dict) -> bool:
        """
        处理订单事件 (buy_order_done / sell_order_done)

        Args:
            plan_id: 计划ID
            event_type: 事件类型 (buy_order_done, sell_order_done)
            order_data: 订单数据

        Returns:
            bool: 是否成功触发
        """
        try:
            logger.info(f"处理订单事件: plan_id={plan_id}, event_type={event_type}, order_id={order_data.get('order_id')}")

            # 检查计划是否存在且启用自动Agent决策
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在")
                    return False

                if not plan.auto_agent_enabled:
                    logger.info(f"计划 {plan_id} 未启用自动Agent决策")
                    return False

                # 获取LLM配置
                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    logger.warning(f"计划 {plan_id} 的LLM配置不存在")
                    return False

            # 获取或创建最新的对话会话
            with get_db() as db:
                latest_conversation = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(AgentConversation.created_at.desc()).first()

                if not latest_conversation:
                    logger.info(f"计划 {plan_id} 没有找到对话会话，创建新会话")
                    # 创建新的对话会话
                    latest_conversation = AgentConversation(
                        plan_id=plan_id,
                        title=f"订单事件 - {order_data.get('inst_id', 'unknown')}",
                        status="active"
                    )
                    db.add(latest_conversation)
                    db.commit()
                    db.refresh(latest_conversation)

                # 记录订单事件到数据库
                order_event_log = OrderEventLog(
                    plan_id=plan_id,
                    event_type=event_type,
                    order_id=order_data.get('order_id', ''),
                    inst_id=order_data.get('inst_id', ''),
                    side=order_data.get('side', ''),
                    event_data=order_data,
                    agent_conversation_id=latest_conversation.id
                )
                db.add(order_event_log)

                # 添加订单事件消息到对话
                order_message = AgentMessage(
                    conversation_id=latest_conversation.id,
                    role="user",
                    message_type="order_event",
                    content=event_type,
                    tool_arguments=order_data
                )
                db.add(order_message)
                db.commit()

            logger.info(f"已为计划 {plan_id} 的订单事件创建消息，会话ID: {latest_conversation.id}")
            return True

        except Exception as e:
            logger.error(f"处理订单事件失败: plan_id={plan_id}, error={e}")
            logger.debug(f"处理订单事件失败详情: {traceback.format_exc()}")
            return False

    async def auto_decision_wrapper(
        self,
        plan_id: int,
        training_id: int = None,
        prediction_data: List[Dict] = None
    ) -> None:
        """
        auto_decision 的包装方法，正确处理 AsyncGenerator
        供 inference_service.py 中的 asyncio.create_task 调用

        增强版本：添加完整上下文日志输出

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID（可选）
            prediction_data: 预测数据（可选，如果不提供则从数据库获取）
        """
        try:
            from config import config
            detailed_logging = config.AGENT_DETAILED_LOGGING

            logger.info(f"🤖 [AGENT推理开始] plan_id={plan_id}, training_id={training_id}")

            # 只获取 system prompt
            if detailed_logging:
                try:
                    with get_db() as db:
                        plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                        if plan:
                            tools_config = plan.agent_tools_config or {}
                            system_prompt = self._build_system_prompt(plan, tools_config)

                            # 只输出 system prompt
                            logger.info(f"🔧 [System Prompt] plan_id={plan_id}")
                            logger.info(f"🔧 [System Prompt Content]\n{system_prompt}")
                        else:
                            logger.error(f"❌ 计划不存在: plan_id={plan_id}")
                            return
                except Exception as e:
                    logger.error(f"❌ 获取 system prompt 失败: {e}")
                    return

            # 消费 AsyncGenerator 的所有输出
            async for messages in self.auto_decision(plan_id, training_id, prediction_data):
                # 不记录任何内容，只执行推理
                pass

            logger.info(f"✅ [AGENT推理完成] plan_id={plan_id}, training_id={training_id}")

        except Exception as e:
            logger.error(f"❌ [AGENT推理失败] plan_id={plan_id}, training_id={training_id}, error={e}")
            logger.debug(f"❌ [AGENT推理失败详情] {traceback.format_exc()}")


# 全局实例
agent_service = LangChainAgentService()