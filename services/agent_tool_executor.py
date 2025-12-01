"""
Agent 工具执行器
负责执行 Agent 调用的交易工具
"""
import asyncio
from typing import Dict, Any, Optional
from api.okx_client import OKXClient
from api.okx_websocket_client import OKXWebSocketClient
from services.agent_tools import get_tool, validate_tool_params, ToolCategory
from utils.logger import setup_logger
from utils.timezone_helper import format_datetime_full_beijing, format_datetime_short_beijing, format_time_range_utc8, format_datetime_beijing
from services.agent_error_handler import AgentErrorHandler

logger = setup_logger(__name__, "agent_tool_executor.log")


class AgentToolExecutor:
    """Agent 工具执行器"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        trading_limits: Optional[Dict] = None,
        plan_id: Optional[int] = None,
        conversation_id: Optional[int] = None
    ):
        """
        初始化工具执行器

        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key
            passphrase: OKX Passphrase
            is_demo: 是否模拟盘
            trading_limits: 交易限制配置
            plan_id: 计划ID（用于错误记录）
            conversation_id: 对话ID（用于错误记录）
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.trading_limits = trading_limits or {}
        self.plan_id = plan_id
        self.conversation_id = conversation_id

        self.environment = "DEMO" if is_demo else "LIVE"

        # 初始化 REST API 客户端
        self.rest_client = OKXClient(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            is_demo=is_demo
        )

        # WebSocket 客户端（懒加载）
        self.ws_client: Optional[OKXWebSocketClient] = None

        logger.info(f"[{self.environment}] Agent 工具执行器初始化完成，plan_id={plan_id}")

    async def _ensure_ws_connected(self):
        """确保 WebSocket 已连接"""
        if not self.ws_client or not self.ws_client.is_connected:
            self.ws_client = OKXWebSocketClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                passphrase=self.passphrase,
                is_demo=self.is_demo
            )
            await self.ws_client.connect()
            await self.ws_client.connect_business()

    @staticmethod
    def _normalize_tool_params(tool_name: str, params: Dict) -> Dict:
        """
        标准化工具参数，处理LLM可能使用的缩写参数名

        Args:
            tool_name: 工具名称
            params: 原始参数字典

        Returns:
            标准化后的参数字典
        """
        normalized_params = params.copy()

        # 处理place_order工具的参数映射
        if tool_name == "place_order":
            # 参数映射表：缩写名 -> 标准名
            param_mapping = {
                "px": "price",          # 价格
                "sz": "size",           # 数量
                "ord_type": "order_type" # 订单类型
            }

            # 应用参数映射
            for short_name, standard_name in param_mapping.items():
                if short_name in normalized_params and standard_name not in normalized_params:
                    normalized_params[standard_name] = normalized_params[short_name]
                    del normalized_params[short_name]

        return normalized_params

    def _check_trading_limits(self, tool_name: str, params: Dict) -> tuple[bool, str]:
        """
        检查交易限制

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            (是否允许, 错误信息)
        """
        # 检查是否允许交易
        if tool_name.startswith("place_") or tool_name.startswith("cancel_") or tool_name.startswith("amend_"):
            # 检查最大单笔交易金额
            max_order_amount = self.trading_limits.get("max_order_amount")
            if max_order_amount and tool_name == "place_limit_order":
                try:
                    price = float(params.get("price", 0))
                    size = float(params.get("size", 0))
                    order_amount = price * size

                    if order_amount > max_order_amount:
                        return False, f"订单金额 {order_amount} 超过限制 {max_order_amount}"
                except (ValueError, TypeError):
                    pass

            # 检查最小单笔交易金额
            min_order_amount = self.trading_limits.get("min_order_amount")
            if min_order_amount and tool_name == "place_limit_order":
                try:
                    price = float(params.get("price", 0))
                    size = float(params.get("size", 0))
                    order_amount = price * size

                    if order_amount < min_order_amount:
                        return False, f"订单金额 {order_amount} 低于最小限制 {min_order_amount}"
                except (ValueError, TypeError):
                    pass

            # 检查允许的交易对
            allowed_inst_ids = self.trading_limits.get("allowed_inst_ids", [])
            if allowed_inst_ids:
                inst_id = params.get("inst_id")
                if inst_id and inst_id not in allowed_inst_ids:
                    return False, f"不允许交易 {inst_id}，允许的交易对: {allowed_inst_ids}"

        return True, ""

    async def execute_tool(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """
        执行工具

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            执行结果
        """
        logger.info(f"[{self.environment}] 执行工具: {tool_name}, 参数: {params}")

        # 验证工具是否存在
        tool = get_tool(tool_name)
        if not tool:
            error_msg = f"工具 {tool_name} 不存在"
            logger.error(f"[{self.environment}] {error_msg}")

            # 记录错误到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=normalized_params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 参数标准化（处理LLM可能使用的缩写参数名）
        normalized_params = self.__class__._normalize_tool_params(tool_name, params)

        # 验证参数
        is_valid, error_msg = validate_tool_params(tool_name, normalized_params)
        if not is_valid:
            logger.error(f"[{self.environment}] 参数验证失败: {error_msg}")

            # 记录错误到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=normalized_params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 检查交易限制
        is_allowed, limit_msg = self._check_trading_limits(tool_name, normalized_params)
        if not is_allowed:
            logger.warning(f"[{self.environment}] 交易限制: {limit_msg}")

            # 记录交易限制错误
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=limit_msg,
                    tool_params=normalized_params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=limit_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 执行工具
        try:
            result = None
            if tool_name == "get_account_balance":
                result = await self._get_account_balance(normalized_params)
            elif tool_name == "get_account_positions":
                result = await self._get_account_positions(normalized_params)
            elif tool_name == "get_order_info":
                return await self._get_order_info(normalized_params)
            elif tool_name == "get_pending_orders":
                return await self._get_pending_orders(normalized_params)
            elif tool_name == "get_order_history":
                return await self._get_order_history(normalized_params)
            elif tool_name == "get_fills":
                return await self._get_fills(normalized_params)
            elif tool_name == "get_current_price":
                return await self._get_current_price(normalized_params)
            elif tool_name == "place_limit_order":
                return await self._place_limit_order(normalized_params)
            elif tool_name == "place_order":
                return await self._place_limit_order(normalized_params)  # 复用限价单逻辑
            elif tool_name == "cancel_order":
                return await self._cancel_order(normalized_params)
            elif tool_name == "amend_order":
                return await self._amend_order(normalized_params)
            elif tool_name == "get_prediction_history":
                return await self._get_prediction_history(normalized_params)
            elif tool_name == "query_prediction_data":
                return await self._query_prediction_data(normalized_params)
            elif tool_name == "query_historical_kline_data":
                return await self._query_historical_kline_data(normalized_params)
            elif tool_name == "run_latest_model_inference":
                return await self._run_latest_model_inference(normalized_params)
            elif tool_name == "get_current_utc_time":
                return await self._get_current_utc_time(normalized_params)
            else:
                return {"success": False, "error": f"工具 {tool_name} 未实现"}

        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            logger.error(f"[{self.environment}] {error_msg}", exc_info=True)

            # 记录异常到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=normalized_params,
                    conversation_id=self.conversation_id
                )

            # 异常情况下的响应，默认继续对话
            fallback_response = AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )
            fallback_response["continue_conversation"] = True  # 异常情况默认继续对话

            return fallback_response

    # ============================================
    # 查询类工具实现
    # ============================================

    async def _get_account_balance(self, params: Dict) -> Dict:
        """查询账户余额"""
        endpoint = "/api/v5/account/balance"
        query_params = {}
        if params.get("ccy"):
            query_params["ccy"] = params["ccy"]

        result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

    async def _get_account_positions(self, params: Dict) -> Dict:
        """查询持仓"""
        endpoint = "/api/v5/account/positions"
        query_params = {}
        if params.get("inst_id"):
            query_params["instId"] = params["inst_id"]

        result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

    async def _get_order_info(self, params: Dict) -> Dict:
        """查询订单信息"""
        try:
            inst_id = params.get("inst_id")
            order_id = params.get("order_id")
            client_order_id = params.get("client_order_id")

            if not inst_id:
                return {"success": False, "error": "缺少必需参数 inst_id"}

            if not order_id and not client_order_id:
                return {"success": False, "error": "必须提供 order_id 或 client_order_id 中的一个"}

            # 使用 REST API 直接查询订单信息
            endpoint = "/api/v5/trade/order"
            query_params = {"instId": inst_id}

            if order_id:
                query_params["ordId"] = order_id
            if client_order_id:
                query_params["clOrdId"] = client_order_id

            result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)

            if result.get("code") == "0":
                return {"success": True, "data": result.get("data", [])}
            else:
                return {"success": False, "error": result.get("msg", "查询失败")}

        except Exception as e:
            logger.error(f"[{self.environment}] 查询订单信息失败: {e}")
            return {"success": False, "error": f"查询订单信息失败: {str(e)}"}

    async def _get_pending_orders(self, params: Dict) -> Dict:
        """查询未成交订单"""
        endpoint = "/api/v5/trade/orders-pending"
        query_params = {"instType": "SPOT"}
        if params.get("inst_id"):
            query_params["instId"] = params["inst_id"]

        result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

    async def _get_order_history(self, params: Dict) -> Dict:
        """查询历史订单"""
        try:
            inst_id = params.get("inst_id")
            begin = params.get("begin")
            end = params.get("end")
            limit = params.get("limit", "100")

            # 使用 REST API 直接查询历史订单
            endpoint = "/api/v5/trade/orders-history-archive"
            query_params = {"instType": "SPOT"}

            if inst_id:
                query_params["instId"] = inst_id
            if begin:
                query_params["begin"] = begin
            if end:
                query_params["end"] = end
            if limit:
                query_params["limit"] = str(limit)

            result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)

            if result.get("code") == "0":
                return {"success": True, "data": result.get("data", [])}
            else:
                return {"success": False, "error": result.get("msg", "查询失败")}

        except Exception as e:
            logger.error(f"[{self.environment}] 查询历史订单失败: {e}")
            return {"success": False, "error": f"查询历史订单失败: {str(e)}"}

    async def _get_fills(self, params: Dict) -> Dict:
        """查询成交明细"""
        endpoint = "/api/v5/trade/fills"
        query_params = {"instType": "SPOT"}

        if params.get("inst_id"):
            query_params["instId"] = params["inst_id"]
        if params.get("order_id"):
            query_params["ordId"] = params["order_id"]
        if params.get("begin"):
            query_params["begin"] = params["begin"]
        if params.get("end"):
            query_params["end"] = params["end"]
        if params.get("limit"):
            query_params["limit"] = params["limit"]

        result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

    async def _get_current_price(self, params: Dict) -> Dict:
        """获取当前价格"""
        endpoint = "/api/v5/market/ticker"
        query_params = {"instId": params["inst_id"]}

        result = self.rest_client._request("GET", endpoint, params=query_params)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

    # ============================================
    # 交易类工具实现（使用 WebSocket）
    # ============================================

    async def _place_limit_order(self, params: Dict) -> Dict:
        """下限价单（集成资金管理策略）"""
        from services.capital_management_service import CapitalManagementService

        inst_id = params["inst_id"]
        side = params["side"]
        price = float(params["price"])
        custom_size = params.get("size")
        custom_amount = params.get("total_amount")
        client_order_id = params.get("client_order_id")

        # 使用资金管理服务进行下单
        capital_service = CapitalManagementService(self.plan_id)

        try:
            result = await capital_service.place_order_with_capital_management(
                inst_id=inst_id,
                side=side,
                price=price,
                custom_amount=custom_amount,
                custom_size=custom_size if custom_size else None,
                client_order_id=client_order_id
            )

            # 格式化返回结果
            if result.get('success'):
                return {
                    "success": True,
                    "data": [{
                        "ordId": result.get('order_id'),
                        "sCode": "0",
                        "sMsg": ""
                    }],
                    "capital_management": result.get('capital_management', {}),
                    "order_details": {
                        "inst_id": inst_id,
                        "side": side,
                        "price": price,
                        "size": result.get('capital_management', {}).get('order_amount', 0) / price,
                        "amount": result.get('capital_management', {}).get('order_amount', 0)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.get('error', '下单失败'),
                    "capital_info": result.get('capital_management', {}).get('capital_info', {})
                }

        except Exception as e:
            logger.error(f"资金管理下单异常: {e}")
            return {
                "success": False,
                "error": f"资金管理下单异常: {str(e)}"
            }

    async def _cancel_order(self, params: Dict) -> Dict:
        """撤单"""
        await self._ensure_ws_connected()

        result = await self.ws_client.cancel_order(
            inst_id=params["inst_id"],
            order_id=params.get("order_id"),
            client_order_id=params.get("client_order_id")
        )

        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "撤单失败")}

    async def _amend_order(self, params: Dict) -> Dict:
        """改单"""
        await self._ensure_ws_connected()

        result = await self.ws_client.amend_order(
            inst_id=params["inst_id"],
            order_id=params.get("order_id"),
            client_order_id=params.get("client_order_id"),
            new_size=params.get("new_size"),
            new_price=params.get("new_price")
        )

        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "改单失败")}

    
    async def _get_prediction_history(self, params: Dict) -> Dict:
        """查询历史预测数据"""
        from services.inference_service import InferenceService
        from database.db import get_db
        from database.models import TrainingRecord, TradingPlan
        from sqlalchemy import and_, desc

        try:
            # 如果没有指定training_id，需要从当前计划获取
            training_id = params.get("training_id")
            inference_batch_id = params.get("inference_batch_id")
            limit = params.get("limit", 10)

            # 限制最大返回数量
            if limit > 50:
                limit = 50

            # 如果未指定training_id，需要通过当前计划获取
            if not training_id:
                # 优先使用self.plan_id，如果没有则从params中获取
                plan_id = getattr(self, 'plan_id', None) or params.get("plan_id")

                if not plan_id:
                    return {
                        "success": False,
                        "error": "未指定training_id且无法获取当前计划的训练记录"
                    }

                with get_db() as db:
                    latest_training = db.query(TrainingRecord).filter(
                        and_(
                            TrainingRecord.plan_id == plan_id,
                            TrainingRecord.status == 'completed',
                            TrainingRecord.is_active == True
                        )
                    ).order_by(desc(TrainingRecord.created_at)).first()

                    if not latest_training:
                        return {
                            "success": False,
                            "error": "未找到可用的训练记录"
                        }

                    training_id = latest_training.id

            # 如果指定了批次ID，返回该批次的详细数据
            if inference_batch_id:
                predictions = InferenceService.get_prediction_data_by_batch(
                    training_id=training_id,
                    inference_batch_id=inference_batch_id
                )

                # 格式化返回数据
                formatted_data = []
                for pred in predictions[:50]:  # 限制最多返回50条
                    formatted_data.append({
                        "时间": format_datetime_beijing(pred['timestamp'], '%Y-%m-%d %H:%M'),
                        "开盘价": f"{pred['open']:.2f}",
                        "最高价": f"{pred['high']:.2f}",
                        "最低价": f"{pred['low']:.2f}",
                        "收盘价": f"{pred['close']:.2f}",
                        "不确定性范围": f"{pred.get('close_min', pred['close']):.2f} ~ {pred.get('close_max', pred['close']):.2f}" if pred.get('close_min') else "N/A",
                        "上涨概率": f"{pred.get('upward_probability', 0) * 100:.1f}%" if pred.get('upward_probability') is not None else "N/A",
                        "波动性放大": f"{pred.get('volatility_amplification_probability', 0) * 100:.1f}%" if pred.get('volatility_amplification_probability') is not None else "N/A"
                    })

                return {
                    "success": True,
                    "message": f"找到{len(predictions)}条预测数据",
                    "data": {
                        "batch_id": inference_batch_id,
                        "total": len(predictions),
                        "predictions": formatted_data
                    }
                }

            # 否则返回所有批次列表
            batches = InferenceService.list_inference_batches(training_id)
            batches = batches[:limit]  # 限制返回数量

            # 格式化批次数据
            formatted_batches = []
            for batch in batches:
                time_range = batch['time_range']
                formatted_batches.append({
                    "批次ID": batch['inference_batch_id'][:16] + "...",  # 缩短显示
                    "完整批次ID": batch['inference_batch_id'],
                    "推理时间": format_datetime_full_beijing(batch['inference_time']),
                    "预测数量": f"{batch['predictions_count']}条",
                    "预测时间范围": format_time_range_utc8(time_range['start'], time_range['end'], '%m-%d %H:%M')
                })

            return {
                "success": True,
                "message": f"找到{len(batches)}个推理批次",
                "data": {
                    "training_id": training_id,
                    "total_batches": len(batches),
                    "batches": formatted_batches
                }
            }

        except Exception as e:
            logger.error(f"查询预测历史失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"查询失败: {str(e)}"
            }

    async def _query_historical_kline_data(self, params: Dict) -> Dict:
        """查询历史K线数据"""
        from database.db import get_db
        from database.models import KlineData
        from sqlalchemy import and_, asc, desc
        from datetime import datetime

        try:
            inst_id = params["inst_id"]
            interval = params.get("interval", "1H")
            start_time = params.get("start_time")
            end_time = params.get("end_time")
            limit = min(params.get("limit", 100), 500)  # 限制最大500条
            order_by = params.get("order_by", "time_asc")

            with get_db() as db:
                # 构建查询条件
                query = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == inst_id,
                        KlineData.interval == interval
                    )
                )

                # 添加时间范围过滤
                if start_time:
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        query = query.filter(KlineData.timestamp >= start_dt)
                    except ValueError:
                        return {"success": False, "error": "开始时间格式错误，请使用 YYYY-MM-DD HH:MM:SS"}

                if end_time:
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        query = query.filter(KlineData.timestamp <= end_dt)
                    except ValueError:
                        return {"success": False, "error": "结束时间格式错误，请使用 YYYY-MM-DD HH:MM:SS"}

                # 排序
                if order_by == "time_asc":
                    query = query.order_by(asc(KlineData.timestamp))
                else:  # time_desc
                    query = query.order_by(desc(KlineData.timestamp))

                # 限制数量
                query = query.limit(limit)

                # 执行查询
                kline_records = query.all()

                if not kline_records:
                    return {
                        "success": True,
                        "message": f"未找到 {inst_id} {interval} 的历史数据",
                        "data": {
                            "inst_id": inst_id,
                            "interval": interval,
                            "count": 0,
                            "records": []
                        }
                    }

                # 格式化数据
                records = []
                for record in kline_records:
                    records.append({
                        "时间(UTC+0)": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "时间戳(毫秒)": int(record.timestamp.timestamp() * 1000),
                        "开盘价": f"{record.open:.6f}",
                        "最高价": f"{record.high:.6f}",
                        "最低价": f"{record.low:.6f}",
                        "收盘价": f"{record.close:.6f}",
                        "成交量": f"{record.volume:.4f}",
                        "成交额": f"{record.amount:.2f}"
                    })

                # 计算统计信息
                closes = [r.close for r in kline_records]
                volumes = [r.volume for r in kline_records]

                price_change = ((closes[-1] - closes[0]) / closes[0] * 100) if len(closes) > 1 and closes[0] != 0 else 0
                high_price = max(closes) if closes else 0
                low_price = min(closes) if closes else 0
                total_volume = sum(volumes) if volumes else 0

                # 获取实际查询的时间范围
                actual_start = min(r.timestamp for r in kline_records)
                actual_end = max(r.timestamp for r in kline_records)

                return {
                    "success": True,
                    "message": f"找到 {len(records)} 条 {inst_id} {interval} 历史数据",
                    "data": {
                        "inst_id": inst_id,
                        "interval": interval,
                        "count": len(records),
                        "time_range": {
                            "start": actual_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "end": actual_end.strftime("%Y-%m-%d %H:%M:%S")
                        },
                        "statistics": {
                            "价格变化(%)": f"{price_change:.2f}%",
                            "最高价": f"{high_price:.6f}",
                            "最低价": f"{low_price:.6f}",
                            "总成交量": f"{total_volume:.4f}"
                        },
                        "records": records
                    }
                }

        except Exception as e:
            logger.error(f"查询历史K线数据失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"查询失败: {str(e)}"
            }

    async def _query_prediction_data(self, params: Dict) -> Dict:
        """查询数据库中的预测数据"""
        from database.db import get_db
        from database.models import PredictionData
        from sqlalchemy import and_, asc, desc
        from datetime import datetime

        try:
            start_time = params.get("start_time")
            end_time = params.get("end_time")
            limit = min(params.get("limit", 100), 500)  # 限制最大500条
            order_by = params.get("order_by", "time_desc")
            inference_batch_id = params.get("inference_batch_id")

            with get_db() as db:
                # 构建查询条件
                query = db.query(PredictionData)

                # 添加时间范围过滤
                if start_time:
                    try:
                        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        query = query.filter(PredictionData.timestamp >= start_dt)
                    except ValueError:
                        return {"success": False, "error": f"开始时间格式错误: {start_time}"}

                if end_time:
                    try:
                        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        query = query.filter(PredictionData.timestamp <= end_dt)
                    except ValueError:
                        return {"success": False, "error": f"结束时间格式错误: {end_time}"}

                # 添加批次ID过滤
                if inference_batch_id:
                    query = query.filter(PredictionData.inference_batch_id == inference_batch_id)

                # 添加排序
                if order_by == "time_asc":
                    query = query.order_by(asc(PredictionData.timestamp))
                else:  # 默认按时间倒序
                    query = query.order_by(desc(PredictionData.timestamp))

                # 应用限制
                predictions = query.limit(limit).all()

                # 转换为字典格式
                data = []
                for pred in predictions:
                    data.append({
                        "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
                        "inference_batch_id": pred.inference_batch_id,
                        "training_record_id": pred.training_record_id,
                        "plan_id": pred.plan_id,
                        "open": pred.open,
                        "high": pred.high,
                        "low": pred.low,
                        "close": pred.close,
                        "volume": pred.volume,
                        "amount": pred.amount,
                        "close_min": pred.close_min,
                        "close_max": pred.close_max,
                        "upward_prob": getattr(pred, 'upward_prob', None),
                        "volatility_prob": getattr(pred, 'volatility_prob', None)
                    })

                return {
                    "success": True,
                    "data": data,
                    "total": len(data),
                    "message": f"查询到 {len(data)} 条预测数据"
                }

        except Exception as e:
            logger.error(f"[{self.environment}] 查询预测数据失败: {e}")
            return {"success": False, "error": f"查询预测数据失败: {str(e)}"}

    async def _run_latest_model_inference(self, params: Dict) -> Dict:
        """运行最新训练模型的推理"""
        from services.inference_service import InferenceService
        from database.db import get_db
        from database.models import TrainingRecord
        from sqlalchemy import and_, desc

        try:
            # 获取当前计划的最新训练记录
            plan_id = getattr(self, 'plan_id', None) or params.get("plan_id")

            if not plan_id:
                return {
                    "success": False,
                    "error": "无法获取当前计划的ID"
                }

            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    return {
                        "success": False,
                        "error": "未找到可用的已完成训练记录"
                    }

                logger.info(f"开始执行模型推理，训练记录ID: {latest_training.id}")

            # 执行模型推理
            success = await InferenceService.start_inference(
                training_id=latest_training.id
            )

            if success:
                return {
                    "success": True,
                    "message": f"模型推理成功完成，训练记录ID: {latest_training.id}",
                    "training_id": latest_training.id
                }
            else:
                return {
                    "success": False,
                    "error": "模型推理失败"
                }

        except Exception as e:
            logger.error(f"[{self.environment}] 执行模型推理失败: {e}")
            return {
                "success": False,
                "error": f"执行模型推理失败: {str(e)}"
            }

    async def _get_current_utc_time(self, params: Dict) -> Dict:
        """获取当前UTC时间"""
        from datetime import datetime, timezone

        try:
            # 获取当前UTC时间
            now_utc = datetime.now(timezone.utc)

            # 格式化不同的时间表示
            timestamp_ms = int(now_utc.timestamp() * 1000)
            formatted_time = now_utc.strftime("%Y-%m-%d %H:%M:%S")
            iso_time = now_utc.isoformat()

            # 转换为北京时间用于对比
            from utils.timezone_helper import convert_to_beijing_time
            beijing_time = convert_to_beijing_time(now_utc)
            formatted_beijing_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")

            return {
                "success": True,
                "message": "当前时间获取成功",
                "data": {
                    "utc_timestamp_ms": timestamp_ms,
                    "utc_time": formatted_time,
                    "utc_iso": iso_time,
                    "beijing_time": formatted_beijing_time,
                    "timezone_info": {
                        "utc": "UTC+0 (协调世界时)",
                        "beijing": "UTC+8 (北京时间)"
                    },
                    "timestamp_description": f"时间戳 {timestamp_ms} 毫秒自1970-01-01 00:00:00 UTC"
                }
            }

        except Exception as e:
            logger.error(f"获取当前时间失败: {e}")
            return {
                "success": False,
                "error": f"获取时间失败: {str(e)}"
            }

    async def close(self):
        """关闭连接"""
        if self.ws_client:
            await self.ws_client.close()
