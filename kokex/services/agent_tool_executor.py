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

logger = setup_logger(__name__, "agent_tool_executor.log")


class AgentToolExecutor:
    """Agent 工具执行器"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        trading_limits: Optional[Dict] = None
    ):
        """
        初始化工具执行器

        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key
            passphrase: OKX Passphrase
            is_demo: 是否模拟盘
            trading_limits: 交易限制配置
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.trading_limits = trading_limits or {}

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

        logger.info(f"[{self.environment}] Agent 工具执行器初始化完成")

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
            return {"success": False, "error": error_msg}

        # 验证参数
        is_valid, error_msg = validate_tool_params(tool_name, params)
        if not is_valid:
            logger.error(f"[{self.environment}] 参数验证失败: {error_msg}")
            return {"success": False, "error": error_msg}

        # 检查交易限制
        is_allowed, limit_msg = self._check_trading_limits(tool_name, params)
        if not is_allowed:
            logger.warning(f"[{self.environment}] 交易限制: {limit_msg}")
            return {"success": False, "error": limit_msg}

        # 执行工具
        try:
            if tool_name == "get_account_balance":
                return await self._get_account_balance(params)
            elif tool_name == "get_account_positions":
                return await self._get_account_positions(params)
            elif tool_name == "get_order_info":
                return await self._get_order_info(params)
            elif tool_name == "get_pending_orders":
                return await self._get_pending_orders(params)
            elif tool_name == "get_order_history":
                return await self._get_order_history(params)
            elif tool_name == "get_fills":
                return await self._get_fills(params)
            elif tool_name == "get_current_price":
                return await self._get_current_price(params)
            elif tool_name == "place_limit_order":
                return await self._place_limit_order(params)
            elif tool_name == "cancel_order":
                return await self._cancel_order(params)
            elif tool_name == "amend_order":
                return await self._amend_order(params)
            elif tool_name == "cancel_all_orders":
                return await self._cancel_all_orders(params)
            elif tool_name == "get_prediction_history":
                return await self._get_prediction_history(params)
            else:
                return {"success": False, "error": f"工具 {tool_name} 未实现"}

        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            logger.error(f"[{self.environment}] {error_msg}", exc_info=True)
            return {"success": False, "error": error_msg}

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
        endpoint = "/api/v5/trade/order"
        query_params = {"instId": params["inst_id"]}

        if params.get("order_id"):
            query_params["ordId"] = params["order_id"]
        if params.get("client_order_id"):
            query_params["clOrdId"] = params["client_order_id"]

        result = self.rest_client._request("GET", endpoint, params=query_params, auth=True)
        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "查询失败")}

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
        endpoint = "/api/v5/trade/orders-history"
        query_params = {"instType": "SPOT"}

        if params.get("inst_id"):
            query_params["instId"] = params["inst_id"]
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
        """下限价单"""
        await self._ensure_ws_connected()

        result = await self.ws_client.place_order(
            inst_id=params["inst_id"],
            side=params["side"],
            order_type="limit",
            size=params["size"],
            price=params["price"],
            client_order_id=params.get("client_order_id")
        )

        if result.get("code") == "0":
            return {"success": True, "data": result.get("data", [])}
        else:
            return {"success": False, "error": result.get("msg", "下单失败")}

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

    async def _cancel_all_orders(self, params: Dict) -> Dict:
        """批量撤单"""
        # 先查询所有挂单
        pending_result = await self._get_pending_orders(params)
        if not pending_result["success"]:
            return pending_result

        orders = pending_result["data"]
        if not orders:
            return {"success": True, "message": "没有需要撤销的订单", "data": []}

        # 逐个撤单
        results = []
        for order in orders:
            cancel_result = await self._cancel_order({
                "inst_id": order["instId"],
                "order_id": order["ordId"]
            })
            results.append(cancel_result)

        success_count = sum(1 for r in results if r["success"])
        return {
            "success": True,
            "message": f"撤销了 {success_count}/{len(orders)} 个订单",
            "data": results
        }

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
            # 这里假设该方法在Agent上下文中调用，需要plan_id
            # 我们从self中获取plan_id（需要在初始化时保存）
            # 由于executor初始化时没有plan_id，我们先检查params中是否有

            if not training_id:
                # 如果params中有plan_id，使用它获取最新训练记录
                plan_id = params.get("plan_id")
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
                        "时间": pred['timestamp'].strftime('%Y-%m-%d %H:%M'),
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
                    "推理时间": batch['inference_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    "预测数量": f"{batch['predictions_count']}条",
                    "预测时间范围": f"{time_range['start'].strftime('%m-%d %H:%M')} ~ {time_range['end'].strftime('%m-%d %H:%M')}"
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

    async def close(self):
        """关闭连接"""
        if self.ws_client:
            await self.ws_client.close()
