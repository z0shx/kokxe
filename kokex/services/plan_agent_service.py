"""
交易计划 Agent 服务
管理计划的 Agent 交互、私有频道订阅和消息处理
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Callable
from database.db import get_db
from database.models import TradingPlan, TradeOrder, SystemLog
from api.okx_websocket_client import OKXWebSocketClient
from services.agent_tool_executor import AgentToolExecutor
from services.agent_tools import get_tools_config
from utils.logger import setup_logger

logger = setup_logger(__name__, "plan_agent_service.log")


class PlanAgentService:
    """交易计划 Agent 服务"""

    def __init__(self, plan_id: int):
        """
        初始化 Agent 服务

        Args:
            plan_id: 交易计划ID
        """
        self.plan_id = plan_id
        self.plan = self._load_plan()

        if not self.plan:
            raise ValueError(f"交易计划不存在: {plan_id}")

        self.environment = "DEMO" if self.plan.is_demo else "LIVE"

        # 初始化 WebSocket 客户端
        self.ws_client: Optional[OKXWebSocketClient] = None

        # 初始化工具执行器
        self.tool_executor = AgentToolExecutor(
            api_key=self.plan.okx_api_key,
            secret_key=self.plan.okx_secret_key,
            passphrase=self.plan.okx_passphrase,
            is_demo=self.plan.is_demo,
            trading_limits=self.plan.trading_limits or {}
        )

        # Agent 消息回调
        self.agent_callback: Optional[Callable] = None

        logger.info(f"[{self.environment}] 计划 {plan_id} Agent 服务初始化完成")

    def _load_plan(self) -> Optional[TradingPlan]:
        """加载交易计划"""
        with get_db() as db:
            return db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()

    def set_agent_callback(self, callback: Callable):
        """
        设置 Agent 消息回调函数

        Args:
            callback: 回调函数 async def callback(event_type, data)
        """
        self.agent_callback = callback

    async def _handle_ws_message(self, message: Dict):
        """
        处理 WebSocket 消息

        Args:
            message: WebSocket 消息
        """
        try:
            logger.debug(f"[{self.environment}] 计划 {self.plan_id} 收到消息: {message}")

            # 解析消息类型
            if "arg" in message and "data" in message:
                channel = message["arg"].get("channel")

                # 订单频道消息
                if channel == "orders":
                    await self._handle_order_message(message["data"])

                # 成交频道消息
                elif channel == "fills":
                    await self._handle_fill_message(message["data"])

                # 账户频道消息
                elif channel == "account":
                    await self._handle_account_message(message["data"])

        except Exception as e:
            logger.error(f"[{self.environment}] 计划 {self.plan_id} 消息处理失败: {e}", exc_info=True)

    async def _handle_order_message(self, orders: list):
        """
        处理订单消息

        Args:
            orders: 订单数据列表
        """
        for order in orders:
            try:
                # 更新数据库中的订单
                with get_db() as db:
                    order_id = order.get("ordId")
                    db_order = db.query(TradeOrder).filter(TradeOrder.order_id == order_id).first()

                    if db_order:
                        # 更新订单信息
                        db_order.status = order.get("state")
                        db_order.filled_size = float(order.get("accFillSz", 0))
                        db_order.avg_price = float(order.get("avgPx", 0)) if order.get("avgPx") else None
                        db_order.updated_at = datetime.utcnow()
                        db.commit()
                    else:
                        # 新订单，创建记录
                        new_order = TradeOrder(
                            plan_id=self.plan_id,
                            order_id=order_id,
                            inst_id=order.get("instId"),
                            side=order.get("side"),
                            order_type=order.get("ordType"),
                            price=float(order.get("px", 0)) if order.get("px") else None,
                            size=float(order.get("sz", 0)),
                            status=order.get("state"),
                            filled_size=float(order.get("accFillSz", 0)),
                            avg_price=float(order.get("avgPx", 0)) if order.get("avgPx") else None,
                            is_demo=self.plan.is_demo
                        )
                        db.add(new_order)
                        db.commit()

                # 记录日志
                self._log_system_event(
                    log_type="order",
                    level="info",
                    message=f"订单更新: {order_id} - {order.get('state')}",
                    details=order
                )

                # 触发 Agent 回调
                if self.agent_callback:
                    await self.agent_callback("order_update", order)

            except Exception as e:
                logger.error(f"[{self.environment}] 计划 {self.plan_id} 处理订单消息失败: {e}", exc_info=True)

    async def _handle_fill_message(self, fills: list):
        """
        处理成交消息

        Args:
            fills: 成交数据列表
        """
        for fill in fills:
            try:
                # 记录成交日志
                self._log_system_event(
                    log_type="fill",
                    level="info",
                    message=f"订单成交: {fill.get('ordId')} - {fill.get('fillSz')}",
                    details=fill
                )

                # 触发 Agent 回调
                if self.agent_callback:
                    await self.agent_callback("fill", fill)

            except Exception as e:
                logger.error(f"[{self.environment}] 计划 {self.plan_id} 处理成交消息失败: {e}", exc_info=True)

    async def _handle_account_message(self, accounts: list):
        """
        处理账户消息

        Args:
            accounts: 账户数据列表
        """
        for account in accounts:
            try:
                # 记录账户变化日志
                self._log_system_event(
                    log_type="account",
                    level="info",
                    message="账户余额变化",
                    details=account
                )

                # 触发 Agent 回调
                if self.agent_callback:
                    await self.agent_callback("account_update", account)

            except Exception as e:
                logger.error(f"[{self.environment}] 计划 {self.plan_id} 处理账户消息失败: {e}", exc_info=True)

    def _log_system_event(self, log_type: str, level: str, message: str, details: Dict = None):
        """记录系统日志"""
        try:
            with get_db() as db:
                log = SystemLog(
                    plan_id=self.plan_id,
                    log_type=log_type,
                    level=level,
                    environment=self.environment,
                    message=message,
                    details=details
                )
                db.add(log)
                db.commit()
        except Exception as e:
            logger.error(f"[{self.environment}] 记录系统日志失败: {e}")

    async def start(self):
        """启动 Agent 服务（连接 WebSocket 并订阅私有频道）"""
        try:
            logger.info(f"[{self.environment}] 计划 {self.plan_id} 启动 Agent 服务")

            # 创建 WebSocket 客户端
            self.ws_client = OKXWebSocketClient(
                api_key=self.plan.okx_api_key,
                secret_key=self.plan.okx_secret_key,
                passphrase=self.plan.okx_passphrase,
                is_demo=self.plan.is_demo,
                on_message=self._handle_ws_message
            )

            # 连接 WebSocket
            await self.ws_client.connect()

            # 订阅订单频道
            await self.ws_client.subscribe(
                channel="orders",
                inst_type="SPOT",
                inst_id=self.plan.inst_id
            )
            logger.info(f"[{self.environment}] 计划 {self.plan_id} 已订阅订单频道")

            # 订阅成交频道
            await self.ws_client.subscribe(
                channel="fills",
                inst_type="SPOT"
            )
            logger.info(f"[{self.environment}] 计划 {self.plan_id} 已订阅成交频道")

            # 订阅账户频道（可选）
            await self.ws_client.subscribe(
                channel="account"
            )
            logger.info(f"[{self.environment}] 计划 {self.plan_id} 已订阅账户频道")

            # 更新数据库状态
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
                if plan:
                    plan.ws_connected = True
                    plan.last_sync_time = datetime.utcnow()
                    db.commit()

            logger.info(f"[{self.environment}] 计划 {self.plan_id} Agent 服务启动成功")

        except Exception as e:
            logger.error(f"[{self.environment}] 计划 {self.plan_id} 启动 Agent 服务失败: {e}", exc_info=True)
            raise

    async def stop(self):
        """停止 Agent 服务"""
        try:
            logger.info(f"[{self.environment}] 计划 {self.plan_id} 停止 Agent 服务")

            # 关闭 WebSocket
            if self.ws_client:
                await self.ws_client.close()

            # 关闭工具执行器
            await self.tool_executor.close()

            # 更新数据库状态
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
                if plan:
                    plan.ws_connected = False
                    db.commit()

            logger.info(f"[{self.environment}] 计划 {self.plan_id} Agent 服务已停止")

        except Exception as e:
            logger.error(f"[{self.environment}] 计划 {self.plan_id} 停止 Agent 服务失败: {e}", exc_info=True)

    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """
        执行 Agent 工具

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            执行结果
        """
        logger.info(f"[{self.environment}] 计划 {self.plan_id} 执行工具: {tool_name}")

        # 记录日志
        self._log_system_event(
            log_type="agent",
            level="info",
            message=f"执行工具: {tool_name}",
            details={"tool": tool_name, "params": params}
        )

        # 为工具参数添加 plan_id（用于查询预测历史等需要plan上下文的工具）
        params_with_context = params.copy()
        if "plan_id" not in params_with_context:
            params_with_context["plan_id"] = self.plan_id

        # 执行工具
        result = await self.tool_executor.execute_tool(tool_name, params_with_context)

        # 记录结果
        self._log_system_event(
            log_type="agent",
            level="info" if result.get("success") else "error",
            message=f"工具执行结果: {tool_name}",
            details={"tool": tool_name, "result": result}
        )

        return result

    def get_available_tools(self) -> list:
        """
        获取可用的工具列表

        Returns:
            工具配置列表
        """
        # 获取计划的工具配置
        plan_tools = self.plan.agent_tools_config or {}
        enabled_tools = plan_tools.get("enabled_tools", [])

        # 获取所有工具配置
        all_tools = get_tools_config()

        # 如果没有配置，返回所有查询类工具
        if not enabled_tools:
            return [tool for tool in all_tools if tool["category"] == "query"]

        # 返回启用的工具
        return [tool for tool in all_tools if tool["name"] in enabled_tools]
