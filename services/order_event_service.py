"""
OKX 订单事件服务
管理订单频道订阅和事件分发，负责将订单事件触发给对应的 Agent
"""
import asyncio
import threading
from typing import Dict, Set, Optional, List
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__, "order_event_service.log")


class OrderEventService:
    """订单事件服务（单例）"""

    def __new__(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = super(OrderEventService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True

        # 计划订阅映射: {plan_id: {connection_key, inst_id, last_event_time}}
        self.plan_subscriptions: Dict[int, Dict] = {}

        # 连接订阅映射: {connection_key: {plan_ids: set, api_credentials, ws_service}}
        self.connection_subscriptions: Dict[str, Dict] = {}

        # 事件循环
        self.loop = None
        self.loop_thread = None
        self._start_event_loop()

        # 统计信息
        self.total_events_processed = 0
        self.total_events_triggered = 0
        self.start_time = datetime.now()

        logger.info("✅ 订单事件服务已初始化")

    def _start_event_loop(self):
        """在后台线程中启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info("订单事件服务事件循环已启动")
            self.loop.run_forever()
            logger.info("订单事件服务事件循环已停止")

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # 等待事件循环启动
        import time
        for _ in range(10):
            if self.loop and self.loop.is_running():
                logger.info("✅ 订单事件服务事件循环线程已就绪")
                break
            time.sleep(0.1)
        else:
            logger.error("❌ 订单事件服务事件循环启动超时")

    def _get_connection_key(self, api_key: str, is_demo: bool) -> str:
        """生成连接键"""
        env = "demo" if is_demo else "live"
        return f"{api_key}_{env}"

    async def subscribe_plan_orders(self, plan_id: int, inst_id: str, api_credentials: dict) -> bool:
        """为计划订阅订单频道"""
        try:
            connection_key = self._get_connection_key(
                api_credentials['api_key'],
                api_credentials['is_demo']
            )

            logger.info(f"为计划 {plan_id} 订阅订单频道: {inst_id}, connection_key: {connection_key}")

            # 检查是否已有连接订阅
            if connection_key not in self.connection_subscriptions:
                # 创建新的连接订阅
                from services.account_ws_service import OKXAccountWebSocket

                # 创建 WebSocket 服务
                ws_service = OKXAccountWebSocket(
                    api_key=api_credentials['api_key'],
                    secret_key=api_credentials['secret_key'],
                    passphrase=api_credentials['passphrase'],
                    is_demo=api_credentials['is_demo'],
                    order_callback=self._handle_order_callback
                )

                # 启动连接
                asyncio.run_coroutine_threadsafe(
                    self._start_websocket_connection(ws_service),
                    self.loop
                )

                # 记录连接订阅
                self.connection_subscriptions[connection_key] = {
                    'plan_ids': {plan_id},
                    'api_credentials': api_credentials,
                    'ws_service': ws_service,
                    'inst_ids': {inst_id}
                }

                logger.info(f"创建新连接订阅: {connection_key}")

            else:
                # 添加到现有连接订阅
                self.connection_subscriptions[connection_key]['plan_ids'].add(plan_id)
                self.connection_subscriptions[connection_key]['inst_ids'].add(inst_id)
                ws_service = self.connection_subscriptions[connection_key]['ws_service']

                # 订阅订单频道
                asyncio.run_coroutine_threadsafe(
                    ws_service.subscribe_orders_channel(inst_id),
                    self.loop
                )

                logger.info(f"添加到现有连接订阅: {connection_key}")

            # 记录计划订阅
            self.plan_subscriptions[plan_id] = {
                'connection_key': connection_key,
                'inst_id': inst_id,
                'last_event_time': None
            }

            logger.info(f"计划 {plan_id} 订单频道订阅成功")
            return True

        except Exception as e:
            logger.error(f"为计划 {plan_id} 订阅订单频道失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def unsubscribe_plan_orders(self, plan_id: int) -> bool:
        """取消计划订单频道订阅"""
        try:
            if plan_id not in self.plan_subscriptions:
                logger.warning(f"计划 {plan_id} 未订阅订单频道")
                return True

            subscription = self.plan_subscriptions[plan_id]
            connection_key = subscription['connection_key']
            inst_id = subscription['inst_id']

            logger.info(f"取消计划 {plan_id} 订单频道订阅: {inst_id}")

            # 从连接订阅中移除计划
            if connection_key in self.connection_subscriptions:
                connection_sub = self.connection_subscriptions[connection_key]
                connection_sub['plan_ids'].discard(plan_id)
                connection_sub['inst_ids'].discard(inst_id)

                # 如果没有计划使用此连接，关闭连接
                if not connection_sub['plan_ids']:
                    logger.info(f"连接 {connection_key} 无计划使用，关闭连接")
                    ws_service = connection_sub['ws_service']
                    asyncio.run_coroutine_threadsafe(
                        ws_service.stop(),
                        self.loop
                    )
                    del self.connection_subscriptions[connection_key]

            # 移除计划订阅
            del self.plan_subscriptions[plan_id]

            logger.info(f"计划 {plan_id} 订单频道订阅已取消")
            return True

        except Exception as e:
            logger.error(f"取消计划 {plan_id} 订单频道订阅失败: {e}")
            return False

    async def _start_websocket_connection(self, ws_service):
        """启动 WebSocket 连接"""
        try:
            await ws_service.start()
        except Exception as e:
            logger.error(f"启动 WebSocket 连接失败: {e}")

    async def _handle_order_callback(self, callback_data: dict):
        """处理订单回调"""
        try:
            order_data = callback_data['order']
            arg = callback_data['arg']
            timestamp = callback_data['timestamp']

            self.total_events_processed += 1

            logger.info(f"处理订单事件: {order_data['inst_id']} {order_data['side']} {order_data['state']}")

            # 查找订阅此交易对的计划
            inst_id = order_data['inst_id']
            interested_plans = self._find_interested_plans(inst_id)

            if not interested_plans:
                logger.debug(f"没有计划订阅交易对 {inst_id}")
                return

            # 为每个感兴趣的计划的 Agent 触发事件
            for plan_id in interested_plans:
                await self._trigger_agent_event(plan_id, order_data)

            logger.info(f"订单事件已处理，触发 {len(interested_plans)} 个计划")

        except Exception as e:
            logger.error(f"处理订单回调失败: {e}")
            import traceback
            traceback.print_exc()

    def _find_interested_plans(self, inst_id: str) -> List[int]:
        """查找订阅此交易对的计划"""
        interested_plans = []

        for plan_id, subscription in self.plan_subscriptions.items():
            if subscription['inst_id'] == inst_id:
                interested_plans.append(plan_id)

        return interested_plans

    async def _trigger_agent_event(self, plan_id: int, order_data: dict):
        """触发 Agent 事件"""
        try:
            # 确定事件类型
            event_type = order_data.get('event_type', 'unknown')

            # 只处理完成事件
            if event_type not in ['buy_order_done', 'sell_order_done']:
                logger.debug(f"跳过事件类型: {event_type}")
                return

            logger.info(f"触发计划 {plan_id} 的 Agent 事件: {event_type}")

            # 调用 Agent 服务处理事件
            from services.langchain_agent import agent_service

            success = await agent_service.handle_order_event(
                plan_id=plan_id,
                event_type=event_type,
                order_data=order_data
            )

            if success:
                self.total_events_triggered += 1
                # 更新最后事件时间
                if plan_id in self.plan_subscriptions:
                    self.plan_subscriptions[plan_id]['last_event_time'] = datetime.now()

                logger.info(f"计划 {plan_id} Agent 事件触发成功: {event_type}")
            else:
                logger.warning(f"计划 {plan_id} Agent 事件触发失败: {event_type}")

        except Exception as e:
            logger.error(f"触发计划 {plan_id} Agent 事件失败: {e}")
            import traceback
            traceback.print_exc()

    def get_subscription_status(self) -> dict:
        """获取订阅状态"""
        try:
            plan_status = {}
            for plan_id, subscription in self.plan_subscriptions.items():
                connection_key = subscription['connection_key']
                connection_sub = self.connection_subscriptions.get(connection_key, {})
                ws_service = connection_sub.get('ws_service')

                plan_status[plan_id] = {
                    'inst_id': subscription['inst_id'],
                    'connection_key': connection_key,
                    'last_event_time': subscription['last_event_time'],
                    'ws_connected': ws_service.get_status()['connected'] if ws_service else False,
                    'ws_status': ws_service.get_status() if ws_service else None
                }

            connection_status = {}
            for connection_key, connection_sub in self.connection_subscriptions.items():
                ws_service = connection_sub['ws_service']
                connection_status[connection_key] = {
                    'plan_ids': list(connection_sub['plan_ids']),
                    'inst_ids': list(connection_sub['inst_ids']),
                    'ws_status': ws_service.get_status() if ws_service else None
                }

            return {
                'total_plans': len(self.plan_subscriptions),
                'total_connections': len(self.connection_subscriptions),
                'plan_subscriptions': plan_status,
                'connection_subscriptions': connection_status,
                'total_events_processed': self.total_events_processed,
                'total_events_triggered': self.total_events_triggered,
                'uptime': str(datetime.now() - self.start_time)
            }

        except Exception as e:
            logger.error(f"获取订阅状态失败: {e}")
            return {
                'error': str(e),
                'total_plans': len(self.plan_subscriptions),
                'total_connections': len(self.connection_subscriptions)
            }

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("开始清理订单事件服务资源")

            # 停止所有 WebSocket 连接
            for connection_key, connection_sub in self.connection_subscriptions.items():
                ws_service = connection_sub['ws_service']
                try:
                    await ws_service.stop()
                    logger.info(f"已停止连接: {connection_key}")
                except Exception as e:
                    logger.error(f"停止连接失败 {connection_key}: {e}")

            # 清空订阅数据
            self.plan_subscriptions.clear()
            self.connection_subscriptions.clear()

            logger.info("✅ 订单事件服务资源清理完成")

        except Exception as e:
            logger.error(f"清理订单事件服务资源失败: {e}")


# 全局实例
order_event_service = OrderEventService()