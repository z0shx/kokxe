"""
OKX 账户 WebSocket 连接管理器
全局单例，管理所有计划的账户WebSocket连接
每个API Key只维护一个连接，多个计划可共享
"""
import asyncio
import threading
from typing import Dict, Optional
from services.account_ws_service import OKXAccountWebSocket
from utils.logger import setup_logger

logger = setup_logger(__name__, "account_ws_manager.log")


class AccountWebSocketManager:
    """账户WebSocket连接管理器（单例）"""

    def __init__(self):
        # key: api_key, value: {'service': OKXAccountWebSocket, 'task': asyncio.Task, 'plan_ids': set()}
        self.connections: Dict[str, Dict] = {}
        self.loop = None
        self.loop_thread = None
        self._start_event_loop()

    def _start_event_loop(self):
        """在后台线程中启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info("账户WebSocket事件循环已启动")
            self.loop.run_forever()
            logger.info("账户WebSocket事件循环已停止")

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # 等待事件循环启动
        import time
        for _ in range(10):
            if self.loop and self.loop.is_running():
                logger.info("✅ 账户WebSocket事件循环线程已就绪")
                break
            time.sleep(0.1)
        else:
            logger.error("❌ 账户WebSocket事件循环启动超时")

    def _get_connection_key(self, api_key: str, is_demo: bool) -> str:
        """生成连接键（API Key + 环境）"""
        env = "demo" if is_demo else "live"
        return f"{api_key}_{env}"

    def get_or_create_connection(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool,
        plan_id: int
    ) -> Optional[OKXAccountWebSocket]:
        """
        获取或创建账户WebSocket连接

        Args:
            api_key: OKX API Key
            secret_key: OKX Secret Key
            passphrase: OKX Passphrase
            is_demo: 是否模拟盘
            plan_id: 计划ID

        Returns:
            WebSocket服务实例，失败返回None
        """
        try:
            conn_key = self._get_connection_key(api_key, is_demo)

            # 如果连接已存在
            if conn_key in self.connections:
                connection = self.connections[conn_key]
                service = connection['service']

                # 添加计划ID
                connection['plan_ids'].add(plan_id)

                logger.info(
                    f"复用已有账户WebSocket连接: key={conn_key}, "
                    f"plan_ids={connection['plan_ids']}"
                )

                return service

            # 创建新连接
            logger.info(f"创建新的账户WebSocket连接: key={conn_key}, plan_id={plan_id}")

            service = OKXAccountWebSocket(
                api_key=api_key,
                secret_key=secret_key,
                passphrase=passphrase,
                is_demo=is_demo,
                callback=None  # 可以添加全局回调
            )

            # 在后台事件循环中启动WebSocket
            if not self.loop or not self.loop.is_running():
                logger.error("事件循环未运行，无法创建连接")
                return None

            # 使用 run_coroutine_threadsafe 在后台线程的事件循环中启动
            future = asyncio.run_coroutine_threadsafe(service.start(), self.loop)

            self.connections[conn_key] = {
                'service': service,
                'future': future,
                'plan_ids': {plan_id},
                'api_key': api_key,
                'is_demo': is_demo
            }

            logger.info(f"✅ 账户WebSocket连接已创建: key={conn_key}")

            return service

        except Exception as e:
            logger.error(f"创建账户WebSocket连接失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def stop_connection(self, api_key: str, is_demo: bool, plan_id: int):
        """
        停止账户WebSocket连接（仅当没有其他计划使用时才真正停止）

        Args:
            api_key: OKX API Key
            is_demo: 是否模拟盘
            plan_id: 计划ID
        """
        try:
            conn_key = self._get_connection_key(api_key, is_demo)

            if conn_key not in self.connections:
                logger.warning(f"连接不存在: key={conn_key}")
                return

            connection = self.connections[conn_key]
            service = connection['service']
            future = connection['future']
            plan_ids = connection['plan_ids']

            # 移除计划ID
            plan_ids.discard(plan_id)

            logger.info(
                f"移除计划ID: key={conn_key}, plan_id={plan_id}, "
                f"剩余plan_ids={plan_ids}"
            )

            # 如果没有计划使用该连接，则停止
            if not plan_ids:
                logger.info(f"停止账户WebSocket连接: key={conn_key}")

                # 停止服务
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(service.stop(), self.loop)

                # 取消future
                if not future.done():
                    future.cancel()

                # 移除连接
                del self.connections[conn_key]

                logger.info(f"✅ 账户WebSocket连接已停止: key={conn_key}")
            else:
                logger.info(f"连接仍被使用，不停止: key={conn_key}")

        except Exception as e:
            logger.error(f"停止账户WebSocket连接失败: {e}")
            import traceback
            traceback.print_exc()

    def get_connection(self, api_key: str, is_demo: bool) -> Optional[OKXAccountWebSocket]:
        """获取现有连接"""
        conn_key = self._get_connection_key(api_key, is_demo)
        connection = self.connections.get(conn_key)
        return connection['service'] if connection else None

    def get_connection_status(self, api_key: str, is_demo: bool) -> Dict:
        """获取连接状态"""
        conn_key = self._get_connection_key(api_key, is_demo)
        connection = self.connections.get(conn_key)

        if not connection:
            return {
                'connected': False,
                'running': False,
                'plan_ids': set(),
                'message': '未连接'
            }

        service = connection['service']
        status = service.get_status()

        return {
            'connected': status['connected'],
            'running': status['running'],
            'plan_ids': connection['plan_ids'],
            'total_received': status['total_received'],
            'last_update': status['last_update'],
            'last_error': status['last_error']
        }

    def get_account_info(self, api_key: str, is_demo: bool) -> Optional[Dict]:
        """获取账户信息"""
        service = self.get_connection(api_key, is_demo)
        if service:
            return service.get_account_info()
        return None

    def get_all_connections(self) -> Dict:
        """获取所有连接状态"""
        result = {}
        for conn_key, connection in self.connections.items():
            service = connection['service']
            status = service.get_status()
            result[conn_key] = {
                'plan_ids': list(connection['plan_ids']),
                'api_key': connection['api_key'][:8] + '...',
                'is_demo': connection['is_demo'],
                'connected': status['connected'],
                'running': status['running'],
                'total_received': status['total_received'],
                'last_update': status['last_update']
            }
        return result

    def stop_all_connections(self):
        """停止所有连接"""
        logger.info("停止所有账户WebSocket连接...")

        for conn_key, connection in list(self.connections.items()):
            service = connection['service']
            future = connection['future']

            if self.loop and self.loop.is_running():
                # 停止服务
                asyncio.run_coroutine_threadsafe(service.stop(), self.loop)

            # 取消future
            if not future.done():
                future.cancel()

        self.connections.clear()
        logger.info("✅ 所有账户WebSocket连接已停止")

        # 停止事件循环
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=5)


# 全局单例
account_ws_manager = AccountWebSocketManager()
