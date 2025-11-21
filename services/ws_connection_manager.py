"""
全局WebSocket连接管理器
确保每个交易对+时间颗粒度全局只有一个WebSocket连接
"""
import asyncio
import threading
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    全局WebSocket连接管理器（单例模式）

    功能：
    1. 管理所有WebSocket连接，确保每个交易对+颗粒度只有一个连接
    2. 提供连接复用机制
    3. 健康检查和自动重连
    4. 统一的状态查询接口
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化管理器"""
        if self._initialized:
            return

        # 连接字典：{(inst_id, interval, is_demo): WebSocketService实例}
        self.connections: Dict[Tuple[str, str, bool], 'WebSocketDataService'] = {}

        # 连接线程字典：{(inst_id, interval, is_demo): Thread实例}
        self.connection_threads: Dict[Tuple[str, str, bool], threading.Thread] = {}

        # 事件循环字典：{(inst_id, interval, is_demo): asyncio.EventLoop}
        self.event_loops: Dict[Tuple[str, str, bool], asyncio.AbstractEventLoop] = {}

        # 健康检查线程
        self.health_check_thread = None
        self.health_check_running = False

        self._initialized = True
        logger.info("WebSocket连接管理器已初始化")

    def get_connection_key(self, inst_id: str, interval: str, is_demo: bool) -> Tuple[str, str, bool]:
        """获取连接键"""
        return (inst_id, interval, is_demo)

    def get_or_create_connection(
        self,
        inst_id: str,
        interval: str,
        is_demo: bool,
        ui_callback=None
    ):
        """
        获取或创建WebSocket连接

        如果已有连接且正在运行，则复用现有连接
        否则创建新连接

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
            ui_callback: UI回调函数（可选）

        Returns:
            WebSocketDataService实例
        """
        # 延迟导入，避免循环导入
        from .ws_data_service import WebSocketDataService

        key = self.get_connection_key(inst_id, interval, is_demo)

        with self._lock:
            # 检查是否已有连接
            if key in self.connections:
                ws_service = self.connections[key]

                # 检查连接是否还活着
                if ws_service.running and ws_service.is_connected:
                    logger.info(f"复用现有WebSocket连接: {inst_id} {interval} demo={is_demo}")

                    # 如果有新的UI回调，添加到回调列表
                    if ui_callback and ui_callback not in ws_service.ui_callbacks:
                        ws_service.ui_callbacks.append(ui_callback)

                    return ws_service
                else:
                    # 连接已断开，清理旧连接
                    logger.warning(f"旧连接已失效，创建新连接: {inst_id} {interval} demo={is_demo}")
                    self._cleanup_connection(key)

            # 创建新连接
            logger.info(f"创建新WebSocket连接: {inst_id} {interval} demo={is_demo}")
            ws_service = WebSocketDataService(
                inst_id=inst_id,
                interval=interval,
                is_demo=is_demo
            )

            # 添加UI回调
            if ui_callback:
                ws_service.ui_callbacks.append(ui_callback)

            # 保存到字典
            self.connections[key] = ws_service

            # 在新线程中启动WebSocket
            self._start_connection_in_thread(key, ws_service)

            return ws_service

    def _start_connection_in_thread(self, key: Tuple, ws_service):
        """在新线程中启动WebSocket连接"""
        def run_ws():
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.event_loops[key] = loop

            try:
                logger.info(f"启动WebSocket线程: {key}")
                loop.run_until_complete(ws_service.start())
            except Exception as e:
                logger.error(f"WebSocket线程异常: {key}, error={e}")
                # 更新数据库状态
                self._update_db_status(key, status='error', is_connected=False, error=str(e))
            finally:
                loop.close()
                logger.info(f"WebSocket线程结束: {key}")

        # 创建并启动线程（非daemon，确保可以正常关闭）
        ws_thread = threading.Thread(
            target=run_ws,
            name=f"ws-{key[0]}-{key[1]}-{key[2]}",
            daemon=False
        )
        ws_thread.start()

        # 保存线程引用
        self.connection_threads[key] = ws_thread

        logger.info(f"WebSocket线程已启动: {ws_thread.name}")

    def stop_connection(self, inst_id: str, interval: str, is_demo: bool):
        """
        停止WebSocket连接

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
        """
        key = self.get_connection_key(inst_id, interval, is_demo)

        with self._lock:
            if key not in self.connections:
                logger.warning(f"连接不存在: {inst_id} {interval} demo={is_demo}")
                return

            ws_service = self.connections[key]

            # 停止WebSocket服务
            logger.info(f"停止WebSocket连接: {inst_id} {interval} demo={is_demo}")
            ws_service.running = False

            # 等待线程结束（最多5秒）
            if key in self.connection_threads:
                thread = self.connection_threads[key]
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"线程未能在5秒内结束: {thread.name}")

            # 清理资源
            self._cleanup_connection(key)

            # 更新数据库状态
            self._update_db_status(key, status='stopped', is_connected=False)

    def _cleanup_connection(self, key: Tuple):
        """清理连接资源"""
        # 移除连接
        if key in self.connections:
            del self.connections[key]

        # 移除线程引用
        if key in self.connection_threads:
            del self.connection_threads[key]

        # 移除事件循环引用
        if key in self.event_loops:
            del self.event_loops[key]

        logger.info(f"连接资源已清理: {key}")

    def get_connection_status(self, inst_id: str, interval: str, is_demo: bool) -> dict:
        """
        获取连接状态（实时）

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘

        Returns:
            状态字典：{
                'exists': bool,          # 连接是否存在
                'running': bool,         # 是否正在运行
                'connected': bool,       # 是否已连接
                'total_received': int,   # 接收消息总数
                'total_saved': int,      # 保存数据条数
                'last_data_time': datetime,  # 最后接收数据时间
                'thread_alive': bool,    # 线程是否活着
            }
        """
        key = self.get_connection_key(inst_id, interval, is_demo)

        # 检查连接是否存在
        if key not in self.connections:
            return {
                'exists': False,
                'running': False,
                'connected': False,
                'total_received': 0,
                'total_saved': 0,
                'last_data_time': None,
                'thread_alive': False,
            }

        ws_service = self.connections[key]
        thread = self.connection_threads.get(key)

        # 获取实时状态
        status = {
            'exists': True,
            'running': ws_service.running,
            'connected': ws_service.is_connected,
            'total_received': ws_service.total_received,
            'total_saved': ws_service.total_saved,
            'last_data_time': ws_service.last_data_time,
            'thread_alive': thread.is_alive() if thread else False,
        }

        # 同步更新数据库状态
        self._update_db_status(
            key,
            status='running' if ws_service.running else 'stopped',
            is_connected=ws_service.is_connected,
            total_received=ws_service.total_received,
            total_saved=ws_service.total_saved,
            last_data_time=ws_service.last_data_time
        )

        return status

    def _update_db_status(self, key: Tuple, **kwargs):
        """更新数据库状态（包括 WebSocketSubscription 和 TradingPlan）"""
        try:
            # 延迟导入，避免循环导入
            from database.db import get_db
            from database.models import WebSocketSubscription, TradingPlan

            # 使用正确的数据库会话管理
            with get_db() as db:
                inst_id, interval, is_demo = key

                # 1. 更新 WebSocketSubscription 表
                subscription = db.query(WebSocketSubscription).filter(
                    WebSocketSubscription.inst_id == inst_id,
                    WebSocketSubscription.interval == interval,
                    WebSocketSubscription.is_demo == is_demo
                ).first()

                if subscription:
                    for k, v in kwargs.items():
                        if k == 'error':
                            subscription.last_error = v
                            subscription.last_error_time = datetime.now()
                            subscription.error_count = (subscription.error_count or 0) + 1
                        else:
                            setattr(subscription, k, v)

                    db.commit()
                    logger.info(f"数据库状态已更新: {key}, {kwargs}")

                # 2. 同步更新 TradingPlan.ws_connected 字段
                if 'is_connected' in kwargs:
                    ws_connected = kwargs['is_connected']

                    # 查找所有匹配的计划
                    plans = db.query(TradingPlan).filter(
                        TradingPlan.inst_id == inst_id,
                        TradingPlan.interval == interval,
                        TradingPlan.is_demo == is_demo
                    ).all()

                    for plan in plans:
                        if plan.ws_connected != ws_connected:
                            plan.ws_connected = ws_connected
                            logger.info(
                                f"同步 TradingPlan.ws_connected: "
                                f"plan_id={plan.id}, ws_connected={ws_connected}"
                            )

                    db.commit()

        except Exception as e:
            logger.error(f"更新数据库状态失败: {e}")

    def start_health_check(self, interval_seconds: int = 30):
        """
        启动健康检查线程

        定期检查所有连接的健康状态，自动清理失效连接

        Args:
            interval_seconds: 检查间隔（秒）
        """
        if self.health_check_running:
            logger.warning("健康检查已在运行")
            return

        def health_check_loop():
            logger.info("健康检查线程已启动")
            self.health_check_running = True

            while self.health_check_running:
                try:
                    self._perform_health_check()
                except Exception as e:
                    logger.error(f"健康检查异常: {e}")

                # 等待下一次检查
                for _ in range(interval_seconds):
                    if not self.health_check_running:
                        break
                    threading.Event().wait(1)

            logger.info("健康检查线程已停止")

        self.health_check_thread = threading.Thread(
            target=health_check_loop,
            name="ws-health-check",
            daemon=True
        )
        self.health_check_thread.start()

    def stop_health_check(self):
        """停止健康检查线程"""
        if not self.health_check_running:
            return

        logger.info("停止健康检查线程")
        self.health_check_running = False

        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)

    def _perform_health_check(self):
        """执行健康检查"""
        now = datetime.now()
        dead_keys = []

        with self._lock:
            for key, ws_service in list(self.connections.items()):
                try:
                    # 检查线程是否活着
                    thread = self.connection_threads.get(key)
                    thread_alive = thread.is_alive() if thread else False

                    # 检查最后接收数据时间（超过5分钟认为异常）
                    if ws_service.last_data_time:
                        time_diff = (now - ws_service.last_data_time).total_seconds()
                        data_timeout = time_diff > 300  # 5分钟
                    else:
                        data_timeout = False

                    # 如果线程死了或数据超时，标记为失效
                    if not thread_alive or (data_timeout and not ws_service.is_connected):
                        logger.warning(f"连接失效: {key}, thread_alive={thread_alive}, data_timeout={data_timeout}")
                        dead_keys.append(key)
                    else:
                        # 更新数据库状态（心跳）
                        self._update_db_status(
                            key,
                            status='running' if ws_service.running else 'stopped',
                            is_connected=ws_service.is_connected,
                            total_received=ws_service.total_received,
                            total_saved=ws_service.total_saved,
                            last_data_time=ws_service.last_data_time
                        )

                except Exception as e:
                    logger.error(f"健康检查失败: {key}, error={e}")

        # 清理失效连接
        for key in dead_keys:
            logger.info(f"清理失效连接: {key}")
            self._cleanup_connection(key)
            self._update_db_status(key, status='error', is_connected=False, error='连接失效（健康检查）')

    def get_all_connections(self) -> Dict[Tuple[str, str, bool], dict]:
        """
        获取所有连接的状态

        Returns:
            字典：{key: status_dict}
        """
        result = {}
        for key in self.connections.keys():
            inst_id, interval, is_demo = key
            result[key] = self.get_connection_status(inst_id, interval, is_demo)
        return result

    def shutdown_all(self):
        """关闭所有连接"""
        logger.info("关闭所有WebSocket连接")

        # 停止健康检查
        self.stop_health_check()

        # 停止所有连接
        keys = list(self.connections.keys())
        for key in keys:
            inst_id, interval, is_demo = key
            self.stop_connection(inst_id, interval, is_demo)

        logger.info("所有连接已关闭")


# 全局单例实例
ws_connection_manager = WebSocketConnectionManager()
