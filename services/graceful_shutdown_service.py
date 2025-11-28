"""
优雅关闭服务
负责在程序关闭时按顺序停止所有服务
"""
import asyncio
import signal
import threading
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from utils.logger import setup_logger

logger = setup_logger(__name__, "graceful_shutdown.log")


class GracefulShutdownService:
    """优雅关闭服务管理器"""

    def __init__(self):
        self.shutdown_handlers = []
        self.is_shutting_down = False
        self.shutdown_lock = threading.Lock()
        self.background_threads = []

    def register_shutdown_handler(self, handler_func, name: str, priority: int = 0):
        """注册关闭处理器

        Args:
            handler_func: 关闭处理函数
            name: 服务名称
            priority: 优先级，数字越小越先关闭
        """
        self.shutdown_handlers.append({
            'handler': handler_func,
            'name': name,
            'priority': priority
        })
        self.shutdown_handlers.sort(key=lambda x: x['priority'])
        logger.info(f"注册关闭处理器: {name} (优先级: {priority})")

    def register_background_thread(self, thread: threading.Thread, name: str):
        """注册后台线程"""
        self.background_threads.append({
            'thread': thread,
            'name': name
        })
        logger.info(f"注册后台线程: {name}")

    async def stop_all_services(self):
        """按顺序停止所有服务"""
        with self.shutdown_lock:
            if self.is_shutting_down:
                logger.warning("关闭程序已在执行中")
                return

            self.is_shutting_down = True
            logger.info("=" * 60)
            logger.info("🚨 开始优雅关闭所有服务...")
            logger.info("=" * 60)

        try:
            # 1. 停止训练服务 (优先级: 0)
            await self._stop_training_services()

            # 2. 停止定时任务调度器 (优先级: 1)
            await self._stop_scheduler()

            # 3. 停止WebSocket连接 (优先级: 2)
            await self._stop_websocket_connections()

            # 4. 停止Agent服务 (优先级: 3)
            await self._stop_agent_services()

            # 5. 停止数据验证服务 (优先级: 4)
            await self._stop_data_validation_service()

            # 6. 停止后台线程 (优先级: 5)
            await self._stop_background_threads()

            # 7. 执行其他注册的关闭处理器
            await self._execute_registered_handlers()

            logger.info("✅ 所有服务已优雅关闭")
            return True

        except Exception as e:
            logger.error(f"❌ 优雅关闭过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _stop_training_services(self):
        """停止训练服务"""
        logger.info("🛑 停止训练服务...")
        try:
            from services.training_service import TrainingService
            from database.db import get_db
            from database.models import TrainingRecord

            with get_db() as db:
                # 查找所有运行中的训练记录
                running_records = db.query(TrainingRecord).filter(
                    TrainingRecord.status == 'running'
                ).all()

                logger.info(f"找到 {len(running_records)} 个运行中的训练记录")

                # 标记为取消状态
                for record in running_records:
                    try:
                        record.status = 'cancelled'
                        record.end_time = time.time()
                        record.completed = False
                        logger.info(f"训练记录 {record.id} 已标记为取消状态")
                    except Exception as e:
                        logger.error(f"更新训练记录 {record.id} 状态失败: {e}")

                db.commit()

            # 停止训练服务
            if hasattr(TrainingService, 'stop_all_training'):
                TrainingService.stop_all_training()
                logger.info("✅ 训练服务已停止")

        except Exception as e:
            logger.error(f"停止训练服务失败: {e}")

    async def _stop_scheduler(self):
        """停止定时任务调度器"""
        logger.info("🛑 停止定时任务调度器...")
        try:
            from services.schedule_service import ScheduleService

            # 停止调度器
            if hasattr(ScheduleService, 'shutdown_scheduler'):
                ScheduleService.shutdown_scheduler()
                logger.info("✅ 定时任务调度器已停止")

        except Exception as e:
            logger.error(f"停止调度器失败: {e}")

    async def _stop_websocket_connections(self):
        """停止WebSocket连接"""
        logger.info("🛑 停止WebSocket连接...")
        try:
            # 停止K线数据WebSocket连接
            from services.ws_connection_manager import ws_connection_manager
            try:
                await ws_connection_manager.stop_all_connections()
                logger.info("✅ K线数据WebSocket连接已停止")
            except Exception as e:
                logger.warning(f"K线数据WebSocket连接停止时出现问题: {e}")
                # 回退到同步方法
                ws_connection_manager.shutdown_all()
                logger.info("✅ K线数据WebSocket连接已停止（同步方式）")

            # 停止账户WebSocket连接
            from services.account_ws_manager import account_ws_manager
            try:
                await account_ws_manager.stop_all_connections_async()
                logger.info("✅ 账户WebSocket连接已停止")
            except Exception as e:
                logger.warning(f"账户WebSocket连接停止时出现问题: {e}")
                # 回退到同步方法
                account_ws_manager.stop_all_connections()
                logger.info("✅ 账户WebSocket连接已停止（同步方式）")

        except Exception as e:
            logger.error(f"停止WebSocket连接失败: {e}")

    async def _stop_agent_services(self):
        """停止Agent服务"""
        logger.info("🛑 停止Agent服务...")
        try:
            from services.langchain_agent_v2 import langchain_agent_v2_service

            # 停止Agent服务
            if hasattr(langchain_agent_v2_service, 'shutdown'):
                langchain_agent_v2_service.shutdown()
                logger.info("✅ Agent服务已停止")

        except Exception as e:
            logger.error(f"停止Agent服务失败: {e}")

    async def _stop_data_validation_service(self):
        """停止数据验证服务"""
        logger.info("🛑 停止数据验证服务...")
        try:
            from services.data_validation_service import data_validation_service

            # 停止数据验证服务
            if hasattr(data_validation_service, 'stop'):
                data_validation_service.stop()
                logger.info("✅ 数据验证服务已停止")

        except Exception as e:
            logger.error(f"停止数据验证服务失败: {e}")

    async def _stop_background_threads(self):
        """停止后台线程"""
        logger.info("🛑 停止后台线程...")
        try:
            # 停止注册的后台线程
            for thread_info in self.background_threads:
                thread = thread_info['thread']
                name = thread_info['name']

                if thread.is_alive():
                    try:
                        # 尝试优雅停止线程
                        if hasattr(thread, 'stop'):
                            thread.stop()
                            logger.info(f"✅ 后台线程 {name} 已停止")
                        else:
                            logger.warning(f"⚠️ 后台线程 {name} 没有停止方法，将等待超时")
                            thread.join(timeout=5.0)
                            if thread.is_alive():
                                logger.warning(f"⚠️ 后台线程 {name} 未在超时时间内停止")
                            else:
                                logger.info(f"✅ 后台线程 {name} 已停止")
                    except Exception as e:
                        logger.error(f"停止后台线程 {name} 失败: {e}")

        except Exception as e:
            logger.error(f"停止后台线程失败: {e}")

    async def _execute_registered_handlers(self):
        """执行注册的关闭处理器"""
        logger.info("🛑 执行注册的关闭处理器...")
        try:
            for handler_info in self.shutdown_handlers:
                handler = handler_info['handler']
                name = handler_info['name']

                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        # 在线程池中执行同步函数
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, handler)

                    logger.info(f"✅ 关闭处理器 {name} 已执行")
                except Exception as e:
                    logger.error(f"执行关闭处理器 {name} 失败: {e}")

        except Exception as e:
            logger.error(f"执行关闭处理器失败: {e}")

    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭...")
            # 在新线程中执行关闭，避免阻塞信号处理
            threading.Thread(
                target=self._shutdown_in_thread,
                daemon=True
            ).start()

        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        logger.info("✅ 信号处理器已设置")

    def _shutdown_in_thread(self):
        """在线程中执行关闭"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 执行关闭
            result = loop.run_until_complete(self.stop_all_services())

            if result:
                logger.info("🎉 优雅关闭完成，程序可以安全退出")
            else:
                logger.error("❌ 优雅关闭过程中出现问题")

            loop.close()

        except Exception as e:
            logger.error(f"关闭过程中发生异常: {e}")
            import traceback
            traceback.print_exc()

    def force_shutdown(self, timeout: int = 30):
        """强制关闭，用于优雅关闭超时时"""
        logger.warning(f"⚠️ 优雅关闭超时 ({timeout}秒)，强制关闭程序")
        import sys
        sys.exit(1)


# 全局实例
graceful_shutdown_service = GracefulShutdownService()


def initialize_graceful_shutdown():
    """初始化优雅关闭服务"""
    logger.info("初始化优雅关闭服务...")

    # 设置信号处理器
    graceful_shutdown_service.setup_signal_handlers()

    # 注册基本的关闭处理器
    graceful_shutdown_service.register_shutdown_handler(
        lambda: logger.info("数据库连接清理完成"),
        "数据库连接清理",
        priority=10
    )

    logger.info("✅ 优雅关闭服务初始化完成")