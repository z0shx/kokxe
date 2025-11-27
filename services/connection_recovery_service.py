"""
连接恢复服务
负责在程序启动时恢复正在运行的计划的WebSocket连接
"""
import asyncio
import threading
from typing import Dict, List
from services.ws_connection_manager import ws_connection_manager
from services.account_ws_manager import account_ws_manager
from services.kline_event_service import kline_event_service
from database.models import TradingPlan
from database.db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__, "connection_recovery.log")


class ConnectionRecoveryService:
    """连接恢复服务"""

    def __init__(self):
        """初始化服务"""
        self.loop = None
        self.loop_thread = None
        self._start_event_loop()

    def _start_event_loop(self):
        """在后台线程中启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info("连接恢复服务事件循环已启动")
            self.loop.run_forever()
            logger.info("连接恢复服务事件循环已停止")

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # 等待事件循环启动
        import time
        for _ in range(10):
            if self.loop and self.loop.is_running():
                logger.info("✅ 连接恢复服务事件循环线程已就绪")
                break
            time.sleep(0.1)
        else:
            logger.error("❌ 连接恢复服务事件循环启动超时")

    def recover_all_connections(self):
        """恢复所有正在运行的计划的连接"""
        try:
            if not self.loop or not self.loop.is_running():
                logger.warning("事件循环未运行，无法恢复连接")
                return False

            # 在事件循环中异步恢复连接
            future = asyncio.run_coroutine_threadsafe(
                self._recover_all_connections_async(),
                self.loop
            )

            # 等待恢复完成
            try:
                result = future.result(timeout=30)  # 30秒超时
                return result
            except Exception as e:
                logger.error(f"恢复连接超时或失败: {e}")
                return False

        except Exception as e:
            logger.error(f"启动连接恢复失败: {e}")
            return False

    async def _recover_all_connections_async(self):
        """异步恢复所有连接"""
        try:
            logger.info("开始恢复WebSocket连接...")

            # 获取所有正在运行的计划
            running_plans = self._get_running_plans()
            logger.info(f"找到 {len(running_plans)} 个正在运行的计划")

            if not running_plans:
                logger.info("没有正在运行的计划，无需恢复连接")
                return True

            # 按API Key分组，避免重复连接
            account_groups = self._group_plans_by_account(running_plans)
            logger.info(f"按账户分组，共 {len(account_groups)} 个账户组")

            # 恢复账户WebSocket连接
            account_recovery_count = await self._recover_account_connections(account_groups)

            # 恢复数据WebSocket连接
            data_recovery_count = await self._recover_data_connections(running_plans)

            # 订阅K线事件
            for plan in running_plans:
                kline_event_service.subscribe_plan(plan.id)

            logger.info(f"连接恢复完成: 账户连接={account_recovery_count}, 数据连接={data_recovery_count}, K线事件订阅={len(running_plans)}")
            return True

        except Exception as e:
            logger.error(f"恢复连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_running_plans(self) -> List[TradingPlan]:
        """获取所有正在运行的计划"""
        try:
            with get_db() as db:
                return db.query(TradingPlan).filter(
                    TradingPlan.status == 'running'
                ).all()
        except Exception as e:
            logger.error(f"获取运行中计划失败: {e}")
            return []

    def _group_plans_by_account(self, plans: List[TradingPlan]) -> Dict[str, List[TradingPlan]]:
        """按API Key分组计划"""
        groups = {}
        for plan in plans:
            if plan.okx_api_key:
                # 使用 (api_key, is_demo) 作为分组键
                key = (plan.okx_api_key, plan.is_demo)
                if key not in groups:
                    groups[key] = []
                groups[key].append(plan)
        return groups

    async def _recover_account_connections(self, account_groups: Dict[tuple, List[TradingPlan]]) -> int:
        """恢复账户WebSocket连接"""
        recovered_count = 0

        for (api_key, is_demo), plans in account_groups.items():
            try:
                # 直接使用API密钥（存储时未加密）
                secret_key = plans[0].okx_secret_key
                passphrase = plans[0].okx_passphrase

                if not secret_key or not passphrase:
                    logger.warning(f"计划 {plans[0].id} 的API凭证不完整，跳过账户连接恢复")
                    continue

                # 为第一个计划创建账户WebSocket连接
                service = account_ws_manager.get_or_create_connection(
                    api_key=api_key,
                    secret_key=secret_key,
                    passphrase=passphrase,
                    is_demo=is_demo,
                    plan_id=plans[0].id
                )

                if service:
                    # 为其他计划添加到现有连接
                    for plan in plans[1:]:
                        account_ws_manager.get_or_create_connection(
                            api_key=api_key,
                            secret_key=secret_key,
                            passphrase=passphrase,
                            is_demo=is_demo,
                            plan_id=plan.id
                        )
                    recovered_count += 1
                    logger.info(f"✅ 账户WebSocket连接已恢复: api_key={api_key[:8]}..., plans={len(plans)}")

            except Exception as e:
                logger.error(f"恢复账户连接失败: api_key={api_key[:8]}..., error={e}")

        return recovered_count

    async def _recover_data_connections(self, plans: List[TradingPlan]) -> int:
        """恢复数据WebSocket连接"""
        recovered_count = 0

        for plan in plans:
            try:
                # 创建或获取数据WebSocket连接
                ws_service = ws_connection_manager.get_or_create_connection(
                    inst_id=plan.inst_id,
                    interval=plan.interval,
                    is_demo=plan.is_demo,
                    ui_callback=None
                )

                if ws_service:
                    # 更新计划的WebSocket连接状态
                    with get_db() as db:
                        db.query(TradingPlan).filter(TradingPlan.id == plan.id).update({
                            'ws_connected': True
                        })
                        db.commit()

                    recovered_count += 1
                    logger.info(f"✅ 数据WebSocket连接已恢复: plan_id={plan.id}, inst_id={plan.inst_id}")

                else:
                    logger.warning(f"数据WebSocket连接恢复失败: plan_id={plan.id}")

            except Exception as e:
                logger.error(f"恢复数据连接失败: plan_id={plan.id}, error={e}")

        return recovered_count

    def get_recovery_status(self) -> Dict:
        """获取恢复状态"""
        try:
            return {
                'loop_running': self.loop.is_running() if self.loop else False,
                'kline_event_subscriptions': kline_event_service.get_active_subscriptions(),
                'account_connections': len(account_ws_manager.connections),
                'data_connections': len(ws_connection_manager.connections)
            }
        except Exception as e:
            logger.error(f"获取恢复状态失败: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """关闭服务"""
        logger.info("正在关闭连接恢复服务...")

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=5)

        logger.info("连接恢复服务已关闭")


# 全局单例
connection_recovery_service = ConnectionRecoveryService()