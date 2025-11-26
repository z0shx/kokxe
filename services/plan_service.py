"""
交易计划业务逻辑服务
"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, List
from database.db import get_db
from database.models import TradingPlan, SystemLog
from services.data_sync_service import DataSyncService
from services.plan_agent_service import PlanAgentService
from services.ws_connection_manager import ws_connection_manager
from utils.logger import setup_logger

logger = setup_logger(__name__, "plan_service.log")


class PlanService:
    """交易计划服务"""

    # 全局 WebSocket 管理器
    ws_managers: Dict[int, DataSyncService] = {}
    ws_tasks: Dict[int, asyncio.Task] = {}

    # 全局 Agent 服务管理器
    agent_services: Dict[int, PlanAgentService] = {}
    agent_tasks: Dict[int, asyncio.Task] = {}

    @classmethod
    def create_plan(
        cls,
        plan_name: str,
        inst_id: str,
        interval: str,
        data_start_time: datetime,
        data_end_time: datetime,
        finetune_params: dict,
        auto_finetune_schedule: list,
        auto_inference_interval_hours: Optional[int] = 4,
        llm_config_id: Optional[int] = None,
        agent_prompt: str = "",
        agent_tools_config: dict = None,
        trading_limits: dict = None,
        okx_api_key: str = "",
        okx_secret_key: str = "",
        okx_passphrase: str = "",
        is_demo: bool = True,
        model_version: Optional[str] = None
    ) -> Optional[int]:
        """
        创建交易计划

        Returns:
            计划ID，失败返回 None
        """
        try:
            with get_db() as db:
                # 根据时间表判断是否启用自动化功能
                auto_enabled = bool(auto_finetune_schedule)
                auto_inference_enabled = bool(auto_inference_interval_hours and auto_inference_interval_hours > 0)

                plan = TradingPlan(
                    plan_name=plan_name,
                    inst_id=inst_id,
                    interval=interval,
                    model_version=model_version,
                    data_start_time=data_start_time,
                    data_end_time=data_end_time,
                    finetune_params=finetune_params,
                    auto_finetune_schedule=auto_finetune_schedule,
                    auto_inference_interval_hours=auto_inference_interval_hours or 4,
                    auto_finetune_enabled=auto_enabled,  # 根据时间表自动设置
                    auto_inference_enabled=auto_inference_enabled,  # 根据预测时间表自动设置
                    auto_agent_enabled=auto_enabled,      # 推理完成后自动触发Agent
                    llm_config_id=llm_config_id,
                    agent_prompt=agent_prompt,
                    agent_tools_config=agent_tools_config or {},
                    trading_limits=trading_limits or {},
                    okx_api_key=okx_api_key,
                    okx_secret_key=okx_secret_key,
                    okx_passphrase=okx_passphrase,
                    is_demo=is_demo,
                    status='created'
                )

                db.add(plan)
                db.commit()
                db.refresh(plan)

                logger.info(f"创建交易计划成功: ID={plan.id}, Name={plan_name}")
                return plan.id

        except Exception as e:
            logger.error(f"创建交易计划失败: {e}")
            return None

    @classmethod
    def get_plan(cls, plan_id: int) -> Optional[TradingPlan]:
        """获取交易计划"""
        with get_db() as db:
            return db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

    @classmethod
    def get_all_plans(cls) -> List[TradingPlan]:
        """获取所有交易计划"""
        with get_db() as db:
            return db.query(TradingPlan).all()

    @classmethod
    def update_plan_status(cls, plan_id: int, status: str) -> bool:
        """更新计划状态"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if plan:
                    plan.status = status
                    db.commit()
                    logger.info(f"计划 {plan_id} 状态更新为: {status}")
                    return True
                return False
        except Exception as e:
            logger.error(f"更新计划状态失败: {e}")
            return False

    @classmethod
    def delete_plan(cls, plan_id: int) -> Dict:
        """
        删除交易计划及其关联数据

        Args:
            plan_id: 计划ID

        Returns:
            结果字典: {'success': bool, 'message': str}
        """
        try:
            with get_db() as db:
                # 获取计划
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'success': False, 'message': '计划不存在'}

                # 检查计划状态
                if plan.status == 'running':
                    return {
                        'success': False,
                        'message': '无法删除运行中的计划，请先停止计划'
                    }

                # 停止WebSocket（如果已连接）
                if plan.ws_connected:
                    from services.ws_connection_manager import ws_connection_manager
                    # stop_connection是同步方法,直接调用
                    ws_connection_manager.stop_connection(
                        inst_id=plan.inst_id,
                        interval=plan.interval,
                        is_demo=plan.is_demo
                    )

                # 删除关联数据
                from database.models import (
                    TrainingRecord, PredictionData, AgentDecision
                )

                # 获取所有训练记录ID
                training_records = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id
                ).all()
                training_ids = [r.id for r in training_records]

                # 删除预测数据
                if training_ids:
                    deleted_predictions = db.query(PredictionData).filter(
                        PredictionData.training_record_id.in_(training_ids)
                    ).delete(synchronize_session=False)
                    logger.info(f"删除预测数据: {deleted_predictions}条")

                # 删除Agent决策记录
                deleted_decisions = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).delete(synchronize_session=False)
                logger.info(f"删除Agent决策: {deleted_decisions}条")

                # 删除训练记录
                deleted_trainings = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id
                ).delete(synchronize_session=False)
                logger.info(f"删除训练记录: {deleted_trainings}条")

                # 删除计划
                plan_name = plan.plan_name
                db.delete(plan)
                db.commit()

                logger.info(f"成功删除计划: ID={plan_id}, Name={plan_name}")
                return {
                    'success': True,
                    'message': f'成功删除计划 "{plan_name}"'
                }

        except Exception as e:
            logger.error(f"删除计划失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'删除失败: {str(e)}'
            }

    @classmethod
    async def start_plan_async(cls, plan_id: int) -> Dict:
        """
        启动计划（启动定时任务）

        Args:
            plan_id: 计划ID

        Returns:
            结果字典
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'success': False, 'message': '计划不存在'}

                if plan.status == 'running':
                    return {'success': False, 'message': '计划已在运行中'}

                # 更新计划状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'status': 'running'
                })
                db.commit()

                # 启动定时任务调度器
                from services.schedule_service import ScheduleService
                success = await ScheduleService.start_schedule(plan_id)

                logger.info(f"计划已启动: plan_id={plan_id}, schedule_success={success}")

                if success:
                    jobs = ScheduleService.get_plan_jobs(plan_id)
                    return {
                        'success': True,
                        'message': f'✅ 计划已启动，创建了 {len(jobs)} 个定时任务'
                    }
                else:
                    return {
                        'success': True,
                        'message': '✅ 计划已启动（未配置定时任务）'
                    }

        except Exception as e:
            logger.error(f"启动计划失败: {e}")
            return {'success': False, 'message': f'启动失败: {str(e)}'}

    @classmethod
    async def stop_plan_async(cls, plan_id: int) -> Dict:
        """
        停止计划（停止定时任务）

        Args:
            plan_id: 计划ID

        Returns:
            结果字典
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'success': False, 'message': '计划不存在'}

                if plan.status == 'stopped':
                    return {'success': False, 'message': '计划已停止'}

                # 更新计划状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'status': 'stopped'
                })
                db.commit()

                # 停止定时任务调度器
                from services.schedule_service import ScheduleService
                success = await ScheduleService.stop_schedule(plan_id)

                logger.info(f"计划已停止: plan_id={plan_id}, schedule_success={success}")
                return {
                    'success': True,
                    'message': '✅ 计划已停止，所有定时任务已移除'
                }

        except Exception as e:
            logger.error(f"停止计划失败: {e}")
            return {'success': False, 'message': f'停止失败: {str(e)}'}

    @classmethod
    def update_ws_status(cls, plan_id: int, connected: bool) -> bool:
        """更新 WebSocket 连接状态"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if plan:
                    plan.ws_connected = connected
                    if connected:
                        plan.last_sync_time = datetime.utcnow()
                    db.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"更新 WebSocket 状态失败: {e}")
            return False

    @classmethod
    async def start_plan(cls, plan_id: int) -> bool:
        """
        启动交易计划（启动 WebSocket 数据同步）

        使用全局连接管理器，确保连接复用

        Args:
            plan_id: 计划ID

        Returns:
            是否成功
        """
        try:
            plan = cls.get_plan(plan_id)
            if not plan:
                logger.error(f"计划不存在: {plan_id}")
                return False

            logger.info(f"计划 {plan_id}: 开始启动")

            # 创建数据同步服务（用于初始化数据和检查）
            sync_service = DataSyncService(
                inst_id=plan.inst_id,
                interval=plan.interval,
                is_demo=plan.is_demo
            )

            # 检查是否需要初始化数据
            if not sync_service.check_data_exists():
                logger.info(f"计划 {plan_id}: 开始初始化数据")
                success = sync_service.initialize_data(days=90)  # 下载90天数据
                if not success:
                    logger.error(f"计划 {plan_id}: 数据初始化失败")
                    return False

            # 检查并填补数据缺失
            sync_service.check_and_fill_gaps()

            # 使用全局管理器获取或创建 WebSocket 连接（会自动复用）
            logger.info(f"计划 {plan_id}: 启动 WebSocket 数据流（使用全局连接管理器）")
            ws_service = ws_connection_manager.get_or_create_connection(
                inst_id=plan.inst_id,
                interval=plan.interval,
                is_demo=plan.is_demo,
                ui_callback=None
            )

            # 保存到本地管理器（用于追踪哪些计划使用了哪些连接）
            cls.ws_managers[plan_id] = ws_service

            logger.info(f"计划 {plan_id}: WebSocket 连接已启动或复用")

            # 启动 Agent 服务（订阅私有频道）
            try:
                agent_service = PlanAgentService(plan_id)
                agent_task = asyncio.create_task(agent_service.start())
                await agent_task  # 等待 Agent 服务启动完成

                cls.agent_services[plan_id] = agent_service
                logger.info(f"计划 {plan_id}: Agent 服务启动成功")
            except Exception as e:
                logger.error(f"计划 {plan_id}: Agent 服务启动失败: {e}")
                # 即使 Agent 服务启动失败，数据同步仍然可以工作

            # 更新计划状态
            cls.update_plan_status(plan_id, 'running')
            cls.update_ws_status(plan_id, True)

            logger.info(f"计划 {plan_id}: 启动成功")
            return True

        except Exception as e:
            logger.error(f"启动计划失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    async def stop_plan(cls, plan_id: int) -> bool:
        """
        停止交易计划

        注意：不会停止全局 WebSocket 连接（因为可能有其他计划在使用）
        只停止 Agent 服务和清理本地引用

        Args:
            plan_id: 计划ID

        Returns:
            是否成功
        """
        try:
            # 停止 Agent 服务
            if plan_id in cls.agent_services:
                await cls.agent_services[plan_id].stop()
                del cls.agent_services[plan_id]
                logger.info(f"计划 {plan_id}: Agent 服务已停止")

            # 清理本地引用（不停止全局连接，因为可能有其他计划在使用）
            if plan_id in cls.ws_managers:
                del cls.ws_managers[plan_id]
                logger.info(f"计划 {plan_id}: 已清理 WebSocket 连接引用")

            if plan_id in cls.ws_tasks:
                cls.ws_tasks[plan_id].cancel()
                del cls.ws_tasks[plan_id]

            # 更新计划状态
            cls.update_plan_status(plan_id, 'stopped')
            cls.update_ws_status(plan_id, False)

            logger.info(f"计划 {plan_id}: 停止成功（WebSocket 连接继续运行，供其他计划使用）")
            return True

        except Exception as e:
            logger.error(f"停止计划失败: {e}")
            import traceback
            traceback.print_exc()
            return False

