"""
统一调度服务
整合原有 SchedulerService 和 ScheduleService 的功能，消除重复实现

主要功能：
1. 自动微调任务调度 (整合两个服务的实现)
2. 自动推理任务调度 (整合两个服务的实现)
3. 自动Agent任务调度 (从 SchedulerService 继承)
4. 每日模型清理任务 (整合两个服务的实现)
5. 统一的任务执行记录和状态管理
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, time, timezone
from typing import Dict, List, Optional
from sqlalchemy import and_, desc, asc
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from database.db import get_db
from database.models import TradingPlan, TaskExecution, TrainingRecord, now_beijing
from services.training_service import TrainingService
from services.inference_service import InferenceService
from services.task_execution_service import TaskExecutionService
from utils.timezone_helper import format_datetime_full_beijing

logger = logging.getLogger(__name__)


class UnifiedScheduler:
    """统一调度服务"""

    # 定义UTC+8时区
    BEIJING_TZ = timezone(timedelta(hours=8))

    def __init__(self):
        self.running = False
        self.scheduler = None
        self.scheduler_task = None
        self._init_scheduler()

    def _init_scheduler(self):
        """初始化 APScheduler 调度器"""
        if self.scheduler is None:
            jobstores = {
                'default': MemoryJobStore()
            }

            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                timezone='Asia/Shanghai'
            )

            logger.info("统一调度器已初始化，时区: Asia/Shanghai")

            # 添加每日模型清理任务 (北京时间凌晨2点执行)
            try:
                cleanup_trigger = CronTrigger(hour=2, minute=0, timezone='Asia/Shanghai')
                self.scheduler.add_job(
                    func=self._daily_model_cleanup_wrapper,
                    trigger=cleanup_trigger,
                    id='daily_model_cleanup',
                    name='Daily Model Cleanup',
                    replace_existing=True,
                    misfire_grace_time=3600  # 允许1小时延迟
                )
                logger.info("✅ 已添加每日模型清理任务 (02:00 Beijing)")
            except Exception as e:
                logger.error(f"❌ 添加每日模型清理任务失败: {e}")

    async def start_scheduler(self):
        """启动统一调度器"""
        if self.running:
            logger.warning("统一调度器已经在运行中")
            return

        try:
            # 启动 APScheduler
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info("APScheduler 已启动")

            # 启动异步调度循环（用于高频检查任务）
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("统一调度器已启动")

            # 输出调度器状态
            self._log_scheduler_status()

        except Exception as e:
            logger.error(f"启动统一调度器失败: {e}")
            self.running = False

    async def stop_scheduler(self):
        """停止统一调度器"""
        if not self.running:
            return

        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("APScheduler 已停止")

        logger.info("统一调度器已停止")

    async def _scheduler_loop(self):
        """调度器主循环 - 处理高频检查任务"""
        while self.running:
            try:
                # 检查计划状态变化和需要立即响应的任务
                await self._check_immediate_tasks()

                # 每30秒检查一次
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度器循环出错: {e}")
                await asyncio.sleep(30)

    async def _check_immediate_tasks(self):
        """检查需要立即响应的任务"""
        try:
            with get_db() as db:
                now = now_beijing()

                # 获取所有启用自动化的运行中计划
                plans = db.query(TradingPlan).filter(
                    TradingPlan.status == 'running'
                ).all()

                for plan in plans:
                    try:
                        # 检查自动Agent任务（需要快速响应）
                        if plan.auto_agent_enabled:
                            await self._check_immediate_agent_tasks(plan, now, db)

                    except Exception as e:
                        logger.error(f"处理计划 {plan.id} 的立即任务时出错: {e}")

        except Exception as e:
            logger.error(f"检查立即任务时出错: {e}")

    async def _check_immediate_agent_tasks(self, plan: TradingPlan, now: datetime, db):
        """检查需要立即执行的Agent任务"""
        try:
            # 检查最近是否有新的推理结果需要Agent处理
            # 这里可以实现更智能的触发逻辑，目前保持简单
            if now.minute % 30 == 0:  # 每30分钟检查一次
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan.id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if latest_training:
                    # 检查是否已经有Agent任务在最近执行过
                    recent_time = now - timedelta(minutes=25)  # 25分钟内不算重复
                    existing_task = db.query(TaskExecution).filter(
                        and_(
                            TaskExecution.plan_id == plan.id,
                            TaskExecution.task_type == 'auto_agent',
                            TaskExecution.created_at >= recent_time,
                            TaskExecution.status.in_(['pending', 'running', 'completed'])
                        )
                    ).first()

                    if not existing_task:
                        await self._create_agent_task(plan, latest_training, now, db)

        except Exception as e:
            logger.error(f"检查立即Agent任务时出错: {e}")

    async def start_plan_schedule(self, plan_id: int) -> bool:
        """
        启动计划的所有定时任务

        Args:
            plan_id: 计划ID

        Returns:
            是否成功启动
        """
        try:
            current_time_beijing = datetime.now(self.BEIJING_TZ)
            logger.info(f"开始启动计划定时任务: plan_id={plan_id}, current_time(UTC+8)={current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 获取计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return False

                logger.info(f"计划信息: plan_id={plan_id}, status={plan.status}, auto_finetune_enabled={plan.auto_finetune_enabled}, auto_inference_enabled={plan.auto_inference_enabled}")

                # 检查是否启用任何自动化功能
                if not (plan.auto_finetune_enabled or plan.auto_inference_enabled or plan.auto_agent_enabled):
                    logger.warning(f"计划未启用任何自动化功能: plan_id={plan_id}")
                    return False

            task_count = 0

            # 处理自动微调任务 (使用 ScheduleService 的实现，更精确)
            if plan.auto_finetune_enabled:
                schedule_times = plan.auto_finetune_schedule or []
                if schedule_times:
                    logger.info(f"启动自动微调任务: plan_id={plan_id}, schedule_times={schedule_times}")

                    for time_str in schedule_times:
                        try:
                            # 解析时间 (HH:MM)
                            hour, minute = map(int, time_str.split(':'))
                            logger.info(f"解析微调时间: time_str={time_str}, hour={hour}, minute={minute}")

                            # 创建cron触发器（每天指定时间执行）
                            trigger = CronTrigger(hour=hour, minute=minute, timezone='Asia/Shanghai')

                            # 任务ID：plan_id + 任务类型 + 时间
                            job_id = f"plan_{plan_id}_finetune_{time_str.replace(':', '')}"

                            # 检查任务是否已存在
                            existing_job = self.scheduler.get_job(job_id)
                            if existing_job:
                                logger.info(f"任务已存在，先移除: {job_id}")
                                self.scheduler.remove_job(job_id)

                            # 添加任务
                            self.scheduler.add_job(
                                func=self._trigger_finetune_wrapper,
                                trigger=trigger,
                                args=[plan_id],
                                id=job_id,
                                name=f"自动微调-计划{plan_id}-{time_str}",
                                replace_existing=True,
                                misfire_grace_time=300  # 允许5分钟的延迟执行
                            )

                            task_count += 1

                            # 立即检查任务的下次执行时间
                            job = self.scheduler.get_job(job_id)
                            next_run_time = job.next_run_time
                            if next_run_time:
                                next_run_beijing = next_run_time.astimezone(self.BEIJING_TZ)
                                logger.info(f"已添加自动微调任务: plan_id={plan_id}, time={time_str}, job_id={job_id}, 下次执行(UTC+8)={next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                            else:
                                logger.warning(f"微调任务创建成功但无下次执行时间: plan_id={plan_id}, time={time_str}, job_id={job_id}")

                        except Exception as e:
                            logger.error(f"创建微调任务失败: time={time_str}, error={e}")
                            continue
                else:
                    logger.warning(f"计划启用了自动微调但未配置时间: plan_id={plan_id}")

            # 处理自动预测任务（使用间隔时间模式）
            if plan.auto_inference_enabled:
                interval_hours = plan.auto_inference_interval_hours or 4
                if interval_hours > 0:
                    logger.info(f"启动自动预测任务: plan_id={plan_id}, interval_hours={interval_hours}")

                    try:
                        # 创建间隔触发器（每N小时执行一次）
                        trigger = IntervalTrigger(hours=interval_hours, timezone='Asia/Shanghai')

                        # 任务ID：plan_id + 任务类型
                        job_id = f"plan_{plan_id}_inference_interval"

                        # 检查任务是否已存在
                        existing_job = self.scheduler.get_job(job_id)
                        if existing_job:
                            logger.info(f"任务已存在，先移除: {job_id}")
                            self.scheduler.remove_job(job_id)

                        # 添加任务
                        self.scheduler.add_job(
                            func=self._trigger_inference_wrapper,
                            trigger=trigger,
                            args=[plan_id],
                            id=job_id,
                            name=f"自动预测-计划{plan_id}-{interval_hours}h间隔",
                            replace_existing=True,
                            misfire_grace_time=300  # 允许5分钟的延迟执行
                        )

                        task_count += 1

                        # 立即检查任务的下次执行时间
                        job = self.scheduler.get_job(job_id)
                        next_run_time = job.next_run_time
                        if next_run_time:
                            next_run_beijing = next_run_time.astimezone(self.BEIJING_TZ)
                            logger.info(f"已添加自动预测任务: plan_id={plan_id}, interval={interval_hours}h, job_id={job_id}, 下次执行(UTC+8)={next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            logger.warning(f"预测任务创建成功但无下次执行时间: plan_id={plan_id}, interval={interval_hours}h, job_id={job_id}")

                    except Exception as e:
                        logger.error(f"创建预测任务失败: interval_hours={interval_hours}, error={e}")
                else:
                    logger.warning(f"计划启用了自动预测但间隔时间无效: plan_id={plan_id}, interval_hours={interval_hours}")

            # 重新输出调度器状态
            self._log_scheduler_status()

            logger.info(f"启动计划定时调度成功: plan_id={plan_id}, 任务数={task_count}")
            return True

        except Exception as e:
            logger.error(f"启动计划定时调度失败: plan_id={plan_id}, error={e}")
            return False

    async def stop_plan_schedule(self, plan_id: int) -> bool:
        """
        停止计划的所有定时任务

        Args:
            plan_id: 计划ID

        Returns:
            是否成功停止
        """
        try:
            # 移除该计划的所有任务
            removed_count = 0
            for job in self.scheduler.get_jobs():
                if job.id.startswith(f"plan_{plan_id}_"):
                    self.scheduler.remove_job(job.id)
                    removed_count += 1
                    logger.info(f"移除任务: {job.id}")

            logger.info(f"停止计划定时调度成功: plan_id={plan_id}, 移除任务数={removed_count}")
            return True

        except Exception as e:
            logger.error(f"停止计划定时调度失败: plan_id={plan_id}, error={e}")
            return False

    async def _trigger_finetune(self, plan_id: int):
        """
        触发微调任务（由调度器调用）
        使用 ScheduleService 的实现，因为它更完善
        """
        try:
            current_time_beijing = datetime.now(self.BEIJING_TZ)
            logger.info(f"⏰ 定时任务触发微调: plan_id={plan_id}, time(UTC+8)={current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 检查计划状态
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return {'success': False, 'error': '计划不存在'}

                logger.info(f"计划状态检查: plan_id={plan_id}, status={plan.status}, auto_finetune_enabled={plan.auto_finetune_enabled}")

                # 检查计划是否运行中
                if plan.status != 'running':
                    logger.warning(f"计划未运行，跳过微调: plan_id={plan_id}, status={plan.status}")
                    return {'success': False, 'error': f'计划未运行: {plan.status}'}

                # 再次检查是否启用自动微调
                if not plan.auto_finetune_enabled:
                    logger.warning(f"计划未启用自动微调，跳过: plan_id={plan_id}")
                    return {'success': False, 'error': '自动微调未启用'}

                # 检查是否有时间表配置
                schedule_times = plan.auto_finetune_schedule or []
                if not schedule_times:
                    logger.warning(f"计划未配置微调时间表，跳过: plan_id={plan_id}")
                    return {'success': False, 'error': '未配置微调时间表'}

                logger.info(f"计划配置检查通过: plan_id={plan_id}, schedule_times={schedule_times}")

            # 创建任务执行记录
            task_execution = None
            try:
                # 从计划配置中找到匹配当前时间的任务
                current_datetime = datetime.now(self.BEIJING_TZ)
                current_time_str = current_datetime.strftime('%H:%M')

                # 找到匹配的时间点
                scheduled_time_str = None
                for time_str in schedule_times:
                    if time_str == current_time_str:
                        scheduled_time_str = time_str
                        break

                # 如果没有精确匹配，使用当前时间
                if not scheduled_time_str:
                    scheduled_time_str = current_time_str

                task_execution = TaskExecutionService.create_scheduled_task(
                    plan_id=plan_id,
                    task_type='auto_finetune',
                    time_str=scheduled_time_str
                )

                # 标记任务开始
                TaskExecutionService.start_task_execution(task_execution.id)

            except Exception as record_error:
                logger.error(f"创建任务执行记录失败: plan_id={plan_id}, error={record_error}")

            # 触发训练
            logger.info(f"开始调用训练服务: plan_id={plan_id}")

            try:
                training_id = await TrainingService.start_training(plan_id, manual=False)

                if training_id:
                    logger.info(f"✅ 定时微调已启动: plan_id={plan_id}, training_id={training_id}")

                    # 等待训练完成（这是关键改进：等待训练完全完成）
                    logger.info(f"等待训练完全完成: plan_id={plan_id}, training_id={training_id}")
                    max_wait_time = 3600  # 最大等待1小时
                    wait_interval = 10   # 每10秒检查一次
                    waited_time = 0

                    while waited_time < max_wait_time:
                        await asyncio.sleep(wait_interval)
                        waited_time += wait_interval

                        # 检查训练状态
                        training_status = TrainingService.get_training_status(training_id)
                        if training_status:
                            logger.info(f"训练状态检查: plan_id={plan_id}, training_id={training_id}, status={training_status['status']}, elapsed={waited_time}s")

                            if training_status['status'] in ['completed', 'failed', 'cancelled']:
                                logger.info(f"✅ 训练已完成: plan_id={plan_id}, training_id={training_id}, final_status={training_status['status']}")

                                # 记录成功结果
                                if task_execution:
                                    TaskExecutionService.complete_task_execution(
                                        task_id=task_execution.id,
                                        success=training_status['status'] == 'completed',
                                        output_data={
                                            'training_id': training_id,
                                            'final_status': training_status['status'],
                                            'duration': training_status.get('train_duration', 0)
                                        }
                                    )

                                return {
                                    'success': training_status['status'] == 'completed',
                                    'training_id': training_id,
                                    'final_status': training_status['status'],
                                    'duration': training_status.get('train_duration', 0)
                                }
                        else:
                            logger.warning(f"无法获取训练状态: plan_id={plan_id}, training_id={training_id}")

                    # 超时处理
                    logger.error(f"训练等待超时: plan_id={plan_id}, training_id={training_id}, waited={waited_time}s")
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=False,
                            error_message='训练等待超时'
                        )

                    return {'success': False, 'error': '训练等待超时', 'training_id': training_id}

                else:
                    logger.error(f"❌ 定时微调启动失败: plan_id={plan_id}")

                    # 记录失败结果
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=False,
                            error_message='训练服务启动失败'
                        )

                    return {'success': False, 'error': '训练服务启动失败'}

            except Exception as training_error:
                logger.error(f"训练服务调用失败: plan_id={plan_id}, error={training_error}")

                # 记录异常结果
                if task_execution:
                    TaskExecutionService.complete_task_execution(
                        task_id=task_execution.id,
                        success=False,
                        error_message=f'训练服务异常: {str(training_error)}'
                    )

                return {'success': False, 'error': f'训练服务异常: {str(training_error)}'}

        except Exception as e:
            logger.error(f"触发微调失败: plan_id={plan_id}, error={e}")
            return {'success': False, 'error': f'触发微调异常: {str(e)}'}

    def _trigger_finetune_wrapper(self, plan_id: int):
        """
        微调触发器包装器，用于在APScheduler中调用async函数
        """
        try:
            logger.info(f"微调触发器包装器开始: plan_id={plan_id}")

            # 检查是否已有事件循环
            try:
                loop = asyncio.get_running_loop()
                logger.info(f"检测到运行中的事件循环，使用新线程执行: plan_id={plan_id}")

                # 如果有运行中的循环，在新线程中运行
                import concurrent.futures
                def run_training_complete():
                    """确保训练完全完成（包括状态更新）的包装函数"""
                    try:
                        logger.info(f"新线程中开始执行训练: plan_id={plan_id}")
                        result = self._run_async_in_new_loop(plan_id, 'finetune')
                        logger.info(f"新线程中训练完成: plan_id={plan_id}, result={result}")
                        return result
                    except Exception as e:
                        logger.error(f"新线程中训练执行失败: plan_id={plan_id}, error={e}")
                        raise

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_training_complete)
                    # 等待训练完全完成，包括状态更新，设置更长的超时时间
                    logger.info(f"等待训练完全完成: plan_id={plan_id}")
                    result = future.result(timeout=10800)  # 3小时超时，给微调训练足够时间完成
                    logger.info(f"✅ 自动训练完全完成: plan_id={plan_id}, result={result}")

            except RuntimeError:
                # 没有运行中的循环，直接运行
                logger.info(f"没有运行中的事件循环，直接执行: plan_id={plan_id}")
                result = asyncio.run(self._trigger_finetune(plan_id))
                logger.info(f"✅ 自动训练完全完成: plan_id={plan_id}, result={result}")

        except Exception as e:
            logger.error(f"微调包装器调用失败: plan_id={plan_id}, error={e}")

    async def _trigger_inference(self, plan_id: int, manual_trigger: bool = False):
        """
        触发预测任务（由调度器调用）
        使用 ScheduleService 的实现，包含智能数据偏移计算
        """
        try:
            current_time_beijing = datetime.now(self.BEIJING_TZ)
            trigger_type = "手动" if manual_trigger else "定时"
            logger.info(f"⏰ {trigger_type}预测任务触发: plan_id={plan_id}, time(UTC+8)={current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 检查计划状态
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return

                logger.info(f"计划状态检查: plan_id={plan_id}, status={plan.status}, auto_inference_enabled={plan.auto_inference_enabled}")

                # 检查计划是否运行中
                if plan.status != 'running':
                    logger.warning(f"计划未运行，跳过预测: plan_id={plan_id}, status={plan.status}")
                    return

                # 再次检查是否启用自动预测
                if not plan.auto_inference_enabled:
                    logger.warning(f"计划未启用自动预测，跳过: plan_id={plan_id}")
                    return

                # 检查是否有间隔时间配置
                interval_hours = plan.auto_inference_interval_hours or 4
                if interval_hours <= 0:
                    logger.warning(f"计划未配置预测间隔时间，跳过: plan_id={plan_id}")
                    return

                logger.info(f"计划配置检查通过: plan_id={plan_id}, interval_hours={interval_hours}")

            # 手动触发时跳过间隔时间检查，自动触发时进行智能预测检查
            if not manual_trigger:
                # 智能预测触发：检查最新预测数据时间
                latest_prediction_time = self.check_latest_prediction_time(plan_id)
                current_time = datetime.now(self.BEIJING_TZ)

                if latest_prediction_time:
                    # 计算时间差
                    time_diff = current_time - latest_prediction_time
                    time_diff_hours = time_diff.total_seconds() / 3600

                    logger.info(f"计划 {plan_id}: 最新预测时间: {latest_prediction_time}, 距今 {time_diff_hours:.2f} 小时")

                    # 如果时间差小于配置的间隔时间，跳过本次预测
                    if time_diff_hours < interval_hours:
                        remaining_hours = interval_hours - time_diff_hours
                        logger.info(f"⏸️ 计划 {plan_id}: 预测间隔未满足，跳过本次预测。还需等待 {remaining_hours:.2f} 小时")
                        return
                    else:
                        logger.info(f"✅ 计划 {plan_id}: 预测间隔已满足，执行新的预测（间隔 {time_diff_hours:.2f} 小时）")
                else:
                    logger.info(f"✅ 计划 {plan_id}: 没有历史预测数据，执行首次预测")
            else:
                logger.info(f"✅ 计划 {plan_id}: 手动触发，跳过间隔时间检查，直接执行预测")

            # 创建任务执行记录
            task_execution = None
            try:
                # 创建预测任务记录
                current_datetime = datetime.now(self.BEIJING_TZ)

                if manual_trigger:
                    task_name = f"手动预测-计划{plan_id}"
                    task_description = "用户手动触发的预测任务"
                    trigger_type = "manual"
                    trigger_source = f"plan_{plan_id}_manual_trigger"
                else:
                    task_name = f"自动预测-计划{plan_id}-{interval_hours}h间隔"
                    task_description = f"每{interval_hours}小时自动执行一次预测"
                    trigger_type = "scheduled"
                    trigger_source = f"plan_{plan_id}_interval_scheduler"

                task_execution = TaskExecutionService.create_task_execution(
                    plan_id=plan_id,
                    task_type="auto_inference",
                    task_name=task_name,
                    task_description=task_description,
                    trigger_type=trigger_type,
                    trigger_source=trigger_source,
                    input_data={"interval_hours": interval_hours, "manual_trigger": manual_trigger}
                )

                # 标记任务开始
                TaskExecutionService.start_task_execution(task_execution.id)

            except Exception as record_error:
                logger.error(f"创建预测任务执行记录失败: plan_id={plan_id}, error={record_error}")

            # 计算智能数据偏移
            from services.inference_data_offset_service import inference_data_offset_service
            logger.info(f"计算智能数据偏移: plan_id={plan_id}")

            try:
                offset_result = inference_data_offset_service.calculate_optimal_data_offset(
                    plan_id=plan_id,
                    target_interval_hours=interval_hours,
                    manual_trigger=manual_trigger
                )

                if offset_result['success']:
                    data_offset = offset_result['data_offset']
                    logger.info(f"✅ 数据偏移计算完成: plan_id={plan_id}, offset={data_offset}")
                    logger.info(f"📊 偏移说明: {offset_result['reasoning']}")

                    # 获取最新训练记录并更新参数
                    with get_db() as db:
                        latest_training = db.query(TrainingRecord).filter(
                            TrainingRecord.plan_id == plan_id,
                            TrainingRecord.status == 'completed',
                            TrainingRecord.is_active == True
                        ).order_by(TrainingRecord.created_at.desc()).first()

                        if latest_training:
                            # 更新推理参数
                            update_success = inference_data_offset_service.update_inference_params_with_offset(
                                plan_id=plan_id,
                                training_id=latest_training.id,
                                data_offset=data_offset
                            )

                            if update_success:
                                logger.info(f"✅ 推理参数已更新: training_id={latest_training.id}, data_offset={data_offset}")
                            else:
                                logger.warning(f"⚠️ 推理参数更新失败，使用默认参数")
                        else:
                            logger.warning(f"⚠️ 未找到训练记录，无法更新推理参数")

                else:
                    logger.warning(f"⚠️ 数据偏移计算失败: {offset_result['reasoning']}")
                    data_offset = 0

            except Exception as offset_error:
                logger.error(f"数据偏移计算异常: plan_id={plan_id}, error={offset_error}")
                data_offset = 0

            # 触发预测
            from services.inference_service import InferenceService
            logger.info(f"开始调用推理服务: plan_id={plan_id}, data_offset={data_offset}")

            try:
                inference_id = await InferenceService.start_inference_by_plan(plan_id, manual=False)

                if inference_id:
                    logger.info(f"✅ 定时预测已启动: plan_id={plan_id}, inference_id={inference_id}, data_offset={data_offset}")

                    # 记录成功结果
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=True,
                            output_data={
                                'inference_id': inference_id,
                                'data_offset': data_offset,
                                'offset_reasoning': offset_result.get('reasoning', '') if 'offset_result' in locals() else ''
                            }
                        )
                else:
                    logger.error(f"❌ 定时预测启动失败: plan_id={plan_id}")

                    # 记录失败结果
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=False,
                            error_message='推理服务启动失败'
                        )

            except Exception as inference_error:
                logger.error(f"推理服务调用失败: plan_id={plan_id}, error={inference_error}")

                # 记录异常结果
                if task_execution:
                    TaskExecutionService.complete_task_execution(
                        task_id=task_execution.id,
                        success=False,
                        error_message=f'推理服务异常: {str(inference_error)}'
                    )

        except Exception as e:
            logger.error(f"触发预测失败: plan_id={plan_id}, error={e}")

    def _trigger_inference_wrapper(self, plan_id: int):
        """
        预测触发器包装器，用于在APScheduler中调用async函数
        """
        try:
            # 检查是否已有事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果有运行中的循环，在新线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_async_in_new_loop, plan_id, 'inference')
                    future.result()
            except RuntimeError:
                # 没有运行中的循环，直接运行
                asyncio.run(self._trigger_inference(plan_id, manual_trigger=True))
        except Exception as e:
            logger.error(f"预测包装器调用失败: plan_id={plan_id}, error={e}")

    def _run_async_in_new_loop(self, plan_id: int, task_type: str):
        """在新的事件循环中运行异步函数"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"新事件循环开始: plan_id={plan_id}, task_type={task_type}")
            if task_type == 'finetune':
                result = loop.run_until_complete(self._trigger_finetune(plan_id))
            elif task_type == 'inference':
                result = loop.run_until_complete(self._trigger_inference(plan_id, manual_trigger=True))
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            logger.info(f"新事件循环完成: plan_id={plan_id}, task_type={task_type}, result={result}")
            return result
        except Exception as e:
            logger.error(f"新事件循环执行失败: plan_id={plan_id}, task_type={task_type}, error={e}")
            raise
        finally:
            loop.close()
            logger.info(f"新事件循环已关闭: plan_id={plan_id}, task_type={task_type}")

    async def _create_agent_task(self, plan: TradingPlan, training_record: TrainingRecord, scheduled_time: datetime, db):
        """创建自动Agent任务"""
        try:
            task = TaskExecution(
                plan_id=plan.id,
                task_type='auto_agent',
                task_name=f'自动Agent决策 - {plan.plan_name}',
                task_description=f'基于推理结果的自动Agent决策任务',
                status='pending',
                priority=3,
                scheduled_time=scheduled_time,
                trigger_type='scheduled',
                trigger_source='immediate_scheduler',
                input_data={
                    'training_record_id': training_record.id,
                    'training_version': training_record.version
                },
                task_metadata={
                    'auto_generated': True,
                    'auto_tool_execution': plan.auto_tool_execution_enabled  # 已废弃字段，保留用于历史记录
                }
            )

            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(f"创建自动Agent任务: {task.id}")
            return task

        except Exception as e:
            logger.error(f"创建Agent任务失败: {e}")
            return None

    def check_latest_prediction_time(self, plan_id: int) -> Optional[datetime]:
        """
        检查计划最新的预测数据时间

        Args:
            plan_id: 计划ID

        Returns:
            最新预测数据的创建时间，如果没有预测数据则返回None
        """
        try:
            with get_db() as db:
                # 获取最新的训练记录ID
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    logger.info(f"计划 {plan_id}: 没有找到完成的训练记录")
                    return None

                # 获取该训练记录的最新预测数据
                from database.models import PredictionData
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.training_record_id == latest_training.id
                ).order_by(PredictionData.created_at.desc()).first()

                if latest_prediction:
                    logger.info(f"计划 {plan_id}: 最新预测数据时间: {latest_prediction.created_at}")
                    return latest_prediction.created_at
                else:
                    logger.info(f"计划 {plan_id}: 训练记录 {latest_training.id} 没有预测数据")
                    return None

        except Exception as e:
            logger.error(f"检查最新预测数据时间失败: plan_id={plan_id}, error={e}")
            return None

    def _log_scheduler_status(self):
        """输出调度器状态信息"""
        try:
            if self.scheduler:
                current_time = datetime.now(self.BEIJING_TZ)
                logger.info(f"统一调度器状态 - 当前时间(UTC+8): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # 输出所有任务
                jobs = self.scheduler.get_jobs()
                logger.info(f"当前任务数: {len(jobs)}")
                for job in jobs:
                    next_run = job.next_run_time
                    if next_run:
                        # 转换为UTC+8时间显示
                        if next_run.tzinfo is None:
                            next_run_beijing = next_run.replace(tzinfo=self.BEIJING_TZ)
                        else:
                            next_run_beijing = next_run.astimezone(self.BEIJING_TZ)
                        logger.info(f"任务 {job.id}: 下次执行 {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        logger.info(f"任务 {job.id}: 无下次执行时间")
        except Exception as e:
            logger.error(f"输出调度器状态失败: {e}")

    def get_task_history(self, plan_id: int, limit: int = 50) -> List[Dict]:
        """获取任务执行历史"""
        try:
            with get_db() as db:
                tasks = db.query(TaskExecution).filter(
                    TaskExecution.plan_id == plan_id
                ).order_by(desc(TaskExecution.created_at)).limit(limit).all()

                task_list = []
                for task in tasks:
                    task_info = {
                        'id': task.id,
                        'task_type': task.task_type,
                        'task_name': task.task_name,
                        'status': task.status,
                        'priority': task.priority,
                        'scheduled_time': format_datetime_full_beijing(task.scheduled_time) if task.scheduled_time else None,
                        'started_at': format_datetime_full_beijing(task.started_at) if task.started_at else None,
                        'completed_at': format_datetime_full_beijing(task.completed_at) if task.completed_at else None,
                        'duration_seconds': task.duration_seconds,
                        'trigger_type': task.trigger_type,
                        'progress_percentage': task.progress_percentage,
                        'error_message': task.error_message,
                        'created_at': format_datetime_full_beijing(task.created_at)
                    }

                    # 添加状态显示
                    status_map = {
                        'pending': '⏳ 等待中',
                        'running': '🔄 执行中',
                        'completed': '✅ 已完成',
                        'failed': '❌ 失败',
                        'cancelled': '⏹️ 已取消'
                    }
                    task_info['status_display'] = status_map.get(task.status, f"❓ {task.status}")

                    # 添加任务类型显示
                    type_map = {
                        'auto_finetune': '🔧 自动微调',
                        'auto_inference': '🔮 自动推理',
                        'auto_agent': '🤖 自动Agent'
                    }
                    task_info['type_display'] = type_map.get(task.task_type, f"📋 {task.task_type}")

                    task_list.append(task_info)

                return task_list

        except Exception as e:
            logger.error(f"获取任务历史失败: {e}")
            return []

    def _daily_model_cleanup_wrapper(self):
        """每日模型清理包装器"""
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._daily_model_cleanup)
                future.result(timeout=300)  # 5分钟超时
        except Exception as e:
            logger.error(f"每日模型清理包装器执行失败: {e}")

    def _daily_model_cleanup(self):
        """执行每日模型清理"""
        try:
            from services.model_cleanup_service import cleanup_all_plans_models
            logger.info("🕰️ 开始每日模型清理")
            cleanup_stats = cleanup_all_plans_models(keep_count=7)
            total_deleted = sum(stats['models_deleted'] for stats in cleanup_stats.values())
            logger.info(f"✅ 每日模型清理完成: 删除{total_deleted}个模型")
            return cleanup_stats
        except Exception as e:
            logger.error(f"❌ 每日模型清理失败: {e}")

    async def reload_all_schedules(self):
        """
        重新加载所有运行中计划的定时任务
        （用于应用启动时）
        """
        try:
            logger.info("重新加载所有定时任务...")

            with get_db() as db:
                # 查询所有运行中的计划
                running_plans = db.query(TradingPlan).filter(
                    TradingPlan.status == 'running'
                ).all()

                logger.info(f"找到 {len(running_plans)} 个运行中的计划")

                for plan in running_plans:
                    success = await self.start_plan_schedule(plan.id)
                    if success:
                        logger.info(f"✅ 重新加载计划 {plan.id} 的定时任务成功")
                    else:
                        logger.warning(f"⚠️ 重新加载计划 {plan.id} 的定时任务失败")

            logger.info("定时任务重新加载完成")

        except Exception as e:
            logger.error(f"重新加载定时任务失败: error={e}")

    # 兼容性方法，保持与原有接口的兼容
    async def start_schedule(self, plan_id: int) -> bool:
        """兼容性方法，调用 start_plan_schedule"""
        return await self.start_plan_schedule(plan_id)

    async def stop_schedule(self, plan_id: int) -> bool:
        """兼容性方法，调用 stop_plan_schedule"""
        return await self.stop_plan_schedule(plan_id)

    def test_scheduler(self):
        """测试调度器是否正常工作"""
        try:
            logger.info("=== 统一调度器测试开始 ===")

            current_time_beijing = datetime.now(self.BEIJING_TZ)
            logger.info(f"当前时间(UTC+8): {current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 获取所有任务
            jobs = self.scheduler.get_jobs()
            logger.info(f"总任务数: {len(jobs)}")

            if not jobs:
                logger.warning("没有找到任何任务")
                return

            for job in jobs:
                next_run = job.next_run_time
                if next_run:
                    next_run_beijing = next_run.astimezone(self.BEIJING_TZ)
                    time_until = next_run_beijing - current_time_beijing
                    logger.info(f"任务 {job.id}: 下次执行 {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}, 距离现在 {time_until}")
                else:
                    logger.warning(f"任务 {job.id}: 无下次执行时间")

            logger.info("=== 统一调度器测试结束 ===")

        except Exception as e:
            logger.error(f"统一调度器测试失败: {e}")


# 全局统一调度器实例
unified_scheduler = UnifiedScheduler()