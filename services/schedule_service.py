"""
定时任务调度服务
负责管理计划的自动微调定时任务
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData
from utils.logger import setup_logger

logger = setup_logger(__name__, "schedule_service.log")

# 全局调度器实例
_scheduler: Optional[BackgroundScheduler] = None
_scheduler_started = False


class ScheduleService:
    """定时任务调度服务"""

    # 定义UTC+8时区
    BEIJING_TZ = timezone(timedelta(hours=8))

    @classmethod
    def check_latest_prediction_time(cls, plan_id: int) -> Optional[datetime]:
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
                from database.models import TrainingRecord
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    logger.info(f"计划 {plan_id}: 没有找到完成的训练记录")
                    return None

                # 获取该训练记录的最新预测数据
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

    @classmethod
    def init_scheduler(cls):
        """初始化调度器"""
        global _scheduler, _scheduler_started

        if _scheduler is None:
            jobstores = {
                'default': MemoryJobStore()
            }

            _scheduler = BackgroundScheduler(
                jobstores=jobstores,
                timezone='Asia/Shanghai'
            )

            logger.info("调度器已初始化，时区: Asia/Shanghai")

        if not _scheduler_started:
            _scheduler.start()
            _scheduler_started = True
            logger.info("调度器已启动")

            # 输出当前时间和下一个任务执行时间
            cls._log_scheduler_status()

    @classmethod
    def _log_scheduler_status(cls):
        """输出调度器状态信息"""
        try:
            if _scheduler:
                current_time = datetime.now(cls.BEIJING_TZ)
                logger.info(f"调度器状态 - 当前时间(UTC+8): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # 输出所有任务
                jobs = _scheduler.get_jobs()
                logger.info(f"当前任务数: {len(jobs)}")
                for job in jobs:
                    next_run = job.next_run_time
                    if next_run:
                        # 转换为UTC+8时间显示
                        if next_run.tzinfo is None:
                            next_run_beijing = next_run.replace(tzinfo=cls.BEIJING_TZ)
                        else:
                            next_run_beijing = next_run.astimezone(cls.BEIJING_TZ)
                        logger.info(f"任务 {job.id}: 下次执行 {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        logger.info(f"任务 {job.id}: 无下次执行时间")
        except Exception as e:
            logger.error(f"输出调度器状态失败: {e}")

    @classmethod
    def get_scheduler(cls) -> BackgroundScheduler:
        """获取调度器实例"""
        if _scheduler is None:
            cls.init_scheduler()
        return _scheduler

    @classmethod
    async def start_schedule(cls, plan_id: int) -> bool:
        """
        启动计划的定时任务

        Args:
            plan_id: 计划ID

        Returns:
            是否成功
        """
        try:
            current_time_beijing = datetime.now(cls.BEIJING_TZ)
            logger.info(f"开始启动定时任务: plan_id={plan_id}, current_time(UTC+8)={current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 获取计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return False

                logger.info(f"计划信息: plan_id={plan_id}, status={plan.status}, auto_finetune_enabled={plan.auto_finetune_enabled}, auto_inference_enabled={plan.auto_inference_enabled}")

                # 检查是否启用自动微调或预测
                if not plan.auto_finetune_enabled and not plan.auto_inference_enabled:
                    logger.warning(f"计划未启用自动微调或预测: plan_id={plan_id}")
                    return False

            # 初始化调度器
            scheduler = cls.get_scheduler()

            task_count = 0

            # 处理自动微调任务
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
                            existing_job = scheduler.get_job(job_id)
                            if existing_job:
                                logger.info(f"任务已存在，先移除: {job_id}")
                                scheduler.remove_job(job_id)

                            # 添加任务
                            scheduler.add_job(
                                func=cls._trigger_finetune_wrapper,
                                trigger=trigger,
                                args=[plan_id],
                                id=job_id,
                                name=f"自动微调-计划{plan_id}-{time_str}",
                                replace_existing=True,
                                misfire_grace_time=300  # 允许5分钟的延迟执行
                            )

                            task_count += 1

                            # 立即检查任务的下次执行时间
                            job = scheduler.get_job(job_id)
                            next_run_time = job.next_run_time
                            if next_run_time:
                                next_run_beijing = next_run_time.astimezone(cls.BEIJING_TZ)
                                logger.info(f"已添加自动微调任务: plan_id={plan_id}, time={time_str}, job_id={job_id}, 下次执行(UTC+8)={next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                            else:
                                logger.warning(f"微调任务创建成功但无下次执行时间: plan_id={plan_id}, time={time_str}, job_id={job_id}")

                        except Exception as e:
                            logger.error(f"创建微调任务失败: time={time_str}, error={e}")
                            import traceback
                            traceback.print_exc()
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
                        from apscheduler.triggers.interval import IntervalTrigger
                        trigger = IntervalTrigger(hours=interval_hours, timezone='Asia/Shanghai')

                        # 任务ID：plan_id + 任务类型
                        job_id = f"plan_{plan_id}_inference_interval"

                        # 检查任务是否已存在
                        existing_job = scheduler.get_job(job_id)
                        if existing_job:
                            logger.info(f"任务已存在，先移除: {job_id}")
                            scheduler.remove_job(job_id)

                        # 添加任务
                        scheduler.add_job(
                            func=cls._trigger_inference_wrapper,
                            trigger=trigger,
                            args=[plan_id],
                            id=job_id,
                            name=f"自动预测-计划{plan_id}-{interval_hours}h间隔",
                            replace_existing=True,
                            misfire_grace_time=300  # 允许5分钟的延迟执行
                        )

                        task_count += 1

                        # 立即检查任务的下次执行时间
                        job = scheduler.get_job(job_id)
                        next_run_time = job.next_run_time
                        if next_run_time:
                            next_run_beijing = next_run_time.astimezone(cls.BEIJING_TZ)
                            logger.info(f"已添加自动预测任务: plan_id={plan_id}, interval={interval_hours}h, job_id={job_id}, 下次执行(UTC+8)={next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            logger.warning(f"预测任务创建成功但无下次执行时间: plan_id={plan_id}, interval={interval_hours}h, job_id={job_id}")

                    except Exception as e:
                        logger.error(f"创建预测任务失败: interval_hours={interval_hours}, error={e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"计划启用了自动预测但间隔时间无效: plan_id={plan_id}, interval_hours={interval_hours}")

            # 重新输出调度器状态
            cls._log_scheduler_status()

            logger.info(f"启动定时调度成功: plan_id={plan_id}, 任务数={task_count}")
            return True

        except Exception as e:
            logger.error(f"启动定时调度失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    def get_plan_jobs(cls, plan_id: int) -> List:
        """
        获取计划的所有定时任务

        Args:
            plan_id: 计划ID

        Returns:
            任务列表
        """
        try:
            scheduler = cls.get_scheduler()
            plan_jobs = []
            for job in scheduler.get_jobs():
                if job.id.startswith(f"plan_{plan_id}_"):
                    plan_jobs.append(job)
            return plan_jobs
        except Exception as e:
            logger.error(f"获取计划任务失败: plan_id={plan_id}, error={e}")
            return []

    @classmethod
    async def stop_schedule(cls, plan_id: int) -> bool:
        """
        停止计划的定时任务

        Args:
            plan_id: 计划ID

        Returns:
            是否成功
        """
        try:
            scheduler = cls.get_scheduler()

            # 移除该计划的所有任务
            removed_count = 0
            for job in scheduler.get_jobs():
                if job.id.startswith(f"plan_{plan_id}_"):
                    scheduler.remove_job(job.id)
                    removed_count += 1
                    logger.info(f"移除任务: {job.id}")

            logger.info(f"停止定时调度成功: plan_id={plan_id}, 移除任务数={removed_count}")
            return True

        except Exception as e:
            logger.error(f"停止定时调度失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    async def _trigger_finetune(cls, plan_id: int):
        """
        触发微调任务（由调度器调用）

        Args:
            plan_id: 计划ID
        """
        try:
            current_time_beijing = datetime.now(cls.BEIJING_TZ)
            logger.info(f"⏰ 定时任务触发: plan_id={plan_id}, time(UTC+8)={current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 检查计划状态
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return

                logger.info(f"计划状态检查: plan_id={plan_id}, status={plan.status}, auto_finetune_enabled={plan.auto_finetune_enabled}")

                # 检查计划是否运行中
                if plan.status != 'running':
                    logger.warning(f"计划未运行，跳过微调: plan_id={plan_id}, status={plan.status}")
                    return

                # 再次检查是否启用自动微调
                if not plan.auto_finetune_enabled:
                    logger.warning(f"计划未启用自动微调，跳过: plan_id={plan_id}")
                    return

                # 检查是否有时间表配置
                schedule_times = plan.auto_finetune_schedule or []
                if not schedule_times:
                    logger.warning(f"计划未配置微调时间表，跳过: plan_id={plan_id}")
                    return

                logger.info(f"计划配置检查通过: plan_id={plan_id}, schedule_times={schedule_times}")

            # 创建任务执行记录
            from services.task_execution_service import TaskExecutionService
            task_execution = None

            try:
                # 从计划配置中找到匹配当前时间的任务
                current_datetime = datetime.now(cls.BEIJING_TZ)
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
            from services.training_service import TrainingService
            logger.info(f"开始调用训练服务: plan_id={plan_id}")

            try:
                training_id = await TrainingService.start_training(plan_id, manual=False)

                if training_id:
                    logger.info(f"✅ 定时微调已启动: plan_id={plan_id}, training_id={training_id}")

                    # 记录成功结果
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=True,
                            output_data={'training_id': training_id}
                        )
                else:
                    logger.error(f"❌ 定时微调启动失败: plan_id={plan_id}")

                    # 记录失败结果
                    if task_execution:
                        TaskExecutionService.complete_task_execution(
                            task_id=task_execution.id,
                            success=False,
                            error_message='训练服务启动失败'
                        )

            except Exception as training_error:
                logger.error(f"训练服务调用失败: plan_id={plan_id}, error={training_error}")
                import traceback
                traceback.print_exc()

                # 记录异常结果
                if task_execution:
                    TaskExecutionService.complete_task_execution(
                        task_id=task_execution.id,
                        success=False,
                        error_message=f'训练服务异常: {str(training_error)}'
                    )

        except Exception as e:
            logger.error(f"触发微调失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def _trigger_finetune_wrapper(cls, plan_id: int):
        """
        包装器方法，用于在APScheduler中调用async函数

        Args:
            plan_id: 计划ID
        """
        try:
            # 检查是否已有事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果有运行中的循环，在新线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(cls._run_async_in_new_loop, plan_id)
                    future.result()
            except RuntimeError:
                # 没有运行中的循环，直接运行
                asyncio.run(cls._trigger_finetune(plan_id))
        except Exception as e:
            logger.error(f"包装器调用失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    async def _trigger_inference(cls, plan_id: int, manual_trigger: bool = False):
        """
        触发预测任务（由调度器调用）

        Args:
            plan_id: 计划ID
            manual_trigger: 是否为手动触发（跳过间隔时间检查）
        """
        try:
            current_time_beijing = datetime.now(cls.BEIJING_TZ)
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
                latest_prediction_time = cls.check_latest_prediction_time(plan_id)
                current_time = datetime.now(cls.BEIJING_TZ)

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
            from services.task_execution_service import TaskExecutionService
            task_execution = None

            try:
                # 创建预测任务记录
                current_datetime = datetime.now(cls.BEIJING_TZ)

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
                    manual_trigger=False
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
                import traceback
                traceback.print_exc()

                # 记录异常结果
                if task_execution:
                    TaskExecutionService.complete_task_execution(
                        task_id=task_execution.id,
                        success=False,
                        error_message=f'推理服务异常: {str(inference_error)}'
                    )

        except Exception as e:
            logger.error(f"触发预测失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def _trigger_inference_wrapper(cls, plan_id: int):
        """
        包装器方法，用于在APScheduler中调用async函数

        Args:
            plan_id: 计划ID
        """
        try:
            # 检查是否已有事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果有运行中的循环，在新线程中运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(cls._run_async_in_new_loop_for_inference, plan_id, manual_trigger=True)
                    future.result()
            except RuntimeError:
                # 没有运行中的循环，直接运行
                asyncio.run(cls._trigger_inference(plan_id, manual_trigger=True))
        except Exception as e:
            logger.error(f"预测包装器调用失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def _run_async_in_new_loop(cls, plan_id: int):
        """在新的事件循环中运行异步函数（微调）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cls._trigger_finetune(plan_id))
        finally:
            loop.close()

    @classmethod
    def _run_async_in_new_loop_for_inference(cls, plan_id: int, manual_trigger: bool = False):
        """在新的事件循环中运行异步函数（预测）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cls._trigger_inference(plan_id, manual_trigger=manual_trigger))
        finally:
            loop.close()

    @classmethod
    def reload_all_schedules(cls):
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
                    # 使用新的事件循环运行异步函数
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        success = loop.run_until_complete(cls.start_schedule(plan.id))
                        if success:
                            logger.info(f"✅ 重新加载计划 {plan.id} 的定时任务成功")
                        else:
                            logger.warning(f"⚠️ 重新加载计划 {plan.id} 的定时任务失败")
                    finally:
                        loop.close()

            logger.info("定时任务重新加载完成")

        except Exception as e:
            logger.error(f"重新加载定时任务失败: error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def trigger_finetune(cls, plan_id: int):
        """
        手动触发微调训练

        Args:
            plan_id: 计划ID

        Returns:
            dict: 触发结果
        """
        try:
            from database.db import get_db
            from database.models import TradingPlan

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {
                        'success': False,
                        'error': '计划不存在'
                    }

                if not plan.auto_finetune_enabled:
                    return {
                        'success': False,
                        'error': '自动微调未启用，请先启用自动微调功能'
                    }

                # 检查是否有正在进行的训练
                from database.models import TrainingRecord
                active_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'training'
                ).first()
                if active_training:
                    return {
                        'success': False,
                        'error': f'已有训练正在进行中 (训练ID: {active_training.id})'
                    }

            logger.info(f"手动触发微调训练: plan_id={plan_id}")

            # 在新线程中执行异步触发
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(cls._trigger_finetune_wrapper, plan_id)
                # 等待一段时间获取初步结果
                try:
                    future.result(timeout=5)  # 5秒超时
                    return {
                        'success': True,
                        'message': '微调训练已启动，请查看任务执行记录'
                    }
                except concurrent.futures.TimeoutError:
                    return {
                        'success': True,
                        'message': '微调训练已启动（正在后台执行）'
                    }

        except Exception as e:
            logger.error(f"手动触发微调训练失败: plan_id={plan_id}, error={e}")
            return {
                'success': False,
                'error': f'触发失败: {str(e)}'
            }

    @classmethod
    def trigger_inference(cls, plan_id: int):
        """
        手动触发预测推理（智能Data Offset）

        Args:
            plan_id: 计划ID

        Returns:
            dict: 触发结果
        """
        try:
            from database.db import get_db
            from database.models import TradingPlan

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {
                        'success': False,
                        'error': '计划不存在'
                    }

                if not plan.auto_inference_enabled:
                    return {
                        'success': False,
                        'error': '自动预测未启用，请先启用自动预测功能'
                    }

                # 检查是否有已完成的训练记录
                from database.models import TrainingRecord
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    return {
                        'success': False,
                        'error': '没有已完成的训练记录，请先完成模型训练'
                    }

            logger.info(f"手动触发预测推理: plan_id={plan_id}")

            # 在新线程中执行异步触发
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(cls._trigger_manual_inference_with_offset, plan_id)
                # 等待一段时间获取初步结果
                try:
                    future.result(timeout=5)  # 5秒超时
                    return {
                        'success': True,
                        'message': '手动预测推理已启动（智能Data Offset），请查看任务执行记录'
                    }
                except concurrent.futures.TimeoutError:
                    return {
                        'success': True,
                        'message': '手动预测推理已启动（正在后台执行）'
                    }

        except Exception as e:
            logger.error(f"手动触发预测推理失败: plan_id={plan_id}, error={e}")
            return {
                'success': False,
                'error': f'触发失败: {str(e)}'
            }

    @classmethod
    def _trigger_manual_inference_with_offset(cls, plan_id: int):
        """
        手动触发推理（带智能Data Offset计算）
        """
        try:
            # 计算智能数据偏移（手动触发模式）
            from services.inference_data_offset_service import inference_data_offset_service
            from database.db import get_db
            from database.models import TradingPlan

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                target_interval_hours = plan.auto_inference_interval_hours or 4

            offset_result = inference_data_offset_service.calculate_optimal_data_offset(
                plan_id=plan_id,
                target_interval_hours=target_interval_hours,
                manual_trigger=True
            )

            if offset_result['success']:
                data_offset = offset_result['data_offset']
                logger.info(f"✅ 手动推理数据偏移计算完成: plan_id={plan_id}, offset={data_offset}")
                logger.info(f"📊 手动推理偏移说明: {offset_result['reasoning']}")

                # 更新推理参数
                with get_db() as db:
                    latest_training = db.query(TrainingRecord).filter(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    ).order_by(TrainingRecord.created_at.desc()).first()

                    if latest_training:
                        update_success = inference_data_offset_service.update_inference_params_with_offset(
                            plan_id=plan_id,
                            training_id=latest_training.id,
                            data_offset=data_offset
                        )

                        if update_success:
                            logger.info(f"✅ 手动推理参数已更新: training_id={latest_training.id}, data_offset={data_offset}")
                        else:
                            logger.warning(f"⚠️ 手动推理参数更新失败，使用默认参数")
                    else:
                        logger.warning(f"⚠️ 未找到训练记录，无法更新手动推理参数")
            else:
                logger.warning(f"⚠️ 手动推理数据偏移计算失败: {offset_result['reasoning']}")
                data_offset = 0

            # 执行推理包装器
            cls._trigger_inference_wrapper(plan_id)

        except Exception as e:
            logger.error(f"手动推理触发失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def test_scheduler(cls):
        """测试调度器是否正常工作"""
        try:
            logger.info("=== 调度器测试开始 ===")
            scheduler = cls.get_scheduler()

            current_time_beijing = datetime.now(cls.BEIJING_TZ)
            logger.info(f"当前时间(UTC+8): {current_time_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

            # 获取所有任务
            jobs = scheduler.get_jobs()
            logger.info(f"总任务数: {len(jobs)}")

            if not jobs:
                logger.warning("没有找到任何任务")
                return

            for job in jobs:
                next_run = job.next_run_time
                if next_run:
                    next_run_beijing = next_run.astimezone(cls.BEIJING_TZ)
                    time_until = next_run_beijing - current_time_beijing
                    logger.info(f"任务 {job.id}: 下次执行 {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}, 距离现在 {time_until}")
                else:
                    logger.warning(f"任务 {job.id}: 无下次执行时间")

            logger.info("=== 调度器测试结束 ===")

        except Exception as e:
            logger.error(f"调度器测试失败: {e}")
            import traceback
            traceback.print_exc()
