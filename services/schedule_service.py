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
from database.models import TradingPlan
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

                logger.info(f"计划信息: plan_id={plan_id}, status={plan.status}, auto_finetune_enabled={plan.auto_finetune_enabled}")

                # 检查是否启用自动微调
                if not plan.auto_finetune_enabled:
                    logger.warning(f"计划未启用自动微调: plan_id={plan_id}")
                    return False

                # 获取时间表
                schedule_times = plan.auto_finetune_schedule or []
                if not schedule_times:
                    logger.warning(f"计划未配置微调时间: plan_id={plan_id}")
                    return False

                logger.info(f"时间表配置: plan_id={plan_id}, schedule_times={schedule_times}")

            # 初始化调度器
            scheduler = cls.get_scheduler()

            # 为每个时间点创建任务
            for time_str in schedule_times:
                try:
                    # 解析时间 (HH:MM)
                    hour, minute = map(int, time_str.split(':'))
                    logger.info(f"解析时间: time_str={time_str}, hour={hour}, minute={minute}")

                    # 创建cron触发器（每天指定时间执行）
                    trigger = CronTrigger(hour=hour, minute=minute, timezone='Asia/Shanghai')
                    logger.info(f"创建Cron触发器: hour={hour}, minute={minute}, timezone=Asia/Shanghai")

                    # 任务ID：plan_id + 时间
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

                    # 立即检查任务的下次执行时间
                    job = scheduler.get_job(job_id)
                    next_run_time = job.next_run_time
                    if next_run_time:
                        next_run_beijing = next_run_time.astimezone(cls.BEIJING_TZ)
                        logger.info(f"已添加定时任务: plan_id={plan_id}, time={time_str}, job_id={job_id}, 下次执行(UTC+8)={next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        logger.warning(f"任务创建成功但无下次执行时间: plan_id={plan_id}, time={time_str}, job_id={job_id}")

                except Exception as e:
                    logger.error(f"创建任务失败: time={time_str}, error={e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # 重新输出调度器状态
            cls._log_scheduler_status()

            logger.info(f"启动定时调度成功: plan_id={plan_id}, 任务数={len(schedule_times)}")
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

            # 触发训练
            from services.training_service import TrainingService
            logger.info(f"开始调用训练服务: plan_id={plan_id}")

            try:
                training_id = await TrainingService.start_training(plan_id, manual=False)

                if training_id:
                    logger.info(f"✅ 定时微调已启动: plan_id={plan_id}, training_id={training_id}")
                else:
                    logger.error(f"❌ 定时微调启动失败: plan_id={plan_id}")
            except Exception as training_error:
                logger.error(f"训练服务调用失败: plan_id={plan_id}, error={training_error}")
                import traceback
                traceback.print_exc()

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
    def _run_async_in_new_loop(cls, plan_id: int):
        """在新的事件循环中运行异步函数"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(cls._trigger_finetune(plan_id))
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
