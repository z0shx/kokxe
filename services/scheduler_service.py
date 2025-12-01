"""
定时任务调度服务

根据用户的自动化配置在指定时间执行自动微调任务
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional
from sqlalchemy import and_, desc, asc

from database.db import get_db
from database.models import TradingPlan, TaskExecution, TrainingRecord, now_beijing
from services.training_service import TrainingService
from services.inference_service import InferenceService
from utils.timezone_helper import format_datetime_full_beijing

logger = logging.getLogger(__name__)


class SchedulerService:
    """定时任务调度服务"""

    def __init__(self):
        self.running = False
        self.scheduler_task = None

    async def start_scheduler(self):
        """启动定时任务调度器"""
        if self.running:
            logger.warning("定时任务调度器已经在运行中")
            return

        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("定时任务调度器已启动")

    async def stop_scheduler(self):
        """停止定时任务调度器"""
        if not self.running:
            return

        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("定时任务调度器已停止")

    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                await self._check_and_execute_scheduled_tasks()
                # 每分钟检查一次
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度器循环出错: {e}")
                await asyncio.sleep(60)

    async def _check_and_execute_scheduled_tasks(self):
        """检查并执行计划中的定时任务"""
        try:
            with get_db() as db:
                now = now_beijing()

                # 获取所有启用自动化的计划
                plans = db.query(TradingPlan).filter(
                    TradingPlan.status.in_(['running', 'created'])  # 运行中或已创建的计划
                ).all()

                for plan in plans:
                    try:
                        # 检查自动微调任务
                        if plan.auto_finetune_enabled and plan.auto_finetune_schedule:
                            await self._check_finetune_tasks(plan, now, db)

                        # 检查自动推理任务
                        if plan.auto_inference_enabled:
                            await self._check_inference_tasks(plan, now, db)

                        # 检查自动Agent任务
                        if plan.auto_agent_enabled:
                            await self._check_agent_tasks(plan, now, db)

                    except Exception as e:
                        logger.error(f"处理计划 {plan.id} 的定时任务时出错: {e}")

        except Exception as e:
            logger.error(f"检查定时任务时出错: {e}")

    async def _check_finetune_tasks(self, plan: TradingPlan, now: datetime, db):
        """检查自动微调任务"""
        if not plan.auto_finetune_schedule:
            return

        try:
            schedule_list = plan.auto_finetune_schedule
            if not isinstance(schedule_list, list):
                return

            # 检查是否到了执行时间
            for schedule_time in schedule_list:
                try:
                    # 解析时间格式 (支持 "HH:MM" 格式)
                    if ':' in str(schedule_time):
                        hour, minute = map(int, str(schedule_time).split(':'))
                        scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                        # 如果当前时间超过计划时间且在5分钟内，则执行任务
                        if now >= scheduled_time and now <= scheduled_time + timedelta(minutes=5):
                            # 检查今天是否已经执行过
                            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                            existing_task = db.query(TaskExecution).filter(
                                and_(
                                    TaskExecution.plan_id == plan.id,
                                    TaskExecution.task_type == 'auto_finetune',
                                    TaskExecution.scheduled_time >= today_start,
                                    TaskExecution.status.in_(['pending', 'running', 'completed'])
                                )
                            ).first()

                            if not existing_task:
                                await self._create_finetune_task(plan, scheduled_time, db)
                                await self._execute_task(existing_task, db)

                except Exception as e:
                    logger.error(f"解析微调时间 {schedule_time} 时出错: {e}")

        except Exception as e:
            logger.error(f"检查微调任务时出错: {e}")

    async def _check_inference_tasks(self, plan: TradingPlan, now: datetime, db):
        """检查自动推理任务"""
        try:
            # 自动推理通常在有新数据或定时执行
            # 这里简化实现：每小时检查一次是否有最新的训练记录需要推理
            if now.minute == 0:  # 整点执行
                # 获取最新的已完成训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan.id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if latest_training:
                    # 检查是否已经有推理任务
                    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    existing_task = db.query(TaskExecution).filter(
                        and_(
                            TaskExecution.plan_id == plan.id,
                            TaskExecution.task_type == 'auto_inference',
                            TaskExecution.scheduled_time >= today_start,
                            TaskExecution.status.in_(['pending', 'running', 'completed'])
                        )
                    ).first()

                    if not existing_task:
                        await self._create_inference_task(plan, latest_training, now, db)
                        await self._execute_task(existing_task, db)

        except Exception as e:
            logger.error(f"检查推理任务时出错: {e}")

    async def _check_agent_tasks(self, plan: TradingPlan, now: datetime, db):
        """检查自动Agent任务"""
        try:
            # 自动Agent通常在推理完成后执行
            # 这里简化实现：每30分钟检查一次
            if now.minute % 30 == 0:
                # 获取最新的已完成推理记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan.id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if latest_training:
                    # 检查是否已经有Agent任务
                    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    existing_task = db.query(TaskExecution).filter(
                        and_(
                            TaskExecution.plan_id == plan.id,
                            TaskExecution.task_type == 'auto_agent',
                            TaskExecution.scheduled_time >= today_start,
                            TaskExecution.status.in_(['pending', 'running', 'completed'])
                        )
                    ).first()

                    if not existing_task:
                        await self._create_agent_task(plan, latest_training, now, db)
                        await self._execute_task(existing_task, db)

        except Exception as e:
            logger.error(f"检查Agent任务时出错: {e}")

    async def _create_finetune_task(self, plan: TradingPlan, scheduled_time: datetime, db):
        """创建自动微调任务"""
        try:
            task = TaskExecution(
                plan_id=plan.id,
                task_type='auto_finetune',
                task_name=f'自动微调 - {plan.plan_name}',
                task_description=f'根据时间表 {plan.auto_finetune_schedule} 自动执行的微调任务',
                status='pending',
                priority=1,
                scheduled_time=scheduled_time,
                trigger_type='scheduled',
                trigger_source='scheduler',
                input_data={
                    'finetune_params': plan.finetune_params,
                    'schedule_time': scheduled_time.isoformat()
                },
                task_metadata={
                    'auto_generated': True,
                    'schedule': plan.auto_finetune_schedule
                }
            )

            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(f"创建自动微调任务: {task.id}")
            return task

        except Exception as e:
            logger.error(f"创建微调任务失败: {e}")
            return None

    async def _create_inference_task(self, plan: TradingPlan, training_record: TrainingRecord, scheduled_time: datetime, db):
        """创建自动推理任务"""
        try:
            task = TaskExecution(
                plan_id=plan.id,
                task_type='auto_inference',
                task_name=f'自动推理 - {plan.plan_name}',
                task_description=f'基于训练版本 v{training_record.version} 的自动推理任务',
                status='pending',
                priority=2,
                scheduled_time=scheduled_time,
                trigger_type='scheduled',
                trigger_source='scheduler',
                input_data={
                    'training_record_id': training_record.id,
                    'training_version': training_record.version
                },
                task_metadata={
                    'auto_generated': True
                }
            )

            db.add(task)
            db.commit()
            db.refresh(task)

            logger.info(f"创建自动推理任务: {task.id}")
            return task

        except Exception as e:
            logger.error(f"创建推理任务失败: {e}")
            return None

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
                trigger_source='scheduler',
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

    async def _execute_task(self, task: TaskExecution, db):
        """执行任务"""
        if not task:
            return

        try:
            # 更新任务状态为运行中
            task.status = 'running'
            task.started_at = now_beijing()
            task.progress_percentage = 0
            db.commit()

            logger.info(f"开始执行任务: {task.id} - {task.task_type}")

            if task.task_type == 'auto_finetune':
                await self._execute_finetune_task(task, db)
            elif task.task_type == 'auto_inference':
                await self._execute_inference_task(task, db)
            elif task.task_type == 'auto_agent':
                await self._execute_agent_task(task, db)

        except Exception as e:
            logger.error(f"执行任务 {task.id} 时出错: {e}")
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = now_beijing()
            if task.started_at:
                task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())
            db.commit()

    async def _execute_finetune_task(self, task: TaskExecution, db):
        """执行自动微调任务"""
        try:
            plan = task.plan

            # 获取数据范围
            finetune_params = task.input_data.get('finetune_params', {})
            train_start_date = finetune_params.get('train_start_date')
            train_end_date = finetune_params.get('train_end_date')

            if not train_start_date or not train_end_date:
                # 使用默认数据范围（最近7天）
                end_date = now_beijing().date()
                start_date = end_date - timedelta(days=7)
            else:
                start_date = datetime.strptime(train_start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(train_end_date, '%Y-%m-%d').date()

            # 调用TrainingService执行微调
            result = await TrainingService.start_training(
                plan_id=plan.id,
                manual=False  # 自动触发，非手动
            )

            if result:  # result是训练记录ID
                task.status = 'completed'
                task.output_data = {
                    'success': True,
                    'training_record_id': result,
                    'message': '微调成功启动'
                }
                logger.info(f"微调任务 {task.id} 成功启动，训练记录ID: {result}")
            else:
                task.status = 'failed'
                task.error_message = '微调启动失败'
                task.output_data = {
                    'success': False,
                    'error': '微调启动失败'
                }
                logger.error(f"微调任务 {task.id} 失败: 无法创建训练记录")

        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            logger.error(f"执行微调任务 {task.id} 时出错: {e}")
        finally:
            task.progress_percentage = 100
            task.completed_at = now_beijing()
            if task.started_at:
                task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())
            db.commit()

    async def _execute_inference_task(self, task: TaskExecution, db):
        """执行自动推理任务"""
        try:
            training_record_id = task.input_data.get('training_record_id')
            if not training_record_id:
                raise ValueError("缺少训练记录ID")

            # 调用InferenceService执行推理
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: InferenceService.start_inference(
                    training_id=training_record_id
                )
            )

            if result:  # result是bool值，表示是否成功
                task.status = 'completed'
                task.output_data = {
                    'success': True,
                    'message': '推理成功启动'
                }
                logger.info(f"推理任务 {task.id} 成功启动")
            else:
                task.status = 'failed'
                task.error_message = '推理启动失败'
                task.output_data = {
                    'success': False,
                    'error': '推理启动失败'
                }
                logger.error(f"推理任务 {task.id} 失败: 无法启动推理")

        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            logger.error(f"执行推理任务 {task.id} 时出错: {e}")
        finally:
            task.progress_percentage = 100
            task.completed_at = now_beijing()
            if task.started_at:
                task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())
            db.commit()

    async def _execute_agent_task(self, task: TaskExecution, db):
        """执行自动Agent任务"""
        try:
            training_record_id = task.input_data.get('training_record_id')
            if not training_record_id:
                raise ValueError("缺少训练记录ID")

            # 调用AgentDecisionService执行Agent决策
            from services.agent_decision_service import AgentDecisionService

            result = await AgentDecisionService.trigger_decision_stream(
                plan_id=task.plan_id,
                training_id=training_record_id
            )

            # 由于trigger_decision_stream是异步生成器，我们这里只记录启动
            task.status = 'completed'
            task.output_data = {
                'success': True,
                'message': 'Agent决策任务已启动',
                'training_record_id': training_record_id
            }
            logger.info(f"Agent任务 {task.id} 已启动")

        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            logger.error(f"执行Agent任务 {task.id} 时出错: {e}")
        finally:
            task.progress_percentage = 100
            task.completed_at = now_beijing()
            if task.started_at:
                task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())
            db.commit()

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


# 全局调度器实例
scheduler_service = SchedulerService()