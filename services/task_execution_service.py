"""
任务执行服务
负责记录和管理定时任务的执行历史
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from database.db import get_db
from database.models import TaskExecution
from utils.logger import setup_logger

logger = setup_logger(__name__, "task_execution_service.log")


class TaskExecutionService:
    """任务执行服务"""

    @classmethod
    def create_task_execution(
        cls,
        plan_id: int,
        task_type: str,
        task_name: str,
        task_description: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
        trigger_type: str = "scheduled",
        trigger_source: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> TaskExecution:
        """
        创建任务执行记录

        Args:
            plan_id: 计划ID
            task_type: 任务类型 (auto_finetune, auto_inference, auto_agent)
            task_name: 任务名称
            task_description: 任务描述
            scheduled_time: 计划执行时间
            trigger_type: 触发类型 (scheduled, manual)
            trigger_source: 触发源
            input_data: 输入参数

        Returns:
            TaskExecution: 创建的任务执行记录
        """
        try:
            with get_db() as db:
                task_execution = TaskExecution(
                    plan_id=plan_id,
                    task_type=task_type,
                    task_name=task_name,
                    task_description=task_description,
                    status='pending',
                    priority=1,
                    scheduled_time=scheduled_time,
                    trigger_type=trigger_type,
                    trigger_source=trigger_source,
                    input_data=input_data,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                db.add(task_execution)
                db.commit()
                db.refresh(task_execution)

                logger.info(f"创建任务执行记录: id={task_execution.id}, type={task_type}, plan_id={plan_id}")
                return task_execution

        except Exception as e:
            logger.error(f"创建任务执行记录失败: {e}")
            raise

    @classmethod
    def update_task_status(
        cls,
        task_id: int,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_seconds: Optional[int] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        progress_percentage: Optional[int] = None,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新任务执行状态

        Args:
            task_id: 任务ID
            status: 任务状态
            started_at: 开始时间
            completed_at: 完成时间
            duration_seconds: 执行时长
            output_data: 输出结果
            error_message: 错误信息
            progress_percentage: 进度百分比
            task_metadata: 额外元数据

        Returns:
            bool: 是否成功
        """
        try:
            with get_db() as db:
                task = db.query(TaskExecution).filter(TaskExecution.id == task_id).first()
                if not task:
                    logger.warning(f"任务不存在: task_id={task_id}")
                    return False

                # 更新字段
                if status:
                    task.status = status
                if started_at:
                    task.started_at = started_at
                if completed_at:
                    task.completed_at = completed_at
                if duration_seconds is not None:
                    task.duration_seconds = duration_seconds
                if output_data:
                    task.output_data = output_data
                if error_message:
                    task.error_message = error_message
                if progress_percentage is not None:
                    task.progress_percentage = progress_percentage
                if task_metadata:
                    task.task_metadata = task_metadata

                task.updated_at = datetime.now()

                db.commit()

                logger.info(f"更新任务状态: id={task_id}, status={status}")
                return True

        except Exception as e:
            logger.error(f"更新任务状态失败: task_id={task_id}, error={e}")
            return False

    @classmethod
    def start_task_execution(cls, task_id: int) -> bool:
        """
        标记任务开始执行

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功
        """
        return cls.update_task_status(
            task_id=task_id,
            status='running',
            started_at=datetime.now(),
            progress_percentage=0
        )

    @classmethod
    def complete_task_execution(
        cls,
        task_id: int,
        success: bool = True,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        标记任务完成

        Args:
            task_id: 任务ID
            success: 是否成功
            output_data: 输出结果
            error_message: 错误信息

        Returns:
            bool: 是否成功
        """
        with get_db() as db:
            task = db.query(TaskExecution).filter(TaskExecution.id == task_id).first()
            if not task:
                logger.warning(f"任务不存在: task_id={task_id}")
                return False

        # 计算执行时长
        duration_seconds = None
        if task.started_at:
            duration_seconds = int((datetime.now() - task.started_at).total_seconds())

        return cls.update_task_status(
            task_id=task_id,
            status='completed' if success else 'failed',
            completed_at=datetime.now(),
            duration_seconds=duration_seconds,
            output_data=output_data,
            error_message=error_message,
            progress_percentage=100
        )

    @classmethod
    def create_scheduled_task(
        cls,
        plan_id: int,
        task_type: str,
        time_str: str
    ) -> TaskExecution:
        """
        创建定时任务执行记录

        Args:
            plan_id: 计划ID
            task_type: 任务类型 (auto_finetune, auto_inference)
            time_str: 时间字符串 (HH:MM)

        Returns:
            TaskExecution: 创建的任务执行记录
        """
        task_type_names = {
            'auto_finetune': '自动微调',
            'auto_inference': '自动预测'
        }

        task_name = f"{task_type_names.get(task_type, task_type)}-计划{plan_id}-{time_str}"
        task_description = f"定时{task_type_names.get(task_type, task_type)}任务"

        # 计算下次执行时间
        from datetime import date, timedelta
        today = date.today()
        hour, minute = map(int, time_str.split(':'))
        scheduled_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))

        # 如果已过今天的时间，则安排到明天
        if scheduled_time <= datetime.now():
            scheduled_time += timedelta(days=1)

        return cls.create_task_execution(
            plan_id=plan_id,
            task_type=task_type,
            task_name=task_name,
            task_description=task_description,
            scheduled_time=scheduled_time,
            trigger_type="scheduled",
            trigger_source=f"plan_{plan_id}_schedule",
            input_data={"scheduled_time_str": time_str}
        )

    @classmethod
    def get_plan_task_history(cls, plan_id: int, limit: int = 50) -> list:
        """
        获取计划的任务执行历史

        Args:
            plan_id: 计划ID
            limit: 返回记录数限制

        Returns:
            list: 任务执行历史列表
        """
        try:
            with get_db() as db:
                tasks = db.query(TaskExecution).filter(
                    TaskExecution.plan_id == plan_id
                ).order_by(
                    TaskExecution.created_at.desc()
                ).limit(limit).all()

                # 转换为字典格式
                result = []
                for task in tasks:
                    result.append({
                        'id': task.id,
                        'task_type': task.task_type,
                        'task_name': task.task_name,
                        'task_description': task.task_description,
                        'status': task.status,
                        'priority': task.priority,
                        'scheduled_time': task.scheduled_time.isoformat() if task.scheduled_time else None,
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'duration_seconds': task.duration_seconds,
                        'trigger_type': task.trigger_type,
                        'trigger_source': task.trigger_source,
                        'input_data': task.input_data,
                        'output_data': task.output_data,
                        'error_message': task.error_message,
                        'progress_percentage': task.progress_percentage,
                        'task_metadata': task.task_metadata,
                        'created_at': task.created_at.isoformat() if task.created_at else None,
                        'updated_at': task.updated_at.isoformat() if task.updated_at else None
                    })

                return result

        except Exception as e:
            logger.error(f"获取计划任务历史失败: plan_id={plan_id}, error={e}")
            return []

    @classmethod
    def get_all_task_history(cls, limit: int = 100) -> list:
        """
        获取所有任务的执行历史

        Args:
            limit: 返回记录数限制

        Returns:
            list: 任务执行历史列表
        """
        try:
            with get_db() as db:
                # 联合查询获取计划名称
                from database.models import TradingPlan
                tasks = db.query(TaskExecution, TradingPlan).join(
                    TradingPlan, TaskExecution.plan_id == TradingPlan.id
                ).order_by(
                    TaskExecution.created_at.desc()
                ).limit(limit).all()

                # 转换为字典格式
                result = []
                for task, plan in tasks:
                    result.append({
                        'id': task.id,
                        'plan_id': task.plan_id,
                        'plan_name': plan.plan_name,
                        'task_type': task.task_type,
                        'task_name': task.task_name,
                        'task_description': task.task_description,
                        'status': task.status,
                        'priority': task.priority,
                        'scheduled_time': task.scheduled_time.isoformat() if task.scheduled_time else None,
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'duration_seconds': task.duration_seconds,
                        'trigger_type': task.trigger_type,
                        'trigger_source': task.trigger_source,
                        'input_data': task.input_data,
                        'output_data': task.output_data,
                        'error_message': task.error_message,
                        'progress_percentage': task.progress_percentage,
                        'task_metadata': task.task_metadata,
                        'created_at': task.created_at.isoformat() if task.created_at else None,
                        'updated_at': task.updated_at.isoformat() if task.updated_at else None
                    })

                return result

        except Exception as e:
            logger.error(f"获取所有任务历史失败: error={e}")
            return []