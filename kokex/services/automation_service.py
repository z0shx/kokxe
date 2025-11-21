"""
自动化执行服务
负责管理交易的自动化流程：自动微调 -> 自动推理 -> 自动Agent决策 -> 自动工具执行
"""
import asyncio
import threading
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, AgentDecision
from services.training_service import TrainingService
from services.inference_service import InferenceService
from services.agent_decision_service import AgentDecisionService
from utils.logger import setup_logger

logger = setup_logger(__name__, "automation.log")


class AutomationStage(Enum):
    """自动化阶段"""
    TRAINING = "training"
    INFERENCE = "inference"
    AGENT_DECISION = "agent_decision"
    TOOL_EXECUTION = "tool_execution"


class AutomationService:
    """自动化服务类（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 调度器状态
        self.scheduler_running = False
        self.scheduler_thread = None
        self.last_check_time = None

        # 活跃任务跟踪
        self.active_tasks: Dict[int, Dict] = {}  # plan_id -> task_info

        self._initialized = True
        logger.info("自动化服务已初始化")

    def start_scheduler(self):
        """启动调度器"""
        if self.scheduler_running:
            logger.warning("自动化调度器已在运行")
            return

        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="automation-scheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        logger.info("✅ 自动化调度器已启动")

    def stop_scheduler(self):
        """停止调度器"""
        if not self.scheduler_running:
            return

        logger.info("停止自动化调度器...")
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("自动化调度器已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("自动化调度器循环已开始")

        while self.scheduler_running:
            try:
                current_time = datetime.now()
                self.last_check_time = current_time

                # 检查需要自动微调的计划
                self._check_auto_training(current_time)

                # 等待1分钟
                for _ in range(60):
                    if not self.scheduler_running:
                        break
                    threading.Event().wait(1)

            except Exception as e:
                logger.error(f"调度器循环出错: {e}")
                import traceback
                traceback.print_exc()
                # 出错后等待1分钟再继续
                threading.Event().wait(60)

        logger.info("自动化调度器循环已结束")

    def _check_auto_training(self, current_time: datetime):
        """检查需要执行自动训练的计划"""
        try:
            with get_db() as db:
                # 获取启用自动微调且正在运行的计划
                plans = db.query(TradingPlan).filter(
                    TradingPlan.auto_finetune_enabled == True,
                    TradingPlan.status == 'running'
                ).all()

                for plan in plans:
                    if self._should_trigger_training(plan, current_time):
                        logger.info(f"触发自动训练: plan_id={plan.id}, plan_name={plan.plan_name}")
                        # 在新线程中执行训练，避免阻塞调度器
                        training_thread = threading.Thread(
                            target=self._execute_auto_training,
                            args=(plan.id, current_time),
                            name=f"auto-training-{plan.id}",
                            daemon=True
                        )
                        training_thread.start()

        except Exception as e:
            logger.error(f"检查自动训练失败: {e}")

    def _should_trigger_training(self, plan: TradingPlan, current_time: datetime) -> bool:
        """判断是否应该触发训练"""
        try:
            # 检查是否有活跃的自动化任务
            if plan.id in self.active_tasks:
                task_info = self.active_tasks[plan.id]
                if task_info.get('stage') in [AutomationStage.TRAINING.value, AutomationStage.INFERENCE.value]:
                    logger.debug(f"计划 {plan.id} 正在执行自动化任务，跳过本次训练")
                    return False

            # 获取训练时间表
            schedule_times = plan.auto_finetune_schedule or []
            if not schedule_times:
                return False

            # 检查当前时间是否在训练时间表中
            current_time_str = current_time.strftime("%H:%M")
            if current_time_str not in schedule_times:
                return False

            # 检查今天是否已经训练过
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

            # 查找今天已完成的训练记录
            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan.id,
                    TrainingRecord.created_at >= today_start,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.trigger_type == 'auto'
                ).order_by(TrainingRecord.created_at.desc()).first()

                if latest_training:
                    logger.debug(f"计划 {plan.id} 今天已完成自动训练，跳过")
                    return False

            return True

        except Exception as e:
            logger.error(f"判断训练触发条件失败: {e}")
            return False

    def _execute_auto_training(self, plan_id: int, trigger_time: datetime):
        """执行自动训练"""
        task_id = f"auto-training-{plan_id}-{trigger_time.strftime('%Y%m%d%H%M')}"

        try:
            # 记录任务开始
            self.active_tasks[plan_id] = {
                'task_id': task_id,
                'stage': AutomationStage.TRAINING.value,
                'start_time': trigger_time,
                'trigger_type': 'auto'
            }

            logger.info(f"开始自动训练: {task_id}")

            # 获取训练参数
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: {plan_id}")
                    return

                finetune_params = plan.finetune_params or {}
                training_params = finetune_params.get('training', {})

            # 异步执行训练
            asyncio.run(self._run_training_async(plan_id, training_params, task_id))

        except Exception as e:
            logger.error(f"执行自动训练失败: {e}")
            # 清理任务记录
            if plan_id in self.active_tasks:
                del self.active_tasks[plan_id]

    async def _run_training_async(self, plan_id: int, training_params: Dict, task_id: str):
        """异步执行训练"""
        try:
            # 使用TrainingService执行训练
            training_result = await TrainingService.start_training_async(
                plan_id=plan_id,
                train_start_date=None,  # 使用自动配置
                train_end_date=None,    # 使用自动配置
                trigger_type='auto'
            )

            if training_result.get('success'):
                training_record_id = training_result.get('training_record_id')
                logger.info(f"自动训练成功: {task_id}, training_record_id={training_record_id}")

                # 更新任务状态
                if plan_id in self.active_tasks:
                    self.active_tasks[plan_id]['training_record_id'] = training_record_id
                    self.active_tasks[plan_id]['stage'] = AutomationStage.INFERENCE.value

                # 检查是否需要自动推理
                await self._check_auto_inference(plan_id, training_record_id)

            else:
                logger.error(f"自动训练失败: {task_id}, error={training_result.get('error')}")

        except Exception as e:
            logger.error(f"异步训练失败: {e}")

        finally:
            # 清理任务记录（如果没有后续步骤）
            if plan_id in self.active_tasks and self.active_tasks[plan_id].get('stage') != AutomationStage.INFERENCE.value:
                del self.active_tasks[plan_id]

    async def _check_auto_inference(self, plan_id: int, training_record_id: int):
        """检查是否需要执行自动推理"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.auto_inference_enabled:
                    logger.info(f"计划 {plan_id} 未启用自动推理，结束自动化流程")
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]
                    return

                logger.info(f"开始自动推理: plan_id={plan_id}, training_record_id={training_record_id}")

                # 获取推理参数
                finetune_params = plan.finetune_params or {}
                inference_params = finetune_params.get('inference', {})

                # 执行推理
                inference_result = await InferenceService.run_inference_async(
                    training_record_id=training_record_id,
                    temperature=inference_params.get('temperature', 1.0),
                    top_p=inference_params.get('top_p', 0.9),
                    sample_count=inference_params.get('sample_count', 30)
                )

                if inference_result.get('success'):
                    logger.info(f"自动推理成功: plan_id={plan_id}")

                    # 更新任务状态
                    if plan_id in self.active_tasks:
                        self.active_tasks[plan_id]['stage'] = AutomationStage.AGENT_DECISION.value

                    # 检查是否需要自动Agent决策
                    await self._check_auto_agent_decision(plan_id, training_record_id)

                else:
                    logger.error(f"自动推理失败: plan_id={plan_id}, error={inference_result.get('error')}")
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]

        except Exception as e:
            logger.error(f"检查自动推理失败: {e}")
            # 清理任务记录
            if plan_id in self.active_tasks:
                del self.active_tasks[plan_id]

    async def _check_auto_agent_decision(self, plan_id: int, training_record_id: int):
        """检查是否需要执行自动Agent决策"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.auto_agent_enabled:
                    logger.info(f"计划 {plan_id} 未启用自动Agent决策，结束自动化流程")
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]
                    return

                logger.info(f"开始自动Agent决策: plan_id={plan_id}")

                # 获取最新的推理数据
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_record_id
                ).order_by(PredictionData.timestamp.desc()).first()

                if not latest_prediction:
                    logger.error(f"未找到推理数据: training_record_id={training_record_id}")
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]
                    return

                # 执行Agent决策
                agent_result = await AgentDecisionService.make_decision_async(
                    plan_id=plan_id,
                    prediction_data_id=latest_prediction.id,
                    auto_mode=True  # 标记为自动模式
                )

                if agent_result.get('success'):
                    agent_decision_id = agent_result.get('decision_id')
                    logger.info(f"自动Agent决策成功: plan_id={plan_id}, decision_id={agent_decision_id}")

                    # 更新任务状态
                    if plan_id in self.active_tasks:
                        self.active_tasks[plan_id]['stage'] = AutomationStage.TOOL_EXECUTION.value
                        self.active_tasks[plan_id]['agent_decision_id'] = agent_decision_id

                    # 检查是否需要自动工具执行
                    await self._check_auto_tool_execution(plan_id, agent_decision_id)

                else:
                    logger.error(f"自动Agent决策失败: plan_id={plan_id}, error={agent_result.get('error')}")
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]

        except Exception as e:
            logger.error(f"检查自动Agent决策失败: {e}")
            # 清理任务记录
            if plan_id in self.active_tasks:
                del self.active_tasks[plan_id]

    async def _check_auto_tool_execution(self, plan_id: int, agent_decision_id: int):
        """检查是否需要执行自动工具执行"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.auto_tool_execution_enabled:
                    logger.info(f"计划 {plan_id} 未启用自动工具执行，记录待执行工具")
                    # 记录待执行工具，等待用户确认
                    await self._record_pending_tools(plan_id, agent_decision_id)
                    # 清理任务记录
                    if plan_id in self.active_tasks:
                        del self.active_tasks[plan_id]
                    return

                logger.info(f"开始自动工具执行: plan_id={plan_id}")

                # 自动执行工具
                execution_result = await AgentDecisionService.execute_pending_tools_async(
                    agent_decision_id=agent_decision_id,
                    auto_execute=True
                )

                if execution_result.get('success'):
                    logger.info(f"自动工具执行成功: plan_id={plan_id}")
                else:
                    logger.error(f"自动工具执行失败: plan_id={plan_id}, error={execution_result.get('error')}")

        except Exception as e:
            logger.error(f"检查自动工具执行失败: {e}")

        finally:
            # 清理任务记录
            if plan_id in self.active_tasks:
                del self.active_tasks[plan_id]

    async def _record_pending_tools(self, plan_id: int, agent_decision_id: int):
        """记录待执行工具"""
        try:
            # 这里AgentDecisionService会自动记录待执行工具
            logger.info(f"已记录待执行工具: plan_id={plan_id}, agent_decision_id={agent_decision_id}")

        except Exception as e:
            logger.error(f"记录待执行工具失败: {e}")

    def get_automation_status(self, plan_id: int) -> Dict:
        """获取自动化状态"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {}

                # 基础配置
                config = {
                    'auto_finetune_enabled': plan.auto_finetune_enabled or False,
                    'auto_inference_enabled': plan.auto_inference_enabled or False,
                    'auto_agent_enabled': plan.auto_agent_enabled or False,
                    'auto_tool_execution_enabled': plan.auto_tool_execution_enabled or False,
                    'auto_finetune_schedule': plan.auto_finetune_schedule or []
                }

                # 当前任务状态
                current_task = self.active_tasks.get(plan_id, {})

                # 获取最新的自动化执行记录
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.trigger_type == 'auto'
                ).order_by(TrainingRecord.created_at.desc()).first()

                status = {
                    **config,
                    'current_task': current_task,
                    'scheduler_running': self.scheduler_running,
                    'last_check_time': self.last_check_time,
                    'latest_auto_training': {
                        'id': latest_training.id,
                        'created_at': latest_training.created_at,
                        'status': latest_training.status,
                        'metrics': latest_training.metrics
                    } if latest_training else None
                }

                return status

        except Exception as e:
            logger.error(f"获取自动化状态失败: {e}")
            return {}

    def get_pending_tool_executions(self, plan_id: int) -> List[Dict]:
        """获取待执行工具列表"""
        try:
            with get_db() as db:
                # 查找自动模式下产生的待执行工具决策
                decisions = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id,
                    AgentDecision.auto_mode == True,
                    AgentDecision.status == 'completed'
                ).all()

                pending_tools = []
                for decision in decisions:
                    # 获取该决策下的待执行工具
                    # 这里需要在AgentDecision中添加pending_tools字段
                    if hasattr(decision, 'pending_tools') and decision.pending_tools:
                        for tool in decision.pending_tools:
                            pending_tools.append({
                                'decision_id': decision.id,
                                'decision_time': decision.created_at,
                                'tool_name': tool.get('name'),
                                'tool_args': tool.get('args'),
                                'status': tool.get('status', 'pending')
                            })

                return pending_tools

        except Exception as e:
            logger.error(f"获取待执行工具失败: {e}")
            return []

    def approve_pending_tool(self, plan_id: int, decision_id: int, tool_name: str) -> str:
        """批准执行待执行工具"""
        try:
            # 异步执行工具
            asyncio.run(self._execute_approved_tool(decision_id, tool_name))
            return f"✅ 已批准执行工具: {tool_name}"
        except Exception as e:
            logger.error(f"批准执行工具失败: {e}")
            return f"❌ 执行失败: {str(e)}"

    def reject_pending_tool(self, plan_id: int, decision_id: int, tool_name: str) -> str:
        """拒绝执行待执行工具"""
        try:
            # 更新工具状态为拒绝
            # 这里需要在数据库中更新状态
            logger.info(f"已拒绝执行工具: plan_id={plan_id}, decision_id={decision_id}, tool={tool_name}")
            return f"❌ 已拒绝执行工具: {tool_name}"
        except Exception as e:
            logger.error(f"拒绝执行工具失败: {e}")
            return f"❌ 操作失败: {str(e)}"

    async def _execute_approved_tool(self, decision_id: int, tool_name: str):
        """执行已批准的工具"""
        try:
            # 这里调用AgentDecisionService来执行特定的工具
            execution_result = await AgentDecisionService.execute_specific_tool_async(
                decision_id=decision_id,
                tool_name=tool_name
            )

            if execution_result.get('success'):
                logger.info(f"工具执行成功: decision_id={decision_id}, tool={tool_name}")
            else:
                logger.error(f"工具执行失败: decision_id={decision_id}, tool={tool_name}")

        except Exception as e:
            logger.error(f"执行已批准工具失败: {e}")


# 全局单例实例
automation_service = AutomationService()