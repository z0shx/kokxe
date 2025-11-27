"""
模型训练服务
负责Kronos模型的微调训练
"""
import asyncio
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, KlineData
from utils.logger import setup_logger
from sqlalchemy import and_, func

logger = setup_logger(__name__, "training_service.log")

# 全局训练锁（确保同时只有一个训练任务）
_training_lock = asyncio.Lock()
_training_queue = []

# 训练进度缓存 {training_id: {'progress': float, 'stage': str, 'message': str}}
_training_progress = {}

# 活跃训练任务缓存 {training_id: task}
_active_training_tasks = {}


def _convert_numpy_to_python(obj):
    """
    递归转换numpy类型为Python原生类型

    Args:
        obj: 任意对象

    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_python(item) for item in obj]
    else:
        return obj


class TrainingService:
    """模型训练服务"""

    @classmethod
    def get_training_progress(cls, training_id: int) -> Optional[Dict]:
        """
        获取训练进度

        Args:
            training_id: 训练记录ID

        Returns:
            进度字典: {'progress': float, 'stage': str, 'message': str} 或 None
        """
        return _training_progress.get(training_id)

    @classmethod
    def _update_progress(cls, training_id: int, progress: float, stage: str, message: str):
        """
        更新训练进度

        Args:
            training_id: 训练记录ID
            progress: 进度 0.0-1.0
            stage: 阶段名称
            message: 进度消息
        """
        _training_progress[training_id] = {
            'progress': progress,
            'stage': stage,
            'message': message,
            'timestamp': datetime.now()
        }
        logger.debug(f"训练进度更新: training_id={training_id}, progress={progress:.2%}, stage={stage}, message={message}")

    @classmethod
    def recover_stuck_training_records(cls):
        """恢复卡住的训练记录"""
        try:
            with get_db() as db:
                stuck_records = db.query(TrainingRecord).filter(
                    TrainingRecord.status == 'training'
                ).all()

                for record in stuck_records:
                    logger.warning(f"发现卡住的训练记录: id={record.id}, version={record.version}, plan_id={record.plan_id}")

                    # 如果训练开始时间超过24小时，标记为失败
                    if record.train_start_time:
                        hours_elapsed = (datetime.utcnow() - record.train_start_time).total_seconds() / 3600
                        if hours_elapsed > 24:
                            logger.error(f"训练记录卡住超过24小时，标记为失败: id={record.id}")
                            record.status = 'failed'
                            record.train_end_time = datetime.utcnow()
                            record.train_duration = int(hours_elapsed * 3600)
                            record.error_message = f"训练卡住超过24小时，自动标记为失败"
                            db.commit()
                        else:
                            logger.info(f"训练记录仍在合理时间内: id={record.id}, 已耗时{hours_elapsed:.1f}小时")
                    else:
                        # 没有开始时间，直接标记为失败
                        logger.error(f"训练记录没有开始时间，标记为失败: id={record.id}")
                        record.status = 'failed'
                        record.error_message = "训练记录没有开始时间，自动标记为失败"
                        db.commit()

                if stuck_records:
                    logger.info(f"处理了 {len(stuck_records)} 个卡住的训练记录")
                else:
                    logger.info("没有发现卡住的训练记录")

        except Exception as e:
            logger.error(f"恢复卡住的训练记录失败: {e}")

    @classmethod
    async def start_training(cls, plan_id: int, manual: bool = False) -> Optional[int]:
        """
        启动模型训练

        Args:
            plan_id: 计划ID
            manual: 是否手动触发（手动触发会立即执行，自动触发会排队）

        Returns:
            训练记录ID，失败返回None
        """
        try:
            # 获取计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return None

                # 生成新版本号
                last_version = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id
                ).order_by(TrainingRecord.created_at.desc()).first()

                if last_version:
                    # 从 v1 -> v2, v2 -> v3
                    last_num = int(last_version.version.replace('v', ''))
                    new_version = f"v{last_num + 1}"
                else:
                    new_version = "v1"

                # 获取训练数据范围（最近N天）
                train_days = (plan.data_end_time - plan.data_start_time).days
                data_end_time = datetime.now()
                data_start_time = data_end_time - timedelta(days=train_days)

                # 检查数据是否充足
                data_count = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= data_start_time,
                        KlineData.timestamp <= data_end_time
                    )
                ).count()

                if data_count < 100:  # 最少需要100条数据
                    logger.error(f"训练数据不足: plan_id={plan_id}, count={data_count}")
                    return None

                # 创建训练记录
                training_record = TrainingRecord(
                    plan_id=plan_id,
                    version=new_version,
                    status='waiting',
                    train_params=plan.finetune_params,
                    data_start_time=data_start_time,
                    data_end_time=data_end_time,
                    data_count=data_count
                )

                db.add(training_record)
                db.commit()
                db.refresh(training_record)

                training_id = training_record.id
                logger.info(
                    f"创建训练记录: plan_id={plan_id}, "
                    f"version={new_version}, training_id={training_id}"
                )

            # 异步执行训练（不阻塞）
            task = asyncio.create_task(cls._execute_training(training_id, plan_id, manual))
            _active_training_tasks[training_id] = task
            logger.info(f"✅ 训练任务已创建: training_id={training_id}")

            return training_id

        except Exception as e:
            logger.error(f"启动训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    async def _execute_training(cls, training_id: int, plan_id: int, manual: bool):
        """
        执行训练任务（异步）

        Args:
            training_id: 训练记录ID
            plan_id: 计划ID
            manual: 是否手动触发
        """
        try:
            # 获取训练锁（确保串行执行）
            async with _training_lock:
                logger.info(f"开始执行训练: training_id={training_id}, plan_id={plan_id}, manual={manual}")

                # 更新状态为训练中
                with get_db() as db:
                    db.query(TrainingRecord).filter(
                        TrainingRecord.id == training_id
                    ).update({
                        'status': 'training',
                        'train_start_time': datetime.utcnow()
                    })
                    db.commit()
                    logger.info(f"✅ 训练状态已更新为training: training_id={training_id}")

                # 执行实际的训练（在线程池中运行，避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                logger.info(f"开始同步训练: training_id={training_id}")
                result = await loop.run_in_executor(
                    None,
                    cls._train_model_sync,
                    training_id,
                    plan_id
                )
                logger.info(f"同步训练完成: training_id={training_id}, success={result.get('success', False)}")

                # 更新训练结果
                try:
                    logger.info(f"开始更新训练结果到数据库: training_id={training_id}")
                    with get_db() as db:
                        train_end_time = datetime.utcnow()
                        logger.info(f"准备更新训练结果: training_id={training_id}, success={result['success']}, end_time={train_end_time}")

                        record = db.query(TrainingRecord).filter(
                            TrainingRecord.id == training_id
                        ).first()

                        if not record:
                            logger.error(f"训练记录不存在: training_id={training_id}")
                            return

                        logger.info(f"当前记录状态: status={record.status}, start_time={record.train_start_time}")

                        # 更新字段
                        old_status = record.status
                        record.status = 'completed' if result['success'] else 'failed'
                        record.train_end_time = train_end_time
                        if record.train_start_time:
                            record.train_duration = int((train_end_time - record.train_start_time).total_seconds())
                        else:
                            record.train_duration = 0
                            logger.warning(f"训练开始时间为空，设置持续时间为0: training_id={training_id}")

                        record.train_metrics = _convert_numpy_to_python(result.get('metrics', {}))
                        record.tokenizer_path = result.get('tokenizer_path')
                        record.predictor_path = result.get('predictor_path')
                        record.error_message = result.get('error')

                        logger.info(f"更新后状态: {old_status} -> {record.status}, duration={record.train_duration}")

                        # 尝试提交
                        db.commit()
                        logger.info(f"✅ 训练记录更新成功: training_id={training_id}, status={record.status}")

                        # 如果成功，更新计划的最新训练记录ID
                        if result['success']:
                            try:
                                update_result = db.query(TradingPlan).filter(
                                    TradingPlan.id == plan_id
                                ).update({
                                    'latest_training_record_id': training_id,
                                    'last_finetune_time': train_end_time
                                })
                                db.commit()
                                logger.info(f"✅ 计划信息更新成功: plan_id={plan_id}, 更新行数={update_result}")
                            except Exception as plan_error:
                                logger.error(f"❌ 更新计划信息失败: {plan_error}")
                                db.rollback()

                except Exception as db_error:
                    logger.error(f"❌ 数据库更新失败: training_id={training_id}, error={db_error}")
                    import traceback
                    traceback.print_exc()

                    # 多次尝试恢复状态更新，确保训练时长被正确记录
                    for attempt in range(3):
                        try:
                            logger.warning(f"尝试状态恢复 (第{attempt+1}次): training_id={training_id}")
                            with get_db() as db:
                                # 重新获取记录以确保数据一致性
                                record = db.query(TrainingRecord).filter(
                                    TrainingRecord.id == training_id
                                ).first()

                                if record:
                                    train_end_time = datetime.utcnow()
                                    duration = 0
                                    if record.train_start_time:
                                        duration = int((train_end_time - record.train_start_time).total_seconds())

                                    # 根据训练结果更新状态，但确保时长被正确记录
                                    if result['success']:
                                        record.status = 'completed'
                                    else:
                                        record.status = 'failed'

                                    record.train_end_time = train_end_time
                                    record.train_duration = duration
                                    record.error_message = f"数据库更新异常(已恢复): {str(db_error)}"

                                    # 保留训练指标和路径信息
                                    if result.get('success') and result.get('metrics'):
                                        record.train_metrics = _convert_numpy_to_python(result.get('metrics', {}))
                                    if result.get('tokenizer_path'):
                                        record.tokenizer_path = result.get('tokenizer_path')
                                    if result.get('predictor_path'):
                                        record.predictor_path = result.get('predictor_path')

                                    db.commit()
                                    logger.warning(f"✅ 状态恢复成功 (第{attempt+1}次): training_id={training_id}, status={record.status}, duration={duration}s")
                                    break
                        except Exception as retry_error:
                            logger.error(f"状态恢复失败 (第{attempt+1}次): {retry_error}")
                            if attempt == 2:  # 最后一次尝试
                                logger.error(f"❌ 所有状态恢复尝试都失败: training_id={training_id}")
                                import traceback
                                traceback.print_exc()

                # 获取最新的训练记录状态用于日志
                final_record = None
                try:
                    with get_db() as db:
                        final_record = db.query(TrainingRecord).filter(
                            TrainingRecord.id == training_id
                        ).first()
                except:
                    pass

                duration_info = f"duration={final_record.train_duration}s" if final_record else "duration=unknown"
                logger.info(
                    f"训练完成: training_id={training_id}, "
                    f"status={result['success']}, "
                    f"{duration_info}"
                )

                # 如果启用了自动推理，触发推理任务
                with get_db() as db:
                    plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                    if plan and plan.auto_inference_enabled and result['success']:
                        logger.info(f"自动触发推理: training_id={training_id}")
                        from services.inference_service import InferenceService
                        asyncio.create_task(InferenceService.start_inference(training_id))

                logger.info(f"✅ 训练任务完全完成: training_id={training_id}")

        except Exception as e:
            logger.error(f"训练执行失败: training_id={training_id}, error={e}")
            import traceback
            traceback.print_exc()

                # 更新状态为失败
            try:
                with get_db() as db:
                    db.query(TrainingRecord).filter(
                        TrainingRecord.id == training_id
                    ).update({
                        'status': 'failed',
                        'train_end_time': datetime.utcnow(),
                        'error_message': str(e)
                    })
                    db.commit()
                    logger.info(f"✅ 失败状态已更新: training_id={training_id}")
            except Exception as db_error:
                logger.error(f"更新失败状态时出错: training_id={training_id}, db_error={db_error}")

        finally:
            # 清理活跃任务缓存
            if training_id in _active_training_tasks:
                del _active_training_tasks[training_id]
                logger.info(f"✅ 已清理活跃任务缓存: training_id={training_id}")

    @classmethod
    def _train_model_sync(cls, training_id: int, plan_id: int) -> Dict:
        """
        同步训练模型（在线程池中执行）

        Returns:
            结果字典: {
                'success': bool,
                'metrics': dict,  # 训练指标
                'tokenizer_path': str,
                'predictor_path': str,
                'error': str  # 错误信息（如有）
            }
        """
        try:
            from services.kronos_trainer import KronosTrainer

            logger.info(f"开始同步训练: training_id={training_id}")

            # 获取训练配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not plan or not training_record:
                    return {'success': False, 'error': '计划或训练记录不存在'}

            # 构建训练配置
            training_config = {
                'plan_id': plan_id,
                'inst_id': plan.inst_id,
                'interval': plan.interval,
                'data_start_time': training_record.data_start_time,
                'data_end_time': training_record.data_end_time,
                'finetune_params': plan.finetune_params,
                'save_path': str(Path(f"./models/plan_{plan_id}/v{training_record.version}"))
            }

            logger.info(f"训练配置: {training_config}")

            # 定义进度回调函数
            def progress_callback(progress: float, stage: str, message: str):
                cls._update_progress(training_id, progress, stage, message)

            # 创建训练器并执行训练
            trainer = KronosTrainer(training_config, progress_callback=progress_callback)
            result = trainer.train()

            logger.info(f"训练完成: success={result['success']}")

            # 清除进度缓存
            if training_id in _training_progress:
                del _training_progress[training_id]

            return result

        except Exception as e:
            logger.error(f"同步训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def get_training_status(cls, training_id: int) -> Optional[Dict]:
        """获取训练状态"""
        try:
            with get_db() as db:
                record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not record:
                    return None

                return {
                    'id': record.id,
                    'plan_id': record.plan_id,
                    'version': record.version,
                    'status': record.status,
                    'is_active': record.is_active,
                    'train_start_time': record.train_start_time,
                    'train_end_time': record.train_end_time,
                    'train_duration': record.train_duration,
                    'train_metrics': record.train_metrics,
                    'error_message': record.error_message
                }
        except Exception as e:
            logger.error(f"获取训练状态失败: {e}")
            return None

    @classmethod
    def list_training_records(cls, plan_id: int) -> list:
        """获取计划的所有训练记录"""
        try:
            with get_db() as db:
                records = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id
                ).order_by(TrainingRecord.created_at.desc()).all()

                result = []
                for record in records:
                    result.append({
                        'id': record.id,
                        'version': record.version,
                        'status': record.status,
                        'is_active': record.is_active,
                        'train_start_time': record.train_start_time,
                        'train_end_time': record.train_end_time,
                        'train_duration': record.train_duration,
                        'data_count': record.data_count,
                        'created_at': record.created_at
                    })

                return result
        except Exception as e:
            logger.error(f"获取训练记录列表失败: {e}")
            return []

    @classmethod
    def toggle_training_version(cls, training_id: int, is_active: bool) -> bool:
        """启用/禁用训练版本"""
        try:
            with get_db() as db:
                db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).update({'is_active': is_active})
                db.commit()

                logger.info(f"训练版本状态已更新: training_id={training_id}, is_active={is_active}")
                return True
        except Exception as e:
            logger.error(f"更新训练版本状态失败: {e}")
            return False

    @classmethod
    def cancel_training(cls, training_id: int) -> Dict:
        """
        取消等待中或训练中的任务

        Args:
            training_id: 训练记录ID

        Returns:
            结果字典: {'success': bool, 'message': str}
        """
        try:
            with get_db() as db:
                record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not record:
                    return {'success': False, 'message': '训练记录不存在'}

                # 只能取消等待中的任务
                if record.status not in ['waiting', 'training']:
                    return {
                        'success': False,
                        'message': f'只能取消等待中或训练中的任务，当前状态: {record.status}'
                    }

                # 更新状态为取消
                db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).update({
                    'status': 'cancelled',
                    'error_message': '用户取消'
                })
                db.commit()

                logger.info(f"训练任务已取消: training_id={training_id}")
                return {
                    'success': True,
                    'message': f'✅ 已取消训练任务 {record.version}'
                }

        except Exception as e:
            logger.error(f"取消训练失败: {e}")
            return {
                'success': False,
                'message': f'取消失败: {str(e)}'
            }

    @classmethod
    def delete_training_record(cls, training_id: int) -> Dict:
        """
        删除训练记录及相关数据

        Args:
            training_id: 训练记录ID

        Returns:
            结果字典: {'success': bool, 'message': str}
        """
        try:
            with get_db() as db:
                record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not record:
                    return {'success': False, 'message': '训练记录不存在'}

                # 不能删除训练中的记录
                if record.status == 'training':
                    return {
                        'success': False,
                        'message': '无法删除训练中的记录，请先取消训练'
                    }

                # 删除预测数据
                from database.models import PredictionData
                deleted_predictions = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                ).delete(synchronize_session=False)

                logger.info(f"删除预测数据: {deleted_predictions}条")

                # 删除训练记录
                version = record.version
                db.delete(record)
                db.commit()

                logger.info(f"成功删除训练记录: training_id={training_id}, version={version}")
                return {
                    'success': True,
                    'message': f'✅ 已删除训练记录 {version} 及 {deleted_predictions} 条预测数据'
                }

        except Exception as e:
            logger.error(f"删除训练记录失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'删除失败: {str(e)}'
            }
