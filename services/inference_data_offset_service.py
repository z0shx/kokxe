"""
推理数据偏移计算服务
智能计算Data Offset确保预测间隔的正确性
"""
import datetime
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from database.db import get_db
from database.models import TradingPlan, PredictionData, TrainingRecord, KlineData
from utils.logger import setup_logger
from utils.timezone_helper import BEIJING_TZ

logger = setup_logger(__name__, "inference_data_offset.log")


class InferenceDataOffsetService:
    """推理数据偏移计算服务"""

    BEIJING_TZ = BEIJING_TZ

    @classmethod
    def calculate_optimal_data_offset(
        cls,
        plan_id: int,
        target_interval_hours: int = 4,
        manual_trigger: bool = False
    ) -> Dict[str, any]:
        """
        计算最优的数据偏移量，确保预测间隔的正确性

        Args:
            plan_id: 计划ID
            target_interval_hours: 目标预测间隔（小时）
            manual_trigger: 是否手动触发

        Returns:
            包含计算结果的字典：
            {
                'success': bool,
                'data_offset': int,
                'reasoning': str,
                'latest_prediction_time': Optional[datetime],
                'time_diff_hours': Optional[float],
                'recommended_offset': int,
                'actual_interval': Optional[float]
            }
        """
        try:
            result = {
                'success': False,
                'data_offset': 0,
                'reasoning': '',
                'latest_prediction_time': None,
                'time_diff_hours': None,
                'recommended_offset': 0,
                'actual_interval': None
            }

            with get_db() as db:
                # 获取计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    result['reasoning'] = '计划不存在'
                    return result

                # 获取最新的训练记录
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    result['reasoning'] = '没有找到可用的训练记录'
                    return result

                # 获取计划的K线间隔（小时）
                kline_interval_hours = cls._convert_interval_to_hours(plan.interval)
                if kline_interval_hours is None:
                    result['reasoning'] = f'不支持的K线间隔: {plan.interval}'
                    return result

                logger.info(f"计划 {plan_id}: K线间隔={kline_interval_hours}小时, 目标预测间隔={target_interval_hours}小时")

                # 获取最新预测数据
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id
                ).order_by(PredictionData.timestamp.desc()).first()

                current_time = datetime.now(cls.BEIJING_TZ)

                if latest_prediction:
                    # 计算时间差
                    latest_prediction_time = latest_prediction.timestamp
                    if hasattr(latest_prediction_time, 'astimezone'):
                        latest_prediction_time = latest_prediction_time.astimezone(cls.BEIJING_TZ)
                    else:
                        # 如果没有时区信息，假设为UTC+8
                        latest_prediction_time = cls.BEIJING_TZ.localize(latest_prediction_time.replace(tzinfo=None))

                    # 确保时间差为正数，如果预测时间在未来，说明时区转换有问题
                    time_diff = current_time - latest_prediction_time
                    time_diff_hours = time_diff.total_seconds() / 3600

                    # 如果时间差为负，说明预测时间在未来，可能是时区问题，使用绝对值
                    if time_diff_hours < 0:
                        logger.warning(f"时间差为负值({time_diff_hours:.2f}h)，可能是时区问题，使用绝对值")
                        time_diff_hours = abs(time_diff_hours)

                    result['latest_prediction_time'] = latest_prediction_time
                    result['time_diff_hours'] = time_diff_hours

                    logger.info(f"计划 {plan_id}: 最新预测时间={latest_prediction_time}, 距今={time_diff_hours:.2f}小时")

                    # 计算需要的偏移量
                    recommended_offset = cls._calculate_data_offset_for_interval(
                        time_diff_hours,
                        target_interval_hours,
                        kline_interval_hours
                    )

                    result['recommended_offset'] = recommended_offset
                    result['data_offset'] = recommended_offset

                    # 计算应用偏移后的实际间隔
                    actual_interval = cls._calculate_actual_interval_after_offset(
                        time_diff_hours,
                        recommended_offset,
                        kline_interval_hours
                    )

                    result['actual_interval'] = actual_interval

                    # 构建推理说明
                    if manual_trigger:
                        result['reasoning'] = cls._build_manual_trigger_reasoning(
                            time_diff_hours, target_interval_hours, recommended_offset, actual_interval, kline_interval_hours
                        )
                    else:
                        result['reasoning'] = cls._build_auto_trigger_reasoning(
                            time_diff_hours, target_interval_hours, recommended_offset, actual_interval, kline_interval_hours
                        )

                    result['success'] = True

                else:
                    # 没有历史预测数据，使用默认偏移
                    result['reasoning'] = f'没有历史预测数据，使用默认偏移量。目标间隔={target_interval_hours}小时，K线间隔={kline_interval_hours}小时'
                    result['data_offset'] = 0
                    result['recommended_offset'] = 0
                    result['success'] = True

                logger.info(f"计划 {plan_id}: Data Offset计算完成 - offset={result['data_offset']}, {result['reasoning']}")

                return result

        except Exception as e:
            logger.error(f"计算数据偏移失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'data_offset': 0,
                'reasoning': f'计算失败: {str(e)}',
                'latest_prediction_time': None,
                'time_diff_hours': None,
                'recommended_offset': 0,
                'actual_interval': None
            }

    @classmethod
    def _convert_interval_to_hours(cls, interval: str) -> Optional[float]:
        """将K线间隔转换为小时数"""
        interval_mapping = {
            '1M': 1/60, '3M': 3/60, '5M': 5/60, '15M': 15/60, '30M': 0.5,
            '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 0.5,
            '1H': 1, '2H': 2, '4H': 4, '6H': 6, '8H': 8, '12H': 12,
            '1D': 24, '3D': 72, '1W': 168,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12,
            '1d': 24, '3d': 72, '1w': 168
        }
        return interval_mapping.get(interval.upper())

    @classmethod
    def _calculate_data_offset_for_interval(
        cls,
        time_diff_hours: float,
        target_interval_hours: int,
        kline_interval_hours: float
    ) -> int:
        """
        计算为满足目标间隔所需的数据偏移量

        Args:
            time_diff_hours: 当前时间差（小时）
            target_interval_hours: 目标间隔（小时）
            kline_interval_hours: K线间隔（小时）

        Returns:
            需要的数据偏移量（K线条数）
        """
        # 计算需要回退的时间以达到目标间隔
        if time_diff_hours < target_interval_hours:
            # 时间差小于目标间隔，需要进一步回退
            additional_backoff = target_interval_hours - time_diff_hours
        else:
            # 时间差大于等于目标间隔，尽量接近目标间隔
            additional_backoff = time_diff_hours % target_interval_hours

        # 将回退时间转换为K线条数
        offset_klines = int(additional_backoff / kline_interval_hours)

        logger.info(f"偏移计算: 时间差={time_diff_hours:.2f}h, 目标={target_interval_hours}h, "
                   f"K线间隔={kline_interval_hours}h, 需要回退={additional_backoff:.2f}h, "
                   f"偏移K线数={offset_klines}")

        return offset_klines

    @classmethod
    def _calculate_actual_interval_after_offset(
        cls,
        time_diff_hours: float,
        data_offset: int,
        kline_interval_hours: float
    ) -> float:
        """
        计算应用偏移后的实际预测间隔

        Args:
            time_diff_hours: 原始时间差（小时）
            data_offset: 数据偏移量（K线条数）
            kline_interval_hours: K线间隔（小时）

        Returns:
            应用偏移后的实际间隔（小时）
        """
        offset_time_hours = data_offset * kline_interval_hours
        actual_interval = time_diff_hours + offset_time_hours
        return actual_interval

    @classmethod
    def _build_manual_trigger_reasoning(
        cls,
        time_diff_hours: float,
        target_interval_hours: int,
        recommended_offset: int,
        actual_interval: float,
        kline_interval_hours: float
    ) -> str:
        """构建手动触发的推理说明"""
        if recommended_offset == 0:
            return (f"手动触发：当前时间差{time_diff_hours:.2f}小时已满足{target_interval_hours}小时目标间隔，"
                   f"使用最新数据直接预测，实际间隔{actual_interval:.2f}小时")
        else:
            return (f"手动触发：当前时间差{time_diff_hours:.2f}小时，"
                   f"为满足{target_interval_hours}小时目标间隔，"
                   f"回退{recommended_offset}条K线数据（{recommended_offset * kline_interval_hours:.2f}小时），"
                   f"实际间隔{actual_interval:.2f}小时")

    @classmethod
    def _build_auto_trigger_reasoning(
        cls,
        time_diff_hours: float,
        target_interval_hours: int,
        recommended_offset: int,
        actual_interval: float,
        kline_interval_hours: float
    ) -> str:
        """构建自动触发的推理说明"""
        return (f"自动触发：距离上次预测{time_diff_hours:.2f}小时，"
               f"为满足{target_interval_hours}小时目标间隔，"
               f"应用{recommended_offset}条K线偏移（{recommended_offset * kline_interval_hours:.2f}小时），"
               f"实际间隔{actual_interval:.2f}小时")

    @classmethod
    def update_inference_params_with_offset(
        cls,
        plan_id: int,
        training_id: int,
        data_offset: int
    ) -> bool:
        """
        更新训练记录的推理参数，应用计算得出的数据偏移

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID
            data_offset: 数据偏移量

        Returns:
            是否更新成功
        """
        try:
            with get_db() as db:
                # 获取计划信息（推理参数存储在计划中）
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return False

                # 获取并更新推理参数
                finetune_params = plan.finetune_params or {}
                if 'inference' not in finetune_params:
                    finetune_params['inference'] = {}

                old_offset = finetune_params['inference'].get('data_offset', 0)
                finetune_params['inference']['data_offset'] = data_offset

                # 保存更新
                plan.finetune_params = finetune_params
                db.commit()

                logger.info(f"更新推理参数: plan_id={plan_id}, training_id={training_id}, "
                           f"old_offset={old_offset}, new_offset={data_offset}")
                return True

        except Exception as e:
            logger.error(f"更新推理参数失败: training_id={training_id}, data_offset={data_offset}, error={e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    def get_prediction_status_summary(cls, plan_id: int) -> Dict[str, any]:
        """
        获取计划预测状态摘要

        Args:
            plan_id: 计划ID

        Returns:
            预测状态摘要
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'error': '计划不存在'}

                # 获取最新预测数据
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id
                ).order_by(PredictionData.timestamp.desc()).first()

                # 获取预测批次统计
                batch_count = db.query(PredictionData.inference_batch_id).filter(
                    PredictionData.plan_id == plan_id
                ).distinct().count()

                current_time = datetime.now(cls.BEIJING_TZ)

                result = {
                    'plan_id': plan_id,
                    'plan_name': plan.plan_name,
                    'inst_id': plan.inst_id,
                    'interval': plan.interval,
                    'auto_inference_enabled': plan.auto_inference_enabled,
                    'auto_inference_interval_hours': plan.auto_inference_interval_hours or 4,
                    'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'latest_prediction_time': None,
                    'time_diff_hours': None,
                    'total_batches': batch_count,
                    'needs_inference': False
                }

                if latest_prediction:
                    latest_time = latest_prediction.timestamp
                    if hasattr(latest_time, 'astimezone'):
                        latest_time = latest_time.astimezone(cls.BEIJING_TZ)
                    else:
                        latest_time = cls.BEIJING_TZ.localize(latest_time.replace(tzinfo=None))

                    time_diff = current_time - latest_time
                    time_diff_hours = time_diff.total_seconds() / 3600

                    result['latest_prediction_time'] = latest_time.strftime('%Y-%m-%d %H:%M:%S')
                    result['time_diff_hours'] = round(time_diff_hours, 2)

                    # 判断是否需要推理
                    if plan.auto_inference_enabled:
                        target_interval = plan.auto_inference_interval_hours or 4
                        result['needs_inference'] = time_diff_hours >= target_interval

                return result

        except Exception as e:
            logger.error(f"获取预测状态摘要失败: plan_id={plan_id}, error={e}")
            return {'error': str(e)}


# 全局实例
inference_data_offset_service = InferenceDataOffsetService()