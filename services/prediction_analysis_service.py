"""
预测分析服务

提供多批次预测数据的分析和统计功能，包括：
- 获取最新批次预测均值数据
- 计算极值预测和时间范围
- 分析多批次预测的重叠度和共识度
"""

from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from database.models import PredictionData, TrainingRecord, KlineData, TradingPlan
from database.db import SessionLocal, get_db
from sqlalchemy import and_, func
from utils.logger import setup_logger

logger = setup_logger(__name__, "prediction_analysis.log")


class PredictionAnalysisService:
    """预测分析服务类"""

    @staticmethod
    def get_latest_training_record(plan_id: int) -> Optional[TrainingRecord]:
        """获取指定计划的最新已完成训练记录"""
        try:
            with get_db() as db:
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).order_by(TrainingRecord.created_at.desc()).first()
                return training_record
        except Exception as e:
            logger.error(f"获取最新训练记录失败: {e}")
            return None

    @staticmethod
    def get_latest_kline_time(plan_id: int) -> Optional[datetime]:
        """获取指定计划的当前最新K线时间"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return None

                latest_kline = db.query(KlineData).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == plan.interval
                ).order_by(KlineData.timestamp.desc()).first()

                return latest_kline.timestamp if latest_kline else None
        except Exception as e:
            logger.error(f"获取最新K线时间失败: {e}")
            return None

    @staticmethod
    def get_future_prediction_data(training_id: int, plan_id: int) -> List[PredictionData]:
        """获取训练记录中只包含未来时间的预测数据"""
        try:
            with get_db() as db:
                # 获取最新K线时间作为"现在"的基准
                latest_kline_time = PredictionAnalysisService.get_latest_kline_time(plan_id)

                query = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                )

                # 只获取未来预测数据（晚于最新K线时间的数据）
                if latest_kline_time:
                    query = query.filter(PredictionData.timestamp > latest_kline_time)

                predictions = query.order_by(PredictionData.timestamp).all()
                logger.info(f"获取到 {len(predictions)} 条未来预测数据，基准时间: {latest_kline_time}")
                return predictions

        except Exception as e:
            logger.error(f"获取未来预测数据失败: {e}")
            return []

    @staticmethod
    def analyze_batch_predictions(predictions: List[PredictionData]) -> Dict:
        """分析多批次预测数据，计算统计指标"""
        if not predictions:
            return {}

        # 按时间点分组数据
        time_stats = defaultdict(lambda: {
            'highs': [], 'lows': [], 'closes': [], 'volumes': [],
            'upward_probs': [], 'volatility_probs': [], 'close_stds': [],
            'batch_ids': []
        })

        for pred in predictions:
            t = pred.timestamp
            time_stats[t]['highs'].append(pred.high)
            time_stats[t]['lows'].append(pred.low)
            time_stats[t]['closes'].append(pred.close)
            time_stats[t]['volumes'].append(pred.volume or 0)
            time_stats[t]['upward_probs'].append(pred.upward_probability or 0)
            time_stats[t]['volatility_probs'].append(pred.volatility_amplification_probability or 0)
            time_stats[t]['close_stds'].append(pred.close_std or 0)
            time_stats[t]['batch_ids'].append(pred.inference_batch_id)

        # 计算每个时间点的统计指标
        results = []
        for timestamp, stats in sorted(time_stats.items()):
            result = {
                'timestamp': timestamp,
                'sample_count': len(stats['highs']),
                'high_mean': np.mean(stats['highs']),
                'high_max': np.max(stats['highs']),
                'high_min': np.min(stats['highs']),
                'high_std': np.std(stats['highs']),
                'low_mean': np.mean(stats['lows']),
                'low_max': np.max(stats['lows']),
                'low_min': np.min(stats['lows']),
                'low_std': np.std(stats['lows']),
                'close_mean': np.mean(stats['closes']),
                'close_max': np.max(stats['closes']),
                'close_min': np.min(stats['closes']),
                'close_std_mean': np.mean(stats['close_stds']),
                'volume_mean': np.mean(stats['volumes']),
                'upward_prob_mean': np.mean(stats['upward_probs']),
                'volatility_prob_mean': np.mean(stats['volatility_probs']),
                'consensus_score': 1 - (np.std(stats['closes']) / np.mean(stats['closes'])) if np.mean(stats['closes']) > 0 else 0,
                'batch_count': len(set(stats['batch_ids']))  # 不同批次数
            }
            results.append(result)

        return results

    @staticmethod
    def find_extreme_predictions(stats_results: List[Dict]) -> Optional[Dict]:
        """从统计结果中找到极值预测"""
        if not stats_results:
            return None

        # 找到最高点和最低点
        max_high_point = max(stats_results, key=lambda x: x['high_max'])
        min_low_point = min(stats_results, key=lambda x: x['low_min'])

        # 找到最一致的时间点（共识度最高）
        highest_consensus = max(stats_results, key=lambda x: x['consensus_score'])

        # 找到预测范围最大的时间点
        max_range_point = max(stats_results, key=lambda x: x['high_max'] - x['low_min'])

        # 计算整体统计
        all_highs = [r['high_mean'] for r in stats_results]
        all_lows = [r['low_mean'] for r in stats_results]
        all_closes = [r['close_mean'] for r in stats_results]

        # 计算时间范围
        time_range = {
            'start': stats_results[0]['timestamp'],
            'end': stats_results[-1]['timestamp'],
            'duration_hours': (stats_results[-1]['timestamp'] - stats_results[0]['timestamp']).total_seconds() / 3600
        }

        return {
            'highest_price': {
                'value': max_high_point['high_max'],
                'time': max_high_point['timestamp'],
                'mean_price': max_high_point['high_mean'],
                'sample_count': max_high_point['sample_count'],
                'batch_count': max_high_point['batch_count']
            },
            'lowest_price': {
                'value': min_low_point['low_min'],
                'time': min_low_point['timestamp'],
                'mean_price': min_low_point['low_mean'],
                'sample_count': min_low_point['sample_count'],
                'batch_count': min_low_point['batch_count']
            },
            'highest_consensus': {
                'time': highest_consensus['timestamp'],
                'consensus_score': highest_consensus['consensus_score'],
                'price_mean': highest_consensus['close_mean'],
                'sample_count': highest_consensus['sample_count'],
                'batch_count': highest_consensus['batch_count']
            },
            'widest_range': {
                'time': max_range_point['timestamp'],
                'range_size': max_range_point['high_max'] - max_range_point['low_min'],
                'high_max': max_range_point['high_max'],
                'low_min': max_range_point['low_min'],
                'sample_count': max_range_point['sample_count'],
                'batch_count': max_range_point['batch_count']
            },
            'time_range': time_range,
            'overall_stats': {
                'high_mean': np.mean(all_highs),
                'low_mean': np.mean(all_lows),
                'close_mean': np.mean(all_closes),
                'high_volatility': np.std(all_highs) / np.mean(all_highs) if np.mean(all_highs) > 0 else 0,
                'low_volatility': np.std(all_lows) / np.mean(all_lows) if np.mean(all_lows) > 0 else 0,
                'close_volatility': np.std(all_closes) / np.mean(all_closes) if np.mean(all_closes) > 0 else 0,
                'prediction_range': max(all_highs) - min(all_lows),
                'total_time_points': len(stats_results),
                'avg_sample_count': np.mean([r['sample_count'] for r in stats_results]),
                'avg_batch_count': np.mean([r['batch_count'] for r in stats_results])
            }
        }

    @classmethod
    def get_latest_prediction_analysis(cls, plan_id: int = 3) -> Dict:
        """
        获取最新批次预测均值数据的主要接口方法

        Args:
            plan_id: 交易计划ID，默认为3

        Returns:
            Dict: 包含极值预测和统计信息的字典
        """
        try:
            logger.info(f"开始分析计划 {plan_id} 的最新预测数据")

            # 获取最新训练记录
            latest_training = cls.get_latest_training_record(plan_id)
            if not latest_training:
                logger.warning(f"计划 {plan_id} 没有找到已完成的训练记录")
                return {'error': '没有找到已完成的训练记录'}

            # 获取未来预测数据
            predictions = cls.get_future_prediction_data(latest_training.id, plan_id)
            if not predictions:
                logger.warning(f"训练记录 {latest_training.id} 没有找到未来预测数据")
                return {'error': '没有找到未来预测数据'}

            # 分析预测数据
            stats_results = cls.analyze_batch_predictions(predictions)
            if not stats_results:
                logger.warning("预测数据分析结果为空")
                return {'error': '预测数据分析结果为空'}

            # 找到极值
            extremes = cls.find_extreme_predictions(stats_results)

            result = {
                'training_id': latest_training.id,
                'training_version': latest_training.version,
                'plan_id': plan_id,
                'analysis_time': datetime.now(),
                'data_points_count': len(predictions),
                'time_points_count': len(stats_results),
                'extremes': extremes,
                'raw_stats': stats_results[:5] if len(stats_results) > 5 else stats_results  # 保留前5个时间点的详细数据
            }

            logger.info(f"分析完成，找到 {len(stats_results)} 个时间点的预测数据")
            return result

        except Exception as e:
            error_msg = f"获取最新预测分析失败: {e}"
            logger.error(error_msg)
            return {'error': error_msg}

    @classmethod
    def format_analysis_result(cls, analysis_result: Dict) -> str:
        """格式化分析结果为可读文本"""
        if 'error' in analysis_result:
            return f"❌ 分析失败: {analysis_result['error']}"

        extremes = analysis_result['extremes']

        result = [
            f"📈 最新预测分析结果",
            f"",
            f"🔹 训练记录: {analysis_result['training_version']} (ID: {analysis_result['training_id']})",
            f"🔹 数据点数: {analysis_result['data_points_count']}",
            f"🔹 时间点数: {analysis_result['time_points_count']}",
            f"",
            f"🎯 极值预测:",
            f"  ⬆️  最高价: {extremes['highest_price']['value']:.2f}",
            f"     时间: {extremes['highest_price']['time']}",
            f"     均值: {extremes['highest_price']['mean_price']:.2f}",
            f"     样本/批次: {extremes['highest_price']['sample_count']}/{extremes['highest_price']['batch_count']}",
            f"",
            f"  ⬇️  最低价: {extremes['lowest_price']['value']:.2f}",
            f"     时间: {extremes['lowest_price']['time']}",
            f"     均值: {extremes['lowest_price']['mean_price']:.2f}",
            f"     样本/批次: {extremes['lowest_price']['sample_count']}/{extremes['lowest_price']['batch_count']}",
            f"",
            f"🎯 预测范围: {extremes['overall_stats']['prediction_range']:.2f}",
            f"",
            f"🔮 共识度最高的时间点:",
            f"  时间: {extremes['highest_consensus']['time']}",
            f"  共识度: {extremes['highest_consensus']['consensus_score']:.3f}",
            f"  价格: {extremes['highest_consensus']['price_mean']:.2f}",
            f"",
            f"📊 整体统计:",
            f"  平均最高价: {extremes['overall_stats']['high_mean']:.2f}",
            f"  平均最低价: {extremes['overall_stats']['low_mean']:.2f}",
            f"  平均收盘价: {extremes['overall_stats']['close_mean']:.2f}",
            f"  价格波动率: {extremes['overall_stats']['close_volatility']:.3f}",
            f"",
            f"⏱️  预测时间范围:",
            f"  开始: {extremes['time_range']['start']}",
            f"  结束: {extremes['time_range']['end']}",
            f"  持续: {extremes['time_range']['duration_hours']:.1f} 小时"
        ]

        return "\n".join(result)


# 全局服务实例
prediction_analysis_service = PredictionAnalysisService()