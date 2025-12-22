#!/usr/bin/env python3
"""
获取最新批次预测均值数据测试脚本

该脚本用于开发和测试多批次预测数据的均值计算算法，
包括重叠度计算和时间范围分析。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import PredictionData, TrainingRecord, KlineData, TradingPlan
from database.db import SessionLocal
from sqlalchemy import and_, func
from datetime import datetime
from collections import defaultdict
import numpy as np

class PredictionAnalyzer:
    """预测数据分析器"""

    def __init__(self):
        self.db = SessionLocal()

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()

    def get_latest_training_record(self, plan_id=3):
        """获取最新的已完成训练记录"""
        training_record = self.db.query(TrainingRecord).filter(
            TrainingRecord.plan_id == plan_id,
            TrainingRecord.status == 'completed'
        ).order_by(TrainingRecord.created_at.desc()).first()
        return training_record

    def get_latest_kline_time(self, plan_id=3):
        """获取当前最新K线时间"""
        plan = self.db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
        if not plan:
            return None

        latest_kline = self.db.query(KlineData).filter(
            KlineData.inst_id == plan.inst_id,
            KlineData.interval == plan.interval
        ).order_by(KlineData.timestamp.desc()).first()

        return latest_kline.timestamp if latest_kline else None

    def get_prediction_data(self, training_id, latest_kline_time=None):
        """获取训练记录的所有预测数据"""
        query = self.db.query(PredictionData).filter(
            PredictionData.training_record_id == training_id
        )

        # 只获取未来预测数据（晚于最新K线时间的数据）
        if latest_kline_time:
            query = query.filter(PredictionData.timestamp > latest_kline_time)

        predictions = query.order_by(PredictionData.inference_batch_id, PredictionData.timestamp).all()
        return predictions

    def analyze_batch_overlap(self, training_id, latest_kline_time=None):
        """分析多批次预测数据的重叠情况"""
        predictions = self.get_prediction_data(training_id, latest_kline_time)

        if not predictions:
            print("没有找到预测数据")
            return None

        # 按批次组织数据
        batch_data = defaultdict(list)
        for pred in predictions:
            batch_data[pred.inference_batch_id].append(pred)

        print(f"找到 {len(batch_data)} 个批次的预测数据")
        for batch_id, batch_preds in batch_data.items():
            time_range = f"{batch_preds[0].timestamp} ~ {batch_preds[-1].timestamp}"
            print(f"  批次 {batch_id[-8:]}: {len(batch_preds)} 条数据, 时间范围: {time_range}")

        # 分析时间点的重叠情况
        time_point_batches = defaultdict(list)
        for pred in predictions:
            time_point_batches[pred.timestamp].append(pred)

        print(f"\n时间点重叠分析:")
        overlapping_points = {t: preds for t, preds in time_point_batches.items() if len(preds) > 1}

        if overlapping_points:
            print(f"找到 {len(overlapping_points)} 个有多个批次预测的时间点")
            for timestamp, preds in list(overlapping_points.items())[:5]:  # 显示前5个示例
                batch_ids = [p.inference_batch_id[-8:] for p in preds]
                print(f"  {timestamp}: {len(preds)} 个批次 {batch_ids}")
        else:
            print("没有找到多批次重叠的时间点")

        return {
            'batch_data': dict(batch_data),
            'time_point_batches': dict(time_point_batches),
            'overlapping_points': overlapping_points
        }

    def calculate_comprehensive_metrics(self, training_id, latest_kline_time=None):
        """计算综合预测指标"""
        predictions = self.get_prediction_data(training_id, latest_kline_time)

        if not predictions:
            return None

        # 按时间点分组计算统计数据
        time_stats = {}
        for pred in predictions:
            t = pred.timestamp
            if t not in time_stats:
                time_stats[t] = {
                    'highs': [], 'lows': [], 'closes': [], 'volumes': [],
                    'upward_probs': [], 'volatility_probs': [],
                    'close_stds': []
                }

            time_stats[t]['highs'].append(pred.high)
            time_stats[t]['lows'].append(pred.low)
            time_stats[t]['closes'].append(pred.close)
            time_stats[t]['volumes'].append(pred.volume or 0)
            time_stats[t]['upward_probs'].append(pred.upward_probability or 0)
            time_stats[t]['volatility_probs'].append(pred.volatility_amplification_probability or 0)
            time_stats[t]['close_stds'].append(pred.close_std or 0)

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
                'close_std': np.mean(stats['close_stds']),
                'volume_mean': np.mean(stats['volumes']),
                'upward_prob_mean': np.mean(stats['upward_probs']),
                'volatility_prob_mean': np.mean(stats['volatility_probs']),
                'price_range_high': np.max(stats['highs']),
                'price_range_low': np.min(stats['lows']),
                'consensus_score': 1 - (np.std(stats['closes']) / np.mean(stats['closes'])) if np.mean(stats['closes']) > 0 else 0
            }
            results.append(result)

        return results

    def find_extreme_predictions(self, stats_results):
        """找到极值预测和时间范围"""
        if not stats_results:
            return None

        # 找到最高点和最低点
        max_high_point = max(stats_results, key=lambda x: x['high_max'])
        min_low_point = min(stats_results, key=lambda x: x['low_min'])

        # 找到最一致的时间点（共识度最高）
        highest_consensus = max(stats_results, key=lambda x: x['consensus_score'])

        # 计算整体统计
        all_highs = [r['high_mean'] for r in stats_results]
        all_lows = [r['low_mean'] for r in stats_results]
        all_closes = [r['close_mean'] for r in stats_results]

        return {
            'highest_price': {
                'value': max_high_point['high_max'],
                'time': max_high_point['timestamp'],
                'mean_high': max_high_point['high_mean'],
                'sample_count': max_high_point['sample_count']
            },
            'lowest_price': {
                'value': min_low_point['low_min'],
                'time': min_low_point['timestamp'],
                'mean_low': min_low_point['low_mean'],
                'sample_count': min_low_point['sample_count']
            },
            'highest_consensus': {
                'time': highest_consensus['timestamp'],
                'consensus_score': highest_consensus['consensus_score'],
                'close_mean': highest_consensus['close_mean'],
                'sample_count': highest_consensus['sample_count']
            },
            'overall_stats': {
                'high_mean': np.mean(all_highs),
                'low_mean': np.mean(all_lows),
                'close_mean': np.mean(all_closes),
                'high_volatility': np.std(all_highs) / np.mean(all_highs) if np.mean(all_highs) > 0 else 0,
                'low_volatility': np.std(all_lows) / np.mean(all_lows) if np.mean(all_lows) > 0 else 0,
                'prediction_range': max(all_highs) - min(all_lows)
            }
        }

def test_prediction_analysis():
    """测试预测数据分析"""
    analyzer = PredictionAnalyzer()

    # 获取最新训练记录（可以通过参数指定）
    training_id = 79  # 可以修改为其他训练ID进行测试

    print(f"=== 分析训练记录 {training_id} 的预测数据 ===\n")

    # 获取最新K线时间
    latest_kline_time = analyzer.get_latest_kline_time()
    print(f"当前最新K线时间: {latest_kline_time}\n")

    # 分析批次重叠情况
    print("1. 批次重叠分析:")
    overlap_analysis = analyzer.analyze_batch_overlap(training_id, latest_kline_time)

    # 计算综合指标
    print("\n2. 综合预测指标计算:")
    stats_results = analyzer.calculate_comprehensive_metrics(training_id, latest_kline_time)

    if stats_results:
        print(f"找到 {len(stats_results)} 个未来时间点的预测数据")
        print("\n前3个时间点的预测统计:")
        for result in stats_results[:3]:
            print(f"  {result['timestamp']}: 样本数={result['sample_count']}, "
                  f"最高价均值={result['high_mean']:.2f}, 最低价均值={result['low_mean']:.2f}, "
                  f"收盘价均值={result['close_mean']:.2f}, 共识度={result['consensus_score']:.3f}")

    # 找到极值预测
    print("\n3. 极值预测分析:")
    extremes = analyzer.find_extreme_predictions(stats_results)

    if extremes:
        print(f"最高价预测:")
        print(f"  价格: {extremes['highest_price']['value']:.2f}")
        print(f"  时间: {extremes['highest_price']['time']}")
        print(f"  均值: {extremes['highest_price']['mean_high']:.2f}")
        print(f"  样本数: {extremes['highest_price']['sample_count']}")

        print(f"\n最低价预测:")
        print(f"  价格: {extremes['lowest_price']['value']:.2f}")
        print(f"  时间: {extremes['lowest_price']['time']}")
        print(f"  均值: {extremes['lowest_price']['mean_low']:.2f}")
        print(f"  样本数: {extremes['lowest_price']['sample_count']}")

        print(f"\n最高共识度时间点:")
        print(f"  时间: {extremes['highest_consensus']['time']}")
        print(f"  共识度: {extremes['highest_consensus']['consensus_score']:.3f}")
        print(f"  收盘价均值: {extremes['highest_consensus']['close_mean']:.2f}")

        print(f"\n整体统计:")
        overall = extremes['overall_stats']
        print(f"  平均最高价: {overall['high_mean']:.2f}")
        print(f"  平均最低价: {overall['low_mean']:.2f}")
        print(f"  平均收盘价: {overall['close_mean']:.2f}")
        print(f"  最高价波动率: {overall['high_volatility']:.3f}")
        print(f"  最低价波动率: {overall['low_volatility']:.3f}")
        print(f"  预测价格范围: {overall['prediction_range']:.2f}")

    return {
        'overlap_analysis': overlap_analysis,
        'stats_results': stats_results,
        'extremes': extremes
    }

if __name__ == "__main__":
    # 支持命令行参数指定训练ID
    if len(sys.argv) > 1:
        training_id = int(sys.argv[1])
        print(f"使用指定的训练ID: {training_id}")

        analyzer = PredictionAnalyzer()
        latest_kline_time = analyzer.get_latest_kline_time()

        overlap_analysis = analyzer.analyze_batch_overlap(training_id, latest_kline_time)
        stats_results = analyzer.calculate_comprehensive_metrics(training_id, latest_kline_time)
        extremes = analyzer.find_extreme_predictions(stats_results)

        if extremes:
            print("\n=== 极值预测结果汇总 ===")
            print(f"最高价: {extremes['highest_price']['value']:.2f} @ {extremes['highest_price']['time']}")
            print(f"最低价: {extremes['lowest_price']['value']:.2f} @ {extremes['lowest_price']['time']}")
            print(f"价格范围: {extremes['overall_stats']['prediction_range']:.2f}")
    else:
        # 默认测试
        test_prediction_analysis()