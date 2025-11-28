#!/usr/bin/env python3
"""
测试智能Data Offset计算功能
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.inference_data_offset_service import inference_data_offset_service

def test_data_offset_calculation():
    """测试数据偏移计算功能"""
    print("🧪 测试智能Data Offset计算功能")
    print("=" * 60)

    # 测试计划ID 2
    plan_id = 2
    print(f"📊 测试计划ID: {plan_id}")

    try:
        # 获取预测状态摘要
        print("\n📈 预测状态摘要:")
        summary = inference_data_offset_service.get_prediction_status_summary(plan_id)

        if 'error' in summary:
            print(f"❌ 获取状态摘要失败: {summary['error']}")
            return False

        print(f"   计划名称: {summary.get('plan_name', 'N/A')}")
        print(f"   交易对: {summary.get('inst_id', 'N/A')}")
        print(f"   K线间隔: {summary.get('interval', 'N/A')}")
        print(f"   自动推理: {summary.get('auto_inference_enabled', False)}")
        print(f"   推理间隔: {summary.get('auto_inference_interval_hours', 4)}小时")
        print(f"   当前时间: {summary.get('current_time', 'N/A')}")

        if summary.get('latest_prediction_time'):
            print(f"   最新预测: {summary.get('latest_prediction_time', 'N/A')}")
            print(f"   时间差: {summary.get('time_diff_hours', 0):.2f}小时")
            print(f"   预测批次: {summary.get('total_batches', 0)}")
            print(f"   需要推理: {'是' if summary.get('needs_inference') else '否'}")
        else:
            print(f"   最新预测: 暂无预测数据")
            print(f"   预测批次: 0")
            print(f"   需要推理: 是")

        # 测试自动触发模式的Data Offset计算
        print("\n🤖 自动触发模式计算:")
        auto_result = inference_data_offset_service.calculate_optimal_data_offset(
            plan_id=plan_id,
            target_interval_hours=4,
            manual_trigger=False
        )

        if auto_result['success']:
            print(f"   ✅ 计算成功")
            print(f"   📊 数据偏移: {auto_result['data_offset']} 条K线")
            print(f"   📝 计算说明: {auto_result['reasoning']}")
            print(f"   ⏰ 实际间隔: {auto_result['actual_interval']:.2f}小时")
        else:
            print(f"   ❌ 计算失败: {auto_result['reasoning']}")

        # 测试手动触发模式的Data Offset计算
        print("\n🖱️  手动触发模式计算:")
        manual_result = inference_data_offset_service.calculate_optimal_data_offset(
            plan_id=plan_id,
            target_interval_hours=4,
            manual_trigger=True
        )

        if manual_result['success']:
            print(f"   ✅ 计算成功")
            print(f"   📊 数据偏移: {manual_result['data_offset']} 条K线")
            print(f"   📝 计算说明: {manual_result['reasoning']}")
            print(f"   ⏰ 实际间隔: {manual_result['actual_interval']:.2f}小时")
        else:
            print(f"   ❌ 计算失败: {manual_result['reasoning']}")

        # 测试不同间隔的计算
        print("\n🔍 不同间隔的计算测试:")
        test_intervals = [2, 4, 6, 8, 12, 24]

        for interval in test_intervals:
            print(f"   ⏰ {interval}小时间隔:")
            interval_result = inference_data_offset_service.calculate_optimal_data_offset(
                plan_id=plan_id,
                target_interval_hours=interval,
                manual_trigger=False
            )

            if interval_result['success']:
                print(f"      ✅ 偏移={interval_result['data_offset']}, "
                      f"实际间隔={interval_result['actual_interval']:.1f}h")
            else:
                print(f"      ❌ {interval_result['reasoning']}")

        # 测试参数更新功能
        print("\n⚙️  推理参数更新测试:")
        try:
            # 获取最新训练记录
            from database.db import get_db
            from database.models import TrainingRecord

            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(TrainingRecord.created_at.desc()).first()

                if latest_training:
                    print(f"   📋 找到训练记录: {latest_training.id}")

                    # 获取当前参数
                    current_offset = 0
                    if latest_training.finetune_params:
                        inference_params = latest_training.finetune_params.get('inference', {})
                        current_offset = inference_params.get('data_offset', 0)

                    print(f"   📊 当前偏移参数: {current_offset}")

                    # 测试更新
                    test_offset = max(0, auto_result.get('data_offset', 0) if 'auto_result' in locals() else 0)
                    update_result = inference_data_offset_service.update_inference_params_with_offset(
                        plan_id=plan_id,
                        training_id=latest_training.id,
                        data_offset=test_offset
                    )

                    if update_result:
                        print(f"   ✅ 参数更新成功: offset={test_offset}")
                    else:
                        print(f"   ❌ 参数更新失败")
                else:
                    print(f"   ⚠️ 未找到完成的训练记录")

        except Exception as e:
            print(f"   ❌ 参数更新测试失败: {e}")

        print("\n" + "=" * 60)
        print("✅ 智能Data Offset计算功能测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 边界情况测试")
    print("-" * 40)

    try:
        # 测试不存在的计划
        print("🔍 测试不存在的计划ID:")
        result = inference_data_offset_service.calculate_optimal_data_offset(
            plan_id=99999,
            target_interval_hours=4,
            manual_trigger=False
        )
        print(f"   结果: {'成功' if result['success'] else '失败'} - {result.get('reasoning', 'N/A')}")

        # 测试不同的K线间隔映射
        print("\n📊 K线间隔转换测试:")
        test_intervals = ['1m', '5m', '15m', '30m', '1H', '4H', '1D']

        for interval in test_intervals:
            hours = inference_data_offset_service._convert_interval_to_hours(interval)
            print(f"   {interval}: {hours}小时")

    except Exception as e:
        print(f"❌ 边界测试失败: {e}")

if __name__ == "__main__":
    print("🚀 开始智能Data Offset计算功能测试\n")

    # 主要测试
    main_test_success = test_data_offset_calculation()

    # 边界测试
    test_edge_cases()

    print(f"\n🎯 测试总结:")
    print(f"   主要功能: {'✅ 通过' if main_test_success else '❌ 失败'}")
    print(f"   边界情况: ✅ 完成")