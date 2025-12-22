#!/usr/bin/env python3
"""
测试预测分析工具的简单脚本

该脚本测试 get_latest_prediction_analysis 工具函数，
验证无参数调用的功能。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trading_tools import get_latest_prediction_analysis
import json

def test_tool_no_params():
    """测试工具的无参数调用"""
    print("🧪 测试预测分析工具（无参数调用）")
    print("=" * 50)

    # 不传递参数，使用默认的 plan_id=3
    result = get_latest_prediction_analysis()

    if result['success']:
        print("✅ 工具调用成功！")
        print()
        print("📋 基本信息:")
        print(f"  训练记录版本: {result['training_version']}")
        print(f"  训练记录ID: {result['training_id']}")
        print(f"  计划ID: {result['plan_id']}")
        print(f"  预测数据点数: {result['data_points_count']}")
        print(f"  预测时间点数: {result['time_points_count']}")
        print()
        print("🎯 极值信息:")
        extremes = result['extremes']
        print(f"  最高价: {extremes['highest_price']['value']:.2f} @ {extremes['highest_price']['time']}")
        print(f"  最低价: {extremes['lowest_price']['value']:.2f} @ {extremes['lowest_price']['time']}")
        print(f"  预测范围: {extremes['overall_stats']['prediction_range']:.2f}")
        print(f"  预测时间跨度: {extremes['time_range']['duration_hours']:.1f} 小时")
        print()
        print("💬 返回消息:")
        print(result['message'])

        # 测试返回结构的完整性
        expected_keys = ['success', 'training_id', 'training_version', 'plan_id',
                        'data_points_count', 'time_points_count', 'extremes',
                        'analysis_summary', 'raw_data', 'message']

        missing_keys = [key for key in expected_keys if key not in result]
        if missing_keys:
            print(f"\n⚠️  缺失的返回字段: {missing_keys}")
        else:
            print(f"\n✅ 返回结构完整，包含所有预期字段")

        return True
    else:
        print(f"❌ 工具调用失败: {result['message']}")
        return False

def test_tool_with_params():
    """测试工具的带参数调用"""
    print("\n🧪 测试预测分析工具（带参数调用）")
    print("=" * 50)

    # 明确传递 plan_id=3
    result = get_latest_prediction_analysis(plan_id=3)

    if result['success']:
        print("✅ 带参数调用成功！")
        print(f"返回结果与无参数调用一致: {result['plan_id'] == 3}")
        return True
    else:
        print(f"❌ 带参数调用失败: {result['message']}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试预测分析工具")
    print()

    success_count = 0
    total_tests = 2

    # 测试1：无参数调用
    if test_tool_no_params():
        success_count += 1

    # 测试2：带参数调用
    if test_tool_with_params():
        success_count += 1

    print(f"\n{'='*50}")
    print(f"📊 测试结果: {success_count}/{total_tests} 通过")

    if success_count == total_tests:
        print("🎉 所有测试通过！预测分析工具工作正常。")
    else:
        print("⚠️  部分测试失败，请检查工具实现。")

if __name__ == "__main__":
    main()