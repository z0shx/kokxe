#!/usr/bin/env python3
"""
KOKEX简化工具测试脚本
仅测试已实现的核心工具
"""
import sys
import os
import json
import asyncio
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.agent_decision_service import AgentDecisionService
from database.db import get_db
from database.models import TradingPlan

def load_test_config():
    """加载测试配置"""
    try:
        with open("tools_test_config_simple.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 找不到 tools_test_config_simple.json 文件")
        return None
    except Exception as e:
        print(f"❌ 加载测试配置失败: {e}")
        return None

async def test_single_tool(plan, tool_name, parameters):
    """测试单个工具"""
    print(f"\n🧪 测试工具: {tool_name}")
    print(f"📋 参数: {json.dumps(parameters, ensure_ascii=False, indent=2)}")

    try:
        result = await AgentDecisionService._execute_single_tool_async(plan, tool_name, parameters)

        if result.get('status') == 'success':
            print(f"✅ 工具调用成功")
            print(f"📊 结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
        else:
            print(f"❌ 工具调用失败: {result.get('message')}")
            return result

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return None

async def run_simple_tests():
    """运行简化工具测试"""
    print("🚀 开始执行KOKEX简化工具测试")
    print("=" * 60)

    # 加载测试配置
    test_config = load_test_config()
    if not test_config:
        return False

    # 获取测试计划
    try:
        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == 2).first()
            if not plan:
                print("❌ 找不到测试计划 (ID=2)")
                return False

            print(f"📋 使用测试计划: {plan.plan_name} (ID: {plan.id})")

    except Exception as e:
        print(f"❌ 获取测试计划失败: {e}")
        return False

    # 执行测试
    total_tests = 0
    successful_tests = 0

    # 定义测试顺序：查询工具 -> 交易工具
    test_order = [
        "get_account_balance",
        "get_current_price",
        "get_prediction_history",
        "query_prediction_data",
        "place_limit_order",
        "place_stop_loss_order"
    ]

    for tool_name in test_order:
        if tool_name not in test_config:
            continue

        tool_config = test_config[tool_name]
        print(f"\n{'='*20} {tool_name} {'='*20}")
        print(f"📝 描述: {tool_config['description']}")
        print(f"🏷️  类别: {tool_config['category']} | ⚠️  风险: {tool_config['risk_level']}")

        for i, test_case in enumerate(tool_config['test_cases'], 1):
            print(f"\n📋 测试用例 {i}: {test_case['name']}")
            print(f"📄 说明: {test_case['description']}")

            result = await test_single_tool(plan, tool_name, test_case['parameters'])
            total_tests += 1
            if result and result.get('status') == 'success':
                successful_tests += 1

    # 测试总结
    print(f"\n{'='*60}")
    print(f"📊 测试完成总结:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功数量: {successful_tests}")
    print(f"   成功率: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "   成功率: 0%")

    return True

def main():
    """主函数"""
    print("KOKEX简化工具测试脚本")
    print("仅测试已实现的核心工具")

    # 检查配置文件
    if not os.path.exists("tools_test_config_simple.json"):
        print("❌ 找不到 tools_test_config_simple.json 文件")
        return False

    # 运行测试
    try:
        success = asyncio.run(run_simple_tests())
        return success
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
        return False
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)