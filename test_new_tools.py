#!/usr/bin/env python3
"""
测试新增的工具功能
"""
import sys
import os
import asyncio
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.agent_tools import get_tool, get_all_tools
from services.trading_tools import OKXTradingTools
from database.db import get_db
from database.models import TradingPlan

def test_tool_definitions():
    """测试工具定义"""
    print("🧪 测试工具定义...")

    # 测试获取新工具
    tools = [
        "query_historical_kline_data",
        "get_current_utc_time",
        "run_latest_model_inference",
        "delete_prediction_data_by_batch"
    ]

    for tool_name in tools:
        tool = get_tool(tool_name)
        if tool:
            print(f"✅ 工具 '{tool_name}' 定义成功")
            print(f"   - 描述: {tool.description[:100]}...")
            print(f"   - 分类: {tool.category}")
            print(f"   - 风险级别: {tool.risk_level}")
            print(f"   - 参数数量: {len(tool.parameters)}")
        else:
            print(f"❌ 工具 '{tool_name}' 未找到")

    return True

def test_trading_tools():
    """测试TradingTools类"""
    print("\n🧪 测试TradingTools类...")

    try:
        # 测试实例化（使用测试参数）
        tools = OKXTradingTools(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            is_demo=True,
            trading_limits={"max_order_amount": 1000}
        )

        print("✅ OKXTradingTools实例化成功")

        # 测试获取UTC时间工具
        result = tools.get_current_utc_time()
        if result.get('success'):
            print(f"✅ get_current_utc_time 工具执行成功")
            print(f"   - 时间戳: {result.get('timestamp', 'N/A')}")
            print(f"   - 格式化时间: {result.get('formatted_time', 'N/A')}")
        else:
            print(f"❌ get_current_utc_time 工具执行失败: {result.get('error')}")

        # 测试删除预测数据工具（不实际删除，只测试参数验证）
        result = tools.delete_prediction_data_by_batch(
            batch_id=999,  # 使用不存在的批次ID
            confirm_delete=False  # 不确认删除
        )

        if result.get('error') and "请设置 confirm_delete=true" in result.get('error'):
            print("✅ delete_prediction_data_by_batch 参数验证成功（安全检查正常）")
        else:
            print(f"❌ delete_prediction_data_by_batch 安全检查异常")

    except Exception as e:
        print(f"❌ TradingTools测试失败: {e}")
        return False

    return True

async def test_agent_decision_service():
    """测试Agent决策服务中的新工具"""
    print("\n🧪 测试Agent决策服务...")

    try:
        from services.agent_decision_service import AgentDecisionService

        # 查找一个测试计划
        with get_db() as db:
            plan = db.query(TradingPlan).first()
            if not plan:
                print("⚠️  没有找到交易计划，跳过实际执行测试")
                return True

        plan_id = plan.id
        print(f"✅ 找到测试计划: ID={plan_id}, 状态={plan.status}")

        # 测试get_current_utc_time工具
        print("测试 get_current_utc_time 工具...")
        try:
            result = await AgentDecisionService._execute_single_tool_async(
                plan, "get_current_utc_time", {}
            )
            if result.get('success'):
                print(f"✅ get_current_utc_time 执行成功")
            else:
                print(f"❌ get_current_utc_time 执行失败: {result.get('error')}")
        except Exception as e:
            print(f"❌ get_current_utc_time 执行异常: {e}")

        # 测试删除预测数据工具（安全模式）
        print("测试 delete_prediction_data_by_batch 工具（安全模式）...")
        try:
            result = await AgentDecisionService._execute_single_tool_async(
                plan, "delete_prediction_data_by_batch", {
                    "batch_id": 999,  # 不存在的批次
                    "confirm_delete": False  # 不确认删除
                }
            )
            if not result.get('success') and "请设置 confirm_delete=true" in result.get('result', {}).get('error', ''):
                print("✅ delete_prediction_data_by_batch 安全检查正常")
            else:
                print(f"❌ delete_prediction_data_by_batch 安全检查异常")
        except Exception as e:
            print(f"❌ delete_prediction_data_by_batch 执行异常: {e}")

    except Exception as e:
        print(f"❌ AgentDecisionService测试失败: {e}")
        return False

    return True

def main():
    """主测试函数"""
    print("🚀 开始测试新增工具功能...\n")

    # 测试工具定义
    definition_success = test_tool_definitions()

    # 测试TradingTools类
    trading_tools_success = test_trading_tools()

    # 测试Agent决策服务
    agent_success = asyncio.run(test_agent_decision_service())

    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print(f"   工具定义: {'✅ 通过' if definition_success else '❌ 失败'}")
    print(f"   TradingTools: {'✅ 通过' if trading_tools_success else '❌ 失败'}")
    print(f"   AgentDecisionService: {'✅ 通过' if agent_success else '❌ 失败'}")

    if definition_success and trading_tools_success and agent_success:
        print("\n🎉 所有新增工具测试通过！")
        return True
    else:
        print("\n❌ 部分测试失败，请检查问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)