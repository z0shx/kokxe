"""
测试LangChain Agent的工具调用功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from services.agent_service import agent_service
from database.models import TradingPlan, LLMConfig
from database.db import get_db


async def test_agent_tools():
    """测试Agent工具调用"""
    print("=" * 60)
    print("测试LangChain Agent工具调用")
    print("=" * 60)

    try:
        # 获取一个可用的计划
        with get_db() as db:
            # 查找有LLM配置的计划
            plan = db.query(TradingPlan).filter(
                TradingPlan.llm_config_id.isnot(None),
                TradingPlan.status == 'running'
            ).first()

            if not plan:
                print("❌ 没有找到可用的计划（需要LLM配置且状态为running）")
                return False

            print(f"✅ 找到计划: {plan.inst_id} (ID: {plan.id})")
            print(f"   状态: {plan.status}")
            print(f"   LLM配置ID: {plan.llm_config_id}")

            # 检查LLM配置
            llm_config = db.query(LLMConfig).filter(
                LLMConfig.id == plan.llm_config_id
            ).first()

            if not llm_config:
                print(f"❌ LLM配置 {plan.llm_config_id} 不存在")
                return False

            print(f"✅ LLM配置: {llm_config.provider} - {llm_config.model_name}")

            # 检查工具配置
            tools_config = plan.agent_tools_config or {}
            enabled_tools = [name for name, enabled in tools_config.items() if enabled]
            print(f"✅ 启用的工具: {enabled_tools if enabled_tools else '无'}")

            if not enabled_tools:
                print("⚠️  没有启用任何工具，请在Agent配置中启用工具")
                return False

        print("\n" + "=" * 60)
        print("开始测试Agent推理...")
        print("=" * 60)

        # 测试推理
        plan_id = plan.id
        response_count = 0

        async for response_chunk in agent_service.stream_manual_inference(plan_id):
            response_count += 1
            print(f"\n--- 响应块 {response_count} ---")

            if isinstance(response_chunk, list):
                print(f"收到 {len(response_chunk)} 条消息")
                for i, msg in enumerate(response_chunk[-3:]):  # 只显示最后3条消息
                    print(f"消息 {i+1}: [{msg.get('role', 'unknown')}] {msg.get('content', '')[:100]}...")

            # 限制测试响应数量
            if response_count >= 5:
                print("\n⏹️  限制测试响应数量，停止测试")
                break

        print(f"\n✅ 测试完成，共收到 {response_count} 个响应块")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_conversation():
    """测试简单对话"""
    print("\n" + "=" * 60)
    print("测试Agent简单对话")
    print("=" * 60)

    try:
        # 获取一个可用的计划
        with get_db() as db:
            plan = db.query(TradingPlan).filter(
                TradingPlan.llm_config_id.isnot(None),
                TradingPlan.status == 'running'
            ).first()

            if not plan:
                print("❌ 没有找到可用的计划")
                return False

        plan_id = plan.id
        test_message = "你好，请简单介绍一下你自己，并告诉我当前的市场情况。"

        print(f"发送消息: {test_message}")

        response_count = 0
        async for response_chunk in agent_service.stream_conversation(plan_id, test_message):
            response_count += 1
            print(f"\n--- 对话响应 {response_count} ---")

            if isinstance(response_chunk, list) and response_chunk:
                last_msg = response_chunk[-1]
                print(f"Agent回复: [{last_msg.get('role')}] {last_msg.get('content', '')[:200]}...")

            if response_count >= 3:  # 限制对话响应
                break

        print(f"\n✅ 对话测试完成，共收到 {response_count} 个响应")
        return True

    except Exception as e:
        print(f"❌ 对话测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("开始LangChain Agent测试...")

    # 测试1: 工具调用推理
    test1_result = await test_agent_tools()

    # 测试2: 简单对话
    test2_result = await test_simple_conversation()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"工具调用测试: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"简单对话测试: {'✅ 通过' if test2_result else '❌ 失败'}")

    if test1_result and test2_result:
        print("\n🎉 所有测试通过！LangChain Agent工作正常。")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查配置和日志。")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)