#!/usr/bin/env python3
"""
测试计划ID 2的Agent真实工具调用情况
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service

async def test_real_agent_tools():
    """测试Agent的真实工具调用"""
    plan_id = 2
    print(f"🚀 测试计划 {plan_id} 的Agent真实工具调用")
    print("=" * 60)

    try:
        # 获取计划信息
        from database.db import get_db
        from database.models import TradingPlan, LLMConfig

        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
            if not plan:
                print("❌ 未找到计划")
                return

            llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()

            print(f"📊 计划信息:")
            print(f"   名称: {plan.plan_name}")
            print(f"   交易对: {plan.inst_id}")
            print(f"   LLM: {llm_config.provider} - {llm_config.model_name}")

            # 检查工具配置
            tools_config = plan.agent_tools_config if isinstance(plan.agent_tools_config, dict) else json.loads(plan.agent_tools_config) if plan.agent_tools_config else {}
            enabled_tools = [name for name, enabled in tools_config.items() if enabled]
            print(f"   启用工具: {len(enabled_tools)} 个 - {enabled_tools[:5]}...")

        print(f"\n🧪 测试工具调用...")
        print("-" * 40)

        message_count = 0
        tool_call_detected = False

        # 使用明确的指令来触发工具调用
        test_message = "请调用get_account_balance和get_positions工具查询账户信息，然后基于这些信息给出交易建议。"

        async for message_batch in langchain_agent_v2_service.stream_conversation(
            plan_id=plan_id,
            user_message=test_message
        ):
            message_count += 1

            for msg in message_batch:
                content = msg.get("content", "")

                # 检测工具调用相关内容
                if any(keyword in content for keyword in ["调用工具", "tool", "执行完成", "余额", "持仓"]):
                    tool_call_detected = True

                # 显示关键信息
                if message_count <= 10:  # 只显示前10条消息避免刷屏
                    print(f"📨 [{message_count}] {msg['role']}: {content[:100]}...")

                    # 如果检测到工具调用，显示详细信息
                    if "调用工具" in content or "余额" in content or "持仓" in content:
                        print(f"🛠️  检测到工具相关内容!")

            # 限制测试消息数量
            if message_count > 15:
                print(f"⏰ 已接收 {message_count} 条消息，停止测试...")
                break

        print(f"\n📊 测试结果:")
        print(f"   总消息数: {message_count}")
        print(f"   检测到工具调用: {'✅ 是' if tool_call_detected else '❌ 否'}")

        if not tool_call_detected:
            print(f"\n⚠️  Agent可能没有正确调用工具!")
            print(f"   当前逻辑是基于文本解析的模拟工具调用")
            print(f"   建议使用真正的LangChain Agent框架")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_agent_tools())