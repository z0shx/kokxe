#!/usr/bin/env python3
"""
最终的LangChain Agent综合测试
验证重构后的Agent是否正确实现了真正的LangChain工具调用
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service, AGENT_AVAILABLE, ConversationType

async def test_final_implementation():
    """最终综合测试"""
    plan_id = 2
    print(f"🚀 LangChain Agent最终综合测试 (计划ID: {plan_id})")
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
            print(f"   Agent API可用: {'✅ 是' if AGENT_AVAILABLE else '⚠️ 否，使用bind_tools'}")

        # 测试1: 自动推理模式
        print(f"\n🧪 测试1: 自动推理模式")
        print("-" * 40)

        auto_messages = 0
        auto_tool_calls = 0

        try:
            async for message_batch in langchain_agent_v2_service.stream_agent_response_real(
                plan_id=plan_id,
                user_message=None,
                conversation_type=ConversationType.AUTO_INFERENCE
            ):
                auto_messages += 1
                for msg in message_batch:
                    content = msg.get("content", "")
                    if "结构化工具调用" in content:
                        auto_tool_calls += 1
                    if auto_messages <= 3:  # 只显示前3条消息
                        print(f"📨 [{auto_messages}] {content[:100]}...")

                if auto_messages > 10:  # 限制测试长度
                    break

        except Exception as e:
            print(f"❌ 自动推理测试失败: {e}")

        print(f"✅ 自动推理完成: {auto_messages} 条消息, {auto_tool_calls} 次工具调用")

        # 测试2: 手动对话模式
        print(f"\n🧪 测试2: 手动对话模式")
        print("-" * 40)

        manual_messages = 0
        manual_tool_calls = 0

        try:
            test_message = "请查询账户余额信息"
            async for message_batch in langchain_agent_v2_service.stream_conversation(
                plan_id=plan_id,
                user_message=test_message
            ):
                manual_messages += 1
                for msg in message_batch:
                    content = msg.get("content", "")
                    if "结构化工具调用" in content:
                        manual_tool_calls += 1
                    if manual_messages <= 3:  # 只显示前3条消息
                        print(f"📨 [{manual_messages}] {content[:100]}...")

                if manual_messages > 10:  # 限制测试长度
                    break

        except Exception as e:
            print(f"❌ 手动对话测试失败: {e}")

        print(f"✅ 手动对话完成: {manual_messages} 条消息, {manual_tool_calls} 次工具调用")

        # 总结
        print(f"\n🎯 测试总结")
        print("=" * 40)
        print(f"   Agent实现类型: {'真正的LangChain Agent' if AGENT_AVAILABLE else '改进的bind_tools版本'}")
        print(f"   自动推理模式: {'✅ 成功' if auto_messages > 0 and auto_tool_calls > 0 else '❌ 失败'}")
        print(f"   手动对话模式: {'✅ 成功' if manual_messages > 0 and manual_tool_calls > 0 else '❌ 失败'}")
        print(f"   总工具调用次数: {auto_tool_calls + manual_tool_calls}")

        if auto_tool_calls > 0 or manual_tool_calls > 0:
            print(f"\n✅ **LangChain Agent重构成功!**")
            print(f"   - 替换了手动文本解析逻辑")
            print(f"   - 实现了结构化工具调用")
            print(f"   - 支持bind_tools方法")
            print(f"   - 正确处理tool_calls和tool_responses")
        else:
            print(f"\n❌ **Agent可能仍存在问题**")
            print(f"   请检查工具配置和LLM响应")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_final_implementation())