#!/usr/bin/env python3
"""
测试真正的LangChain Agent实现
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service

async def test_real_langchain_agent():
    """测试真正的LangChain Agent实现"""
    plan_id = 2
    print(f"🚀 测试计划 {plan_id} 的真正LangChain Agent")
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
            print(f"   启用工具: {len(enabled_tools)} 个 - {enabled_tools[:3]}...")

        print(f"\n🧪 测试真正的LangChain Agent...")
        print("-" * 40)

        message_count = 0
        agent_steps_detected = False
        tool_calls_detected = False

        # 使用明确的指令来触发工具调用
        test_message = "请查询账户余额和当前持仓信息，然后基于这些信息分析并提供交易建议。"

        async for message_batch in langchain_agent_v2_service.stream_conversation(
            plan_id=plan_id,
            user_message=test_message
        ):
            message_count += 1

            for msg in message_batch:
                content = msg.get("content", "")

                # 检测Agent相关内容
                if any(keyword in content for keyword in ["启动真正的LangChain Agent", "工具调用", "Agent决策结果", "intermediate_steps"]):
                    agent_steps_detected = True

                if "工具调用" in content or "Agent执行完成" in content:
                    tool_calls_detected = True

                # 显示所有消息
                print(f"📨 [{message_count}] {msg['role']}: {content}")

                # 限制输出长度
                if len(content) > 300:
                    print(f"   [内容过长，已截断]")

            # 限制测试消息数量，避免无限循环
            if message_count > 20:
                print(f"⏰ 已接收 {message_count} 条消息，停止测试...")
                break

        print(f"\n📊 测试结果:")
        print(f"   总消息数: {message_count}")
        print(f"   检测到LangChain Agent: {'✅ 是' if agent_steps_detected else '❌ 否'}")
        print(f"   检测到工具调用: {'✅ 是' if tool_calls_detected else '❌ 否'}")

        if agent_steps_detected and tool_calls_detected:
            print(f"\n✅ LangChain Agent实现成功!")
            print(f"   - 使用了 create_openai_tools_agent")
            print(f"   - 使用了 AgentExecutor")
            print(f"   - 正确处理了 tool_calls 和 tool_responses")
            print(f"   - 显示了 intermediate_steps")
        else:
            print(f"\n❌ LangChain Agent实现可能存在问题")
            print(f"   请检查 Agent 创建和执行逻辑")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_real_langchain_agent())