#!/usr/bin/env python3
"""
测试Agent输出格式 - 包含Qwen分析和工具调用的独立显示
"""

import asyncio
from services.langchain_agent_v2 import langchain_agent_v2_service

async def test_agent_format():
    """测试Agent输出格式"""
    plan_id = 2
    print(f"🚀 测试Agent输出格式（计划ID: {plan_id}）...")
    print("=" * 80)

    try:
        from services.langchain_agent_v2 import ConversationType

        message_count = 0
        qwen_analysis_count = 0
        qwen_output_count = 0
        tool_call_count = 0
        tool_result_count = 0

        async for message_batch in langchain_agent_v2_service.stream_agent_response(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            message_count += 1

            print(f"📨 消息批次 {message_count}:")

            for i, msg in enumerate(message_batch):
                role = msg['role']
                content = msg['content']

                # 检查特殊标记
                if "<QWEN_ANALYSIS_START>" in content:
                    qwen_analysis_count += 1
                    print(f"   {i+1}. [QWEN_ANALYSIS] 🤖 Qwen分析开始")
                elif "<QWEN_OUTPUT>" in content:
                    qwen_output_count += 1
                    output_content = content.replace("<QWEN_OUTPUT>", "").replace("</QWEN_OUTPUT>", "").strip()
                    print(f"   {i+1}. [QWEN_OUTPUT] 📝 {output_content[:50]}...")
                elif "<TOOL_CALL>" in content:
                    tool_call_count += 1
                    call_content = content.replace("<TOOL_CALL>", "").replace("</TOOL_CALL>", "").strip()
                    print(f"   {i+1}. [TOOL_CALL] 🛠️ {call_content[:60]}...")
                elif "<TOOL_RESULT>" in content:
                    tool_result_count += 1
                    result_content = content.replace("<TOOL_RESULT>", "").replace("</TOOL_RESULT>", "").strip()
                    print(f"   {i+1}. [TOOL_RESULT] ✅ {result_content[:60]}...")
                elif role == "assistant":
                    print(f"   {i+1}. [ASSISTANT] {content[:60]}...")
                elif role == "user":
                    print(f"   {i+1}. [USER] {content[:60]}...")
                else:
                    print(f"   {i+1}. [{role}] {content[:60]}...")

            print("-" * 40)

            # 限制测试时间
            if message_count > 20:
                print("⏰ 达到测试限制，停止测试")
                break

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n📊 测试统计:")
    print(f"  总消息批次: {message_count}")
    print(f"  Qwen分析开始: {qwen_analysis_count}")
    print(f"  Qwen输出块: {qwen_output_count}")
    print(f"  工具调用: {tool_call_count}")
    print(f"  工具结果: {tool_result_count}")

if __name__ == "__main__":
    asyncio.run(test_agent_format())