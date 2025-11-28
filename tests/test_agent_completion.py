#!/usr/bin/env python3
"""
测试Agent完成后继续发送文本的问题
"""

import asyncio
import time
from services.langchain_agent_v2 import langchain_agent_v2_service

async def test_agent_completion():
    """测试Agent完成后继续发送"""
    plan_id = 2
    print(f"🚀 测试Agent完成后继续发送功能（计划ID: {plan_id}）...")
    print("=" * 60)

    try:
        from services.langchain_agent_v2 import ConversationType

        message_count = 0
        is_completed = False

        # 第一阶段：执行自动推理
        print("📋 第一阶段：执行自动推理")
        async for message_batch in langchain_agent_v2_service.stream_agent_response(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            message_count += 1
            print(f"📨 消息批次 {message_count}")

            # 检查是否完成
            if any("执行完成" in msg.get("content", "") for msg in message_batch):
                is_completed = True
                print("✅ 检测到推理完成")

            # 限制测试时间
            if message_count > 25:
                print("⏰ 第一阶段完成，停止推理测试")
                break

        print(f"\n📊 第一阶段统计: {message_count} 个消息批次")
        print(f"推理状态: {'已完成' if is_completed else '未完成'}")

        # 第二阶段：测试继续对话
        if is_completed:
            print(f"\n📋 第二阶段：测试继续对话")
            conversation_history = []

            # 发送新的用户消息
            new_message = "请总结刚才的分析结果"
            print(f"📝 用户消息: {new_message}")

            try:
                async for message_batch in langchain_agent_v2_service.stream_conversation(
                    plan_id=plan_id,
                    user_message=new_message
                ):
                    print(f"📨 继续对话消息批次:")
                    for msg in message_batch:
                        print(f"   [{msg['role']}]: {msg['content'][:100]}...")

                    # 限制继续对话的测试
                    conversation_history.extend(message_batch)
                    if len(conversation_history) > 5:
                        print("⏰ 继续对话测试完成")
                        break

            except Exception as e:
                print(f"❌ 继续对话测试失败: {e}")

        print(f"\n✅ Agent完成测试完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_completion())