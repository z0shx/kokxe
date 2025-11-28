#!/usr/bin/env python3
"""
测试实时流式输出 - 检查是否真的流式显示
"""

import asyncio
import time
from services.langchain_agent_v2 import langchain_agent_v2_service


async def test_real_time_stream():
    """测试实时流式输出"""
    plan_id = 2
    print(f"🚀 开始测试实时流式输出（计划ID: {plan_id}）...")
    print("⏰ 每条消息都会显示时间戳，以便观察流式效果")
    print("=" * 60)

    message_count = 0
    start_time = time.time()

    try:
        # 使用正确的枚举类型
        from services.langchain_agent_v2 import ConversationType

        async for message_batch in langchain_agent_v2_service.stream_agent_response(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            current_time = time.time()
            elapsed = current_time - start_time
            message_count += 1

            print(f"📨 [{elapsed:.1f}s] 消息批次 {message_count} ({len(message_batch)} 条消息):")

            for i, msg in enumerate(message_batch):
                role = msg['role']
                content = msg['content']
                content_preview = content[:100] + "..." if len(content) > 100 else content

                print(f"   {i+1}. [{role}]: {content_preview}")

            print("-" * 40)

            # 模拟流式显示的延迟
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"❌ 流式输出错误: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - start_time
    print(f"✅ 测试完成！")
    print(f"📊 统计: {message_count} 个消息批次, 总耗时 {total_time:.1f} 秒")
    print(f"⚡ 平均每批次间隔: {total_time/max(message_count-1, 1):.2f} 秒")


if __name__ == "__main__":
    asyncio.run(test_real_time_stream())