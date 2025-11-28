#!/usr/bin/env python3
"""
测试新的LangChain Agent服务 v2
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from services.langchain_agent_v2 import langchain_agent_v2_service, ConversationType

async def test_langchain_agent_v2():
    print('=== 测试新的LangChain Agent服务 v2 ===')

    plan_id = 2
    user_message = '你好，请介绍一下自己，并查看当前的市场情况。'

    print(f'计划ID: {plan_id}')
    print(f'用户消息: {user_message}')
    print('\n开始流式对话...')

    try:
        message_count = 0
        async for message_batch in langchain_agent_v2_service.stream_agent_response(
            plan_id=plan_id,
            user_message=user_message,
            conversation_type=ConversationType.MANUAL_CHAT
        ):
            message_count += 1
            print(f'\n--- 消息批次 {message_count} ---')

            if message_batch:
                for msg in message_batch:
                    content_preview = msg["content"][:100] if len(msg["content"]) > 100 else msg["content"]
                    print(f'[{msg["role"]}]: {content_preview}{"..." if len(msg["content"]) > 100 else ""}')

            # 限制测试消息数量
            if message_count >= 10:
                break

        print(f'\n✅ 测试完成，共收到 {message_count} 个消息批次')

    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_langchain_agent_v2())