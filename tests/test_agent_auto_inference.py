#!/usr/bin/env python3
"""
测试新的LangChain Agent服务 v2 - 自动推理模式
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service, ConversationType

async def test_agent_auto_inference():
    print('=== 测试LangChain Agent v2 - 自动推理模式 ===')

    plan_id = 2  # 自动推理模式不需要用户消息

    print(f'计划ID: {plan_id}')
    print('\n开始自动推理流程...')
    print('(应该显示：系统提示 -> 预测数据 -> 思考过程 -> 工具调用 -> AI分析)\n')

    try:
        message_count = 0
        async for message_batch in langchain_agent_v2_service.stream_agent_response_real(
            plan_id=plan_id,
            user_message=None,  # 自动推理不需要用户消息
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            message_count += 1
            print(f'--- 消息批次 {message_count} ---')

            if message_batch:
                for msg in message_batch:
                    role = msg["role"]
                    content = msg["content"]

                    # 截断长内容显示
                    if len(content) > 150:
                        content_preview = content[:150] + "..."
                    else:
                        content_preview = content

                    print(f'[{role}]: {content_preview}')

            # 限制测试消息数量
            if message_count >= 15:
                break

        print(f'\n✅ 自动推理测试完成，共收到 {message_count} 个消息批次')

    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_agent_auto_inference())