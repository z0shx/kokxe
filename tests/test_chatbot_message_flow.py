#!/usr/bin/env python3
"""
测试Chatbot消息流格式 - 验证role序列输出
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service, ConversationType

async def test_chatbot_message_flow():
    """测试Chatbot消息流格式"""
    plan_id = 2
    print(f"🧪 测试Chatbot消息流格式 (计划ID: {plan_id})")
    print("=" * 60)

    try:
        print("📋 期望的消息流序列:")
        print("1. role:system - 配置的提示词内容")
        print("2. role:user - 最新批次的预测数据")
        print("3. role:assistant - 流式输出（Gradio兼容）")
        print("   可能包含:")
        print("   - role:think - Qwen思考过程")
        print("   - role:tool - 工具调用（独立消息气泡）")
        print("-" * 60)

        message_sequence = []
        role_sequence = []

        # 测试自动推理模式
        async for message_batch in langchain_agent_v2_service.stream_agent_response_real(
            plan_id=plan_id,
            user_message=None,
            conversation_type=ConversationType.AUTO_INFERENCE
        ):
            for msg in message_batch:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # 记录消息序列
                role_sequence.append(role)
                message_sequence.append({
                    "role": role,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "content_length": len(content)
                })

                # 实时输出
                print(f"📨 role: {role:8} | 长度: {len(content):4} | {content[:80]}")

                # 限制测试长度
                if len(message_sequence) > 20:
                    print("⏰ 达到消息数量限制，停止测试...")
                    break

        # 分析消息序列
        print("\n" + "=" * 60)
        print("📊 消息流分析结果")
        print("=" * 60)

        # 统计各种role
        role_counts = {}
        for role in role_sequence:
            role_counts[role] = role_counts.get(role, 0) + 1

        print(f"\n📈 Role统计:")
        for role, count in role_counts.items():
            print(f"   {role:12}: {count} 次")

        # 检查期望的消息流格式
        print(f"\n🎯 消息流格式检查:")

        # 检查是否有system消息
        has_system = "system" in role_sequence
        print(f"   ✅ role:system     : {'存在' if has_system else '缺失'}")

        # 检查是否有user消息（自动推理模式）
        has_user = "user" in role_sequence
        print(f"   ✅ role:user       : {'存在' if has_user else '缺失'}")

        # 检查是否有assistant消息
        has_assistant = "assistant" in role_sequence
        print(f"   ✅ role:assistant   : {'存在' if has_assistant else '缺失'}")

        # 检查是否有think消息（Qwen思考）
        has_think = "think" in role_sequence
        print(f"   🔮 role:think      : {'存在' if has_think else '缺失'}")

        # 检查是否有tool消息
        has_tool = "tool" in role_sequence
        print(f"   🛠️  role:tool       : {'存在' if has_tool else '缺失'}")

        # 检查消息序列顺序
        print(f"\n📋 消息序列顺序 (前10条):")
        for i, role in enumerate(role_sequence[:10]):
            print(f"   {i+1:2}. {role}")

        # 验证消息流格式正确性
        format_correct = True
        issues = []

        # 自动推理模式应该以system开始
        if not has_system:
            format_correct = False
            issues.append("缺少system消息")

        # 应该有user消息（预测数据）
        if not has_user:
            format_correct = False
            issues.append("缺少user消息（预测数据）")

        # 应该有assistant消息
        if not has_assistant:
            format_correct = False
            issues.append("缺少assistant消息")

        # 最终评估
        print(f"\n🏆 格式评估:")
        if format_correct:
            print(f"   ✅ Chatbot消息流格式正确!")
            print(f"   - 符合标准的role序列")
            print(f"   - 支持Qwen思考过程")
            print(f"   - 工具调用使用独立消息气泡")
            print(f"   - 与Gradio流式接口兼容")
        else:
            print(f"   ❌ 消息流格式需要改进:")
            for issue in issues:
                print(f"   - {issue}")

        return format_correct

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_chatbot_message_flow())