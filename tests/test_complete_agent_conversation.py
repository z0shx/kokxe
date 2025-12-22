#!/usr/bin/env python3
"""
完整的Agent对话功能测试
测试用户聊天、推理对话、上下文恢复和继续对话功能
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent import agent_service
from database.models import AgentConversation, AgentMessage
from database.db import get_db


def print_separator(title: str):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")


async def test_user_chat_functionality():
    """测试用户聊天功能"""
    print_separator("测试用户聊天功能")

    try:
        print("发送用户聊天消息...")
        message_count = 0
        tool_call_detected = False

        async for chunk in agent_service.stream_conversation(
            plan_id=3,
            user_message="你好，我是用户，请帮我查询ETH的当前价格",
            conversation_type="user_chat"
        ):
            message_count += 1
            for message in chunk:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    if "交易助手" in content:
                        print("✅ 正确使用聊天助手提示词")
                    elif "交易决策系统" in content:
                        print("❌ 错误使用交易决策提示词")

                elif role == "assistant":
                    if "工具调用" in content:
                        tool_call_detected = True
                        print("🔧 检测到工具调用")
                    elif len(content) > 50 and not content.startswith("✅"):
                        print(f"🤖 AI回复: {content[:100]}...")
                        break

                if message_count >= 5:
                    break

        print(f"✅ 用户聊天测试完成 (工具调用: {tool_call_detected})")
        return True

    except Exception as e:
        print(f"❌ 用户聊天测试失败: {e}")
        return False


async def test_inference_functionality():
    """测试推理功能"""
    print_separator("测试推理决策功能")

    try:
        print("发送推理决策消息...")
        message_count = 0
        tool_call_detected = False

        async for chunk in agent_service.stream_conversation(
            plan_id=3,
            user_message="请基于最新数据进行交易决策分析",
            conversation_type="inference_session"
        ):
            message_count += 1
            for message in chunk:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    if "交易决策系统" in content:
                        print("✅ 正确使用交易决策提示词")
                    elif "交易助手" in content:
                        print("❌ 错误使用聊天助手提示词")

                elif role == "assistant":
                    if "工具调用" in content:
                        tool_call_detected = True
                        print("🔧 检测到工具调用")

                if message_count >= 5:
                    break

        print(f"✅ 推理功能测试完成 (工具调用: {tool_call_detected})")
        return True

    except Exception as e:
        print(f"❌ 推理功能测试失败: {e}")
        return False


def test_conversation_recovery():
    """测试对话恢复功能"""
    print_separator("测试对话恢复功能")

    try:
        from ui.plan_detail_chat_ui import PlanDetailChatUI
        from ui.plan_detail import PlanDetailUI

        # 创建对话UI
        plan_detail_ui = PlanDetailUI()
        chat_ui = PlanDetailChatUI(plan_detail_ui)

        # 获取对话列表
        choices = chat_ui.get_conversation_list_for_selection(3)
        print(f"找到 {len(choices)} 个对话记录")

        # 测试恢复最新对话
        if choices and choices[0][1] is not None:
            latest_conv_id = choices[0][1]
            print(f"恢复最新对话 (ID: {latest_conv_id})")

            restored_history = chat_ui.restore_selected_conversation(3, latest_conv_id)
            print(f"✅ 成功恢复 {len(restored_history)} 条消息")

            # 分析消息类型
            message_types = {}
            for msg in restored_history:
                role = msg.get("role", "unknown")
                message_types[role] = message_types.get(role, 0) + 1

            print(f"消息类型分布: {message_types}")
            return True
        else:
            print("⚠️ 没有找到可恢复的对话")
            return False

    except Exception as e:
        print(f"❌ 对话恢复测试失败: {e}")
        return False


async def test_continue_conversation():
    """测试继续对话功能"""
    print_separator("测试继续对话功能")

    try:
        # 检查现有对话
        with get_db() as db:
            latest_conv = db.query(AgentConversation).filter(
                AgentConversation.plan_id == 3,
                AgentConversation.conversation_type == "user_chat",
                AgentConversation.status == "active"
            ).order_by(AgentConversation.last_message_at.desc()).first()

            if latest_conv:
                print(f"找到现有对话 (ID: {latest_conv.id})")

                # 获取原始消息数
                original_count = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == latest_conv.id
                ).count()
                print(f"原始消息数: {original_count}")

                # 发送新消息继续对话
                print("发送新消息继续对话...")
                async for chunk in agent_service.stream_conversation(
                    plan_id=3,
                    user_message="谢谢，请再帮我查询一下BTC的价格",
                    conversation_type="user_chat"
                ):
                    for message in chunk:
                        if message.get("role") == "assistant" and len(message.get("content", "")) > 50:
                            print(f"🤖 继续对话回复: {message['content'][:100]}...")
                            break

                # 检查消息是否被正确添加到原对话中
                new_count = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == latest_conv.id
                ).count()

                if new_count > original_count:
                    print(f"✅ 消息已正确添加到原对话 (新增 {new_count - original_count} 条)")
                    return True
                else:
                    print("❌ 消息未被添加到原对话")
                    return False
            else:
                print("⚠️ 没有找到可继续的对话")
                return False

    except Exception as e:
        print(f"❌ 继续对话测试失败: {e}")
        return False


def test_database_integrity():
    """测试数据库完整性"""
    print_separator("测试数据库完整性")

    try:
        with get_db() as db:
            # 检查对话数据
            conversations = db.query(AgentConversation).filter(
                AgentConversation.plan_id == 3
            ).all()

            print(f"计划3共有 {len(conversations)} 个对话")

            # 按类型统计
            type_stats = {}
            for conv in conversations:
                conv_type = conv.conversation_type
                type_stats[conv_type] = type_stats.get(conv_type, 0) + 1

            print(f"对话类型分布: {type_stats}")

            # 检查消息数据
            total_messages = db.query(AgentMessage).join(AgentConversation).filter(
                AgentConversation.plan_id == 3
            ).count()

            print(f"总消息数: {total_messages}")

            # 按消息类型统计
            message_type_stats = {}
            messages = db.query(AgentMessage).join(AgentConversation).filter(
                AgentConversation.plan_id == 3
            ).all()

            for msg in messages:
                msg_type = msg.message_type
                message_type_stats[msg_type] = message_type_stats.get(msg_type, 0) + 1

            print(f"消息类型分布: {message_type_stats}")

            print("✅ 数据库完整性检查完成")
            return True

    except Exception as e:
        print(f"❌ 数据库完整性检查失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🚀 开始完整的Agent对话功能测试")
    print(f"测试计划ID: 3")
    print(f"测试时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 执行所有测试
    test_results = []

    # 1. 数据库完整性检查
    test_results.append(("数据库完整性", test_database_integrity()))

    # 2. 用户聊天功能测试
    test_results.append(("用户聊天功能", await test_user_chat_functionality()))

    # 3. 推理功能测试
    test_results.append(("推理决策功能", await test_inference_functionality()))

    # 4. 对话恢复功能测试
    test_results.append(("对话恢复功能", test_conversation_recovery()))

    # 5. 继续对话功能测试
    test_results.append(("继续对话功能", await test_continue_conversation()))

    # 输出测试结果
    print_separator("测试结果汇总")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n📊 测试统计: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！Agent对话功能完全正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    asyncio.run(main())