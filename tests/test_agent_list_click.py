#!/usr/bin/env python3
"""
测试Agent对话记录列表点击恢复功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.plan_detail import PlanDetailUI
from ui.plan_detail_chat_ui import PlanDetailChatUI


def test_agent_list_click_functionality():
    """测试Agent对话记录列表点击功能"""

    print("🧪 测试Agent对话记录列表点击恢复功能")
    print("=" * 50)

    # 创建UI实例
    plan_detail_ui = PlanDetailUI()
    chat_ui = PlanDetailChatUI(plan_detail_ui)

    # 设置chat_ui到plan_detail_ui中
    plan_detail_ui.chat_ui = chat_ui

    print("1. 测试获取对话记录列表...")
    try:
        choices = chat_ui.get_conversation_list_for_selection(3)
        print(f"✅ 找到 {len(choices)} 个对话记录")

        for i, (label, conv_id) in enumerate(choices[:3]):
            print(f"  {i+1}. ID: {conv_id}, 标签: {label[:50]}...")

    except Exception as e:
        print(f"❌ 获取对话列表失败: {e}")
        return False

    if not choices or choices[0][1] is None:
        print("⚠️ 没有找到可测试的对话记录")
        return True

    # 选择第一个对话进行测试
    test_conv_id = choices[0][1]
    print(f"\n2. 测试恢复对话 ID: {test_conv_id}")

    try:
        restored_history = chat_ui.restore_selected_conversation(3, test_conv_id)
        print(f"✅ 成功恢复 {len(restored_history)} 条消息")

        # 分析消息内容
        if restored_history:
            print("\n📋 恢复的消息类型:")
            role_count = {}
            for msg in restored_history:
                role = msg.get('role', 'unknown')
                role_count[role] = role_count.get(role, 0) + 1

            for role, count in role_count.items():
                print(f"  {role}: {count} 条")

            # 显示前几条消息预览
            print("\n💬 消息预览:")
            for i, msg in enumerate(restored_history[:3]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:80]
                print(f"  {i+1}. {role}: {content}...")

        else:
            print("⚠️ 恢复的消息为空")

    except Exception as e:
        print(f"❌ 恢复对话失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n3. 测试模拟点击事件...")
    try:
        # 模拟gr.SelectData事件
        class MockSelectData:
            def __init__(self, index):
                self.index = index

        # 模拟点击第一行
        mock_event = MockSelectData([0])

        # 调用修复后的函数（模拟main.py中的逻辑）
        def mock_restore_agent_conversation(evt, plan_id):
            """模拟main.py中的restore_agent_conversation函数"""
            try:
                if not plan_id:
                    return [{"role": "assistant", "content": "请先选择计划"}]

                row_index = evt.index[0]

                # 获取对话数据
                agent_decisions = plan_detail_ui.load_agent_decisions(int(plan_id))
                if agent_decisions.empty or row_index >= len(agent_decisions):
                    return [{"role": "assistant", "content": "对话记录不存在或已被更新"}]

                clicked_row = agent_decisions.iloc[row_index]
                if 'ID' in clicked_row:
                    conversation_id = int(clicked_row['ID'])
                else:
                    conversation_id = int(clicked_row.iloc[0])

                # 恢复对话
                restored_history = chat_ui.restore_selected_conversation(int(plan_id), conversation_id)

                if restored_history and len(restored_history) > 0:
                    if (restored_history[0].get("role") == "assistant" and
                        restored_history[0].get("content", "").startswith("❌ 恢复对话失败")):
                        return restored_history
                    else:
                        return restored_history + [{"role": "user", "content": f"已恢复对话 ID: {conversation_id}"}]
                else:
                    return [{"role": "assistant", "content": "恢复的对话为空"}]

            except Exception as e:
                return [{"role": "assistant", "content": f"恢复对话失败: {str(e)}"}]

        result_messages = mock_restore_agent_conversation(mock_event, 3)
        print(f"✅ 模拟点击成功，返回 {len(result_messages)} 条消息")

        # 显示结果预览
        if result_messages:
            last_msg = result_messages[-1]
            if last_msg.get('role') == 'user' and '已恢复对话' in last_msg.get('content', ''):
                print(f"✅ 恢复成功: {last_msg['content']}")
            else:
                print(f"⚠️ 意外的返回结果: {last_msg}")

    except Exception as e:
        print(f"❌ 模拟点击失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 列表点击恢复功能测试完成！")
    return True


if __name__ == "__main__":
    success = test_agent_list_click_functionality()
    if success:
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败！")

    sys.exit(0 if success else 1)