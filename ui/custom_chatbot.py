"""
自定义 Chatbot 组件，支持新的消息格式
- think 模式内容展示
- tool 调用独立消息气泡
- play 投资结果展示
"""
import gradio as gr
import json
import re
from typing import List, Dict, Any, Tuple


def format_message_for_display(message: Dict[str, Any]) -> Tuple[str, str]:
    """
    格式化消息用于显示

    Args:
        message: 消息字典，包含 role 和 content

    Returns:
        Tuple[avatar_name, formatted_content]: 头像名称和格式化后的内容
    """
    role = message.get("role", "assistant")
    content = message.get("content", "")

    if role == "system":
        # 系统消息 - 显示系统提示词
        return "💻", content

    elif role == "user":
        return "👤", content

    elif role == "think":
        # 专门的思考角色处理，使用大脑图标
        return "🧠", _format_thinking_message(content)

    elif role == "assistant":
        # 检查是否是思考过程（向后兼容）
        if content.startswith("💭 **思考过程**") or content.startswith("🧠 **思考过程**"):
            return "🧠", content
        else:
            return "🤖", content

    elif role == "tool_call":
        # 工具调用
        return "🔧", _format_tool_call_message(content)

    elif role == "tool_result":
        # 工具执行结果
        return "✅", _format_tool_result_message(content)

    elif role == "tool":
        # 兼容旧格式，尝试自动检测
        try:
            tool_data = json.loads(content)
            if tool_data.get("status") == "calling":
                return "🔧", _format_tool_call_message(content)
            else:
                return "✅", _format_tool_result_message(content)
        except:
            return "🔧", f"🔧 **工具消息**:\n{content}"

    elif role == "play":
        return "📊", _format_play_message(content)

    else:
        return "❓", content


def _format_thinking_message(content: str) -> str:
    """格式化思考过程消息"""
    # 如果内容已经有格式，保持不变
    if content.startswith("🧠 **思考过程**") or content.startswith("💭 **思考过程**"):
        return content
    # 否则添加格式化标题
    return f"🧠 **思考过程**:\n{content}"


def _format_tool_call_message(content: str) -> str:
    """格式化工具调用消息"""
    try:
        tool_data = json.loads(content)
        tool_name = tool_data.get("tool_name", "unknown")
        args = tool_data.get("arguments", {})

        args_str = ", ".join([f"{k}=`{v}`" for k, v in args.items()])
        return f"🔧 **调用工具**: `{tool_name}`\n\n**参数**: {args_str}"
    except (json.JSONDecodeError, Exception):
        return f"🔧 **工具调用**: {content}"


def _format_tool_result_message(content: str) -> str:
    """格式化工具执行结果消息"""
    try:
        tool_data = json.loads(content)
        tool_name = tool_data.get("tool_name", "unknown")
        args = tool_data.get("arguments", {})
        result = tool_data.get("result", {})
        status = tool_data.get("status", "success")

        status_emoji = "✅" if status == "success" else "❌"

        # 格式化参数
        args_str = ", ".join([f"{k}=`{v}`" for k, v in args.items()])

        # 格式化结果
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            result_str = str(result)

        return f"""{status_emoji} **工具执行完成**: `{tool_name}`

**参数**: {args_str}

**结果**:
```json
{result_str}
```"""
    except (json.JSONDecodeError, Exception):
        return f"✅ **工具结果**: {content}"


def _format_tool_message(content: str) -> str:
    """格式化工具调用消息"""
    try:
        tool_data = json.loads(content)
        tool_name = tool_data.get("tool_name", "unknown")

        if tool_data.get("status") == "calling":
            # 工具调用中
            args = tool_data.get("arguments", {})
            args_str = ", ".join([f"{k}=`{v}`" for k, v in args.items()])
            return f"**🔧 调用工具**: `{tool_name}`\n\n**参数**: {args_str}"

        elif tool_data.get("status") in ["success", "error"]:
            # 工具执行结果
            args = tool_data.get("arguments", {})
            result = tool_data.get("result", {})

            status_emoji = "✅" if tool_data.get("status") == "success" else "❌"

            # 格式化参数
            args_str = ", ".join([f"{k}=`{v}`" for k, v in args.items()])

            # 格式化结果
            if isinstance(result, dict):
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                result_str = str(result)

            return f"""**{status_emoji} 工具执行完成**: `{tool_name}`

**参数**: {args_str}

**结果**:
```json
{result_str}
```"""

        else:
            return f"**🔧 工具消息**: {tool_name}"

    except (json.JSONDecodeError, Exception):
        # 如果不是JSON格式，直接显示
        return f"**🔧 工具消息**: {content}"


def _format_play_message(content: str) -> str:
    """格式化投资结果消息"""
    try:
        play_data = json.loads(content)
        decisions = play_data.get("investment_decisions", [])
        total_decisions = play_data.get("total_decisions", 0)
        session_id = play_data.get("session_id")
        timestamp = play_data.get("timestamp")

        if not decisions:
            return "**📊 投资决策**: 无投资决策"

        result_lines = [
            f"**📊 投资决策总结** ({total_decisions} 个决策)",
            "",
            f"**会话ID**: {session_id}",
            f"**时间**: {timestamp}",
            ""
        ]

        for i, decision in enumerate(decisions, 1):
            action = decision.get("action", "unknown")
            params = decision.get("parameters", {})
            result = decision.get("result", {})
            decision_time = decision.get("timestamp")

            # 确定操作类型和图标
            action_icons = {
                "place_order": "📈",
                "cancel_order": "❌",
                "amend_order": "✏️"
            }
            icon = action_icons.get(action, "🔧")

            result_lines.append(f"**{i}. {icon} {action}**")
            result_lines.append(f"   - 时间: {decision_time}")

            # 显示参数
            if params:
                result_lines.append("   - 参数:")
                for k, v in params.items():
                    result_lines.append(f"     • {k}: `{v}`")

            # 显示结果
            if isinstance(result, dict):
                success = not result.get("error") if isinstance(result, dict) else True
                status = "✅ 成功" if success else "❌ 失败"
                result_lines.append(f"   - 结果: {status}")

                if isinstance(result, dict) and result:
                    result_lines.append("   - 详情:")
                    for k, v in result.items():
                        if k != "error":
                            result_lines.append(f"     • {k}: `{v}`")
                        elif result.get("error"):
                            result_lines.append(f"     • 错误: `{v}`")

            result_lines.append("")

        return "\n".join(result_lines)

    except (json.JSONDecodeError, Exception):
        return f"**📊 投资决策**: {content}"


def create_custom_chatbot(height: int = 600) -> gr.Chatbot:
    """
    创建自定义 Chatbot 组件

    Args:
        height: 聊天框高度

    Returns:
        gr.Chatbot: 配置好的 Chatbot 组件
    """

    custom_css = """
    .message.user {
        background-color: #e3f2fd;
    }
    .message.assistant {
        background-color: #f5f5f5;
    }
    .message.system {
        background-color: #e8eaf6;
        border-left: 4px solid #3f51b5;
        font-weight: 500;
    }
    .message.think {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        font-style: italic;
    }
    .message.tool_call {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .message.tool_result {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .message.tool {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .message.play {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
    .tool-details, .play-details, .thinking-details {
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        white-space: pre-wrap;
        margin: 8px 0;
        padding: 8px;
        background-color: #f8f9fa;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    .thinking-process {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        font-style: italic;
    }
    """

    return gr.Chatbot(
        height=height,
        label="AI Agent 对话",
        show_label=True,
        avatar_images=["👤", "🤖"],
        bubble_full_width=False,
        type="messages",
        latex_delimiters=[
            {"left": "$", "right": "$", "display": False},
            {"left": "$$", "right": "$$", "display": True},
        ]
    )


def process_streaming_messages(messages: List[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    处理流式消息，转换为 Chatbot 可显示的格式

    Args:
        messages: 消息批次列表，每批次包含多个消息

    Returns:
        List[Dict[str, str]]: Chatbot 格式的消息列表，每个消息包含 role 和 content
    """
    chatbot_messages = []

    for batch in messages:
        for message in batch:
            if not message.get("content"):
                continue

            role = message.get("role", "assistant")
            content = message.get("content", "")

            # 根据消息类型进行特殊处理
            if role == "system":
                # 系统提示词 - 保持系统角色
                formatted_content = f"💻 System: {content}"
                chatbot_messages.append({"role": "system", "content": formatted_content})

            elif role == "user":
                # 用户消息 - 直接显示
                chatbot_messages.append({"role": "user", "content": content})

            elif role == "think":
                # 思考过程 - 使用专门的思考角色
                formatted_content = _format_thinking_message(content)
                chatbot_messages.append({"role": "think", "content": formatted_content})

            elif role == "assistant":
                # 助手消息 - 检查是否是思考过程（向后兼容）
                if content.startswith("💭 **思考过程**") or content.startswith("🧠 **思考过程**"):
                    # 思考过程 - 使用 think 角色
                    formatted_content = _format_thinking_message(content)
                    chatbot_messages.append({"role": "think", "content": formatted_content})
                else:
                    # 普通助手回复
                    chatbot_messages.append({"role": "assistant", "content": content})

            elif role == "tool_call":
                # 工具调用
                formatted_content = _format_tool_call_message(content)
                chatbot_messages.append({"role": "tool_call", "content": formatted_content})

            elif role == "tool_result":
                # 工具执行结果
                formatted_content = _format_tool_result_message(content)
                chatbot_messages.append({"role": "tool_result", "content": formatted_content})

            elif role == "tool":
                # 兼容旧格式，尝试自动检测
                try:
                    tool_data = json.loads(content)
                    if tool_data.get("status") == "calling":
                        formatted_content = _format_tool_call_message(content)
                        chatbot_messages.append({"role": "tool_call", "content": formatted_content})
                    else:
                        formatted_content = _format_tool_result_message(content)
                        chatbot_messages.append({"role": "tool_result", "content": formatted_content})
                except (json.JSONDecodeError, Exception):
                    # 如果不是JSON格式，直接显示
                    formatted_content = f"🔧 **工具消息**:\n{content}"
                    chatbot_messages.append({"role": "tool", "content": formatted_content})

            elif role == "play":
                # 投资结果 - 格式化显示
                try:
                    play_data = json.loads(content)
                    formatted_content = _format_play_message(content)
                    chatbot_messages.append({"role": "play", "content": formatted_content})
                except (json.JSONDecodeError, Exception):
                    formatted_content = f"📊 **投资结果**:\n{content}"
                    chatbot_messages.append({"role": "play", "content": formatted_content})

            else:
                # 其他类型的消息
                chatbot_messages.append({"role": "assistant", "content": content})

    return chatbot_messages


def format_conversation_history(messages: List[Dict]) -> List[Dict]:
    """
    格式化对话历史，用于从数据库加载消息后显示

    Args:
        messages: 数据库消息列表

    Returns:
        List[Dict]: Chatbot 格式的消息列表，每个消息包含 role 和 content
    """
    chatbot_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content or ""

        # 根据消息类型进行特殊处理
        if msg.message_type == "thinking":
            formatted_content = f"💭 **思考过程**:\n{content}"
            chatbot_messages.append({"role": "assistant", "content": formatted_content})
        elif msg.message_type in ["tool_call", "tool_result"]:
            tool_data = {
                "tool_name": msg.tool_name or "unknown",
                "arguments": msg.tool_arguments or {},
                "result": msg.tool_result or {},
                "status": "success" if msg.message_type == "tool_result" else "calling"
            }
            tool_content = json.dumps(tool_data, ensure_ascii=False)
            chatbot_messages.append({"role": "tool", "content": tool_content})
        elif msg.message_type == "play_result":
            chatbot_messages.append({"role": "play", "content": content})
        else:
            # 普通消息
            chatbot_messages.append({"role": role, "content": content})

    return chatbot_messages