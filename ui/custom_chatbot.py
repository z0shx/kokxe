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
        return "💻", f"💻 **系统提示词**:\n{content}"

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
    """增强的工具调用消息格式化，使用markdown和代码块模拟tag效果"""
    try:
        tool_data = json.loads(content)
        tool_name = tool_data.get("tool_name", "unknown")
        args = tool_data.get("arguments", {})
        status = tool_data.get("status", "calling")
        tool_call_id = tool_data.get("tool_call_id", "")

        # 状态图标
        status_icon = "🔄" if status == "calling" else "🔧"

        # 提取关键参数用于简洁显示
        key_params = []
        if isinstance(args, dict):
            for key in ['inst_id', 'side', 'order_type', 'size']:
                if key in args:
                    key_params.append(f"{key}: `{args[key]}`")

        params_summary = " | ".join(key_params) if key_params else "无参数"

        # 创建markdown格式的工具调用标签
        tool_tag = f"""
### {status_icon} 调用工具: `{tool_name}`

**状态**: `{status}` {f'| **ID**: `{tool_call_id[:8]}`' if tool_call_id else ''}

**参数**: {params_summary}

<details>
<summary>📋 点击查看详细参数和API数据</summary>

**参数详情**:
```json
{json.dumps(args, indent=2, ensure_ascii=False)}
```

**完整数据**:
```json
{json.dumps(tool_data, indent=2, ensure_ascii=False)}
```

</details>

---

"""

        return tool_tag.strip()

    except (json.JSONDecodeError, Exception):
        # 降级显示
        return f"""
### 🔧 工具调用

**状态**: 格式错误

**原始数据**: `{content[:100]}...`

---

"""


def _format_tool_result_message(content: str) -> str:
    """增强的工具执行结果消息格式化，使用markdown和可折叠的详情"""
    try:
        tool_data = json.loads(content)
        tool_name = tool_data.get("tool_name", "unknown")
        args = tool_data.get("arguments", {})
        result = tool_data.get("result", {})
        status = tool_data.get("status", "success")
        tool_call_id = tool_data.get("tool_call_id", "")

        status_icon = "✅" if status == "success" else "❌"

        # 提取关键信息用于简洁显示
        key_info = ""
        if isinstance(result, dict) and 'order_id' in result:
            key_info = f"| **订单ID**: `{result['order_id']}`"
        elif isinstance(result, dict) and 'success' in result:
            key_info = f"| **结果**: `{result['success']}`"

        # 创建markdown格式的工具结果标签
        result_tag = f"""
### {status_icon} 工具执行: `{tool_name}`

**状态**: `{status}` {f'| **ID**: `{tool_call_id[:8]}`' if tool_call_id else ''} {key_info}

<details>
<summary>📊 点击查看执行结果和参数详情</summary>

**执行结果**:
```json
{json.dumps(result, indent=2, ensure_ascii=False)}
```

**调用参数**:
```json
{json.dumps(args, indent=2, ensure_ascii=False)}
```

**完整数据**:
```json
{json.dumps(tool_data, indent=2, ensure_ascii=False)}
```

</details>

---

"""

        return result_tag.strip()

    except (json.JSONDecodeError, Exception):
        # 降级显示
        return f"""
### ❌ 工具结果

**状态**: 格式错误

**原始数据**: `{content[:100]}...`

---

"""


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
                # 系统提示词 - 保持system角色，添加格式化
                formatted_content = f"💻 **系统提示词**:\n\n{content}"
                chatbot_messages.append({"role": "system", "content": formatted_content})

            elif role == "user":
                # 用户消息 - 直接显示
                chatbot_messages.append({"role": "user", "content": content})

            elif role == "think":
                # 思考过程 - 转换为 assistant 角色，但保持格式化显示
                formatted_content = _format_thinking_message(content)
                chatbot_messages.append({"role": "assistant", "content": formatted_content})

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
                # 工具调用 - 转换为 assistant 角色，但保持格式化显示
                formatted_content = _format_tool_call_message(content)
                chatbot_messages.append({"role": "assistant", "content": formatted_content})

            elif role == "tool_result":
                # 工具执行结果 - 转换为 assistant 角色，但保持格式化显示
                formatted_content = _format_tool_result_message(content)
                chatbot_messages.append({"role": "assistant", "content": formatted_content})

            elif role == "tool":
                # 兼容旧格式，尝试自动检测 - 转换为 assistant 角色
                try:
                    tool_data = json.loads(content)
                    if tool_data.get("status") == "calling":
                        formatted_content = _format_tool_call_message(content)
                        chatbot_messages.append({"role": "assistant", "content": formatted_content})
                    else:
                        formatted_content = _format_tool_result_message(content)
                        chatbot_messages.append({"role": "assistant", "content": formatted_content})
                except (json.JSONDecodeError, Exception):
                    # 如果不是JSON格式，直接显示
                    formatted_content = f"🔧 **工具消息**:\n{content}"
                    chatbot_messages.append({"role": "assistant", "content": formatted_content})

            elif role == "play":
                # 投资结果 - 转换为 assistant 角色，但保持格式化显示
                try:
                    play_data = json.loads(content)
                    formatted_content = _format_play_message(content)
                    chatbot_messages.append({"role": "assistant", "content": formatted_content})
                except (json.JSONDecodeError, Exception):
                    formatted_content = f"📊 **投资结果**:\n{content}"
                    chatbot_messages.append({"role": "assistant", "content": formatted_content})

            else:
                # 其他类型的消息
                chatbot_messages.append({"role": "assistant", "content": content})

    return chatbot_messages


def format_agent_message_for_display(msg, include_order_details: bool = True) -> Dict[str, str]:
    """
    统一的消息格式化函数，将 AgentMessage 数据库对象转换为 Chatbot 显示格式

    这是核心的统一格式化函数，确保 UI 恢复和 Agent 加载使用相同的格式。

    Args:
        msg: AgentMessage 数据库对象
        include_order_details: 是否包含订单详情（需要数据库访问）

    Returns:
        Dict[str, str]: Chatbot 格式的消息，包含 role 和 content
    """
    role = msg.role
    content = msg.content or ""
    message_type = msg.message_type or "text"

    # 系统消息 - 统一格式
    if role == "system":
        formatted_content = f"💻 **系统提示词**:\n\n{content}"
        return {"role": "system", "content": formatted_content}

    # 思考过程 - 使用统一的格式化函数
    if message_type == "thinking":
        formatted_content = _format_thinking_message(content)
        return {"role": "assistant", "content": formatted_content}

    # 工具调用 - 使用统一的格式化函数
    if message_type == "tool_call":
        # 解析工具参数（可能是字符串或dict）
        arguments = {}
        if msg.tool_arguments:
            if isinstance(msg.tool_arguments, str):
                try:
                    arguments = json.loads(msg.tool_arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": msg.tool_arguments}
            elif isinstance(msg.tool_arguments, dict):
                arguments = msg.tool_arguments

        tool_data = {
            "tool_name": msg.tool_name or "unknown",
            "arguments": arguments,
            "result": {},
            "status": "calling",
            "tool_call_id": msg.tool_call_id or ""
        }
        tool_content = json.dumps(tool_data, ensure_ascii=False)
        formatted_content = _format_tool_call_message(tool_content)
        return {"role": "assistant", "content": formatted_content}

    # 工具结果 - 使用统一的格式化函数，可选包含订单详情
    if message_type == "tool_result":
        # 解析工具参数（可能是字符串或dict）
        arguments = {}
        if msg.tool_arguments:
            if isinstance(msg.tool_arguments, str):
                try:
                    arguments = json.loads(msg.tool_arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": msg.tool_arguments}
            elif isinstance(msg.tool_arguments, dict):
                arguments = msg.tool_arguments

        # 解析工具结果（可能是字符串或dict）
        result = {}
        if msg.tool_result:
            if isinstance(msg.tool_result, str):
                try:
                    result = json.loads(msg.tool_result)
                except json.JSONDecodeError:
                    result = {"raw": msg.tool_result}
            elif isinstance(msg.tool_result, dict):
                result = msg.tool_result

        tool_data = {
            "tool_name": msg.tool_name or "unknown",
            "arguments": arguments,
            "result": result,
            "status": "success" if msg.tool_status == "success" else "failed",
            "tool_call_id": msg.tool_call_id or ""
        }

        # 如果需要订单详情且是交易相关工具
        if include_order_details and msg.related_order_id and tool_data['tool_name'] in ['place_order', 'amend_order', 'cancel_order']:
            try:
                from database.db import get_db
                from database.models import TradeOrder
                with get_db() as db:
                    order = db.query(TradeOrder).filter(
                        TradeOrder.order_id == msg.related_order_id
                    ).first()
                    if order:
                        # 将订单详情添加到结果中
                        order_info = {
                            "order_id": order.order_id,
                            "inst_id": order.inst_id,
                            "side": order.side,
                            "order_type": order.order_type,
                            "size": order.size,
                            "price": order.price,
                            "status": order.status,
                            "filled_size": order.filled_size,
                            "avg_price": order.avg_price,
                            "created_at": order.created_at.isoformat() if order.created_at else None
                        }
                        # 将订单详情合并到结果中
                        if isinstance(tool_data['result'], dict):
                            tool_data['result']['_order_info'] = order_info
                        else:
                            tool_data['result'] = {"_order_info": order_info}
            except Exception as e:
                # 如果查询订单失败，继续使用原始数据
                pass

        tool_content = json.dumps(tool_data, ensure_ascii=False)
        formatted_content = _format_tool_result_message(tool_content)
        return {"role": "assistant", "content": formatted_content}

    # 投资结果
    if message_type == "play_result":
        formatted_content = _format_play_message(content)
        return {"role": "assistant", "content": formatted_content}

    # 普通消息 - user 和 assistant
    if role in ["user", "assistant"]:
        return {"role": role, "content": content}

    # 默认 - 作为 assistant 消息处理
    return {"role": "assistant", "content": content}


def format_conversation_history(messages: List[Dict]) -> List[Dict]:
    """
    格式化对话历史，用于从数据库加载消息后显示

    注意：Gradio Chatbot 在 type="messages" 模式下只渲染 user 和 assistant 角色。
    其他角色（如 system）会被忽略，因此需要将 system 转换为 assistant 角色。

    Args:
        messages: 数据库消息列表 (AgentMessage 对象)

    Returns:
        List[Dict]: Chatbot 格式的消息列表，每个消息包含 role 和 content
    """
    chatbot_messages = []

    for msg in messages:
        formatted_msg = format_agent_message_for_display(msg, include_order_details=True)

        # 确保 system 消息也显示：将 system role 转换为 assistant
        # 但保留 "💻 **系统提示词**:" 格式标识
        if formatted_msg.get("role") == "system":
            formatted_msg["role"] = "assistant"

        chatbot_messages.append(formatted_msg)

    return chatbot_messages