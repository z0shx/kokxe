"""
计划详情页 AI Agent 对话界面模块
"""
import asyncio
import gradio as gr
from utils.logger import setup_logger

logger = setup_logger(__name__, "plan_detail_chat_ui.log")


class PlanDetailChatUI:
    """计划详情页 AI Agent 对话 UI"""

    def __init__(self, plan_detail_ui):
        self.plan_detail_ui = plan_detail_ui

    async def _collect_stream_messages(self, plan_id: int, user_message: str, conversation_type: str):
        """收集流式消息的辅助方法"""
        from services.langchain_agent import agent_service

        messages = []
        try:
            async for message_batch in agent_service.stream_conversation(
                plan_id=plan_id,
                user_message=user_message,
                conversation_type=conversation_type
            ):
                for message in message_batch:
                    messages.append(message)
        except Exception as e:
            logger.error(f"收集流式消息失败: {e}")
            messages = [{"role": "assistant", "content": f"❌ 对话失败: {str(e)}"}]

        return messages

    def build_ui(self):
        """构建 AI Agent 对话界面"""
        components = {}

        # AI Agent 对话
        gr.Markdown("**AI Agent 对话**")
        agent_chatbot = gr.Chatbot(
            label="AI Agent 推理过程",
            height=500,
            show_copy_button=True,
            type='messages'
        )

        # AI Agent 对话交互界面
        with gr.Row():
            with gr.Column(scale=4):
                agent_user_input = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的消息或指令...",
                    lines=2,
                    max_lines=5
                )
            with gr.Column(scale=1):
                with gr.Row():
                    agent_send_btn = gr.Button("📤 发送", variant="primary", size="sm")
                    execute_inference_btn = gr.Button("🧠 执行推理", variant="secondary", size="sm")
                with gr.Row():
                    agent_clear_btn = gr.Button("🗑️ 清除对话", variant="secondary", size="sm")

        # 对话状态显示
        agent_status = gr.Markdown("", visible=False)

        # 保存组件引用
        components.update({
            'agent_chatbot': agent_chatbot,
            'agent_user_input': agent_user_input,
            'agent_send_btn': agent_send_btn,
            'execute_inference_btn': execute_inference_btn,
            'agent_clear_btn': agent_clear_btn,
            'agent_status': agent_status
        })

        # 定义简化的同步事件处理函数
        def agent_send_message_wrapper(pid, user_message, history):
            """发送消息给AI Agent（简化版本）"""
            from utils.common import validate_plan_exists

            is_valid, plan_id, error_msg = validate_plan_exists(pid)

            if not is_valid:
                return history, gr.update(value=""), gr.update(visible=True, value=f"❌ {error_msg}")

            if not user_message or not user_message.strip():
                return history, gr.update(value=""), gr.update(visible=True, value=f"❌ 请输入消息内容")

            try:
                # 调用真实的 Agent 服务进行对话
                from services.langchain_agent import agent_service
                import asyncio

                # 创建异步运行器
                def run_async_generator():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._collect_stream_messages(plan_id, user_message.strip(), "user_chat")
                        )
                    finally:
                        loop.close()

                # 获取所有消息后更新界面
                messages = run_async_generator()
                return history + messages, gr.update(value=""), gr.update(visible=False, value="")

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 发送失败: {str(e)}"}]
                return history + error_message, gr.update(value=""), gr.update(visible=False, value="")

        def agent_execute_inference_wrapper(pid, history):
            """执行AI Agent推理（简化版本）"""
            from utils.common import validate_plan_exists

            is_valid, plan_id, error_msg = validate_plan_exists(pid)

            if not is_valid:
                return history, gr.update(visible=True, value=f"❌ {error_msg}")

            try:
                # 创建异步运行器
                def run_async_inference():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._collect_stream_messages(plan_id, "请基于最新数据进行分析和决策", "auto_inference")
                        )
                    finally:
                        loop.close()

                # 获取所有消息后更新界面
                messages = run_async_inference()
                return history + messages, gr.update(visible=False, value="")

            except Exception as e:
                logger.error(f"执行推理失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 执行推理失败: {str(e)}"}]
                return history + error_message, gr.update(visible=False, value="")

        def agent_clear_conversation_wrapper(pid, history):
            """清除AI Agent对话"""
            from utils.common import validate_plan_exists

            is_valid, plan_id, error_msg = validate_plan_exists(pid)

            if not is_valid:
                return history, gr.update(value=""), gr.update(visible=False, value=f"❌ {error_msg}")

            try:
                result = self.plan_detail_ui.clear_agent_records(plan_id)
                # 清空聊天历史
                empty_history = []
                status_message = f"✅ {result}"
                return empty_history, gr.update(value=""), gr.update(visible=True, value=status_message)

            except Exception as e:
                logger.error(f"清除对话失败: {e}")
                return history, gr.update(value=""), gr.update(visible=False, value=f"❌ 清除失败: {str(e)}")

        # 保存事件处理函数
        components.update({
            'agent_send_message_wrapper': agent_send_message_wrapper,
            'agent_execute_inference_wrapper': agent_execute_inference_wrapper,
            'agent_clear_conversation_wrapper': agent_clear_conversation_wrapper
        })

        return components

    def bind_events(self, components, plan_id_input):
        """绑定事件处理器"""
        # 绑定事件处理器 - 使用组件引用而不是字典键
        components['agent_send_btn'].click(
            fn=components['agent_send_message_wrapper'],
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status']],
            show_progress=True
        )

        components['execute_inference_btn'].click(
            fn=components['agent_execute_inference_wrapper'],
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_status']],
            show_progress=True
        )

        components['agent_clear_btn'].click(
            fn=components['agent_clear_conversation_wrapper'],
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status']]
        )

        # 支持回车发送消息
        components['agent_user_input'].submit(
            fn=components['agent_send_message_wrapper'],
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status']],
            show_progress=True
        )