"""
增强的聊天机器人界面，支持取消推理功能
"""
import asyncio
import gradio as gr
from utils.logger import setup_logger
from typing import List, Optional, Tuple, Dict, Any
import threading
import queue
import time

logger = setup_logger(__name__, "enhanced_chatbot.log")


class EnhancedChatbot:
    """增强的聊天机器人，支持取消功能"""

    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queues: Dict[str, queue.Queue] = {}
        self.cancel_flags: Dict[str, bool] = {}

    def create_chat_interface(self, plan_id_input, components):
        """
        创建支持取消功能的聊天界面

        Args:
            plan_id_input: 计划ID输入组件
            components: UI组件字典
        """

        # 添加取消按钮状态
        with gr.Row():
            agent_send_btn = gr.Button("📤 发送", variant="primary", size="sm", interactive=False)
            agent_cancel_btn = gr.Button("❌ 取消推理", variant="stop", size="sm", interactive=False, visible=False)
            agent_execute_inference_btn = gr.Button("🧠 执行推理", variant="secondary", size="sm", interactive=False)

        with gr.Row():
            agent_clear_btn = gr.Button("🗑️ 清除对话", variant="secondary", size="sm")

        # 更新组件引用
        components.update({
            'agent_send_btn': agent_send_btn,
            'agent_cancel_btn': agent_cancel_btn,
            'agent_execute_inference_btn': agent_execute_inference_btn,
            'agent_clear_btn': agent_clear_btn
        })

        return components

    def update_button_states(self, is_running: bool, has_context: bool = False, has_input: bool = False):
        """
        更新按钮状态

        Args:
            is_running: 是否正在运行
            has_context: 是否有对话上下文
            has_input: 是否有输入内容
        """
        if is_running:
            return (
                gr.update(interactive=False, visible=False),  # 发送按钮
                gr.update(interactive=True, visible=True),   # 取消按钮
                gr.update(interactive=False, visible=False)   # 执行推理按钮
            )
        else:
            send_interactive = has_input or has_context
            return (
                gr.update(interactive=send_interactive, visible=True),  # 发送按钮
                gr.update(interactive=False, visible=False),            # 取消按钮
                gr.update(interactive=has_context, visible=True)         # 执行推理按钮
            )

    def generate_session_id(self, plan_id: str, input_text: str = "") -> str:
        """生成会话ID"""
        import hashlib
        content = f"{plan_id}_{input_text}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def cancel_task(self, session_id: str) -> bool:
        """
        取消指定会话的任务

        Args:
            session_id: 会话ID

        Returns:
            是否成功取消
        """
        try:
            # 设置取消标志
            self.cancel_flags[session_id] = True

            # 取消异步任务
            if session_id in self.active_tasks:
                task = self.active_tasks[session_id]
                if not task.done():
                    task.cancel()
                    logger.info(f"任务已取消: {session_id}")
                    return True

            return False
        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False

    def is_cancelled(self, session_id: str) -> bool:
        """检查任务是否被取消"""
        return self.cancel_flags.get(session_id, False)

    def cleanup_session(self, session_id: str):
        """清理会话资源"""
        self.active_tasks.pop(session_id, None)
        self.task_queues.pop(session_id, None)
        self.cancel_flags.pop(session_id, None)
        logger.info(f"会话资源已清理: {session_id}")

    def async_to_sync_stream_with_cancel(self, async_func, session_id: str, initial_history=None, **kwargs):
        """
        带取消功能的异步流转同步处理

        Args:
            async_func: 异步函数
            session_id: 会话ID
            initial_history: 初始历史
            **kwargs: 传递给异步函数的参数

        Yields:
            元组 (历史记录, 用户输入更新, 状态更新, 按钮状态更新)
        """
        import asyncio
        import sys
        import threading
        import queue

        # 创建队列和线程
        result_queue = queue.Queue()
        error_queue = queue.Queue()

        def run_async_in_thread():
            try:
                # 创建新的事件循环
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)

                async def stream_processor():
                    try:
                        current_history = initial_history.copy() if initial_history else []

                        # 发送开始信号
                        result_queue.put(("start", current_history.copy(), gr.update(value=""), gr.update(visible=False, value=""), self.update_button_states(True)))

                        # 处理异步流
                        async for message_batch in async_func(**kwargs):
                            # 检查取消标志
                            if self.is_cancelled(session_id):
                                logger.info(f"检测到取消信号，停止处理: {session_id}")
                                # 发送取消完成信号
                                result_queue.put(("cancelled", current_history.copy(), gr.update(value=""), gr.update(visible=True, value="推理已取消"), self.update_button_states(False, has_input=False, has_context=len(current_history) > 0)))
                                break

                            # 处理消息批次
                            for message in message_batch:
                                current_history.append(message)
                                result_queue.put(("message", current_history.copy(), gr.update(value=""), gr.update(visible=False, value=""), self.update_button_states(True)))

                        # 如果没有被取消，发送完成信号
                        if not self.is_cancelled(session_id):
                            result_queue.put(("complete", current_history.copy(), gr.update(value=""), gr.update(visible=False, value=""), self.update_button_states(False, has_input=False, has_context=len(current_history) > 0)))

                    except asyncio.CancelledError:
                        logger.info(f"异步任务被取消: {session_id}")
                        result_queue.put(("cancelled", initial_history.copy() if initial_history else [], gr.update(value=""), gr.update(visible=True, value="推理已取消"), self.update_button_states(False, has_input=False, has_context=False)))
                    except Exception as e:
                        logger.error(f"异步流处理失败: {e}")
                        error_queue.put(e)

                # 创建并运行任务
                task = new_loop.create_task(stream_processor())
                self.active_tasks[session_id] = task

                # 运行事件循环
                new_loop.run_until_complete(task)

            except Exception as e:
                logger.error(f"线程执行失败: {e}")
                error_queue.put(e)
            finally:
                try:
                    new_loop.close()
                except:
                    pass
                # 清理会话资源
                self.cleanup_session(session_id)

        # 启动线程
        thread = threading.Thread(target=run_async_in_thread, daemon=True)
        thread.start()

        # 处理队列结果
        while True:
            try:
                # 检查是否有错误
                if not error_queue.empty():
                    error = error_queue.get_nowait()
                    error_message = [{"role": "assistant", "content": f"❌ 执行失败: {str(error)}"}]
                    yield initial_history + error_message if initial_history else error_message, gr.update(value=""), gr.update(visible=True, value=str(error)), self.update_button_states(False, has_input=False, has_context=False)
                    break

                # 获取结果
                if not result_queue.empty():
                    result_type, history, input_update, status_update, button_update = result_queue.get_nowait()

                    if result_type in ["start", "message", "cancelled", "complete"]:
                        yield history, input_update, status_update, button_update

                        if result_type in ["cancelled", "complete"]:
                            break
                    else:
                        logger.warning(f"未知结果类型: {result_type}")

                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理队列结果失败: {e}")
                break


# 创建全局实例
enhanced_chatbot = EnhancedChatbot()


def create_enhanced_chat_interface(plan_id_input):
    """
    创建增强的聊天界面

    Args:
        plan_id_input: 计划ID输入组件

    Returns:
        Gradio界面组件
    """
    with gr.Row():
        agent_chatbot = gr.Chatbot(
            label="AI交易助手",
            height=400,
            show_label=True,
            avatar_images=(
                "👤",  # 用户头像
                "🤖"   # AI头像
            ),
            bubble_full_width=False
        )

    with gr.Row():
        with gr.Column(scale=3):
            agent_user_input = gr.Textbox(
                label="",
                placeholder="请输入您的问题或交易指令...",
                lines=2,
                max_lines=5
            )
        with gr.Column(scale=1):
            # 这里会被 EnhancedChatbot.create_chat_interface 替换
            pass

    return {
        'agent_chatbot': agent_chatbot,
        'agent_user_input': agent_user_input
    }


def setup_enhanced_events(components, plan_id_input):
    """
    设置增强的事件处理

    Args:
        components: UI组件字典
        plan_id_input: 计划ID输入组件
    """

    def get_current_state(pid, user_input, history):
        """获取当前状态"""
        has_input = bool(user_input and user_input.strip())
        has_context = bool(history and len(history) > 0)
        return has_input, has_context

    def cancel_inference(pid, history):
        """取消推理"""
        # 这里需要从某个地方获取当前会话ID
        # 简化版本：暂时返回空
        logger.info("取消推理请求")
        return history, gr.update(visible=False, value="推理已取消"), enhanced_chatbot.update_button_states(False, has_input=False, has_context=bool(history))

    # 设置取消按钮事件
    if 'agent_cancel_btn' in components:
        components['agent_cancel_btn'].click(
            fn=cancel_inference,
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )

    # 设置输入框变化事件
    def on_input_change(pid, user_input, history):
        """输入变化时更新按钮状态"""
        has_input, has_context = get_current_state(pid, user_input, history)
        return enhanced_chatbot.update_button_states(False, has_context=has_context, has_input=has_input)

    if 'agent_user_input' in components:
        components['agent_user_input'].change(
            fn=on_input_change,
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )