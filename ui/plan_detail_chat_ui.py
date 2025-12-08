"""
计划详情页 AI Agent 对话界面模块
"""
import asyncio
import gradio as gr
from utils.logger import setup_logger
from ui.custom_chatbot import create_custom_chatbot, process_streaming_messages
from ui.enhanced_chatbot import enhanced_chatbot

logger = setup_logger(__name__, "plan_detail_chat_ui.log")


class PlanDetailChatUI:
    """计划详情页 AI Agent 对话 UI"""

    def __init__(self, plan_detail_ui):
        self.plan_detail_ui = plan_detail_ui
        self.current_session_id = None

    def _async_to_sync_stream(self, async_func, initial_history=None, **kwargs):
        """
        重写的异步流转同步处理方法
        支持真正的实时流式输出
        """
        import asyncio
        import sys
        import threading
        import queue
        from ui.streaming_handler import StreamingHandler

        # 统一使用线程处理，避免事件循环冲突
        result_queue = queue.Queue()
        error_queue = queue.Queue()

        def run_async_in_thread():
            try:
                # 创建新的事件循环
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)

                async def stream_processor():
                    handler = StreamingHandler()
                    current_history = initial_history.copy() if initial_history else []
                    session_id = "session_" + str(hash(str(kwargs)))

                    async for message_batch in handler.process_agent_stream_realtime(
                        async_func(**kwargs), session_id
                    ):
                        # 立即处理每个消息批次
                        for message in message_batch:
                            current_history.append(message)
                            result_queue.put(("message", current_history.copy(), gr.update(value=""), gr.update(visible=False, value="")))

                    result_queue.put(("done", None, None, None))

                new_loop.run_until_complete(stream_processor())
                new_loop.close()

            except Exception as e:
                error_queue.put(e)

        # 启动异步处理线程
        thread = threading.Thread(target=run_async_in_thread, daemon=True)
        thread.start()

        # 实时获取结果并yield
        while True:
            try:
                # 检查错误
                if not error_queue.empty():
                    raise error_queue.get()

                # 获取消息
                status, history, user_input_update, status_update = result_queue.get(timeout=0.1)
                if status == "message":
                    yield history, user_input_update, status_update
                elif status == "done":
                    break

            except queue.Empty:
                continue

            except Exception as e:
                # 降级处理：使用简单的同步包装
                current_history = initial_history.copy() if initial_history else []
                error_message = [{"role": "assistant", "content": f"❌ 流式处理错误: {str(e)}"}]
                yield current_history + error_message, gr.update(value=""), gr.update(visible=True, value=f"❌ 流式处理错误: {str(e)}")
                break

    def _validate_plan_and_message(self, pid, user_message, history):
        """验证计划存在性和消息内容"""
        from utils.common import validate_plan_exists

        is_valid, plan_id, error_msg = validate_plan_exists(pid)
        if not is_valid:
            return False, None, None, history, f"❌ {error_msg}"

        if not user_message or not user_message.strip():
            return False, None, None, history, "❌ 请输入消息内容"

        return True, plan_id, user_message.strip(), history, None

    async def _collect_stream_messages(self, plan_id: int, user_message: str, conversation_type: str):
        """收集流式消息的辅助方法"""
        from services.langchain_agent import agent_service

        messages = []
        message_batches = []
        try:
            async for message_batch in agent_service.stream_conversation(
                plan_id=plan_id,
                user_message=user_message,
                conversation_type=conversation_type
            ):
                message_batches.append(message_batch)
                for message in message_batch:
                    messages.append(message)
        except Exception as e:
            logger.error(f"收集流式消息失败: {e}")
            messages = [{"role": "assistant", "content": f"❌ 对话失败: {str(e)}"}]

        # 使用增强的消息处理
        return process_streaming_messages(message_batches)

    def _get_latest_prediction_data(self, plan_id: int):
        """获取最新预测数据（原始数据）"""
        try:
            from database.models import TrainingRecord, PredictionData
            from database.db import get_db
            from sqlalchemy import desc, and_

            with get_db() as db:
                # 获取最新有预测数据的训练记录
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).join(PredictionData, TrainingRecord.id == PredictionData.training_record_id).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    return None

                # 获取最新批次的预测数据
                latest_batch = db.query(PredictionData.inference_batch_id).filter(
                    PredictionData.training_record_id == latest_training.id
                ).order_by(desc(PredictionData.created_at)).first()

                if not latest_batch:
                    return None

                return db.query(PredictionData).filter(
                    and_(
                        PredictionData.training_record_id == latest_training.id,
                        PredictionData.inference_batch_id == latest_batch.inference_batch_id
                    )
                ).order_by(PredictionData.timestamp.asc()).all()

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return None

    def _format_prediction_as_csv(self, predictions) -> str:
        """格式化预测数据为CSV"""
        if not predictions:
            return None

        # CSV头部
        csv_lines = ["timestamp,open,high,low,close,volume,amount,upward_probability,volatility_amplification_probability"]

        # 数据行
        for pred in predictions:
            timestamp_str = pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            upward_prob = (pred.upward_probability or 0) * 100
            vol_prob = (pred.volatility_amplification_probability or 0) * 100

            csv_lines.append(
                f"{timestamp_str},{pred.open:.2f},{pred.high:.2f},"
                f"{pred.low:.2f},{pred.close:.2f},"
                f"{pred.volume or 0:.2f},{pred.amount or 0:.2f},"
                f"{upward_prob:.2f}%,{vol_prob:.2f}%"
            )

        return "\n".join(csv_lines)

    def _get_latest_prediction_csv_data(self, plan_id: int) -> str:
        """获取最新预测数据的CSV格式文本"""
        predictions = self._get_latest_prediction_data(plan_id)
        return self._format_prediction_as_csv(predictions)

    def build_ui(self):
        """构建 AI Agent 对话界面"""
        components = {}

        # AI Agent 对话
        gr.Markdown("**AI Agent 对话**")
        agent_chatbot = create_custom_chatbot(height=500)

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
                # 创建按钮组件
                agent_send_btn = gr.Button("📤 发送", variant="primary", size="sm")
                agent_cancel_btn = gr.Button("❌ 取消推理", variant="stop", size="sm", interactive=False, visible=False)
                agent_execute_inference_btn = gr.Button("🧠 执行推理", variant="secondary", size="sm")
            with gr.Row():
                agent_clear_btn = gr.Button("🗑️ 清除对话", variant="secondary", size="sm")

        # 对话状态显示
        agent_status = gr.Markdown("", visible=False)

        # 保存组件引用
        components.update({
            'agent_chatbot': agent_chatbot,
            'agent_user_input': agent_user_input,
            'agent_send_btn': agent_send_btn,
            'agent_cancel_btn': agent_cancel_btn,
            'agent_execute_inference_btn': agent_execute_inference_btn,
            'agent_clear_btn': agent_clear_btn,
            'agent_status': agent_status
        })

        # 定义简化的同步事件处理函数
        def update_button_state_on_input(pid, user_input, history):
            """输入变化时更新按钮状态"""
            has_input = bool(user_input and user_input.strip())
            has_context = bool(history and len(history) > 0)
            return enhanced_chatbot.update_button_states(False, has_context=has_context, has_input=has_input)
        def agent_send_message_wrapper(pid, user_message, history):
            """发送消息给AI Agent（支持取消的流式版本）"""
            # 验证输入
            is_valid, plan_id, clean_message, current_history, error_msg = self._validate_plan_and_message(pid, user_message, history)
            if not is_valid:
                yield history + [{"role": "assistant", "content": error_msg}], gr.update(value=""), gr.update(visible=True, value=error_msg), enhanced_chatbot.update_button_states(False, has_input=False, has_context=len(history) > 0)
                return

            try:
                # 生成会话ID
                self.current_session_id = enhanced_chatbot.generate_session_id(str(plan_id), clean_message)

                # 使用带取消功能的异步流转同步处理
                from services.langchain_agent import agent_service
                for result in enhanced_chatbot.async_to_sync_stream_with_cancel(
                    agent_service.stream_conversation,
                    session_id=self.current_session_id,
                    initial_history=current_history,
                    plan_id=plan_id,
                    user_message=clean_message,
                    conversation_type="user_chat"
                ):
                    yield result

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 发送失败: {str(e)}"}]
                yield current_history + error_message, gr.update(value=""), gr.update(visible=True, value=str(e)), enhanced_chatbot.update_button_states(False, has_input=False, has_context=len(current_history) > 0)

        def agent_cancel_inference_wrapper(pid, history):
            """取消推理"""
            if self.current_session_id:
                success = enhanced_chatbot.cancel_task(self.current_session_id)
                if success:
                    logger.info(f"推理已取消: {self.current_session_id}")
                    # 保留当前对话上下文
                    return history, gr.update(visible=True, value="推理已取消，保留当前上下文"), enhanced_chatbot.update_button_states(False, has_input=False, has_context=len(history) > 0)
                else:
                    logger.warning(f"取消推理失败: {self.current_session_id}")
                    return history, gr.update(visible=True, value="取消推理失败"), enhanced_chatbot.update_button_states(False, has_input=False, has_context=len(history) > 0)
            else:
                return history, gr.update(visible=True, value="没有正在进行的推理"), enhanced_chatbot.update_button_states(False, has_input=False, has_context=len(history) > 0)

        def agent_execute_inference_wrapper(pid, history):
            """执行AI Agent推理（重置上下文 - 流式版本）"""
            # 验证计划存在性（重用验证方法，但传入空消息以跳过消息验证）
            _, plan_id, _, _, error_msg = self._validate_plan_and_message(pid, "dummy", [])
            if error_msg and "计划不存在" in error_msg:
                yield history + [{"role": "assistant", "content": error_msg}], gr.update(visible=True, value=error_msg)
                return

            try:
                # 清除现有对话历史，重置上下文
                empty_history = []

                # 获取最新25小时内的实际交易数据
                from services.historical_data_service import historical_data_service
                historical_data = historical_data_service.get_optimal_historical_data(plan_id)
                if not historical_data:
                    yield empty_history + [{"role": "assistant", "content": "❌ 未找到可用的历史K线数据"}], gr.update(visible=False, value="")
                    return

                # 获取最新预测交易数据
                prediction_data = self._get_latest_prediction_csv_data(plan_id)
                if not prediction_data:
                    yield empty_history + [{"role": "assistant", "content": "❌ 未找到可用的预测数据，请先执行模型推理"}], gr.update(visible=False, value="")
                    return

                # 构建推理请求
                inference_request = f"""【最新25小时实际交易数据】
{historical_data}

【最新预测交易数据（最新批次）】
{prediction_data}

请基于以上数据进行交易决策。"""

                # 使用通用异步流转同步处理（推理版本，重置历史）
                from services.langchain_agent import agent_service
                for history, user_input_update, status_update in self._async_to_sync_stream(
                    agent_service.stream_conversation,
                    initial_history=empty_history,  # 重置历史
                    plan_id=plan_id,
                    user_message=inference_request,
                    conversation_type="inference_session"
                ):
                    # 推理函数只需要返回 chatbot 和 status，忽略 user_input_update
                    yield history, status_update

            except Exception as e:
                logger.error(f"执行推理失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 执行推理失败: {str(e)}"}]
                yield history + error_message, gr.update(visible=False, value="")

        # 保存事件处理函数
        components.update({
            'agent_send_message_wrapper': agent_send_message_wrapper,
            'agent_cancel_inference_wrapper': agent_cancel_inference_wrapper,
            'agent_execute_inference_wrapper': agent_execute_inference_wrapper,
            'agent_clear_conversation_wrapper': self.agent_clear_conversation_wrapper,
            'update_button_state_on_input': update_button_state_on_input
        })

        
        return components

    async def _collect_all_messages_async(self, plan_id: int, user_message: str):
        """收集所有流式消息"""
        from services.langchain_agent import agent_service

        messages = []
        message_batches = []

        async for message_batch in agent_service.stream_conversation(
            plan_id=plan_id,
            user_message=user_message,
            conversation_type="auto_inference"
        ):
            message_batches.append(message_batch)
            for message in message_batch:
                messages.append(message)

        # 使用增强的消息处理
        return process_streaming_messages(message_batches)

    def agent_clear_conversation_wrapper(self, pid, history):
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

    def bind_events(self, components, plan_id_input):
        """绑定事件处理器"""
        # 绑定事件处理器 - 使用组件引用而不是字典键
        components['agent_send_btn'].click(
            fn=components['agent_send_message_wrapper'],
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']],
            show_progress=True
        )

        components['agent_cancel_btn'].click(
            fn=components['agent_cancel_inference_wrapper'],
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )

        components['agent_execute_inference_btn'].click(
            fn=components['agent_execute_inference_wrapper'],
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']],
            show_progress=True
        )

        components['agent_clear_btn'].click(
            fn=components['agent_clear_conversation_wrapper'],
            inputs=[plan_id_input, components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )

        # 支持回车发送消息
        components['agent_user_input'].submit(
            fn=components['agent_send_message_wrapper'],
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status'], components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']],
            show_progress=True
        )

        # 添加输入变化监听，动态更新按钮状态
        def update_button_state_on_input(pid, user_input, history):
            """输入变化时更新按钮状态"""
            has_input = bool(user_input and user_input.strip())
            has_context = bool(history and len(history) > 0)
            return enhanced_chatbot.update_button_states(False, has_context=has_context, has_input=has_input)

        components['agent_user_input'].change(
            fn=update_button_state_on_input,
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )