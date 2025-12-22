"""
计划详情页 AI Agent 对话界面模块
"""
import asyncio
import gradio as gr
from utils.logger import setup_logger
from ui.custom_chatbot import create_custom_chatbot, process_streaming_messages

logger = setup_logger(__name__, "plan_detail_chat_ui.log")


class ChatButtonStateManager:
    """简化的聊天按钮状态管理器"""

    # 类变量用于跟踪活动任务
    _active_tasks = {}

    @staticmethod
    def update_button_states(is_running: bool, has_context: bool = False, has_input: bool = False):
        """更新按钮状态"""
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
                gr.update(interactive=True, visible=True)               # 执行推理按钮 - 始终可用
            )

    @staticmethod
    def generate_session_id(plan_id: str, input_text: str = "") -> str:
        """生成会话ID"""
        import hashlib
        import time
        content = f"{plan_id}_{input_text}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @classmethod
    def register_task(cls, session_id: str, task_info: dict):
        """注册任务"""
        cls._active_tasks[session_id] = task_info

    @classmethod
    def cancel_task(cls, session_id: str) -> bool:
        """取消任务"""
        if session_id in cls._active_tasks:
            task_info = cls._active_tasks[session_id]
            try:
                # 简化的取消逻辑
                task_info['cancelled'] = True
                del cls._active_tasks[session_id]
                return True
            except Exception:
                return False
        return False


class PlanDetailChatUI:
    """计划详情页 AI Agent 对话 UI"""

    def __init__(self, plan_detail_ui):
        self.plan_detail_ui = plan_detail_ui
        self.current_session_id = None
        self.active_tasks = {}  # 简化的任务管理

    def _async_to_sync_stream(self, async_func, initial_history=None, **kwargs):
        """简化的异步流转同步处理方法"""
        import asyncio

        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            current_history = initial_history.copy() if initial_history else []

            async def stream_processor():
                try:
                    async for message_batch in async_func(**kwargs):
                        if isinstance(message_batch, list):
                            for message in message_batch:
                                current_history.append(message)
                                yield current_history.copy(), gr.update(value=""), gr.update(visible=False, value="")
                        else:
                            current_history.append(message_batch)
                            yield current_history.copy(), gr.update(value=""), gr.update(visible=False, value="")

                except Exception as e:
                    error_message = [{"role": "assistant", "content": f"❌ 流式处理错误: {str(e)}"}]
                    current_history.extend(error_message)
                    yield current_history.copy(), gr.update(value=""), gr.update(visible=True, value=f"❌ 流式处理错误: {str(e)}")

            # 运行异步生成器
            async_gen = stream_processor()
            try:
                while True:
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
            except StopAsyncIteration:
                pass

        except Exception as e:
            # 错误处理
            current_history = initial_history.copy() if initial_history else []
            error_message = [{"role": "assistant", "content": f"❌ 异步处理错误: {str(e)}"}]
            yield current_history + error_message, gr.update(value=""), gr.update(visible=True, value=f"❌ 异步处理错误: {str(e)}")

        finally:
            try:
                loop.close()
            except:
                pass

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

    def _format_prediction_with_monte_carlo_stats(self, predictions) -> str:
        """
        格式化预测数据为CSV格式，保持精确时间戳并显示所有蒙特卡罗路径信息

        Args:
            predictions: 预测数据列表

        Returns:
            str: 包含精确时间戳的详细预测数据文本
        """
        if not predictions:
            return "无预测数据可用"

        # 生成输出文本
        output_lines = []
        output_lines.append("【蒙特卡罗路径预测数据（保持精确时间）】")
        output_lines.append(f"总预测数据点: {len(predictions)}")
        output_lines.append("")

        # 按时间戳排序，保持时间顺序
        predictions_sorted = sorted(predictions, key=lambda x: x.timestamp)

        # 添加所有预测数据（保持精确时间戳）
        output_lines.append("timestamp,path_id,open,high,low,close,volume,amount,upward_probability,volatility_amplification_probability")

        # 为每个预测数据添加路径ID
        path_groups = {}
        for i, pred in enumerate(predictions_sorted):
            timestamp_str = pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')

            # 按时间戳分组，为每个时间点的路径分配ID
            if timestamp_str not in path_groups:
                path_groups[timestamp_str] = 0
            else:
                path_groups[timestamp_str] += 1

            path_id = f"path_{path_groups[timestamp_str]}"
            upward_prob = (pred.upward_probability or 0) * 100
            vol_prob = (pred.volatility_amplification_probability or 0) * 100

            output_lines.append(
                f"{timestamp_str},{path_id},{pred.open:.2f},{pred.high:.2f},"
                f"{pred.low:.2f},{pred.close:.2f},"
                f"{pred.volume or 0:.2f},{pred.amount or 0:.2f},"
                f"{upward_prob:.2f}%,{vol_prob:.2f}%"
            )

        # 添加每个时间点的统计信息（用于快速概览）
        output_lines.append("")
        output_lines.append("【时间点统计概览】")
        output_lines.append("timestamp,path_count,open_min,open_max,high_min,high_max,low_min,low_max,close_min,close_max")

        # 计算每个时间点的统计信息
        time_stats = {}
        for pred in predictions_sorted:
            timestamp_str = pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            if timestamp_str not in time_stats:
                time_stats[timestamp_str] = {
                    'count': 0,
                    'opens': [],
                    'highs': [],
                    'lows': [],
                    'closes': []
                }

            time_stats[timestamp_str]['count'] += 1
            time_stats[timestamp_str]['opens'].append(pred.open)
            time_stats[timestamp_str]['highs'].append(pred.high)
            time_stats[timestamp_str]['lows'].append(pred.low)
            time_stats[timestamp_str]['closes'].append(pred.close)

        # 添加统计信息（显示前15个时间点）
        for i, (timestamp_str, stats) in enumerate(sorted(time_stats.items())[:15]):
            output_lines.append(
                f"{timestamp_str},{stats['count']},"
                f"{min(stats['opens']):.2f},{max(stats['opens']):.2f},"
                f"{min(stats['highs']):.2f},{max(stats['highs']):.2f},"
                f"{min(stats['lows']):.2f},{max(stats['lows']):.2f},"
                f"{min(stats['closes']):.2f},{max(stats['closes']):.2f}"
            )

        if len(time_stats) > 15:
            output_lines.append(f"... （还有 {len(time_stats) - 15} 个时间点的统计）")

        return "\n".join(output_lines)

    def _get_latest_prediction_csv_data(self, plan_id: int) -> str:
        """获取最新预测数据的CSV格式文本（包含蒙特卡罗路径统计）"""
        predictions = self._get_latest_prediction_data(plan_id)
        return self._format_prediction_with_monte_carlo_stats(predictions)

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

        # 保存组件引用（移除重复的对话恢复下拉选择器组件）
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
            return ChatButtonStateManager.update_button_states(False, has_context=has_context, has_input=has_input)
        def agent_send_message_wrapper(pid, user_message, history):
            """发送消息给AI Agent（支持取消的流式版本）"""
            # 验证输入
            is_valid, plan_id, clean_message, current_history, error_msg = self._validate_plan_and_message(pid, user_message, history)
            if not is_valid:
                button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                yield history + [{"role": "assistant", "content": error_msg}], gr.update(value=""), gr.update(visible=True, value=error_msg), button_states[0], button_states[1], button_states[2]
                return

            try:
                # 生成会话ID
                self.current_session_id = ChatButtonStateManager.generate_session_id(str(plan_id), clean_message)

                # 使用带取消功能的异步流转同步处理
                from services.langchain_agent import agent_service
                final_result = None
                for result in ChatButtonStateManager.async_to_sync_stream(
                    agent_service.stream_conversation,
                    session_id=self.current_session_id,
                    initial_history=current_history,
                    plan_id=plan_id,
                    user_message=clean_message,
                    conversation_type="user_chat"
                ):
                    final_result = result
                    yield result

                # 对话完成后，自动刷新对话列表
                if final_result:
                    try:
                        # 获取最新的对话列表
                        choices = self.get_conversation_list_for_selection(plan_id)
                        # 添加对话列表刷新到yield结果中
                        history, _, _, button_updates = final_result
                        yield history, _, _, button_updates, gr.update(choices=choices), gr.update(interactive=True)
                    except Exception as refresh_error:
                        logger.error(f"自动刷新对话列表失败: {refresh_error}")
                        # 即使刷新失败，也要返回原始结果
                        yield final_result

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 发送失败: {str(e)}"}]
                button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(current_history) > 0)
                yield current_history + error_message, gr.update(value=""), gr.update(visible=True, value=str(e)), button_states[0], button_states[1], button_states[2]

        def agent_cancel_inference_wrapper(pid, history):
            """取消推理"""
            if self.current_session_id:
                success = ChatButtonStateManager.cancel_task(self.current_session_id)
                if success:
                    logger.info(f"推理已取消: {self.current_session_id}")
                    # 保留当前对话上下文
                    button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                    return history, gr.update(value=""), gr.update(visible=True, value="推理已取消，保留当前上下文"), button_states[0], button_states[1], button_states[2]
                else:
                    logger.warning(f"取消推理失败: {self.current_session_id}")
                    button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                    return history, gr.update(value=""), gr.update(visible=True, value="取消推理失败"), button_states[0], button_states[1], button_states[2]
            else:
                button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                return history, gr.update(value=""), gr.update(visible=True, value="没有正在进行的推理"), button_states[0], button_states[1], button_states[2]

        def agent_execute_inference_wrapper(pid, history):
            """执行AI Agent推理（重置上下文 - 流式版本）"""
            # 验证计划存在性（重用验证方法，但传入空消息以跳过消息验证）
            _, plan_id, _, _, error_msg = self._validate_plan_and_message(pid, "dummy", [])
            if error_msg and "计划不存在" in error_msg:
                button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                yield history + [{"role": "assistant", "content": error_msg}], gr.update(visible=True, value=error_msg), button_states[0], button_states[1], button_states[2]
                return

            try:
                # 清除现有对话历史，重置上下文
                empty_history = []

                # 获取最新25小时内的实际交易数据
                from services.historical_data_service import historical_data_service
                historical_data = historical_data_service.get_optimal_historical_data(plan_id)
                if not historical_data:
                    button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(empty_history) > 0)
                    yield empty_history + [{"role": "assistant", "content": "❌ 未找到可用的历史K线数据"}], gr.update(visible=True, value="未找到可用的历史K线数据"), button_states[0], button_states[1], button_states[2]
                    return

                # 获取最新预测交易数据
                prediction_data = self._get_latest_prediction_csv_data(plan_id)
                if not prediction_data:
                    button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(empty_history) > 0)
                    yield empty_history + [{"role": "assistant", "content": "❌ 未找到可用的预测数据，请先执行模型推理"}], gr.update(visible=True, value="未找到可用的预测数据，请先执行模型推理"), button_states[0], button_states[1], button_states[2]
                    return

                # 构建推理请求
                inference_request = f"""【最新25小时实际交易数据】
{historical_data}

【最新预测交易数据（最新批次）】
{prediction_data}

请基于以上数据进行交易决策。"""

                # 使用通用异步流转同步处理（推理版本，重置历史）
                from services.langchain_agent import agent_service
                final_result = None
                for history, user_input_update, status_update in self._async_to_sync_stream(
                    agent_service.stream_conversation,
                    initial_history=empty_history,  # 重置历史
                    plan_id=plan_id,
                    user_message=inference_request,
                    conversation_type="inference_session"
                ):
                    # 推理函数需要返回完整的5个值：chatbot, status, send_btn, cancel_btn, inference_btn
                    button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                    final_result = (history, status_update, button_states[0], button_states[1], button_states[2])
                    yield final_result

                # 推理完成后，自动刷新对话列表
                if final_result:
                    try:
                        choices = self.get_conversation_list_for_selection(plan_id)
                        # 发送刷新信号，但界面需要处理额外的输出参数
                        yield (*final_result, gr.update(choices=choices), gr.update(interactive=True))
                    except Exception as refresh_error:
                        logger.error(f"推理完成后自动刷新对话列表失败: {refresh_error}")
                        # 即使刷新失败，也要返回原始结果
                        yield final_result

            except Exception as e:
                logger.error(f"执行推理失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 执行推理失败: {str(e)}"}]
                button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
                yield history + error_message, gr.update(visible=True, value=f"执行推理失败: {str(e)}"), button_states[0], button_states[1], button_states[2]

        # 保存事件处理函数（移除重复的对话恢复下拉选择器相关处理函数）
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
            button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=False)
            return history, gr.update(value=""), gr.update(visible=True, value=f"❌ {error_msg}"), button_states[0], button_states[1], button_states[2]

        try:
            result = self.plan_detail_ui.clear_agent_records(plan_id)
            # 清空聊天历史
            empty_history = []
            status_message = f"✅ {result}"
            button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=False)
            return empty_history, gr.update(value=""), gr.update(visible=True, value=status_message), button_states[0], button_states[1], button_states[2]

        except Exception as e:
            logger.error(f"清除对话失败: {e}")
            button_states = ChatButtonStateManager.update_button_states(False, has_input=False, has_context=len(history) > 0)
            return history, gr.update(value=""), gr.update(visible=True, value=f"❌ 清除失败: {str(e)}"), button_states[0], button_states[1], button_states[2]

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
            return ChatButtonStateManager.update_button_states(False, has_context=has_context, has_input=has_input)

        components['agent_user_input'].change(
            fn=update_button_state_on_input,
            inputs=[plan_id_input, components['agent_user_input'], components['agent_chatbot']],
            outputs=[components['agent_send_btn'], components['agent_cancel_btn'], components['agent_execute_inference_btn']]
        )

        # 移除重复的对话恢复下拉选择器事件绑定 - 直接使用列表点击恢复功能

    def setup_auto_refresh_and_initial_load(self, plan_id_input, components):
        """设置自动刷新和初始加载（简化版本 - 列表点击恢复无需额外初始化）"""
        return components

    def restore_conversation_from_records(self, plan_id: int, conversation_id: int = None):
        """
        从对话记录恢复完整对话到chatbot

        Args:
            plan_id: 计划ID
            conversation_id: 指定对话ID，如果为None则恢复最新对话
        """
        try:
            from database.models import AgentConversation, AgentMessage
            from database.db import get_db
            from ui.custom_chatbot import format_conversation_history
            from sqlalchemy import desc

            with get_db() as db:
                # 如果没有指定对话ID，获取最新的对话
                if conversation_id is None:
                    conversation = db.query(AgentConversation).filter(
                        AgentConversation.plan_id == plan_id
                    ).order_by(desc(AgentConversation.last_message_at)).first()
                else:
                    conversation = db.query(AgentConversation).filter(
                        AgentConversation.id == conversation_id,
                        AgentConversation.plan_id == plan_id
                    ).first()

                if not conversation:
                    return []  # 返回空历史，表示没有对话记录

                # 获取该对话的所有消息，按时间顺序排列
                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation.id
                ).order_by(AgentMessage.created_at).all()

                if messages:
                    # 格式化消息为chatbot格式
                    formatted_history = format_conversation_history(messages)
                    logger.info(f"恢复了对话 {conversation.id} 的 {len(messages)} 条消息")
                    return formatted_history
                else:
                    return []

        except Exception as e:
            logger.error(f"恢复对话记录失败: {e}")
            return []

    def get_conversation_list_for_selection(self, plan_id: int):
        """
        获取对话列表用于选择恢复

        Returns:
            gr.Dropdown choices 格式的对话列表
        """
        try:
            from database.models import AgentConversation, AgentMessage
            from database.db import get_db
            from database.models import now_beijing
            from sqlalchemy import desc, func

            with get_db() as db:
                conversations = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(desc(AgentConversation.last_message_at)).limit(10).all()

                choices = []
                for conv in conversations:
                    # 统计消息数量
                    message_count = db.query(func.count(AgentMessage.id)).filter(
                        AgentMessage.conversation_id == conv.id
                    ).scalar()

                    # 获取最新消息预览
                    latest_message = db.query(AgentMessage).filter(
                        AgentMessage.conversation_id == conv.id
                    ).order_by(desc(AgentMessage.created_at)).first()

                    preview = ""
                    if latest_message:
                        content = latest_message.content or ""
                        preview = content[:30] + "..." if len(content) > 30 else content

                    # 格式化选择项
                    status_emoji = {
                        'active': '💬',
                        'completed': '✅',
                        'error': '❌',
                        'paused': '⏸️'
                    }.get(conv.status, '💬')

                    time_str = conv.last_message_at.strftime("%m-%d %H:%M") if conv.last_message_at else "N/A"
                    label = f"{status_emoji} {time_str} | {message_count}条消息 | {preview}"

                    choices.append((label, conv.id))

                # 添加默认选项
                if not choices:
                    choices = [("无对话记录", None)]

                return choices

        except Exception as e:
            logger.error(f"获取对话列表失败: {e}")
            return [("无对话记录", None)]

    def restore_selected_conversation(self, plan_id: int, selected_conversation_id):
        """
        恢复选择的对话

        Args:
            plan_id: 计划ID
            selected_conversation_id: 选择的对话ID
        """
        try:
            logger.info(f"开始恢复对话: plan_id={plan_id}, conversation_id={selected_conversation_id}")

            if not selected_conversation_id:
                logger.warning("选择的对话ID为空")
                return []  # 返回空历史

            restored_history = self.restore_conversation_from_records(plan_id, selected_conversation_id)
            logger.info(f"成功恢复对话，包含 {len(restored_history)} 条消息")
            return restored_history

        except Exception as e:
            logger.error(f"恢复对话失败: {e}")
            import traceback
            logger.error(f"恢复对话失败详情: {traceback.format_exc()}")
            # 返回错误消息而不是空历史
            return [{"role": "assistant", "content": f"❌ 恢复对话失败: {str(e)}"}]