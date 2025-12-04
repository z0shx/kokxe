"""
计划详情页 AI Agent 对话界面模块
"""
import asyncio
import gradio as gr
from utils.logger import setup_logger
from ui.custom_chatbot import create_custom_chatbot, process_streaming_messages

logger = setup_logger(__name__, "plan_detail_chat_ui.log")


class PlanDetailChatUI:
    """计划详情页 AI Agent 对话 UI"""

    def __init__(self, plan_detail_ui):
        self.plan_detail_ui = plan_detail_ui

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

    def _get_latest_prediction_csv_data(self, plan_id: int) -> str:
        """获取最新预测数据的CSV格式文本"""
        try:
            from database.models import TrainingRecord, PredictionData
            from database.db import get_db
            from sqlalchemy import desc, and_

            # 获取最新的有预测数据的已完成训练记录
            with get_db() as db:
                completed_trainings = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed'
                ).order_by(desc(TrainingRecord.created_at)).all()

                # 找到第一个有预测数据的训练记录
                latest_training_with_pred = None
                for training in completed_trainings:
                    pred_count = db.query(PredictionData).filter(
                        PredictionData.training_record_id == training.id
                    ).count()
                    if pred_count > 0:
                        latest_training_with_pred = training
                        break

                if not latest_training_with_pred:
                    return None

                latest_training = latest_training_with_pred

                # 获取该训练记录的最新推理批次的预测数据
                latest_batch = db.query(PredictionData.inference_batch_id).filter(
                    PredictionData.training_record_id == latest_training.id
                ).order_by(desc(PredictionData.created_at)).first()

                if not latest_batch:
                    return None

                # 获取最新批次的预测数据（按时间排序）
                predictions_query = db.query(PredictionData).filter(
                    and_(
                        PredictionData.training_record_id == latest_training.id,
                        PredictionData.inference_batch_id == latest_batch.inference_batch_id
                    )
                ).order_by(PredictionData.timestamp.asc()).all()

                if not predictions_query:
                    return None

                # 转换为字典格式
                predictions = []
                for pred in predictions_query:
                    predictions.append({
                        'timestamp': pred.timestamp,
                        'open': pred.open,
                        'high': pred.high,
                        'low': pred.low,
                        'close': pred.close,
                        'volume': pred.volume or 0,
                        'amount': pred.amount or 0,
                        'upward_probability': pred.upward_probability or 0,
                        'volatility_amplification_probability': pred.volatility_amplification_probability or 0,
                        'close_min': pred.close_min,
                        'close_max': pred.close_max
                    })

            # 构建CSV格式的预测数据
            csv_lines = []

            # CSV头部
            csv_lines.append("timestamp,open,high,low,close,volume,amount,upward_probability,volatility_amplification_probability")

            # 数据行
            for pred in predictions:
                timestamp_str = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                upward_prob = pred.get('upward_probability', 0) * 100
                vol_prob = pred.get('volatility_amplification_probability', 0) * 100

                csv_lines.append(
                    f"{timestamp_str},{pred['open']:.2f},{pred['high']:.2f},"
                    f"{pred['low']:.2f},{pred['close']:.2f},"
                    f"{pred.get('volume', 0):.2f},{pred.get('amount', 0):.2f},"
                    f"{upward_prob:.2f}%,{vol_prob:.2f}%"
                )

            return "\n".join(csv_lines)

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return None

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
                with gr.Row():
                    agent_send_btn = gr.Button("📤 发送", variant="primary", size="sm")
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
            'agent_execute_inference_btn': agent_execute_inference_btn,
            'agent_clear_btn': agent_clear_btn,
            'agent_status': agent_status
        })

        # 定义简化的同步事件处理函数
        def agent_send_message_wrapper(pid, user_message, history):
            """发送消息给AI Agent（流式版本）"""
            from utils.common import validate_plan_exists

            is_valid, plan_id, error_msg = validate_plan_exists(pid)

            if not is_valid:
                return history, gr.update(value=""), gr.update(visible=True, value=f"❌ {error_msg}")

            if not user_message or not user_message.strip():
                return history, gr.update(value=""), gr.update(visible=True, value=f"❌ 请输入消息内容")

            try:
                # 调用真实的 Agent 服务进行流式对话
                from services.langchain_agent import agent_service

                # 创建同步消息收集器
                def collect_messages():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def async_collect():
                        messages = []
                        async for message_batch in agent_service.stream_conversation(
                            plan_id=plan_id,
                            user_message=user_message.strip(),
                            conversation_type="user_chat"
                        ):
                            # 使用增强的消息处理
                            processed = process_streaming_messages([message_batch])
                            messages.extend(processed)
                        return messages

                    try:
                        return loop.run_until_complete(async_collect())
                    finally:
                        loop.close()

                # 收集所有消息用于显示
                messages = collect_messages()

                return history + messages, gr.update(value=""), gr.update(visible=False, value="")

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 发送失败: {str(e)}"}]
                return history + error_message, gr.update(value=""), gr.update(visible=False, value="")

        def agent_execute_inference_wrapper(pid, history):
            """执行AI Agent推理（增强版本 - 包含历史数据）"""
            from utils.common import validate_plan_exists
            from services.historical_data_service import historical_data_service

            is_valid, plan_id, error_msg = validate_plan_exists(pid)

            if not is_valid:
                return history + [{"role": "assistant", "content": f"❌ {error_msg}"}], gr.update(visible=True, value=f"❌ {error_msg}")

            try:
                # 获取历史K线数据
                historical_data = historical_data_service.get_optimal_historical_data(plan_id)
                if not historical_data:
                    return history + [{"role": "assistant", "content": "❌ 未找到可用的历史K线数据"}], gr.update(visible=False, value="")

                # 获取最新预测数据
                prediction_data = self._get_latest_prediction_csv_data(plan_id)

                if not prediction_data:
                    error_message = [{"role": "assistant", "content": "❌ 未找到可用的预测数据，请先执行模型推理"}]
                    return history + error_message, gr.update(visible=False, value="")

                # 构建包含历史和预测数据的完整分析请求
                analysis_request = f"""请基于以下数据进行分析和决策：

【最近24小时K线数据】
{historical_data}

【最新预测交易数据】
{prediction_data}

请综合分析：
1. 历史价格趋势和模式
2. 预测数据的可信度和趋势
3. 交易机会识别和风险评估
4. 具体交易建议和执行策略

如需执行交易操作，请使用相应的工具。"""

                # 创建同步消息收集器
                def collect_inference_messages():
                    from services.langchain_agent import agent_service
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def async_collect():
                        messages = []
                        async for message_batch in agent_service.stream_conversation(
                            plan_id=plan_id,
                            user_message=analysis_request,
                            conversation_type="auto_inference"
                        ):
                            # 使用增强的消息处理
                            processed = process_streaming_messages([message_batch])
                            messages.extend(processed)
                        return messages

                    try:
                        return loop.run_until_complete(async_collect())
                    finally:
                        loop.close()

                # 收集所有消息用于显示
                messages = collect_inference_messages()

                return history + messages, gr.update(visible=False, value="")

            except Exception as e:
                logger.error(f"执行推理失败: {e}")
                error_message = [{"role": "assistant", "content": f"❌ 执行推理失败: {str(e)}"}]
                return history + error_message, gr.update(visible=False, value="")

        # 保存事件处理函数
        components.update({
            'agent_send_message_wrapper': agent_send_message_wrapper,
            'agent_execute_inference_wrapper': agent_execute_inference_wrapper,
            'agent_clear_conversation_wrapper': self.agent_clear_conversation_wrapper
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
            outputs=[components['agent_chatbot'], components['agent_user_input'], components['agent_status']],
            show_progress=True
        )

        components['agent_execute_inference_btn'].click(
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