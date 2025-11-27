"""
增强的推理服务 - 修复版本
解决thinking模式文本重叠和流式输出问题
"""
import json
import asyncio
from typing import Dict, List, AsyncGenerator, Optional
from database.models import TradingPlan, PredictionData
from database.db import get_db
from utils.logger import setup_logger
from services.enhanced_conversation_service import enhanced_conversation_service, ConversationType
from services.enhanced_agent_stream_service import enhanced_agent_stream_service
from services.kline_event_service import kline_event_service

logger = setup_logger(__name__, "enhanced_inference_fixed.log")


class EnhancedInferenceService:
    """增强的推理服务 - 修复版本"""

    @classmethod
    async def execute_manual_inference(cls, plan_id: int) -> AsyncGenerator[List[Dict], None]:
        """
        执行手动推理 - 正确的流式输出，避免文本重叠

        Args:
            plan_id: 计划ID

        Yields:
            Chatbot消息列表 - 增量更新，分离thinking和正文
        """
        try:
            # 1. 初始化对话（重置上下文）
            conversation_id = await enhanced_agent_stream_service.initialize_conversation(
                plan_id=plan_id,
                conversation_type=ConversationType.AUTO_INFERENCE,
                reset_context=True
            )

            # 2. 添加预测数据消息
            prediction_success = await enhanced_agent_stream_service.add_prediction_data_message(
                conversation_id=conversation_id,
                plan_id=plan_id,
                trigger_event="manual_inference"
            )

            if not prediction_success:
                yield [{"role": "assistant", "content": "❌ 没有可用的预测数据，请先完成模型训练"}]
                return

            # 3. 获取当前对话状态用于显示（完整上下文）
            current_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
            formatted_messages = enhanced_conversation_service.format_for_chatbot(current_messages)

            # 立即返回完整上下文（包括系统提示词和预测数据）
            yield formatted_messages

            # 4. 开始AI分析对话 - 重新设计为增量流式输出
            thinking_buffer = ""      # thinking内容缓冲区
            analysis_buffer = ""      # 分析内容缓冲区
            last_sent_thinking = ""   # 上次发送的thinking内容
            last_sent_analysis = ""   # 上次发送的分析内容
            thinking_complete = False # 标记thinking是否完成

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message="",  # 空消息，让AI基于预测数据进行分析
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")

                    should_update = False

                    if chunk_type == "thinking":
                        # 累积thinking内容
                        new_thinking = chunk_data.get("content", "")
                        thinking_buffer += new_thinking
                        should_update = True

                    elif chunk_type == "content":
                        # 累积分析内容
                        new_content = chunk_data.get("content", "")
                        analysis_buffer += new_content
                        should_update = True
                        thinking_complete = True  # 开始正文意味着thinking完成

                    elif chunk_type == "tool_call":
                        tool_name = chunk_data.get("tool_name", "")
                        arguments = chunk_data.get("arguments", {})
                        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."
                        analysis_buffer += tool_section
                        should_update = True

                    elif chunk_type == "tool_result":
                        tool_name = chunk_data.get("tool_name", "")
                        result = chunk_data.get("result", {})
                        success = result.get("success", False)
                        status_emoji = "✅" if success else "❌"

                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续分析..."
                        analysis_buffer += tool_section
                        should_update = True

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("content", "未知错误")
                        error_section = f"\n\n❌ **推理错误**: {error_msg}"
                        analysis_buffer += error_section
                        should_update = True

                    # 只有内容更新且内容确实发生变化时才生成新消息
                    if should_update:
                        thinking_changed = thinking_buffer != last_sent_thinking
                        analysis_changed = analysis_buffer != last_sent_analysis

                        if thinking_changed or analysis_changed:
                            # 构建增量消息内容 - 分离thinking和正文
                            content_parts = []

                            # 添加thinking部分（如果有）
                            if thinking_buffer:
                                content_parts.append(f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_buffer}\n</details>")

                            # 添加分析部分（如果有）
                            if analysis_buffer:
                                if content_parts:  # 如果已有thinking，添加分隔符
                                    content_parts.append("\n\n---\n\n")
                                content_parts.append(analysis_buffer)

                            if content_parts:
                                content_update = "".join(content_parts)

                                # 创建新的assistant消息
                                new_assistant_message = {
                                    "role": "assistant",
                                    "content": content_update,
                                    "metadata": {
                                        "streaming": True,
                                        "has_thinking": bool(thinking_buffer),
                                        "thinking_completed": thinking_complete,
                                        "incremental": True,  # 标记为增量更新
                                        "chunk_type": chunk_type
                                    }
                                }

                                # 返回历史消息 + 新的增量消息
                                response_messages = formatted_messages + [new_assistant_message]
                                yield response_messages

                                # 更新最后发送的内容记录
                                last_sent_thinking = thinking_buffer
                                last_sent_analysis = analysis_buffer

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理推理块失败: {e}")
                    continue

            # 5. 推理完成 - 发送最终完成消息
            final_content_parts = []

            if thinking_buffer:
                final_content_parts.append(f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_buffer}\n\n✅ 思考完成\n</details>")

            if analysis_buffer:
                if final_content_parts:
                    final_content_parts.append("\n\n---\n\n")
                final_content_parts.append(analysis_buffer + "\n\n✅ **推理完成**")

            if final_content_parts:
                final_content = "".join(final_content_parts)
                final_message = {
                    "role": "assistant",
                    "content": final_content,
                    "metadata": {
                        "completed": True,
                        "final": True,
                        "has_thinking": bool(thinking_buffer),
                        "thinking_completed": True,
                        "incremental": False  # 最终消息不是增量更新
                    }
                }

                final_response = formatted_messages + [final_message]
                yield final_response

        except Exception as e:
            logger.error(f"执行手动推理失败: {e}")
            import traceback
            traceback.print_exc()

            yield [{"role": "assistant", "content": f"❌ 推理过程出错: {str(e)}"}]

    @classmethod
    async def continue_conversation(
        cls,
        plan_id: int,
        user_message: str,
        conversation_type: ConversationType = ConversationType.MANUAL_CHAT
    ) -> AsyncGenerator[List[Dict], None]:
        """
        继续对话 - 修复版本
        """
        try:
            # 获取或创建对话会话
            conversation_id = await enhanced_agent_stream_service.initialize_conversation(
                plan_id=plan_id,
                conversation_type=conversation_type,
                reset_context=False
            )

            # 获取对话历史
            current_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
            formatted_messages = enhanced_conversation_service.format_for_chatbot(current_messages)

            # 立即返回历史消息
            yield formatted_messages

            # 开始流式对话
            thinking_buffer = ""
            analysis_buffer = ""
            last_sent_thinking = ""
            last_sent_analysis = ""
            thinking_complete = False

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message=user_message,
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")

                    should_update = False

                    if chunk_type == "thinking":
                        new_thinking = chunk_data.get("content", "")
                        thinking_buffer += new_thinking
                        should_update = True

                    elif chunk_type == "content":
                        new_content = chunk_data.get("content", "")
                        analysis_buffer += new_content
                        should_update = True
                        thinking_complete = True

                    elif chunk_type in ["tool_call", "tool_result", "error"]:
                        # 处理工具相关和错误消息
                        if chunk_type == "tool_call":
                            tool_name = chunk_data.get("tool_name", "")
                            arguments = chunk_data.get("arguments", {})
                            args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                            tool_section = f"\n\n🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."
                        elif chunk_type == "tool_result":
                            tool_name = chunk_data.get("tool_name", "")
                            result = chunk_data.get("result", {})
                            success = result.get("success", False)
                            status_emoji = "✅" if success else "❌"
                            result_str = json.dumps(result, indent=2, ensure_ascii=False)
                            tool_section = f"\n\n🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续分析..."
                        else:  # error
                            error_msg = chunk_data.get("content", "未知错误")
                            tool_section = f"\n\n❌ **推理错误**: {error_msg}"

                        analysis_buffer += tool_section
                        should_update = True

                    if should_update:
                        thinking_changed = thinking_buffer != last_sent_thinking
                        analysis_changed = analysis_buffer != last_sent_analysis

                        if thinking_changed or analysis_changed:
                            content_parts = []

                            if thinking_buffer:
                                content_parts.append(f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_buffer}\n</details>")

                            if analysis_buffer:
                                if content_parts:
                                    content_parts.append("\n\n---\n\n")
                                content_parts.append(analysis_buffer)

                            if content_parts:
                                content_update = "".join(content_parts)

                                new_assistant_message = {
                                    "role": "assistant",
                                    "content": content_update,
                                    "metadata": {
                                        "streaming": True,
                                        "has_thinking": bool(thinking_buffer),
                                        "thinking_completed": thinking_complete,
                                        "incremental": True,
                                        "chunk_type": chunk_type
                                    }
                                }

                                response_messages = formatted_messages + [new_assistant_message]
                                yield response_messages

                                last_sent_thinking = thinking_buffer
                                last_sent_analysis = analysis_buffer

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理对话块失败: {e}")
                    continue

            # 对话完成
            final_content_parts = []

            if thinking_buffer:
                final_content_parts.append(f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_buffer}\n\n✅ 思考完成\n</details>")

            if analysis_buffer:
                if final_content_parts:
                    final_content_parts.append("\n\n---\n\n")
                final_content_parts.append(analysis_buffer + "\n\n✅ **对话完成**")

            if final_content_parts:
                final_content = "".join(final_content_parts)
                final_message = {
                    "role": "assistant",
                    "content": final_content,
                    "metadata": {
                        "completed": True,
                        "final": True,
                        "has_thinking": bool(thinking_buffer),
                        "thinking_completed": True,
                        "incremental": False
                    }
                }

                final_response = formatted_messages + [final_message]
                yield final_response

        except Exception as e:
            logger.error(f"继续对话失败: {e}")
            import traceback
            traceback.print_exc()

            yield [{"role": "assistant", "content": f"❌ 对话过程出错: {str(e)}"}]

    @classmethod
    async def handle_kline_event_trigger(cls, plan_id: int, inst_id: str, kline_data: dict):
        """
        处理K线事件触发
        """
        try:
            # 获取或创建K线事件对话
            conversation_id = await enhanced_agent_stream_service.initialize_conversation(
                plan_id=plan_id,
                conversation_type=ConversationType.KLINE_EVENT,
                reset_context=False  # 不重置上下文，继续之前的对话
            )

            # 构建事件消息
            event_message = f"""🔔 **新K线数据事件**

**交易对**: {inst_id}
**更新时间**: {kline_data.get('timestamp', 'N/A')}
**收盘价**: {kline_data.get('close', 0)}
**成交量**: {kline_data.get('volume', 0)}

请基于最新市场数据更新分析并考虑是否需要调整交易策略。"""

            # 自动继续对话（基于事件数据）
            await enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message=event_message,
                use_thinking_mode=True
            )

            logger.info(f"K线事件触发对话完成: plan_id={plan_id}, conversation_id={conversation_id}")

        except Exception as e:
            logger.error(f"处理K线事件触发失败: {e}")

    @classmethod
    def get_latest_prediction_data(cls, plan_id: int) -> Optional[PredictionData]:
        """获取最新的预测数据"""
        try:
            with get_db() as db:
                return db.query(PredictionData).filter(
                    PredictionData.plan_id == plan_id,
                    PredictionData.status == "success"
                ).order_by(PredictionData.created_at.desc()).first()

        except Exception as e:
            logger.error(f"获取最新预测数据失败: {e}")
            return None


# 创建全局实例
enhanced_inference_service = EnhancedInferenceService()