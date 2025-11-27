"""
增强的推理服务 V2 - 完全重构版本
彻底解决thinking和文本混合问题
"""
import json
import asyncio
from typing import Dict, List, AsyncGenerator, Optional
from database.models import TradingPlan, PredictionData
from database.db import get_db
from utils.logger import setup_logger
from services.enhanced_conversation_service import enhanced_conversation_service, ConversationType
from services.enhanced_agent_stream_service import enhanced_agent_stream_service

logger = setup_logger(__name__, "enhanced_inference_v2.log")


class EnhancedInferenceServiceV2:
    """增强的推理服务 - V2 完全重构版本"""

    @classmethod
    async def execute_manual_inference(cls, plan_id: int) -> AsyncGenerator[List[Dict], None]:
        """
        执行手动推理 - V2版本，完全分离thinking和正文

        Args:
            plan_id: 计划ID

        Yields:
            Chatbot消息列表 - 每次只更新一个部分，避免混合
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

            # 4. 开始AI分析对话 - 完全重构的流式处理
            thinking_complete = False
            analysis_complete = False

            # 状态跟踪
            current_thinking_display = ""
            current_analysis_display = ""
            last_sent_message = None

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message="",  # 空消息，让AI基于预测数据进行分析
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")
                    chunk_content = chunk_data.get("content", "")

                    should_send_update = False
                    new_message = None

                    if chunk_type == "thinking":
                        # 处理thinking增量
                        current_thinking_display += chunk_content
                        thinking_complete = False

                        # 只发送thinking部分，不包含analysis
                        new_message = {
                            "role": "assistant",
                            "content": cls._format_thinking_section(current_thinking_display),
                            "metadata": {
                                "streaming": True,
                                "section": "thinking",
                                "complete": False,
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    elif chunk_type == "content":
                        # 处理analysis增量
                        current_analysis_display += chunk_content
                        analysis_complete = False
                        thinking_complete = True  # 开始analysis意味着thinking可能完成了

                        # 构建完整显示：thinking + analysis
                        display_content = cls._build_combined_display(
                            current_thinking_display,
                            current_analysis_display
                        )

                        new_message = {
                            "role": "assistant",
                            "content": display_content,
                            "metadata": {
                                "streaming": True,
                                "section": "combined",
                                "thinking_complete": thinking_complete,
                                "analysis_complete": False,
                                "has_thinking": bool(current_thinking_display),
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    elif chunk_type in ["tool_call", "tool_result", "error"]:
                        # 处理工具相关消息
                        tool_content = cls._format_tool_message(chunk_data)
                        current_analysis_display += tool_content

                        # 重新构建完整显示
                        display_content = cls._build_combined_display(
                            current_thinking_display,
                            current_analysis_display
                        )

                        new_message = {
                            "role": "assistant",
                            "content": display_content,
                            "metadata": {
                                "streaming": True,
                                "section": "tool_update",
                                "tool_type": chunk_type,
                                "thinking_complete": thinking_complete,
                                "analysis_complete": False,
                                "has_thinking": bool(current_thinking_display),
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    # 只有当需要更新且内容确实发生变化时才发送
                    if should_send_update and new_message:
                        # 避免发送重复内容
                        if not last_sent_message or new_message["content"] != last_sent_message["content"]:
                            response_messages = formatted_messages + [new_message]
                            yield response_messages
                            last_sent_message = new_message

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理推理块失败: {e}")
                    continue

            # 5. 推理完成 - 发送最终完成消息
            thinking_complete = True
            analysis_complete = True

            # 构建最终完整显示
            final_content = cls._build_final_display(
                current_thinking_display,
                current_analysis_display
            )

            final_message = {
                "role": "assistant",
                "content": final_content,
                "metadata": {
                    "completed": True,
                    "final": True,
                    "thinking_complete": True,
                    "analysis_complete": True,
                    "has_thinking": bool(current_thinking_display),
                    "has_analysis": bool(current_analysis_display)
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
    def _format_thinking_section(cls, thinking_content: str) -> str:
        """格式化thinking部分，使用折叠显示"""
        if not thinking_content.strip():
            return ""

        return f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_content}\n</details>"

    @classmethod
    def _build_combined_display(cls, thinking_content: str, analysis_content: str) -> str:
        """构建组合显示：thinking + analysis"""
        parts = []

        # 添加thinking部分（如果有）
        if thinking_content.strip():
            parts.append(cls._format_thinking_section(thinking_content))

        # 添加analysis部分（如果有）
        if analysis_content.strip():
            if parts:  # 如果已有thinking，添加分隔符
                parts.append("\n\n---\n\n")
            parts.append(analysis_content)

        # 如果都没有内容，显示占位符
        if not parts:
            return "🤔 AI正在思考中..."

        return "".join(parts)

    @classmethod
    def _build_final_display(cls, thinking_content: str, analysis_content: str) -> str:
        """构建最终完成显示"""
        parts = []

        # 添加thinking部分（如果有）
        if thinking_content.strip():
            parts.append(f"<details>\n<summary>🧠 AI思考过程</summary>\n\n{thinking_content}\n\n✅ 思考完成\n</details>")

        # 添加analysis部分（如果有）
        if analysis_content.strip():
            if parts:
                parts.append("\n\n---\n\n")
            parts.append(analysis_content + "\n\n✅ **推理完成**")

        # 如果都没有内容，显示完成消息
        if not parts:
            return "✅ 推理完成"

        return "".join(parts)

    @classmethod
    def _format_tool_message(cls, chunk_data: Dict) -> str:
        """格式化工具消息"""
        chunk_type = chunk_data.get("type", "")

        if chunk_type == "tool_call":
            tool_name = chunk_data.get("tool_name", "未知工具")
            arguments = chunk_data.get("arguments", {})
            args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
            return f"\n\n🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."

        elif chunk_type == "tool_result":
            tool_name = chunk_data.get("tool_name", "未知工具")
            result = chunk_data.get("result", {})
            success = result.get("success", False)
            status_emoji = "✅" if success else "❌"
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            return f"\n\n🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续分析..."

        elif chunk_type == "error":
            error_msg = chunk_data.get("content", "未知错误")
            return f"\n\n❌ **推理错误**: {error_msg}"

        return ""

    @classmethod
    async def continue_conversation(
        cls,
        plan_id: int,
        user_message: str,
        conversation_type: ConversationType = ConversationType.MANUAL_CHAT
    ) -> AsyncGenerator[List[Dict], None]:
        """
        继续对话 - V2版本，完全分离thinking和正文
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

            # 开始流式对话 - 使用相同的分离逻辑
            thinking_complete = False
            analysis_complete = False
            current_thinking_display = ""
            current_analysis_display = ""
            last_sent_message = None

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message=user_message,
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")
                    chunk_content = chunk_data.get("content", "")

                    should_send_update = False
                    new_message = None

                    if chunk_type == "thinking":
                        current_thinking_display += chunk_content
                        thinking_complete = False

                        new_message = {
                            "role": "assistant",
                            "content": cls._format_thinking_section(current_thinking_display),
                            "metadata": {
                                "streaming": True,
                                "section": "thinking",
                                "complete": False,
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    elif chunk_type == "content":
                        current_analysis_display += chunk_content
                        analysis_complete = False
                        thinking_complete = True

                        display_content = cls._build_combined_display(
                            current_thinking_display,
                            current_analysis_display
                        )

                        new_message = {
                            "role": "assistant",
                            "content": display_content,
                            "metadata": {
                                "streaming": True,
                                "section": "combined",
                                "thinking_complete": thinking_complete,
                                "analysis_complete": False,
                                "has_thinking": bool(current_thinking_display),
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    elif chunk_type in ["tool_call", "tool_result", "error"]:
                        tool_content = cls._format_tool_message(chunk_data)
                        current_analysis_display += tool_content

                        display_content = cls._build_combined_display(
                            current_thinking_display,
                            current_analysis_display
                        )

                        new_message = {
                            "role": "assistant",
                            "content": display_content,
                            "metadata": {
                                "streaming": True,
                                "section": "tool_update",
                                "tool_type": chunk_type,
                                "thinking_complete": thinking_complete,
                                "analysis_complete": False,
                                "has_thinking": bool(current_thinking_display),
                                "has_analysis": bool(current_analysis_display)
                            }
                        }
                        should_send_update = True

                    if should_send_update and new_message:
                        if not last_sent_message or new_message["content"] != last_sent_message["content"]:
                            response_messages = formatted_messages + [new_message]
                            yield response_messages
                            last_sent_message = new_message

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理对话块失败: {e}")
                    continue

            # 对话完成
            thinking_complete = True
            analysis_complete = True

            final_content = cls._build_final_display(
                current_thinking_display,
                current_analysis_display
            )

            final_message = {
                "role": "assistant",
                "content": final_content,
                "metadata": {
                    "completed": True,
                    "final": True,
                    "thinking_complete": True,
                    "analysis_complete": True,
                    "has_thinking": bool(current_thinking_display),
                    "has_analysis": bool(current_analysis_display)
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
enhanced_inference_service = EnhancedInferenceServiceV2()