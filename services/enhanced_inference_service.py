"""
增强的推理服务
重构版：使用新的对话管理和流式服务
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

logger = setup_logger(__name__, "enhanced_inference.log")


class EnhancedInferenceService:
    """增强的推理服务"""

    @classmethod
    async def execute_manual_inference(cls, plan_id: int) -> AsyncGenerator[List[Dict], None]:
        """
        执行手动推理

        Args:
            plan_id: 计划ID

        Yields:
            Chatbot消息列表
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

            # 4. 开始AI分析对话
            thinking_content = ""      # 分离thinking内容
            analysis_content = ""      # 分离正文分析内容
            chunk_count = 0
            thinking_completed = False  # 标记thinking是否完成

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message="",  # 空消息，让AI基于预测数据进行分析
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")
                    content = chunk_data.get("content", "")
                    chunk_count = chunk_data.get("chunk_count", 0)

                    # 分别处理thinking和正文内容
                    if chunk_type == "thinking_start":
                        thinking_content = "🧠 **AI思考过程**\n\n"
                        thinking_completed = False

                    elif chunk_type == "thinking":
                        # 累积thinking内容
                        thinking_content += content
                        thinking_completed = False

                    elif chunk_type == "content":
                        # 正文开始，标记thinking完成
                        if not thinking_completed:
                            thinking_completed = True

                        # 累积正文内容
                        if analysis_content:
                            # 如果已有内容，添加分隔符
                            analysis_content += "\n\n" + content
                        else:
                            analysis_content = content

                    elif chunk_type == "tool_call_start":
                        thinking_completed = True  # 工具调用开始时thinking应该完成
                        tool_name = chunk_data.get("tool_name", "")
                        tool_section = f"\n\n🛠️ **调用工具**: `{tool_name}`\n\n⏳ 正在执行..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "tool_call":
                        thinking_completed = True
                        tool_name = chunk_data.get("tool_name", "")
                        arguments = chunk_data.get("arguments", {})
                        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "tool_result":
                        tool_name = chunk_data.get("tool_name", "")
                        result = chunk_data.get("result", {})
                        success = result.get("success", False)
                        status_emoji = "✅" if success else "❌"

                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续分析..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("content", "未知错误")
                        error_section = f"\n\n❌ **推理错误**: {error_msg}"

                        if analysis_content:
                            analysis_content += error_section
                        else:
                            analysis_content = error_section

                    # 构建组合内容：thinking（如果存在） + 正文
                    combined_content = ""
                    message_metadata = {
                        "streaming": True,
                        "chunk_count": chunk_count,
                        "has_thinking": bool(thinking_content),
                        "thinking_completed": thinking_completed
                    }

                    if thinking_content:
                        combined_content = thinking_content
                        # 添加thinking部分的折叠元数据
                        message_metadata.update({
                            "collapsible_sections": [{
                                "type": "thinking",
                                "default_collapsed": True,  # thinking部分默认折叠
                                "title": "🧠 AI思考过程",
                                "completed": thinking_completed
                            }]
                        })

                    if analysis_content:
                        if combined_content:
                            # 在thinking和正文之间添加分隔线
                            combined_content += "\n\n---\n\n**分析结果**\n\n" + analysis_content
                        else:
                            combined_content = analysis_content

                    # 构建完整消息列表：系统上下文 + AI回复
                    assistant_message = {
                        "role": "assistant",
                        "content": combined_content,
                        "metadata": message_metadata
                    }

                    complete_messages = formatted_messages + [assistant_message]
                    yield complete_messages

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理推理块失败: {e}")
                    continue

            # 推理完成 - 最终整理
            thinking_completed = True
            final_content = ""

            if thinking_content:
                final_content = thinking_content

            if analysis_content:
                if final_content:
                    final_content += "\n\n---\n\n**最终分析**\n\n" + analysis_content + "\n\n✅ **推理完成**"
                else:
                    final_content = analysis_content + "\n\n✅ **推理完成**"

            # 构建最终消息
            final_metadata = {
                "completed": True,
                "final": True,
                "has_thinking": bool(thinking_content),
                "thinking_completed": True
            }

            if thinking_content:
                # 添加thinking部分的折叠元数据
                final_metadata.update({
                    "collapsible_sections": [{
                        "type": "thinking",
                        "default_collapsed": True,  # thinking部分默认折叠
                        "title": "🧠 AI思考过程",
                        "completed": True
                    }]
                })

            final_message = {
                "role": "assistant",
                "content": final_content,
                "metadata": final_metadata
            }

            final_complete_messages = formatted_messages + [final_message]
            yield final_complete_messages

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
        继续对话

        Args:
            plan_id: 计划ID
            user_message: 用户消息
            conversation_type: 对话类型

        Yields:
            Chatbot消息列表
        """
        try:
            # 获取或创建对话（不重置上下文）
            conversation_id = await enhanced_agent_stream_service.initialize_conversation(
                plan_id=plan_id,
                conversation_type=conversation_type,
                reset_context=False
            )

            # 获取当前对话状态（完整上下文）
            current_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
            formatted_messages = enhanced_conversation_service.format_for_chatbot(current_messages)

            # 添加用户消息到显示
            messages_with_user = formatted_messages + [
                {"role": "user", "content": user_message}
            ]

            yield messages_with_user

            # 开始AI回复
            thinking_content = ""      # 分离thinking内容
            analysis_content = ""      # 分离正文分析内容
            chunk_count = 0
            thinking_completed = False  # 标记thinking是否完成

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message=user_message,
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")
                    content = chunk_data.get("content", "")
                    chunk_count = chunk_data.get("chunk_count", 0)

                    # 分别处理thinking和正文内容
                    if chunk_type == "thinking_start":
                        thinking_content = "🧠 **AI思考过程**\n\n"
                        thinking_completed = False

                    elif chunk_type == "thinking":
                        # 累积thinking内容
                        thinking_content += content
                        thinking_completed = False

                    elif chunk_type == "content":
                        # 正文开始，标记thinking完成
                        if not thinking_completed:
                            thinking_completed = True

                        # 累积正文内容
                        if analysis_content:
                            analysis_content += "\n\n" + content
                        else:
                            analysis_content = content

                    elif chunk_type == "tool_call_start":
                        thinking_completed = True
                        tool_name = chunk_data.get("tool_name", "")
                        tool_section = f"\n\n🛠️ **调用工具**: `{tool_name}`\n\n⏳ 正在执行..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "tool_call":
                        thinking_completed = True
                        tool_name = chunk_data.get("tool_name", "")
                        arguments = chunk_data.get("arguments", {})
                        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "tool_result":
                        tool_name = chunk_data.get("tool_name", "")
                        result = chunk_data.get("result", {})
                        success = result.get("success", False)
                        status_emoji = "✅" if success else "❌"

                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        tool_section = f"\n\n🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续对话..."

                        if analysis_content:
                            analysis_content += tool_section
                        else:
                            analysis_content = tool_section

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("content", "未知错误")
                        error_section = f"\n\n❌ **回复错误**: {error_msg}"

                        if analysis_content:
                            analysis_content += error_section
                        else:
                            analysis_content = error_section

                    # 构建组合内容：thinking（如果存在） + 正文
                    combined_content = ""
                    message_metadata = {
                        "streaming": True,
                        "chunk_count": chunk_count,
                        "has_thinking": bool(thinking_content),
                        "thinking_completed": thinking_completed
                    }

                    if thinking_content:
                        combined_content = thinking_content
                        # 添加thinking部分的折叠元数据
                        message_metadata.update({
                            "collapsible_sections": [{
                                "type": "thinking",
                                "default_collapsed": True,  # thinking部分默认折叠
                                "title": "🧠 AI思考过程",
                                "completed": thinking_completed
                            }]
                        })

                    if analysis_content:
                        if combined_content:
                            combined_content += "\n\n---\n\n**回复内容**\n\n" + analysis_content
                        else:
                            combined_content = analysis_content

                    # 构建完整消息列表：历史消息 + 用户消息 + AI回复
                    assistant_message = {
                        "role": "assistant",
                        "content": combined_content,
                        "metadata": message_metadata
                    }

                    complete_messages = messages_with_user + [assistant_message]
                    yield complete_messages

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理对话块失败: {e}")
                    continue

            # 对话完成 - 最终整理
            thinking_completed = True
            final_content = ""

            if thinking_content:
                final_content = thinking_content

            if analysis_content:
                if final_content:
                    final_content += "\n\n---\n\n**最终回复**\n\n" + analysis_content + "\n\n✅ **回复完成**"
                else:
                    final_content = analysis_content + "\n\n✅ **回复完成**"

            # 构建最终消息
            final_metadata = {
                "completed": True,
                "final": True,
                "has_thinking": bool(thinking_content),
                "thinking_completed": True
            }

            if thinking_content:
                # 添加thinking部分的折叠元数据
                final_metadata.update({
                    "collapsible_sections": [{
                        "type": "thinking",
                        "default_collapsed": True,  # thinking部分默认折叠
                        "title": "🧠 AI思考过程",
                        "completed": True
                    }]
                })

            final_message = {
                "role": "assistant",
                "content": final_content,
                "metadata": final_metadata
            }

            final_complete_messages = messages_with_user + [final_message]
            yield final_complete_messages

        except Exception as e:
            logger.error(f"继续对话失败: {e}")
            import traceback
            traceback.print_exc()

            yield [{"role": "assistant", "content": f"❌ 对话过程出错: {str(e)}"}]

    @classmethod
    async def handle_kline_event_trigger(cls, plan_id: int, inst_id: str, kline_data: dict):
        """
        处理K线事件触发

        Args:
            plan_id: 计划ID
            inst_id: 交易对
            kline_data: K线数据
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
**更新时间**: {kline_data.get('timestamp', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S UTC')}
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
    def get_latest_conversation_messages(
        cls,
        plan_id: int,
        conversation_type: ConversationType = ConversationType.MANUAL_CHAT
    ) -> List[Dict]:
        """
        获取最新的对话消息

        Args:
            plan_id: 计划ID
            conversation_type: 对话类型

        Returns:
            Chatbot格式的消息列表
        """
        try:
            conversation = enhanced_conversation_service.get_latest_conversation_by_type(
                plan_id=plan_id,
                conversation_type=conversation_type
            )

            if not conversation:
                return [{"role": "assistant", "content": "暂无对话记录"}]

            messages = enhanced_conversation_service.get_conversation_messages(conversation.id)
            return enhanced_conversation_service.format_for_chatbot(messages)

        except Exception as e:
            logger.error(f"获取最新对话消息失败: {e}")
            return [{"role": "assistant", "content": f"获取对话失败: {str(e)}"}]


# 全局实例
enhanced_inference_service = EnhancedInferenceService()