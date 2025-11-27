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
            current_assistant_msg = ""
            chunk_count = 0

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

                    # 重新获取完整的对话上下文
                    updated_context_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
                    full_context = enhanced_conversation_service.format_for_chatbot(updated_context_messages)

                    if chunk_type == "thinking_start":
                        current_assistant_msg = "🧠 **开始思考分析...**\n\n"

                    elif chunk_type == "thinking":
                        current_assistant_msg = f"🧠 **AI思考过程**\n\n{content}"

                    elif chunk_type == "content":
                        current_assistant_msg = content

                    elif chunk_type == "tool_call_start":
                        tool_name = chunk_data.get("tool_name", "")
                        current_assistant_msg = f"🛠️ **调用工具**: `{tool_name}`\n\n⏳ 正在执行..."

                    elif chunk_type == "tool_call":
                        tool_name = chunk_data.get("tool_name", "")
                        arguments = chunk_data.get("arguments", {})
                        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                        current_assistant_msg = f"🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."

                    elif chunk_type == "tool_result":
                        tool_name = chunk_data.get("tool_name", "")
                        result = chunk_data.get("result", {})
                        success = result.get("success", False)
                        status_emoji = "✅" if success else "❌"

                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        current_assistant_msg = f"🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续分析..."

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("content", "未知错误")
                        current_assistant_msg = f"❌ **推理错误**: {error_msg}"

                    # 构建完整的显示消息（包括所有上下文 + 当前AI响应）
                    updated_messages = full_context + [
                        {"role": "assistant", "content": current_assistant_msg, "metadata": {"streaming": True}}
                    ]

                    yield updated_messages

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理推理块失败: {e}")
                    continue

            # 推理完成 - 获取最终完整上下文
            final_context_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
            final_full_context = enhanced_conversation_service.format_for_chatbot(final_context_messages)

            final_assistant_msg = current_assistant_msg + "\n\n✅ **推理完成**"
            final_messages = final_full_context + [
                {"role": "assistant", "content": final_assistant_msg, "metadata": {"completed": True}}
            ]

            yield final_messages

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
            current_assistant_msg = ""

            async for chunk_str in enhanced_agent_stream_service.chat_with_tools_stream(
                conversation_id=conversation_id,
                user_message=user_message,
                use_thinking_mode=True
            ):
                try:
                    chunk_data = json.loads(chunk_str)
                    chunk_type = chunk_data.get("type", "")
                    content = chunk_data.get("content", "")

                    # 重新获取完整的对话上下文
                    updated_context_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
                    full_context = enhanced_conversation_service.format_for_chatbot(updated_context_messages)

                    if chunk_type == "thinking_start":
                        current_assistant_msg = "🧠 **开始思考分析...**\n\n"

                    elif chunk_type == "thinking":
                        current_assistant_msg = f"🧠 **AI思考过程**\n\n{content}"

                    elif chunk_type == "content":
                        current_assistant_msg = content

                    elif chunk_type == "tool_call":
                        tool_name = chunk_data.get("tool_name", "")
                        arguments = chunk_data.get("arguments", {})
                        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                        current_assistant_msg = f"🛠️ **工具调用**: `{tool_name}`\n\n📋 **参数**:\n```json\n{args_str}\n```\n\n⏳ 正在执行..."

                    elif chunk_type == "tool_result":
                        tool_name = chunk_data.get("tool_name", "")
                        result = chunk_data.get("result", {})
                        success = result.get("success", False)
                        status_emoji = "✅" if success else "❌"

                        result_str = json.dumps(result, indent=2, ensure_ascii=False)
                        current_assistant_msg = f"🛠️ **工具执行结果**: `{tool_name}` {status_emoji}\n\n```json\n{result_str}\n```\n\n🔄 继续对话..."

                    elif chunk_type == "error":
                        error_msg = chunk_data.get("content", "未知错误")
                        current_assistant_msg = f"❌ **回复错误**: {error_msg}"

                    # 构建完整的显示消息（包括所有上下文 + 用户消息 + 当前AI响应）
                    updated_messages = full_context + [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": current_assistant_msg, "metadata": {"streaming": True}}
                    ]

                    yield updated_messages

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"处理对话块失败: {e}")
                    continue

            # 对话完成 - 获取最终完整上下文
            final_context_messages = enhanced_conversation_service.get_conversation_messages(conversation_id)
            final_full_context = enhanced_conversation_service.format_for_chatbot(final_context_messages)

            final_messages = final_full_context + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": current_assistant_msg, "metadata": {"completed": True}}
            ]

            yield final_messages

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