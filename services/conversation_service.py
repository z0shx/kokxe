"""
AI Agent 对话管理服务
负责管理对话会话、消息记录和对话展示
"""
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from enum import Enum
import logging
from utils.time_utils import now_beijing

from database.db import get_db
from database.models import (
    AgentConversation, AgentMessage, TradingPlan, TrainingRecord,
    PredictionData, AgentDecision
)
from utils.logger import setup_logger

logger = setup_logger(__name__, "conversation.log")


class MessageType(Enum):
    """消息类型"""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"




class ConversationService:
    """对话服务类"""

    @staticmethod
    def create_conversation(
        plan_id: int,
        training_record_id: Optional[int] = None,
        session_name: Optional[str] = None,
        conversation_type: str = "auto_inference"
    ) -> AgentConversation:
        """创建新的对话会话"""
        try:
            with get_db() as db:
                conversation = AgentConversation(
                    plan_id=plan_id,
                    training_record_id=training_record_id,
                    session_name=session_name or f"推理会话_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    conversation_type=conversation_type,
                    status='active',
                    started_at=now_beijing(),
                    last_message_at=now_beijing()
                )

                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                logger.info(f"创建对话会话成功: conversation_id={conversation.id}, plan_id={plan_id}")
                return conversation

        except Exception as e:
            logger.error(f"创建对话会话失败: {e}")
            raise

    @staticmethod
    def add_message(
        conversation_id: int,
        role: str,
        content: str,
        message_type: str = "text",
        react_iteration: Optional[int] = None,
        react_stage: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[Dict] = None,
        tool_result: Optional[Dict] = None,
        tool_status: str = "pending",
        llm_model: Optional[str] = None
    ) -> AgentMessage:
        """添加消息到对话会话"""
        try:
            with get_db() as db:
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    message_type=message_type,
                    react_iteration=react_iteration,
                    react_stage=react_stage,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    tool_result=tool_result,
                    tool_status=tool_status,
                    llm_model=llm_model,
                    timestamp=now_beijing()
                )

                db.add(message)

                # 更新会话统计信息
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).first()
                if conversation:
                    conversation.total_messages += 1
                    if message_type in ["tool_call", "tool_result"]:
                        conversation.total_tool_calls += 1
                    conversation.last_message_at = now_beijing()

                db.commit()
                db.refresh(message)

                logger.debug(f"添加消息成功: conversation_id={conversation_id}, role={role}, type={message_type}")
                return message

        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            raise

    @staticmethod
    def get_conversation_messages(conversation_id: int) -> List[AgentMessage]:
        """获取对话会话的所有消息"""
        try:
            with get_db() as db:
                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id
                ).order_by(AgentMessage.timestamp.asc()).all()

                return messages

        except Exception as e:
            logger.error(f"获取对话消息失败: {e}")
            return []

    @staticmethod
    def get_plan_conversations(plan_id: int, limit: int = 10) -> List[AgentConversation]:
        """获取计划的对话会话列表"""
        try:
            with get_db() as db:
                conversations = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(AgentConversation.started_at.desc()).limit(limit).all()

                return conversations

        except Exception as e:
            logger.error(f"获取计划对话会话失败: {e}")
            return []

    @staticmethod
    def get_latest_conversation(plan_id: int, conversation_type: Optional[str] = None) -> Optional[AgentConversation]:
        """获取计划的最新对话会话"""
        try:
            with get_db() as db:
                query = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                )

                if conversation_type:
                    query = query.filter(AgentConversation.conversation_type == conversation_type)

                conversation = query.order_by(AgentConversation.started_at.desc()).first()
                return conversation

        except Exception as e:
            logger.error(f"获取最新对话会话失败: {e}")
            return None

    @staticmethod
    def format_messages_for_chatbot(messages: List[AgentMessage]) -> List[Dict]:
        """格式化消息为Chatbot格式"""
        formatted_messages = []

        for message in messages:
            if message.message_type == MessageType.TEXT.value:
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            elif message.message_type == MessageType.THINKING.value:
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"💭 **思考过程**:\n{message.content}"
                })
            elif message.message_type == MessageType.TOOL_CALL.value:
                tool_args_str = json.dumps(message.tool_arguments, indent=2, ensure_ascii=False) if message.tool_arguments else "{}"
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"🔧 **工具调用**: {message.tool_name}\n**参数**: {tool_args_str}"
                })
            elif message.message_type == MessageType.TOOL_RESULT.value:
                tool_result_str = json.dumps(message.tool_result, indent=2, ensure_ascii=False) if message.tool_result else "{}"
                status_icon = "✅" if message.tool_status == "success" else "❌"
                formatted_messages.append({
                    "role": "assistant",
                    "content": f"{status_icon} **工具结果**: {message.tool_name}\n**结果**: {tool_result_str}"
                })

        return formatted_messages

    @staticmethod
    def get_tool_calls_summary(conversation_id: int) -> List[Dict]:
        """获取对话的工具调用摘要"""
        try:
            with get_db() as db:
                tool_calls = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id,
                    AgentMessage.message_type == MessageType.TOOL_CALL.value
                ).order_by(AgentMessage.timestamp.asc()).all()

                summary = []
                for call in tool_calls:
                    summary.append({
                        "tool_name": call.tool_name,
                        "arguments": call.tool_arguments,
                        "status": call.tool_status,
                        "iteration": call.react_iteration,
                        "timestamp": call.timestamp
                    })

                return summary

        except Exception as e:
            logger.error(f"获取工具调用摘要失败: {e}")
            return []

    @staticmethod
    def complete_conversation(conversation_id: int) -> bool:
        """完成对话会话"""
        try:
            with get_db() as db:
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).first()

                if conversation:
                    conversation.status = 'completed'
                    conversation.completed_at = now_beijing()
                    db.commit()
                    logger.info(f"对话会话已完成: conversation_id={conversation_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"完成对话会话失败: {e}")
            return False

    @staticmethod
    def get_conversation_with_messages(conversation_id: int) -> Optional[Tuple[AgentConversation, List[AgentMessage]]]:
        """获取对话会话及其所有消息"""
        try:
            with get_db() as db:
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).first()

                if conversation:
                    messages = db.query(AgentMessage).filter(
                        AgentMessage.conversation_id == conversation_id
                    ).order_by(AgentMessage.timestamp.asc()).all()
                    return conversation, messages

                return None

        except Exception as e:
            logger.error(f"获取对话会话及消息失败: {e}")
            return None