"""
AI Agent 对话管理服务
负责管理对话会话、消息记录和对话展示
"""
import json
from typing import Dict, List, Optional
from utils.time_utils import now_beijing

from database.db import get_db
from database.models import AgentConversation, AgentMessage
from utils.logger import setup_logger

logger = setup_logger(__name__, "conversation.log")




class ConversationService:
    """对话服务类"""

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
    def get_tool_calls_summary(conversation_id: int) -> List[Dict]:
        """获取对话的工具调用摘要"""
        try:
            with get_db() as db:
                tool_calls = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id,
                    AgentMessage.message_type == "tool_call"
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