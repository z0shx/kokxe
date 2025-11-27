"""
Agent工具调用错误记录服务
负责记录Agent工具调用错误并确保对话能够继续
"""
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from database.db import get_db
from database.models import AgentDecision, TradingPlan
from utils.logger import setup_logger
from utils.timezone_helper import format_datetime_full_beijing

logger = setup_logger(__name__, "agent_error_handler.log")


class AgentErrorHandler:
    """Agent工具调用错误处理器"""

    @classmethod
    def record_tool_error(
        cls,
        plan_id: int,
        tool_name: str,
        error_message: str,
        tool_params: Optional[Dict] = None,
        conversation_id: Optional[int] = None,
        decision_context: Optional[Dict] = None
    ) -> int:
        """
        记录工具调用错误到数据库

        Args:
            plan_id: 计划ID
            tool_name: 工具名称
            error_message: 错误信息
            tool_params: 工具参数（可选）
            conversation_id: 对话ID（可选）
            decision_context: 决策上下文（可选）

        Returns:
            错误记录ID
        """
        try:
            with get_db() as db:
                # 创建错误决策记录
                error_decision = AgentDecision(
                    plan_id=plan_id,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    tool_params=tool_params or {},
                    llm_response="",  # 空响应，因为这是错误记录
                    decision="ERROR",
                    reason=f"工具调用失败: {error_message}",
                    confidence=0.0,
                    risk_level="low",  # 错误记录设为低风险
                    status="failed",
                    error_message=error_message,
                    created_at=datetime.utcnow()
                )

                db.add(error_decision)
                db.commit()
                db.refresh(error_decision)

                logger.info(f"记录工具调用错误: plan_id={plan_id}, tool={tool_name}, error_id={error_decision.id}")
                return error_decision.id

        except Exception as e:
            logger.error(f"记录工具错误失败: plan_id={plan_id}, tool={tool_name}, error={e}")
            return 0

    @classmethod
    def should_continue_conversation(
        cls,
        error_message: str,
        tool_name: str,
        consecutive_errors: int = 1
    ) -> bool:
        """
        判断是否应该继续对话

        Args:
            error_message: 错误信息
            tool_name: 工具名称
            consecutive_errors: 连续错误次数

        Returns:
            是否继续对话
        """
        # 致命错误列表，这些错误应该停止对话
        fatal_errors = [
            "API密钥无效",
            "认证失败",
            "账户被冻结",
            "权限不足",
            "配置错误"
        ]

        # 网络相关错误，通常可以重试
        network_errors = [
            "连接超时",
            "网络错误",
            "502 Bad Gateway",
            "503 Service Unavailable",
            "504 Gateway Timeout"
        ]

        # 数据相关错误，通常可以继续
        data_errors = [
            "数据不存在",
            "参数验证失败",
            "查询结果为空",
            "格式错误"
        ]

        # 检查致命错误
        for fatal_error in fatal_errors:
            if fatal_error in error_message:
                logger.warning(f"检测到致命错误，停止对话: {fatal_error}")
                return False

        # 检查连续错误次数
        if consecutive_errors >= 3:
            logger.warning(f"连续错误次数过多({consecutive_errors})，停止对话")
            return False

        # 网络错误和数据错误通常可以继续
        for network_error in network_errors:
            if network_error in error_message:
                logger.info(f"检测到网络错误，建议继续对话: {network_error}")
                return True

        for data_error in data_errors:
            if data_error in error_message:
                logger.info(f"检测到数据错误，建议继续对话: {data_error}")
                return True

        # 默认策略：继续对话
        logger.info(f"未知错误类型，默认继续对话: {error_message[:100]}...")
        return True

    @classmethod
    def get_recent_error_count(
        cls,
        plan_id: int,
        time_window_minutes: int = 30
    ) -> int:
        """
        获取最近的错误次数

        Args:
            plan_id: 计划ID
            time_window_minutes: 时间窗口（分钟）

        Returns:
            错误次数
        """
        try:
            with get_db() as db:
                time_threshold = datetime.utcnow() - timedelta(minutes=time_window_minutes)

                error_count = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id,
                    AgentDecision.status == "failed",
                    AgentDecision.created_at >= time_threshold
                ).count()

                return error_count

        except Exception as e:
            logger.error(f"获取错误次数失败: plan_id={plan_id}, error={e}")
            return 0

    @classmethod
    def format_error_for_agent(
        cls,
        tool_name: str,
        error_message: str,
        suggestion: Optional[str] = None
    ) -> str:
        """
        格式化错误信息供Agent理解

        Args:
            tool_name: 工具名称
            error_message: 错误信息
            suggestion: 建议（可选）

        Returns:
            格式化的错误信息
        """
        formatted_error = f"工具 {tool_name} 调用失败: {error_message}"

        if suggestion:
            formatted_error += f"\n建议: {suggestion}"
        else:
            # 根据错误类型提供建议
            if "连接超时" in error_message or "网络" in error_message:
                formatted_error += "\n建议: 稍后重试，或检查网络连接"
            elif "参数验证失败" in error_message:
                formatted_error += "\n建议: 检查参数格式和必需字段"
            elif "数据不存在" in error_message:
                formatted_error += "\n建议: 尝试其他查询条件或确认数据是否存在"
            elif "权限" in error_message or "认证" in error_message:
                formatted_error += "\n建议: 检查API密钥和权限配置"
            else:
                formatted_error += "\n建议: 检查输入参数或尝试其他方法"

        return formatted_error

    @classmethod
    def create_fallback_response(
        cls,
        tool_name: str,
        error_message: str,
        plan_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        创建失败响应，确保对话能够继续

        Args:
            tool_name: 工具名称
            error_message: 错误信息
            plan_context: 计划上下文（可选）

        Returns:
            标准化的失败响应
        """
        return {
            "success": False,
            "error": error_message,
            "tool_name": tool_name,
            "fallback_message": f"工具 {tool_name} 调用失败，但我们可以继续分析其他方面",
            "continue_conversation": True,
            "timestamp": datetime.utcnow().isoformat(),
            "context": plan_context or {}
        }