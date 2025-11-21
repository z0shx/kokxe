"""
待确认工具调用管理服务
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from database.db import get_db
from database.models import PendingToolCall, TradingPlan
from utils.logger import setup_logger

logger = setup_logger(__name__, "pending_tool_service.log")


class PendingToolService:
    """待确认工具调用服务"""

    @classmethod
    def create_pending_tool(
        cls,
        plan_id: int,
        tool_name: str,
        tool_arguments: Dict,
        agent_decision_id: Optional[int] = None,
        expected_effect: Optional[str] = None,
        risk_warning: Optional[str] = None,
        expires_in_minutes: int = 30
    ) -> int:
        """
        创建待确认工具调用

        Args:
            plan_id: 计划ID
            tool_name: 工具名称
            tool_arguments: 工具参数
            agent_decision_id: Agent决策ID
            expected_effect: 预期效果说明
            risk_warning: 风险提示
            expires_in_minutes: 超时时间（分钟）

        Returns:
            创建的工具调用ID
        """
        try:
            with get_db() as db:
                # 创建新的待确认工具
                pending_tool = PendingToolCall(
                    plan_id=plan_id,
                    agent_decision_id=agent_decision_id,
                    tool_name=tool_name,
                    tool_arguments=tool_arguments,
                    expected_effect=expected_effect,
                    risk_warning=risk_warning,
                    expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
                    status='pending'
                )

                db.add(pending_tool)
                db.commit()
                db.refresh(pending_tool)

                logger.info(
                    f"创建待确认工具: id={pending_tool.id}, plan_id={plan_id}, "
                    f"tool={tool_name}, expires_in={expires_in_minutes}min"
                )

                return pending_tool.id

        except Exception as e:
            logger.error(f"创建待确认工具失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def expire_old_pending_tools(cls, plan_id: int):
        """
        将旧的待确认工具标记为过期（每次新推理时调用）

        Args:
            plan_id: 计划ID
        """
        try:
            with get_db() as db:
                # 将该计划的所有pending状态的工具标记为expired
                updated = db.query(PendingToolCall).filter(
                    PendingToolCall.plan_id == plan_id,
                    PendingToolCall.status == 'pending'
                ).update({
                    'status': 'expired',
                    'updated_at': datetime.utcnow()
                })

                db.commit()

                if updated > 0:
                    logger.info(f"已将计划 {plan_id} 的 {updated} 个待确认工具标记为过期")

        except Exception as e:
            logger.error(f"过期旧工具失败: {e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def get_pending_tools(cls, plan_id: int) -> List[Dict]:
        """
        获取计划的待确认工具列表

        Args:
            plan_id: 计划ID

        Returns:
            待确认工具列表
        """
        try:
            with get_db() as db:
                tools = db.query(PendingToolCall).filter(
                    PendingToolCall.plan_id == plan_id,
                    PendingToolCall.status == 'pending'
                ).order_by(PendingToolCall.created_at.desc()).all()

                result = []
                for tool in tools:
                    result.append({
                        'id': tool.id,
                        'tool_name': tool.tool_name,
                        'tool_arguments': tool.tool_arguments,
                        'expected_effect': tool.expected_effect,
                        'risk_warning': tool.risk_warning,
                        'created_at': tool.created_at,
                        'expires_at': tool.expires_at
                    })

                return result

        except Exception as e:
            logger.error(f"获取待确认工具失败: {e}")
            return []

    @classmethod
    def confirm_tool(cls, tool_id: int, confirmed_by: str = 'manual') -> Dict:
        """
        确认工具调用

        Args:
            tool_id: 工具ID
            confirmed_by: 确认人

        Returns:
            执行结果
        """
        try:
            with get_db() as db:
                tool = db.query(PendingToolCall).filter(
                    PendingToolCall.id == tool_id
                ).first()

                if not tool:
                    return {'success': False, 'error': '工具不存在'}

                if tool.status != 'pending':
                    return {'success': False, 'error': f'工具状态为 {tool.status}，无法确认'}

                # 检查是否过期
                if tool.expires_at and datetime.utcnow() > tool.expires_at:
                    tool.status = 'expired'
                    db.commit()
                    return {'success': False, 'error': '工具已过期'}

                # 更新状态为已确认
                tool.status = 'confirmed'
                tool.confirmed_at = datetime.utcnow()
                tool.confirmed_by = confirmed_by
                db.commit()

                logger.info(f"工具 {tool_id} 已确认，准备执行")

                # TODO: 执行工具（需要集成 AgentToolExecutor）
                # 这里返回待执行状态，实际执行在外部完成
                return {
                    'success': True,
                    'message': '工具已确认',
                    'tool_name': tool.tool_name,
                    'tool_arguments': tool.tool_arguments
                }

        except Exception as e:
            logger.error(f"确认工具失败: {e}")
            return {'success': False, 'error': str(e)}

    @classmethod
    def reject_tool(cls, tool_id: int, reason: str = '') -> Dict:
        """
        拒绝工具调用

        Args:
            tool_id: 工具ID
            reason: 拒绝原因

        Returns:
            操作结果
        """
        try:
            with get_db() as db:
                tool = db.query(PendingToolCall).filter(
                    PendingToolCall.id == tool_id
                ).first()

                if not tool:
                    return {'success': False, 'error': '工具不存在'}

                if tool.status != 'pending':
                    return {'success': False, 'error': f'工具状态为 {tool.status}，无法拒绝'}

                # 更新状态为已拒绝
                tool.status = 'rejected'
                tool.error_message = reason
                tool.updated_at = datetime.utcnow()
                db.commit()

                logger.info(f"工具 {tool_id} 已拒绝: {reason}")

                return {'success': True, 'message': '工具已拒绝'}

        except Exception as e:
            logger.error(f"拒绝工具失败: {e}")
            return {'success': False, 'error': str(e)}

    @classmethod
    def auto_expire_tools(cls):
        """
        自动过期超时的工具（可以定期调用）
        """
        try:
            with get_db() as db:
                now = datetime.utcnow()

                updated = db.query(PendingToolCall).filter(
                    PendingToolCall.status == 'pending',
                    PendingToolCall.expires_at < now
                ).update({
                    'status': 'expired',
                    'updated_at': now
                })

                db.commit()

                if updated > 0:
                    logger.info(f"自动过期了 {updated} 个超时工具")

        except Exception as e:
            logger.error(f"自动过期工具失败: {e}")
