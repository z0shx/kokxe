"""
Agent 工具确认服务
处理AI Agent调用的工具需要用户确认的情况
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from database.db import get_db
from database.models import TradingPlan, AgentDecision, PendingToolCall
from utils.logger import setup_logger
from utils.timezone_helper import format_datetime_full_beijing, get_current_beijing_time

logger = setup_logger(__name__, "agent_confirmation_service.log")


class ConfirmationMode(Enum):
    """确认模式"""
    AUTO = "auto"  # 自动执行
    MANUAL = "manual"  # 手动确认
    DISABLED = "disabled"  # 禁用工具调用


class ConfirmationStatus(Enum):
    """确认状态"""
    PENDING = "pending"  # 等待确认
    APPROVED = "approved"  # 已同意
    REJECTED = "rejected"  # 已拒绝
    EXPIRED = "expired"  # 已过期
    EXECUTED = "executed"  # 已执行


class AgentConfirmationService:
    """Agent工具确认服务"""

    def __init__(self):
        self.default_timeout_minutes = 30  # 默认30分钟超时
        self.max_pending_tools = 50  # 最大待确认工具数量

    def get_confirmation_mode(self, plan: TradingPlan) -> ConfirmationMode:
        """
        获取计划的确认模式

        Args:
            plan: 交易计划

        Returns:
            确认模式
        """
        # 从agent_tools_config中读取确认模式
        tools_config = plan.agent_tools_config or {}
        auto_tool_execution = tools_config.get('auto_execution', False)

        if not auto_tool_execution:
            return ConfirmationMode.MANUAL
        elif plan.auto_tool_execution_enabled:
            return ConfirmationMode.AUTO
        else:
            return ConfirmationMode.MANUAL

    def should_require_confirmation(self, plan: TradingPlan, tool_name: str, tool_args: Dict) -> bool:
        """
        判断是否需要确认

        Args:
            plan: 交易计划
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            是否需要确认
        """
        mode = self.get_confirmation_mode(plan)

        if mode == ConfirmationMode.AUTO:
            return False
        elif mode == ConfirmationMode.DISABLED:
            return True  # 实际上不执行任何工具
        else:  # MANUAL
            return True

    async def create_pending_tool_call(
        self,
        plan_id: int,
        agent_decision_id: int,
        tool_name: str,
        tool_args: Dict,
        expected_effect: str = "",
        risk_warning: str = "",
        timeout_minutes: Optional[int] = None
    ) -> int:
        """
        创建待确认的工具调用

        Args:
            plan_id: 计划ID
            agent_decision_id: Agent决策ID
            tool_name: 工具名称
            tool_args: 工具参数
            expected_effect: 预期效果
            risk_warning: 风险提示
            timeout_minutes: 超时时间（分钟）

        Returns:
            待确认工具调用ID
        """
        try:
            timeout = timeout_minutes or self.default_timeout_minutes
            expires_at = get_current_beijing_time() + timedelta(minutes=timeout)

            with get_db() as db:
                # 检查待确认工具数量限制
                pending_count = db.query(PendingToolCall).filter(
                    PendingToolCall.plan_id == plan_id,
                    PendingToolCall.status == ConfirmationStatus.PENDING.value
                ).count()

                if pending_count >= self.max_pending_tools:
                    logger.warning(f"计划 {plan_id} 待确认工具数量已达到上限 {self.max_pending_tools}")
                    raise Exception("待确认工具数量过多，请先处理现有的待确认工具")

                # 创建待确认工具调用记录
                pending_tool = PendingToolCall(
                    plan_id=plan_id,
                    agent_decision_id=agent_decision_id,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                    expected_effect=expected_effect,
                    risk_warning=risk_warning,
                    status=ConfirmationStatus.PENDING.value,
                    expires_at=expires_at
                )

                db.add(pending_tool)
                db.commit()
                db.refresh(pending_tool)

                logger.info(f"创建待确认工具调用: ID={pending_tool.id}, tool={tool_name}, plan_id={plan_id}")
                return pending_tool.id

        except Exception as e:
            logger.error(f"创建待确认工具调用失败: {e}")
            raise

    def get_pending_tools(self, plan_id: Optional[int] = None) -> List[Dict]:
        """
        获取待确认的工具列表

        Args:
            plan_id: 计划ID，None表示获取所有

        Returns:
            待确认工具列表
        """
        try:
            with get_db() as db:
                query = db.query(PendingToolCall).filter(
                    PendingToolCall.status == ConfirmationStatus.PENDING.value
                )

                if plan_id is not None:
                    query = query.filter(PendingToolCall.plan_id == plan_id)

                # 排序：先到期的在前
                pending_tools = query.order_by(PendingToolCall.expires_at.asc()).all()

                result = []
                for tool in pending_tools:
                    # 检查是否过期
                    if tool.expires_at:
                        current_time = get_current_beijing_time()
                        # 确保两个时间对象都有timezone信息
                        if tool.expires_at.tzinfo is None:
                            import pytz
                            beijing_tz = pytz.timezone('Asia/Shanghai')
                            expires_at = beijing_tz.localize(tool.expires_at)
                        else:
                            expires_at = tool.expires_at

                        if current_time > expires_at:
                            tool.status = ConfirmationStatus.EXPIRED.value
                            db.commit()
                            continue

                    result.append({
                        'id': tool.id,
                        'plan_id': tool.plan_id,
                        'agent_decision_id': tool.agent_decision_id,
                        'tool_name': tool.tool_name,
                        'tool_args': tool.tool_arguments,  # 改为tool_args以匹配UI期望
                        'expected_effect': tool.expected_effect,
                        'risk_warning': tool.risk_warning,
                        'created_at': tool.created_at,
                        'expires_at': tool.expires_at,
                        'remaining_time': self._get_remaining_time(tool.expires_at),
                        'status': ConfirmationStatus.PENDING.value  # 添加状态字段
                    })

                return result

        except Exception as e:
            logger.error(f"获取待确认工具列表失败: {e}")
            return []

    def _get_remaining_time(self, expires_at: Optional[datetime]) -> str:
        """
        获取剩余时间字符串

        Args:
            expires_at: 过期时间

        Returns:
            剩余时间字符串
        """
        if not expires_at:
            return "无限制"

        now = get_current_beijing_time()
        if expires_at <= now:
            return "已过期"

        delta = expires_at - now
        if delta.days > 0:
            return f"{delta.days}天{delta.seconds // 3600}小时"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}小时{(delta.seconds % 3600) // 60}分钟"
        else:
            return f"{delta.seconds // 60}分钟"

    def get_tool_detail(self, tool_id: int) -> Optional[Dict[str, Any]]:
        """
        获取工具详情

        Args:
            tool_id: 工具ID

        Returns:
            工具详情字典
        """
        try:
            with get_db() as db:
                tool_call = db.query(PendingToolCall).filter(PendingToolCall.id == tool_id).first()
                if not tool_call:
                    return None

                # 格式化时间
                created_at_str = ""
                expires_at_str = ""
                if tool_call.created_at:
                    created_at_str = tool_call.created_at.strftime('%Y-%m-%d %H:%M:%S')
                if tool_call.expires_at:
                    expires_at_str = tool_call.expires_at.strftime('%Y-%m-%d %H:%M:%S')

                return {
                    'id': tool_call.id,
                    'plan_id': tool_call.plan_id,
                    'tool_name': tool_call.tool_name,
                    'tool_arguments': tool_call.tool_arguments,
                    'expected_effect': tool_call.expected_effect,
                    'risk_warning': tool_call.risk_warning,
                    'created_at': created_at_str,
                    'expires_at': expires_at_str,
                    'status': tool_call.status,
                    'execution_result': tool_call.execution_result
                }

        except Exception as e:
            logger.error(f"获取工具详情失败: {e}")
            return None

    async def confirm_tool_call(
        self,
        pending_tool_id: int,
        approved: bool,
        confirmed_by: str = "user",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        确认工具调用

        Args:
            pending_tool_id: 待确认工具ID
            approved: 是否同意
            confirmed_by: 确认人
            notes: 备注

        Returns:
            确认结果
        """
        try:
            with get_db() as db:
                pending_tool = db.query(PendingToolCall).filter(
                    PendingToolCall.id == pending_tool_id
                ).first()

                if not pending_tool:
                    return {'success': False, 'message': '未找到待确认工具'}

                if pending_tool.status != ConfirmationStatus.PENDING.value:
                    return {
                        'success': False,
                        'message': f'工具状态为 {pending_tool.status}，无法确认'
                    }

                # 检查是否过期
                if pending_tool.expires_at and get_current_beijing_time() > pending_tool.expires_at:
                    pending_tool.status = ConfirmationStatus.EXPIRED.value
                    db.commit()
                    return {'success': False, 'message': '工具确认已过期'}

                # 更新状态
                pending_tool.status = ConfirmationStatus.APPROVED.value if approved else ConfirmationStatus.REJECTED.value
                pending_tool.confirmed_at = get_current_beijing_time()
                pending_tool.confirmed_by = confirmed_by

                # 记录确认结果
                confirmation_result = {
                    'pending_tool_id': pending_tool_id,
                    'plan_id': pending_tool.plan_id,
                    'tool_name': pending_tool.tool_name,
                    'tool_arguments': pending_tool.tool_arguments,
                    'approved': approved,
                    'confirmed_by': confirmed_by,
                    'confirmed_at': pending_tool.confirmed_at,
                    'notes': notes
                }

                db.commit()

                logger.info(f"工具确认: ID={pending_tool_id}, approved={approved}, by={confirmed_by}")

                # 如果同意，执行工具
                if approved:
                    execution_result = await self._execute_confirmed_tool(pending_tool)
                    confirmation_result['execution_result'] = execution_result

                return {
                    'success': True,
                    'message': '已' + ('同意' if approved else '拒绝') + '工具调用',
                    'result': confirmation_result
                }

        except Exception as e:
            logger.error(f"确认工具调用失败: {e}")
            return {'success': False, 'message': f'确认失败: {str(e)}'}

    async def _execute_confirmed_tool(self, pending_tool: PendingToolCall) -> Dict[str, Any]:
        """
        执行已确认的工具

        Args:
            pending_tool: 待确认工具对象

        Returns:
            执行结果
        """
        try:
            from services.trading_tools import OKXTradingTools
            from database.models import TradingPlan

            # 获取交易计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(
                    TradingPlan.id == pending_tool.plan_id
                ).first()

                if not plan:
                    return {'success': False, 'error': '未找到交易计划'}

            # 创建交易工具实例
            trading_tools = OKXTradingTools(
                api_key=plan.okx_api_key,
                secret_key=plan.okx_secret_key,
                passphrase=plan.okx_passphrase,
                is_demo=plan.is_demo,
                trading_limits=plan.trading_limits
            )

            # 执行工具
            tool_name = pending_tool.tool_name
            tool_args = pending_tool.tool_arguments

            if tool_name == 'place_order':
                result = await trading_tools.place_order(**tool_args)
            elif tool_name == 'place_limit_order':
                result = await trading_tools.place_limit_order(**tool_args)
            elif tool_name == 'place_stop_loss_order':
                result = await trading_tools.place_stop_loss_order(**tool_args)
            elif tool_name == 'cancel_order':
                result = await trading_tools.cancel_order(**tool_args)
            elif tool_name == 'get_positions':
                result = await trading_tools.get_positions(**tool_args)
            elif tool_name == 'get_current_price':
                result = await trading_tools.get_current_price(**tool_args)
            elif tool_name == 'get_trading_limits':
                result = await trading_tools.get_trading_limits(**tool_args)
            else:
                result = {'success': False, 'error': f'未知工具: {tool_name}'}

            # 更新待确认工具状态
            with get_db() as db:
                pending_tool.status = ConfirmationStatus.EXECUTED.value
                pending_tool.execution_result = result
                db.commit()

            return result

        except Exception as e:
            logger.error(f"执行已确认工具失败: {e}")
            return {'success': False, 'error': str(e)}

    async def batch_confirm_tools(
        self,
        pending_tool_ids: List[int],
        approved: bool,
        confirmed_by: str = "user"
    ) -> Dict[str, Any]:
        """
        批量确认工具调用

        Args:
            pending_tool_ids: 待确认工具ID列表
            approved: 是否同意
            confirmed_by: 确认人

        Returns:
            批量确认结果
        """
        results = []
        success_count = 0

        for tool_id in pending_tool_ids:
            result = await self.confirm_tool_call(tool_id, approved, confirmed_by)
            results.append({
                'tool_id': tool_id,
                'result': result
            })
            if result['success']:
                success_count += 1

        return {
            'success': True,
            'message': f'批量确认完成: {success_count}/{len(pending_tool_ids)} 个工具已处理',
            'results': results
        }

    def get_tool_execution_history(self, plan_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """
        获取工具执行历史

        Args:
            plan_id: 计划ID，None表示获取所有
            limit: 返回记录数限制

        Returns:
            执行历史列表
        """
        try:
            with get_db() as db:
                query = db.query(PendingToolCall)

                if plan_id is not None:
                    query = query.filter(PendingToolCall.plan_id == plan_id)

                # 只返回已处理（非pending）的记录
                tools = query.filter(
                    PendingToolCall.status != ConfirmationStatus.PENDING.value
                ).order_by(PendingToolCall.created_at.desc()).limit(limit).all()

                result = []
                for tool in tools:
                    result.append({
                        'id': tool.id,
                        'plan_id': tool.plan_id,
                        'tool_name': tool.tool_name,
                        'tool_arguments': tool.tool_arguments,
                        'status': tool.status,
                        'created_at': tool.created_at,
                        'confirmed_at': tool.confirmed_at,
                        'confirmed_by': tool.confirmed_by,
                        'execution_result': tool.execution_result,
                        'error_message': tool.error_message
                    })

                return result

        except Exception as e:
            logger.error(f"获取工具执行历史失败: {e}")
            return []

    def cleanup_expired_tools(self) -> Dict[str, Any]:
        """
        清理过期的待确认工具

        Returns:
            清理结果
        """
        try:
            with get_db() as db:
                expired_tools = db.query(PendingToolCall).filter(
                    PendingToolCall.status == ConfirmationStatus.PENDING.value,
                    PendingToolCall.expires_at < get_current_beijing_time()
                ).all()

                count = 0
                for tool in expired_tools:
                    tool.status = ConfirmationStatus.EXPIRED.value
                    count += 1

                db.commit()

                logger.info(f"清理过期工具: {count} 个")
                return {
                    'success': True,
                    'cleaned_count': count,
                    'message': f'已清理 {count} 个过期的待确认工具'
                }

        except Exception as e:
            logger.error(f"清理过期工具失败: {e}")
            return {'success': False, 'error': str(e)}


# 全局实例
confirmation_service = AgentConfirmationService()