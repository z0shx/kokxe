"""
K线数据事件服务
负责处理新交易数据接收时的事件触发对话
"""
import asyncio
import threading
from datetime import datetime
from typing import Dict, Optional, Set
from services.conversation_service import ConversationService
from database.models import TradingPlan, AgentConversation, AgentMessage
from database.db import get_db
from utils.logger import setup_logger
from sqlalchemy import and_, desc

logger = setup_logger(__name__, "kline_event_service.log")


class KlineEventService:
    """K线数据事件服务（单例）"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化服务"""
        if self._initialized:
            return

        # 事件循环
        self.loop = None
        self.loop_thread = None
        self._start_event_loop()

        # 订阅的计划ID集合
        self.subscribed_plans: Set[int] = set()

        self._initialized = True
        logger.info("K线事件服务初始化完成")

    def _start_event_loop(self):
        """在后台线程中启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            logger.info("K线事件服务事件循环已启动")
            self.loop.run_forever()
            logger.info("K线事件服务事件循环已停止")

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # 等待事件循环启动
        import time
        for _ in range(10):
            if self.loop and self.loop.is_running():
                logger.info("✅ K线事件服务事件循环线程已就绪")
                break
            time.sleep(0.1)
        else:
            logger.error("❌ K线事件服务事件循环启动超时")

    def subscribe_plan(self, plan_id: int):
        """订阅计划的K线事件"""
        self.subscribed_plans.add(plan_id)
        logger.info(f"订阅K线事件: plan_id={plan_id}, total_subscribed={len(self.subscribed_plans)}")

    def unsubscribe_plan(self, plan_id: int):
        """取消订阅计划的K线事件"""
        self.subscribed_plans.discard(plan_id)
        logger.info(f"取消订阅K线事件: plan_id={plan_id}, total_subscribed={len(self.subscribed_plans)}")

    def trigger_new_kline_event(self, inst_id: str, interval: str, kline_data: dict):
        """
        触发新K线数据事件

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            kline_data: K线数据字典
        """
        try:
            if not self.loop or not self.loop.is_running():
                logger.warning("事件循环未运行，无法处理K线事件")
                return

            # 在事件循环中异步处理
            asyncio.run_coroutine_threadsafe(
                self._handle_new_kline_event(inst_id, interval, kline_data),
                self.loop
            )

        except Exception as e:
            logger.error(f"触发K线事件失败: {e}")

    async def _handle_new_kline_event(self, inst_id: str, interval: str, kline_data: dict):
        """异步处理新K线数据事件"""
        try:
            # 查找订阅了该交易对的所有计划
            with get_db() as db:
                plans = db.query(TradingPlan).filter(
                    and_(
                        TradingPlan.inst_id == inst_id,
                        TradingPlan.interval == interval,
                        TradingPlan.status == 'running',
                        TradingPlan.id.in_(self.subscribed_plans) if self.subscribed_plans else True
                    )
                ).all()

                if not plans:
                    return

                logger.info(f"找到 {len(plans)} 个订阅的计划，处理新K线事件")

                # 为每个计划触发事件
                for plan in plans:
                    try:
                        await self._trigger_plan_event(plan, kline_data)
                    except Exception as e:
                        logger.error(f"处理计划 {plan.id} 的K线事件失败: {e}")

        except Exception as e:
            logger.error(f"处理新K线事件失败: {e}")

    async def _trigger_plan_event(self, plan: TradingPlan, kline_data: dict):
        """为单个计划触发事件"""
        try:
            # 使用增强推理服务处理K线事件
            from services.enhanced_inference_service import enhanced_inference_service

            await enhanced_inference_service.handle_kline_event_trigger(
                plan_id=plan.id,
                inst_id=plan.inst_id,
                kline_data=kline_data
            )

            logger.info(f"已为计划 {plan.id} 触发K线事件对话")

        except Exception as e:
            logger.error(f"为计划 {plan.id} 触发事件失败: {e}")

    def _get_or_create_event_conversation(self, plan_id: int) -> Optional[AgentConversation]:
        """获取或创建事件对话会话"""
        try:
            with get_db() as db:
                # 查找现有的事件对话会话
                conversation = db.query(AgentConversation).filter(
                    and_(
                        AgentConversation.plan_id == plan_id,
                        AgentConversation.conversation_type == "kline_event",
                        AgentConversation.status == 'active'
                    )
                ).order_by(desc(AgentConversation.last_message_at)).first()

                if conversation:
                    # 更新最后消息时间
                    conversation.last_message_at = datetime.utcnow()
                    db.commit()
                    return conversation

                # 创建新的事件对话会话
                conversation = AgentConversation(
                    plan_id=plan_id,
                    conversation_type="kline_event",
                    session_name=f"K线事件监听_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    status='active',
                    started_at=datetime.utcnow(),
                    last_message_at=datetime.utcnow()
                )

                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                logger.info(f"创建K线事件对话会话: conversation_id={conversation.id}, plan_id={plan_id}")
                return conversation

        except Exception as e:
            logger.error(f"获取或创建事件对话会话失败: {e}")
            return None

    def _build_event_message(self, plan: TradingPlan, kline_data: dict) -> str:
        """构建事件消息"""
        timestamp = kline_data.get('timestamp', datetime.utcnow())
        close_price = kline_data.get('close', 0)
        volume = kline_data.get('volume', 0)

        # 格式化时间
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

        message = f"""🔔 **新K线数据通知**

**交易对**: {plan.inst_id}
**时间颗粒度**: {plan.interval}
**更新时间**: {time_str}
**收盘价**: {close_price}
**成交量**: {volume}

请分析最新市场数据并考虑是否需要调整交易策略。"""

        return message

    def get_active_subscriptions(self) -> Dict:
        """获取活跃的订阅信息"""
        return {
            'subscribed_plans': list(self.subscribed_plans),
            'total_count': len(self.subscribed_plans),
            'loop_running': self.loop.is_running() if self.loop else False
        }

    def shutdown(self):
        """关闭服务"""
        logger.info("正在关闭K线事件服务...")

        # 清空订阅
        self.subscribed_plans.clear()

        # 停止事件循环
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=5)

        logger.info("K线事件服务已关闭")


# 全局单例 - 懒加载
kline_event_service = None

def get_kline_event_service():
    """获取K线事件服务实例（懒加载）"""
    global kline_event_service
    if kline_event_service is None:
        kline_event_service = KlineEventService()
        logger.info("K线事件服务已创建并启动")
    return kline_event_service