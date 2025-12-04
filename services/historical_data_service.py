"""
历史数据服务
用于获取最近24小时的K线数据作为AI Agent分析上下文
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from sqlalchemy import and_, desc
from database.models import TradingPlan, KlineData
from database.db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__, "historical_data_service.log")


class HistoricalDataService:
    """历史数据服务"""

    @staticmethod
    def get_recent_24h_kline_data(plan_id: int) -> Optional[str]:
        """
        获取计划最近24小时的K线数据

        Args:
            plan_id: 计划ID

        Returns:
            Optional[str]: CSV格式的K线数据，如果没有数据则返回None
        """
        try:
            with get_db() as db:
                # 获取计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return None

                # 计算24小时前的时间
                now = datetime.now()
                twenty_four_hours_ago = now - timedelta(hours=24)

                # 查询最近24小时的K线数据
                kline_data = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= twenty_four_hours_ago,
                        KlineData.timestamp <= now
                    )
                ).order_by(KlineData.timestamp.asc()).all()

                if not kline_data:
                    logger.warning(f"计划 {plan_id} 最近24小时没有K线数据")
                    return None

                # 转换为CSV格式
                csv_lines = []
                # CSV头部
                csv_lines.append("timestamp,open,high,low,close,volume,amount")

                # 数据行
                for kline in kline_data:
                    timestamp_str = kline.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    csv_lines.append(
                        f"{timestamp_str},{kline.open:.2f},{kline.high:.2f},"
                        f"{kline.low:.2f},{kline.close:.2f},"
                        f"{kline.volume or 0:.2f},{kline.amount or 0:.2f}"
                    )

                logger.info(f"计划 {plan_id} 获取到 {len(kline_data)} 条K线数据")
                return "\n".join(csv_lines)

        except Exception as e:
            logger.error(f"获取历史K线数据失败: {e}")
            return None

    @staticmethod
    def get_recent_kline_data_by_count(plan_id: int, count: int = 24) -> Optional[str]:
        """
        根据数量获取最近的K线数据（考虑时间颗粒度）

        Args:
            plan_id: 计划ID
            count: 要获取的数据条数（默认24条）

        Returns:
            Optional[str]: CSV格式的K线数据
        """
        try:
            with get_db() as db:
                # 获取计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    logger.error(f"计划不存在: plan_id={plan_id}")
                    return None

                # 查询最近的N条K线数据
                kline_data = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval
                    )
                ).order_by(desc(KlineData.timestamp)).limit(count).all()

                if not kline_data:
                    logger.warning(f"计划 {plan_id} 没有K线数据")
                    return None

                # 按时间正序排列（从早到晚）
                kline_data.reverse()

                # 转换为CSV格式
                csv_lines = []
                # CSV头部
                csv_lines.append("timestamp,open,high,low,close,volume,amount")

                # 数据行
                for kline in kline_data:
                    timestamp_str = kline.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    csv_lines.append(
                        f"{timestamp_str},{kline.open:.2f},{kline.high:.2f},"
                        f"{kline.low:.2f},{kline.close:.2f},"
                        f"{kline.volume or 0:.2f},{kline.amount or 0:.2f}"
                    )

                logger.info(f"计划 {plan_id} 获取到 {len(kline_data)} 条K线数据")
                return "\n".join(csv_lines)

        except Exception as e:
            logger.error(f"获取历史K线数据失败: {e}")
            return None

    @staticmethod
    def get_optimal_historical_data(plan_id: int) -> Optional[str]:
        """
        获取最优的历史数据：优先24小时数据，不足时使用最近N条数据

        Args:
            plan_id: 计划ID

        Returns:
            Optional[str]: CSV格式的K线数据
        """
        # 首先尝试获取24小时数据
        data = HistoricalDataService.get_recent_24h_kline_data(plan_id)

        if data:
            return data

        # 如果24小时数据不足，获取最近24条数据
        logger.info(f"计划 {plan_id} 24小时数据不足，使用最近24条数据")
        return HistoricalDataService.get_recent_kline_data_by_count(plan_id, 24)


# 创建全局服务实例
historical_data_service = HistoricalDataService()