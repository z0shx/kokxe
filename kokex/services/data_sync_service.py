"""
数据同步服务
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import and_, desc
from database.db import get_db
from database.models import KlineData, SystemLog
from api.okx_client import OKXClient
from api.okx_websocket import OKXWebSocket
from utils.logger import setup_logger

logger = setup_logger(__name__, "data_sync.log")


class DataSyncService:
    """K线数据同步服务"""

    def __init__(
        self,
        inst_id: str,
        interval: str,
        is_demo: bool = True
    ):
        """
        初始化数据同步服务

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
        """
        self.inst_id = inst_id
        self.interval = interval
        self.is_demo = is_demo
        self.environment = "DEMO" if is_demo else "LIVE"

        self.okx_client = OKXClient(is_demo=is_demo)
        self.ws_client = None
        self.running = False

        logger.info(
            f"[{self.environment}] 数据同步服务初始化: "
            f"{inst_id} {interval}"
        )

    def get_interval_seconds(self) -> int:
        """获取时间颗粒度对应的秒数"""
        mapping = {
            "30m": 30 * 60,
            "1H": 60 * 60,
            "2H": 2 * 60 * 60,
            "4H": 4 * 60 * 60
        }
        return mapping.get(self.interval, 60 * 60)

    def check_data_exists(self) -> bool:
        """
        检查数据库中是否已有该交易对的K线数据

        Returns:
            是否存在数据
        """
        with get_db() as db:
            count = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                )
            ).count()

            exists = count > 0
            logger.info(
                f"[{self.environment}] 数据库中 {self.inst_id} {self.interval} "
                f"{'已有' if exists else '无'} 数据 (共 {count} 条)"
            )
            return exists

    def get_latest_timestamp(self) -> Optional[datetime]:
        """
        获取数据库中最新的K线时间戳

        Returns:
            最新时间戳，如果没有数据则返回 None
        """
        with get_db() as db:
            latest = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                )
            ).order_by(desc(KlineData.timestamp)).first()

            if latest:
                logger.info(
                    f"[{self.environment}] 最新K线时间: {latest.timestamp}"
                )
                return latest.timestamp
            return None

    def initialize_data(self, days: int = 30) -> bool:
        """
        初始化历史K线数据

        Args:
            days: 下载最近多少天的数据

        Returns:
            是否成功
        """
        logger.info(
            f"[{self.environment}] 开始初始化数据: "
            f"{self.inst_id} {self.interval}, 最近 {days} 天"
        )

        try:
            # 计算起始时间（毫秒时间戳）
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            # 转换为毫秒时间戳
            after_ts = int(start_time.timestamp() * 1000)
            all_candles = []

            # 分批下载（每次最多100条）
            current_after = after_ts
            max_iterations = 1000  # 防止无限循环

            for i in range(max_iterations):
                candles = self.okx_client.get_history_candles(
                    inst_id=self.inst_id,
                    bar=self.interval,
                    after=str(current_after),
                    limit=100
                )

                if not candles:
                    break

                all_candles.extend(candles)
                logger.info(
                    f"[{self.environment}] 已下载 {len(all_candles)} 条数据"
                )

                # 更新 after 参数为最后一条数据的时间戳
                last_ts = int(candles[-1][0])
                if last_ts == current_after:
                    # 没有更多数据了
                    break
                current_after = last_ts

                # 避免请求过快
                import time
                time.sleep(0.2)

            # 保存到数据库
            saved_count = self._save_candles(all_candles)

            logger.info(
                f"[{self.environment}] 数据初始化完成: "
                f"下载 {len(all_candles)} 条, 保存 {saved_count} 条"
            )

            # 记录系统日志
            self._log_system(
                "data_init",
                "info",
                f"数据初始化完成: 下载 {len(all_candles)} 条, 保存 {saved_count} 条"
            )

            return True

        except Exception as e:
            logger.error(f"[{self.environment}] 数据初始化失败: {e}")
            self._log_system("data_init", "error", f"数据初始化失败: {e}")
            return False

    def _save_candles(self, candles: List[list]) -> int:
        """
        保存K线数据到数据库（避免重复）

        Args:
            candles: K线数据列表

        Returns:
            实际保存的数量
        """
        saved_count = 0

        with get_db() as db:
            for candle in candles:
                parsed = self.okx_client.parse_candle_data(candle)
                if not parsed:
                    continue

                # 检查是否已存在
                exists = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == self.inst_id,
                        KlineData.interval == self.interval,
                        KlineData.timestamp == parsed['timestamp']
                    )
                ).first()

                if exists:
                    continue

                # 插入新数据
                kline = KlineData(
                    inst_id=self.inst_id,
                    interval=self.interval,
                    **parsed
                )
                db.add(kline)
                saved_count += 1

            db.commit()

        return saved_count

    def check_and_fill_gaps(self) -> int:
        """
        检查并填补数据缺失

        Returns:
            填补的数据点数量
        """
        logger.info(f"[{self.environment}] 开始检查数据缺失")

        with get_db() as db:
            # 获取所有数据点的时间戳
            klines = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                )
            ).order_by(KlineData.timestamp).all()

            if len(klines) < 2:
                logger.info(f"[{self.environment}] 数据点不足，无需检查")
                return 0

            # 检查时间间隔
            interval_seconds = self.get_interval_seconds()
            gaps = []

            for i in range(len(klines) - 1):
                current_time = klines[i].timestamp
                next_time = klines[i + 1].timestamp
                expected_time = current_time + timedelta(seconds=interval_seconds)

                if next_time > expected_time:
                    # 发现缺失
                    gap_start = current_time
                    gap_end = next_time
                    gaps.append((gap_start, gap_end))
                    logger.warning(
                        f"[{self.environment}] 发现数据缺失: "
                        f"{gap_start} -> {gap_end}"
                    )

            if not gaps:
                logger.info(f"[{self.environment}] 未发现数据缺失")
                return 0

            # 填补缺失
            total_filled = 0
            for gap_start, gap_end in gaps:
                filled = self._fill_gap(gap_start, gap_end)
                total_filled += filled

            logger.info(
                f"[{self.environment}] 数据缺失检查完成: "
                f"发现 {len(gaps)} 处缺失, 填补 {total_filled} 条"
            )

            return total_filled

    def _fill_gap(self, start_time: datetime, end_time: datetime) -> int:
        """
        填补指定时间段的数据缺失

        Args:
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            填补的数据点数量
        """
        # 转换为毫秒时间戳
        after_ts = int(start_time.timestamp() * 1000)
        before_ts = int(end_time.timestamp() * 1000)

        # 从 OKX API 获取数据
        candles = self.okx_client.get_history_candles(
            inst_id=self.inst_id,
            bar=self.interval,
            after=str(after_ts),
            before=str(before_ts),
            limit=100
        )

        # 保存数据
        filled_count = self._save_candles(candles)

        logger.info(
            f"[{self.environment}] 填补数据: "
            f"{start_time} -> {end_time}, 填补 {filled_count} 条"
        )

        return filled_count

    async def start_websocket(self, plan_id: Optional[int] = None):
        """
        启动 WebSocket 数据流

        Args:
            plan_id: 关联的计划ID
        """
        logger.info(f"[{self.environment}] 启动 WebSocket 数据流")

        # 创建消息处理回调
        async def on_message(candle):
            await self._handle_ws_candle(candle, plan_id)

        # 创建 WebSocket 客户端
        self.ws_client = OKXWebSocket(
            inst_id=self.inst_id,
            interval=self.interval,
            on_message=on_message,
            is_demo=self.is_demo
        )

        # 启动 WebSocket
        self.running = True
        await self.ws_client.start()

    async def _handle_ws_candle(self, candle: list, plan_id: Optional[int]):
        """
        处理 WebSocket 接收到的K线数据

        Args:
            candle: K线数据
            plan_id: 计划ID
        """
        parsed = self.ws_client.parse_candle(candle)
        if not parsed:
            return

        # 只保存已确认的K线
        if not parsed.get('confirmed', True):
            logger.debug(f"[{self.environment}] 跳过未确认的K线数据")
            return

        # 保存到数据库
        try:
            with get_db() as db:
                # 检查是否已存在
                exists = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == parsed['inst_id'],
                        KlineData.interval == parsed['interval'],
                        KlineData.timestamp == parsed['timestamp']
                    )
                ).first()

                if exists:
                    logger.debug(f"[{self.environment}] K线数据已存在，跳过")
                    return

                # 插入新数据
                kline = KlineData(
                    inst_id=parsed['inst_id'],
                    interval=parsed['interval'],
                    timestamp=parsed['timestamp'],
                    open=parsed['open'],
                    high=parsed['high'],
                    low=parsed['low'],
                    close=parsed['close'],
                    volume=parsed['volume'],
                    amount=parsed['amount']
                )
                db.add(kline)
                db.commit()

                logger.info(
                    f"[{self.environment}] 保存K线数据: "
                    f"{parsed['timestamp']}, close={parsed['close']}"
                )

                # 检查数据缺失
                asyncio.create_task(self._async_check_gaps())

        except Exception as e:
            logger.error(f"[{self.environment}] 保存K线数据失败: {e}")
            self._log_system("ws_save", "error", f"保存K线数据失败: {e}", plan_id)

    async def _async_check_gaps(self):
        """异步检查数据缺失"""
        await asyncio.sleep(1)  # 延迟1秒后检查
        self.check_and_fill_gaps()

    async def stop_websocket(self):
        """停止 WebSocket 数据流"""
        self.running = False
        if self.ws_client:
            await self.ws_client.stop()
        logger.info(f"[{self.environment}] WebSocket 数据流已停止")

    def _log_system(
        self,
        log_type: str,
        level: str,
        message: str,
        plan_id: Optional[int] = None,
        details: dict = None
    ):
        """
        记录系统日志到数据库

        Args:
            log_type: 日志类型
            level: 日志级别
            message: 日志消息
            plan_id: 计划ID
            details: 详细信息
        """
        try:
            with get_db() as db:
                log = SystemLog(
                    plan_id=plan_id,
                    log_type=log_type,
                    level=level,
                    environment=self.environment,
                    message=message,
                    details=details
                )
                db.add(log)
                db.commit()
        except Exception as e:
            logger.error(f"[{self.environment}] 记录系统日志失败: {e}")
