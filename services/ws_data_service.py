"""
WebSocket 数据服务（增强版）
支持断线重连、数据去重、缺失填补
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable
from sqlalchemy.dialects.postgresql import insert
from api.okx_websocket import OKXWebSocket
from database.models import KlineData, WebSocketSubscription, TradingPlan, now_beijing
from database.db import get_db
from utils.logger import get_ws_logger
from utils.data_downloader import DataDownloader
from services.kline_event_service import kline_event_service
import json


class WebSocketDataService:
    """WebSocket 数据服务（增强版）"""

    def __init__(
        self,
        inst_id: str,
        interval: str,
        is_demo: bool = True,
        on_data_callback: Optional[Callable] = None
    ):
        """
        初始化 WebSocket 数据服务

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
            on_data_callback: 数据回调函数（用于实时更新UI）
        """
        self.inst_id = inst_id
        self.interval = interval
        self.is_demo = is_demo
        self.on_data_callback = on_data_callback
        self.environment = "DEMO" if is_demo else "LIVE"

        # 日志
        self.logger = get_ws_logger(inst_id, interval)

        # WebSocket 客户端
        self.ws_client = None
        self.running = False

        # 最后接收数据的时间
        self.last_data_time = None

        # 数据下载器（用于填补缺失）
        self.downloader = DataDownloader(inst_id, interval, is_demo)

        # 统计信息
        self.total_received = 0
        self.total_saved = 0

        # 订阅记录 ID
        self.subscription_id = None

    @property
    def is_connected(self) -> bool:
        """WebSocket 是否已连接"""
        return (
            self.running and
            self.ws_client is not None and
            hasattr(self.ws_client, 'subscribed') and
            self.ws_client.subscribed
        )

    def _get_or_create_subscription(self):
        """获取或创建订阅记录"""
        with get_db() as db:
            subscription = db.query(WebSocketSubscription).filter(
                WebSocketSubscription.inst_id == self.inst_id,
                WebSocketSubscription.interval == self.interval,
                WebSocketSubscription.is_demo == self.is_demo
            ).first()

            if not subscription:
                subscription = WebSocketSubscription(
                    inst_id=self.inst_id,
                    interval=self.interval,
                    is_demo=self.is_demo,
                    status='stopped'
                )
                db.add(subscription)
                db.commit()
                db.refresh(subscription)

            return subscription.id

    def _update_subscription_status(self, **kwargs):
        """更新订阅状态（同时同步到 TradingPlan）"""
        if not self.subscription_id:
            return

        with get_db() as db:
            # 1. 更新 WebSocketSubscription 表
            db.query(WebSocketSubscription).filter(
                WebSocketSubscription.id == self.subscription_id
            ).update(kwargs)
            db.commit()

            # 2. 同步更新 TradingPlan.ws_connected 字段
            if 'is_connected' in kwargs:
                ws_connected = kwargs['is_connected']

                # 查找所有匹配的计划
                plans = db.query(TradingPlan).filter(
                    TradingPlan.inst_id == self.inst_id,
                    TradingPlan.interval == self.interval,
                    TradingPlan.is_demo == self.is_demo
                ).all()

                for plan in plans:
                    if plan.ws_connected != ws_connected:
                        plan.ws_connected = ws_connected
                        self.logger.info(
                            f"[{self.environment}] 同步 TradingPlan.ws_connected: "
                            f"plan_id={plan.id}, ws_connected={ws_connected}"
                        )

                db.commit()

    async def start(self):
        """启动 WebSocket 服务"""
        if self.running:
            self.logger.warning(
                f"[{self.environment}] WebSocket 服务已在运行中"
            )
            return

        # 获取或创建订阅记录
        self.subscription_id = self._get_or_create_subscription()

        self.running = True
        self.logger.info(
            f"[{self.environment}] 启动 WebSocket 数据服务: "
            f"{self.inst_id} {self.interval}"
        )

        # 更新订阅状态
        self._update_subscription_status(
            status='running',
            is_connected=False,
            started_at=now_beijing()
        )

        # ⚠️ 在 WebSocket 连接之前，先检查并填补缺失的历史数据
        self.logger.info(f"[{self.environment}] 检查历史数据完整性...")
        await self._check_and_backfill_before_connect()

        # 创建 WebSocket 客户端
        self.ws_client = OKXWebSocket(
            inst_id=self.inst_id,
            interval=self.interval,
            on_message=self._handle_ws_message,
            is_demo=self.is_demo,
            on_connect_callback=self._on_connect_success,
            on_disconnect_callback=self._on_disconnect
        )

        # 启动 WebSocket 连接
        try:
            await self.ws_client.start()
        except Exception as e:
            self.logger.error(
                f"[{self.environment}] WebSocket 服务启动失败: {e}"
            )
            self.running = False
            self._update_subscription_status(
                status='error',
                is_connected=False,
                last_error=str(e),
                last_error_time=now_beijing(),
                error_count=self._get_error_count() + 1
            )

    async def _on_connect_success(self):
        """WebSocket连接和订阅成功回调"""
        self.logger.info(f"[{self.environment}] WebSocket订阅成功，更新连接状态")
        # 只有在订阅成功后才更新连接状态
        self._update_subscription_status(is_connected=True)

    async def _on_disconnect(self):
        """WebSocket断开连接回调"""
        self.logger.info(f"[{self.environment}] WebSocket断开连接，更新连接状态")
        # 更新连接状态为断开
        self._update_subscription_status(is_connected=False)

    def _get_error_count(self) -> int:
        """获取当前错误次数"""
        if not self.subscription_id:
            return 0
        with get_db() as db:
            sub = db.query(WebSocketSubscription).get(self.subscription_id)
            return sub.error_count if sub else 0

    async def _check_and_backfill_before_connect(self):
        """
        连接前检查并填补缺失数据

        在 WebSocket 连接建立之前调用，确保历史数据完整后再开始实时订阅
        """
        try:
            # 查询数据库中最后一条数据的时间
            with get_db() as db:
                last_record = db.query(KlineData).filter(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                ).order_by(KlineData.timestamp.desc()).first()

                if not last_record:
                    self.logger.info(
                        f"[{self.environment}] 数据库中无历史数据，跳过填补检查"
                    )
                    return

                # 数据库存储的是UTC+8北京时间的naive datetime，需要添加时区信息用于比较
                from database.models import from_beijing_naive
                last_timestamp_beijing = from_beijing_naive(last_record.timestamp)
                self.last_data_time = last_timestamp_beijing

            # 获取当前北京时间
            from database.models import BEIJING_TZ
            current_time_beijing = datetime.now(BEIJING_TZ)
            interval_minutes = self.downloader.checker.interval_minutes

            # 将当前时间向下对齐到周期边界（找到当前周期的开始时间）
            def align_to_period_start(dt, minutes):
                ts_seconds = int(dt.timestamp())
                period_seconds = minutes * 60
                aligned_seconds = (ts_seconds // period_seconds) * period_seconds
                return datetime.fromtimestamp(aligned_seconds, tz=dt.tzinfo)

            current_period_start = align_to_period_start(current_time_beijing, interval_minutes)

            # 最后一个完整周期 = 当前周期的前一个周期
            # 因为当前周期还没结束（或者刚开始）
            last_complete_period_start = current_period_start - timedelta(minutes=interval_minutes)

            # 计算时间差（从数据库最后时间到最后完整周期）
            time_diff = last_complete_period_start - last_timestamp_beijing

            self.logger.info(
                f"[{self.environment}] 数据完整性检查(UTC+8): "
                f"数据库最后={last_timestamp_beijing.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"最后完整周期={last_complete_period_start.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"差距={time_diff}"
            )

            # 如果有缺失的周期
            if time_diff > timedelta(minutes=0):
                self.logger.warning(
                    f"[{self.environment}] ⚠️ 检测到数据缺失: 缺少 {int(time_diff.total_seconds() / (interval_minutes * 60))} 个周期"
                )

                # 计算填补范围：
                # - start_backfill: 数据库最后时间的下一个周期（第一个需要填补的周期）
                # - end_backfill: 最后完整周期的下一个周期（因为 before 参数是不包含的）
                start_backfill = last_timestamp_beijing + timedelta(minutes=interval_minutes)
                # before 参数获取 < before 的数据，所以要 +1 周期才能包含 last_complete_period_start
                end_backfill = last_complete_period_start + timedelta(minutes=interval_minutes)

                self.logger.info(
                    f"[{self.environment}] 🔧 开始填补缺失数据(UTC+8): "
                    f"从 {start_backfill.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"到 {last_complete_period_start.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(API before参数={end_backfill.strftime('%Y-%m-%d %H:%M:%S')})"
                )

                # 同步执行数据填补（阻塞式，确保填补完成后再连接 WebSocket）
                loop = asyncio.get_event_loop()
                filled_count = await loop.run_in_executor(
                    None,
                    self._fill_gap_sync,
                    start_backfill,
                    end_backfill
                )

                self.logger.info(
                    f"[{self.environment}] ✅ 历史数据填补完成: 共填补 {filled_count} 条数据"
                )

                # 更新统计信息
                self.total_saved += filled_count
                self._update_subscription_status(
                    total_saved=self.total_saved
                )
            else:
                self.logger.info(
                    f"[{self.environment}] ✅ 历史数据完整，无需填补"
                )

        except Exception as e:
            self.logger.error(
                f"[{self.environment}] 填补检查失败: {e}"
            )
            import traceback
            traceback.print_exc()

    async def stop(self):
        """停止 WebSocket 服务"""
        self.running = False

        # 更新订阅状态
        self._update_subscription_status(
            status='stopped',
            is_connected=False,
            stopped_at=now_beijing()
        )

        if self.ws_client:
            try:
                await self.ws_client.stop()
            except Exception as e:
                self.logger.warning(f"[{self.environment}] 停止 WebSocket 时出错: {e}")

        self.logger.info(
            f"[{self.environment}] WebSocket 数据服务已停止"
        )

    async def _handle_ws_message(self, candle_data):
        """
        处理 WebSocket 消息

        Args:
            candle_data: K线数据（数组格式）
        """
        # 检查是否应该停止
        if not self.running:
            return

        try:
            # 统计接收消息数
            self.total_received += 1

            # 解析K线数据
            parsed = self.ws_client.parse_candle(candle_data)

            if not parsed:
                return

            # 检查是否需要填补缺失数据
            await self._check_and_fill_gaps(parsed['timestamp'])

            # 保存数据（新增或更新）
            is_new = await self._save_candle_data(parsed)

            # 只在新数据时增加计数
            if is_new:
                self.total_saved += 1

                # 触发K线数据事件（仅在新数据时）
                try:
                    kline_event_service.trigger_new_kline_event(
                        inst_id=self.inst_id,
                        interval=self.interval,
                        kline_data=parsed
                    )
                except Exception as e:
                    self.logger.error(f"触发K线事件失败: {e}")

            # 更新最后数据时间
            self.last_data_time = parsed['timestamp']

            # 更新订阅统计
            self._update_subscription_status(
                total_received=self.total_received,
                total_saved=self.total_saved,
                last_data_time=parsed['timestamp'],
                last_message=json.dumps({
                    'timestamp': parsed['timestamp'].isoformat(),
                    'close': parsed['close']
                })
            )

            # 触发回调（无论新数据还是更新都触发，用于实时图表更新）
            if self.on_data_callback:
                try:
                    await self.on_data_callback(parsed)
                except Exception as e:
                    self.logger.error(
                        f"[{self.environment}] 回调函数执行失败: {e}"
                    )

        except Exception as e:
            self.logger.error(
                f"[{self.environment}] 处理 WebSocket 消息失败: {e}"
            )
            # 更新错误信息
            self._update_subscription_status(
                last_error=str(e),
                last_error_time=now_beijing(),
                error_count=self._get_error_count() + 1
            )

    async def _check_and_fill_gaps(self, current_time: datetime):
        """
        检查并填补缺失的数据

        Args:
            current_time: 当前接收到的数据时间
        """
        if self.last_data_time is None:
            # 第一次接收数据，从数据库查询最后一条数据的时间
            with get_db() as db:
                last_record = db.query(KlineData).filter(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                ).order_by(KlineData.timestamp.desc()).first()

                if last_record:
                    # 数据库存储的是UTC+8北京时间的naive datetime，需要添加时区信息
                    from database.models import from_beijing_naive
                    self.last_data_time = from_beijing_naive(last_record.timestamp)

        if self.last_data_time:
            # 计算时间差
            time_diff = current_time - self.last_data_time
            interval_minutes = self.downloader.checker.interval_minutes
            expected_diff = timedelta(minutes=interval_minutes)

            # 如果时间差大于预期，说明有缺失
            if time_diff > expected_diff * 1.5:  # 允许50%的误差
                self.logger.warning(
                    f"[{self.environment}] 检测到数据缺失(UTC+8): "
                    f"最后数据时间 {self.last_data_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"当前数据时间 {current_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"差距 {time_diff}"
                )

                # 异步填补缺失数据
                asyncio.create_task(self._fill_gap_async(
                    self.last_data_time,
                    current_time
                ))

    async def _fill_gap_async(self, start_time: datetime, end_time: datetime):
        """
        异步填补缺失数据

        Args:
            start_time: 开始时间
            end_time: 结束时间
        """
        try:
            self.logger.info(
                f"[{self.environment}] 开始填补缺失数据: "
                f"{start_time} ~ {end_time}"
            )

            # 使用 loop.run_in_executor 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._fill_gap_sync,
                start_time,
                end_time
            )

            self.logger.info(
                f"[{self.environment}] 缺失数据填补完成: {result}"
            )

        except Exception as e:
            self.logger.error(
                f"[{self.environment}] 填补缺失数据失败: {e}"
            )

    def _fill_gap_sync(self, start_time: datetime, end_time: datetime) -> int:
        """
        同步填补缺失数据

        使用简化策略：直接获取最新 300 条已确认数据，保存到数据库（会自动去重）

        Args:
            start_time: 开始时间 (UTC timezone-aware) - 填补的最早时间
            end_time: 结束时间 (UTC timezone-aware) - 填补的最晚时间

        Returns:
            填补的数据条数
        """
        self.logger.info(
            f"[{self.environment}] 填补缺失数据: "
            f"从 {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"到 {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        # 直接获取最新已确认的数据（不传 after/before 参数）
        candles = self.downloader.okx_client.get_history_candles(
            inst_id=self.inst_id,
            bar=self.interval,
            limit=300
        )

        if not candles:
            self.logger.warning(
                f"[{self.environment}] 无法获取历史数据进行填补"
            )
            return 0

        # 筛选出需要填补的数据（在 start_time 和 end_time 之间的）
        filled_count = 0
        for candle in candles:
            candle_time_ms = int(candle[0])
            candle_time = datetime.fromtimestamp(candle_time_ms / 1000, tz=timezone.utc)

            # 只保存在时间范围内的数据
            if start_time <= candle_time < end_time:
                parsed = self.downloader.okx_client.parse_candle_data(candle)
                if parsed:
                    # 保存到数据库
                    with get_db() as db:
                        # 使用北京时间存储（统一时区标准）
                        # 优先使用 timestamp_beijing 字段，如果不存在则从 timestamp 转换
                        if 'timestamp_beijing' in parsed:
                            timestamp_beijing = parsed['timestamp_beijing']
                        else:
                            # 兼容旧格式，从UTC时间转换为北京时间
                            from database.models import to_beijing_naive
                            timestamp_beijing = to_beijing_naive(parsed['timestamp'])

                        # 检查是否已存在（使用北京时间比较）
                        existing = db.query(KlineData).filter(
                            KlineData.inst_id == self.inst_id,
                            KlineData.interval == self.interval,
                            KlineData.timestamp == timestamp_beijing
                        ).first()

                        if not existing:
                            new_data = KlineData(
                                inst_id=self.inst_id,
                                interval=self.interval,
                                timestamp=timestamp_beijing,  # 使用北京时间存储
                                open=parsed['open'],
                                high=parsed['high'],
                                low=parsed['low'],
                                close=parsed['close'],
                                volume=parsed['volume'],
                                amount=parsed['amount']
                            )
                            db.add(new_data)
                            db.commit()
                            filled_count += 1
                            self.logger.info(
                                f"[{self.environment}] 填补数据: {candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                            )

        return filled_count

    async def _save_candle_data(self, parsed_data: dict) -> bool:
        """
        保存K线数据（带去重）

        Args:
            parsed_data: 解析后的K线数据

        Returns:
            是否为新数据（True=新插入，False=更新已有数据）
        """
        try:
            # 检查服务是否仍在运行
            if not self.running:
                self.logger.warning(
                    f"[{self.environment}] 服务已停止，跳过数据保存"
                )
                return False

            # 检查事件循环是否仍在运行
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    self.logger.warning(
                        f"[{self.environment}] 事件循环已关闭，跳过数据保存"
                    )
                    return False
            except RuntimeError:
                # 没有运行中的事件循环，创建新的
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    is_new = loop.run_until_complete(self._save_to_db_sync(parsed_data))
                    loop.close()
                except Exception as e:
                    self.logger.error(
                        f"[{self.environment}] 同步保存数据失败: {e}"
                    )
                    return False

                # 只在新数据时打印日志
                if is_new:
                    self.logger.info(
                        f"[{self.environment}] 新增K线数据: "
                        f"{parsed_data['timestamp']}, "
                        f"close={parsed_data['close']}"
                    )
                return is_new

            # 事件循环正常，使用异步方式
            is_new = await loop.run_in_executor(
                None,
                self._save_to_db,
                parsed_data
            )

            # 只在新数据时打印日志
            if is_new:
                self.logger.info(
                    f"[{self.environment}] 新增K线数据: "
                    f"{parsed_data['timestamp']}, "
                    f"close={parsed_data['close']}"
                )

            return is_new

        except Exception as e:
            self.logger.error(
                f"[{self.environment}] 保存K线数据失败: {e}"
            )
            return False

    async def _save_to_db_sync(self, parsed_data: dict) -> bool:
        """
        同步保存数据到数据库（用于事件循环关闭时）
        """
        return self._save_to_db(parsed_data)

    def _save_to_db(self, parsed_data: dict) -> bool:
        """
        保存数据到数据库

        Args:
            parsed_data: 解析后的数据

        Returns:
            是否为新数据（True=新插入，False=更新已有数据）
        """
        # 使用北京时间存储（统一时区标准）
        # 优先使用 timestamp_beijing 字段，如果不存在则从 timestamp 转换
        if 'timestamp_beijing' in parsed_data:
            # 新的数据格式，直接使用北京时间naive datetime
            timestamp_beijing = parsed_data['timestamp_beijing']
        else:
            # 兼容旧格式，从UTC时间转换为北京时间
            from database.models import to_beijing_naive
            timestamp_beijing = to_beijing_naive(parsed_data['timestamp'])

        with get_db() as db:
            # 先检查数据是否存在（使用北京时间比较）
            existing = db.query(KlineData).filter(
                KlineData.inst_id == self.inst_id,
                KlineData.interval == self.interval,
                KlineData.timestamp == timestamp_beijing
            ).first()

            if existing:
                # 数据已存在，更新
                existing.open = parsed_data['open']
                existing.high = parsed_data['high']
                existing.low = parsed_data['low']
                existing.close = parsed_data['close']
                existing.volume = parsed_data['volume']
                existing.amount = parsed_data['amount']
                db.commit()
                return False  # 不是新数据
            else:
                # 新数据，插入（使用北京时间）
                new_data = KlineData(
                    inst_id=self.inst_id,
                    interval=self.interval,
                    timestamp=timestamp_beijing,
                    open=parsed_data['open'],
                    high=parsed_data['high'],
                    low=parsed_data['low'],
                    close=parsed_data['close'],
                    volume=parsed_data['volume'],
                    amount=parsed_data['amount']
                )
                db.add(new_data)
                db.commit()
                return True  # 是新数据

    def get_status(self) -> dict:
        """
        获取服务状态

        Returns:
            状态字典
        """
        return {
            'running': self.running,
            'connected': self.ws_client.running if self.ws_client else False,
            'inst_id': self.inst_id,
            'interval': self.interval,
            'last_data_time': self.last_data_time,
            'environment': self.environment
        }
