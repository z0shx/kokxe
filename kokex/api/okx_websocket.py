"""
OKX WebSocket 客户端
"""
import json
import asyncio
import websockets
from typing import Callable, Optional
from datetime import datetime, timezone
from config import config
from utils.logger import get_ws_logger

class OKXWebSocket:
    """OKX WebSocket 客户端"""

    def __init__(
        self,
        inst_id: str,
        interval: str,
        on_message: Callable,
        is_demo: bool = True,
        on_connect_callback: Optional[Callable] = None,
        on_disconnect_callback: Optional[Callable] = None
    ):
        """
        初始化 WebSocket 客户端

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            on_message: 消息处理回调函数
            is_demo: 是否模拟盘
            on_connect_callback: 连接成功回调函数
            on_disconnect_callback: 连接断开回调函数
        """
        self.inst_id = inst_id
        self.interval = interval
        self.on_message = on_message
        self.is_demo = is_demo
        self.on_connect_callback = on_connect_callback
        self.on_disconnect_callback = on_disconnect_callback

        # 设置 WebSocket URL
        if is_demo:
            self.ws_url = config.OKX_DEMO_WS_BUSINESS
        else:
            self.ws_url = config.OKX_WS_BUSINESS

        # 设置频道（现货K线）
        self.channel = config.WS_CHANNEL_MAPPING.get(interval, "candle1H")

        self.ws = None
        self.running = False
        self.reconnect_interval = 5  # 重连间隔（秒）
        self.max_reconnect_attempts = 10  # 最大重连次数
        self.reconnect_count = 0
        self.subscribed = False  # 新增：订阅状态

        self.environment = "DEMO" if is_demo else "LIVE"
        self.logger = get_ws_logger(inst_id, interval)
        self.logger.info(f"[{self.environment}] WebSocket 客户端初始化: {inst_id} {interval}")

    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            # 设置代理环境变量（如果启用代理）
            import os
            original_proxy = os.environ.get('http_proxy')
            original_https_proxy = os.environ.get('https_proxy')

            if config.PROXY_ENABLED and config.PROXY_URL:
                os.environ['http_proxy'] = config.PROXY_URL
                os.environ['https_proxy'] = config.PROXY_URL
                self.logger.info(f"[{self.environment}] 使用代理: {config.PROXY_URL}")

            try:
                self.ws = await websockets.connect(self.ws_url)
                self.logger.info(f"[{self.environment}] WebSocket 连接成功")

                # 订阅频道
                await self.subscribe()
                self.reconnect_count = 0  # 重置重连计数

                return True

            finally:
                # 恢复原始代理设置
                if original_proxy is not None:
                    os.environ['http_proxy'] = original_proxy
                elif 'http_proxy' in os.environ:
                    del os.environ['http_proxy']

                if original_https_proxy is not None:
                    os.environ['https_proxy'] = original_https_proxy
                elif 'https_proxy' in os.environ:
                    del os.environ['https_proxy']

        except Exception as e:
            self.logger.error(f"[{self.environment}] WebSocket 连接失败: {e}")
            return False

    async def subscribe(self):
        """订阅K线频道"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {
                    "channel": self.channel,
                    "instId": self.inst_id
                }
            ]
        }

        await self.ws.send(json.dumps(subscribe_msg))
        self.logger.info(
            f"[{self.environment}] 订阅频道: {self.channel}, "
            f"交易对: {self.inst_id}"
        )

    async def start(self):
        """启动 WebSocket 连接（带自动重连）"""
        self.running = True
        self.logger.info(f"[{self.environment}] 启动 WebSocket 服务")

        while self.running:
            try:
                # 尝试连接
                connected = await self.connect()
                if not connected:
                    await self._handle_reconnect()
                    continue

                # 接收消息
                async for message in self.ws:
                    # 检查是否应该停止（在处理消息前检查）
                    if not self.running:
                        self.logger.info(f"[{self.environment}] 检测到停止信号，退出消息接收循环")
                        break

                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"[{self.environment}] JSON 解析错误: {e}")
                    except Exception as e:
                        self.logger.error(f"[{self.environment}] 消息处理错误: {e}")

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"[{self.environment}] WebSocket 连接关闭")
                if self.running:
                    await self._handle_reconnect()
                else:
                    # 主动停止导致的连接关闭，不需要重连
                    break

            except Exception as e:
                self.logger.error(f"[{self.environment}] WebSocket 错误: {e}")
                if self.running:
                    await self._handle_reconnect()
                else:
                    break

        self.logger.info(f"[{self.environment}] WebSocket 服务循环已退出")

    async def _handle_message(self, data: dict):
        """
        处理接收到的消息

        Args:
            data: 消息数据
        """
        # 检查是否应该停止
        if not self.running:
            return

        # 检查是否是订阅确认消息
        if data.get('event') == 'subscribe':
            self.subscribed = True
            self.logger.info(f"[{self.environment}] 订阅成功: {data}")

            # 调用连接成功回调
            if self.on_connect_callback:
                try:
                    await self.on_connect_callback()
                except Exception as e:
                    self.logger.error(f"[{self.environment}] 连接回调执行错误: {e}")
            return

        # 检查是否是错误消息
        if data.get('event') == 'error':
            self.logger.error(f"[{self.environment}] 订阅错误: {data}")
            return

        # 处理K线数据
        if 'data' in data:
            candles = data['data']
            # 改为 debug 级别，避免过于频繁的日志输出
            self.logger.debug(
                f"[{self.environment}] 接收到 {len(candles)} 条K线数据"
            )

            # 调用回调函数
            for candle in candles:
                try:
                    await self.on_message(candle)
                except Exception as e:
                    self.logger.error(f"[{self.environment}] 回调函数执行错误: {e}")

    async def _handle_reconnect(self):
        """处理重连逻辑"""
        if self.reconnect_count >= self.max_reconnect_attempts:
            self.logger.error(
                f"[{self.environment}] 达到最大重连次数 "
                f"({self.max_reconnect_attempts})，停止重连"
            )
            self.running = False
            return

        self.reconnect_count += 1
        wait_time = self.reconnect_interval * self.reconnect_count

        self.logger.info(
            f"[{self.environment}] {wait_time} 秒后尝试重连 "
            f"(第 {self.reconnect_count} 次)"
        )

        await asyncio.sleep(wait_time)

    async def stop(self):
        """停止 WebSocket 连接"""
        self.logger.info(f"[{self.environment}] 正在停止 WebSocket 服务...")
        self.running = False
        self.subscribed = False

        # 先关闭 WebSocket 连接，这会导致 async for 循环立即退出
        if self.ws:
            try:
                await self.ws.close()
                self.logger.info(f"[{self.environment}] WebSocket 连接已关闭")
            except Exception as e:
                self.logger.warning(f"[{self.environment}] 关闭 WebSocket 时出错: {e}")

        # 调用断开连接回调
        if self.on_disconnect_callback:
            try:
                await self.on_disconnect_callback()
            except Exception as e:
                self.logger.error(f"[{self.environment}] 断开连接回调执行错误: {e}")

        self.logger.info(f"[{self.environment}] WebSocket 服务已停止")

    def parse_candle(self, candle: list) -> Optional[dict]:
        """
        解析现货K线数据

        Args:
            candle: K线数据数组 [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                - ts: 时间戳（毫秒，UTC 时区）
                - o: 开盘价
                - h: 最高价
                - l: 最低价
                - c: 收盘价
                - vol: 交易量（币本位）
                - volCcy: 交易量（计价货币）
                - volCcyQuote: 交易额（USDT）
                - confirm: 是否确认（0=未确认，1=已确认）

        Returns:
            解析后的字典，只返回 confirm=1 的数据
            注意: timestamp 是 UTC 时区的 datetime 对象 (timezone-aware)
        """
        try:
            if len(candle) < 9:
                return None

            # 只处理已确认的K线数据
            if candle[8] != '1':
                return None

            # ⚠️ 重要: OKX WebSocket 返回的是 UTC 时间戳,必须显式指定 UTC 时区
            # 使用 timezone.utc 确保 datetime 对象是 timezone-aware 的
            return {
                'inst_id': self.inst_id,
                'interval': self.interval,
                'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),  # vol - 交易量（币本位）
                'amount': float(candle[7])   # volCcyQuote - 交易额（USDT）
            }
        except (ValueError, IndexError) as e:
            self.logger.error(f"[{self.environment}] K线数据解析错误: {e}")
            return None
