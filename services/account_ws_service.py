"""
OKX 账户私有 WebSocket 服务
负责连接OKX私有WebSocket并实时更新账户余额和订单信息
"""
import asyncio
import json
import hmac
import base64
import time
from datetime import datetime
from typing import Optional, Dict, Callable
import websockets
from websockets.asyncio.client import connect
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "account_ws.log")


class OKXAccountWebSocket:
    """OKX账户WebSocket连接"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        callback: Optional[Callable] = None
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.callback = callback

        # WebSocket URL
        if is_demo:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"

        # 连接状态
        self.ws = None
        self.running = False
        self.connected = False

        # 数据缓存
        self.account_data = {}
        self.positions_data = []

        # 统计
        self.total_received = 0
        self.last_update_time = None
        self.last_error = None

    def _generate_signature(self, timestamp: str, method: str, request_path: str) -> str:
        """生成签名"""
        message = timestamp + method + request_path
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

    async def _login(self):
        """登录认证"""
        try:
            timestamp = str(int(time.time()))
            sign = self._generate_signature(timestamp, 'GET', '/users/self/verify')

            login_msg = {
                "op": "login",
                "args": [{
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": sign
                }]
            }

            logger.debug(f"发送登录请求: api_key={self.api_key[:8]}..., is_demo={self.is_demo}, timestamp={timestamp}")
            await self.ws.send(json.dumps(login_msg))

            # 等待登录响应
            response = await self.ws.recv()
            data = json.loads(response)

            if data.get('event') == 'login' and data.get('code') == '0':
                logger.info("✅ 登录成功")
                return True
            else:
                logger.error(f"❌ 登录失败: {data}")
                logger.error(f"  - API Key (前8位): {self.api_key[:8]}")
                logger.error(f"  - 模拟盘环境: {self.is_demo}")
                logger.error(f"  - WebSocket URL: {self.ws_url}")
                return False

        except Exception as e:
            logger.error(f"登录出错: {e}")
            return False

    async def _subscribe_channels(self):
        """订阅频道"""
        try:
            # 订阅账户余额和持仓频道
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    {"channel": "balance_and_position"}
                ]
            }

            await self.ws.send(json.dumps(subscribe_msg))
            logger.debug("订阅账户余额和持仓频道")

        except Exception as e:
            logger.error(f"订阅频道失败: {e}")

    async def _handle_message(self, message: str):
        """处理接收到的消息"""
        try:
            data = json.loads(message)

            # 处理订阅响应
            if data.get('event') == 'subscribe':
                logger.debug(f"订阅成功: {data.get('arg')}")
                return

            # 处理数据推送
            if 'data' in data and 'arg' in data:
                channel = data['arg'].get('channel')
                payload = data['data']

                if channel == 'balance_and_position':
                    self._update_balance_and_position(payload)

                self.total_received += 1
                self.last_update_time = datetime.now()

                # 触发回调
                if self.callback:
                    try:
                        self.callback({
                            'channel': channel,
                            'data': payload,
                            'timestamp': self.last_update_time
                        })
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始消息: {message}")
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_balance_and_position(self, data: list):
        """更新账户余额和持仓数据"""
        for item in data:
            # 更新余额数据
            bal_data = item.get('balData', [])
            for bal in bal_data:
                ccy = bal.get('ccy')
                self.account_data[ccy] = {
                    'available': float(bal.get('availBal', 0)),
                    'balance': float(bal.get('cashBal', 0)),
                    'frozen': float(bal.get('frozenBal', 0)),
                    'equity': float(bal.get('bal', 0)) if bal.get('bal') else float(bal.get('cashBal', 0)),
                    'currency': ccy,
                    'update_time': datetime.now()
                }

            # 更新持仓数据
            pos_data = item.get('posData', [])
            self.positions_data = []
            for pos in pos_data:
                self.positions_data.append({
                    'inst_id': pos.get('instId'),
                    'pos_side': pos.get('posSide'),
                    'pos': float(pos.get('pos', 0)),
                    'avg_price': float(pos.get('avgPx', 0)) if pos.get('avgPx') else 0,
                    'upl': float(pos.get('upl', 0)),
                    'upl_ratio': float(pos.get('uplRatio', 0)) if pos.get('uplRatio') else 0,
                    'margin': float(pos.get('margin', 0)) if pos.get('margin') else 0,
                    'lever': float(pos.get('lever', 0)) if pos.get('lever') else 0,
                    'update_time': datetime.now()
                })

        logger.debug(f"账户数据已更新: {len(self.account_data)} 币种, {len(self.positions_data)} 个持仓")

    async def start(self):
        """启动WebSocket连接"""
        self.running = True
        retry_count = 0
        max_retries = 5

        # 设置代理环境变量（如果启用代理）
        import os
        original_proxy = os.environ.get('http_proxy')
        original_https_proxy = os.environ.get('https_proxy')

        try:
            if config.PROXY_ENABLED and config.PROXY_URL:
                os.environ['http_proxy'] = config.PROXY_URL
                os.environ['https_proxy'] = config.PROXY_URL
                logger.info(f"使用代理: {config.PROXY_URL}")

            while self.running and retry_count < max_retries:
                try:
                    logger.debug(f"正在连接 OKX 账户 WebSocket: {self.ws_url}")

                    async with websockets.connect(
                        self.ws_url,
                        ping_interval=20,
                        ping_timeout=10
                    ) as websocket:
                        self.ws = websocket
                        self.connected = True
                        retry_count = 0  # 重置重试计数

                        # 登录
                        login_success = await self._login()
                        if not login_success:
                            logger.error("登录失败，停止连接")
                            self.connected = False
                            break

                        # 订阅频道
                        await self._subscribe_channels()

                        logger.info("✅ OKX 账户 WebSocket 已启动")

                        # 接收消息循环
                        while self.running:
                            try:
                                message = await asyncio.wait_for(
                                    websocket.recv(),
                                    timeout=30
                                )
                                await self._handle_message(message)

                            except asyncio.TimeoutError:
                                # 发送ping保持连接
                                await websocket.ping()
                                continue
                            except websockets.exceptions.ConnectionClosed:
                                logger.warning("WebSocket连接关闭")
                                self.connected = False
                                break

                except Exception as e:
                    self.connected = False
                    self.last_error = str(e)
                    retry_count += 1
                    logger.error(f"WebSocket错误 (尝试 {retry_count}/{max_retries}): {e}")

                    if retry_count < max_retries and self.running:
                        wait_time = min(2 ** retry_count, 30)
                        logger.info(f"等待 {wait_time} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("达到最大重试次数，停止连接")
                        break

            self.connected = False
            self.running = False
            logger.info("OKX 账户 WebSocket 已停止")

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

    async def stop(self):
        """停止WebSocket连接"""
        logger.info("正在停止 OKX 账户 WebSocket...")
        self.running = False
        self.connected = False

        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"关闭 WebSocket 失败: {e}")

    def get_account_info(self) -> Dict:
        """获取账户信息"""
        return {
            'balances': self.account_data,
            'positions': self.positions_data,
            'connected': self.connected,
            'last_update': self.last_update_time,
            'total_received': self.total_received
        }

    def get_status(self) -> Dict:
        """获取连接状态"""
        return {
            'connected': self.connected,
            'running': self.running,
            'api_key': self.api_key[:8] + '...',
            'is_demo': self.is_demo,
            'total_received': self.total_received,
            'last_update': self.last_update_time,
            'last_error': self.last_error
        }
