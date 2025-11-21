"""
OKX WebSocket 客户端 - 私有频道
支持订单频道、成交频道、下单、撤单、改单等操作
"""
import json
import time
import hmac
import base64
import asyncio
import websockets
from typing import Dict, List, Optional, Callable
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__, "okx_websocket.log")


class OKXWebSocketClient:
    """OKX WebSocket 私有频道客户端"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        on_message: Optional[Callable] = None
    ):
        """
        初始化 WebSocket 客户端

        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            is_demo: 是否模拟盘
            on_message: 消息回调函数
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.on_message = on_message

        self.environment = "DEMO" if is_demo else "LIVE"

        # WebSocket 地址
        if is_demo:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/private"
            self.ws_business_url = "wss://wspap.okx.com:8443/ws/v5/business"
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/private"
            self.ws_business_url = "wss://ws.okx.com:8443/ws/v5/business"

        self.ws = None
        self.ws_business = None
        self.is_connected = False
        self.is_authenticated = False
        self.subscriptions = []

        logger.info(f"[{self.environment}] OKX WebSocket 客户端初始化完成")

    def _generate_signature(self, timestamp: str, method: str, request_path: str) -> str:
        """
        生成 WebSocket 认证签名

        Args:
            timestamp: 时间戳
            method: 请求方法 (GET)
            request_path: 请求路径 (/users/self/verify)

        Returns:
            签名字符串
        """
        message = timestamp + method + request_path
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

    async def _login(self, ws):
        """
        WebSocket 登录认证

        Args:
            ws: WebSocket 连接
        """
        timestamp = str(int(time.time()))
        sign = self._generate_signature(timestamp, "GET", "/users/self/verify")

        login_msg = {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": sign
            }]
        }

        await ws.send(json.dumps(login_msg))
        logger.info(f"[{self.environment}] 发送登录请求")

        # 等待登录响应
        response = await ws.recv()
        resp_data = json.loads(response)

        if resp_data.get("event") == "login" and resp_data.get("code") == "0":
            self.is_authenticated = True
            logger.info(f"[{self.environment}] WebSocket 登录成功")
        else:
            logger.error(f"[{self.environment}] WebSocket 登录失败: {resp_data}")
            raise Exception(f"WebSocket 登录失败: {resp_data}")

    async def connect(self):
        """连接到 WebSocket 私有频道"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.is_connected = True
            logger.info(f"[{self.environment}] WebSocket 私有频道连接成功")

            # 登录认证
            await self._login(self.ws)

            # 启动消息接收循环
            asyncio.create_task(self._receive_messages())

        except Exception as e:
            logger.error(f"[{self.environment}] WebSocket 连接失败: {e}")
            self.is_connected = False
            raise

    async def connect_business(self):
        """连接到 WebSocket 业务频道 (用于下单、撤单等操作)"""
        try:
            self.ws_business = await websockets.connect(self.ws_business_url)
            logger.info(f"[{self.environment}] WebSocket 业务频道连接成功")

            # 登录认证
            await self._login(self.ws_business)

        except Exception as e:
            logger.error(f"[{self.environment}] WebSocket 业务频道连接失败: {e}")
            raise

    async def _receive_messages(self):
        """接收 WebSocket 消息循环"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                logger.debug(f"[{self.environment}] 收到消息: {data}")

                # 调用回调函数
                if self.on_message:
                    try:
                        await self.on_message(data)
                    except Exception as e:
                        logger.error(f"[{self.environment}] 消息回调处理失败: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[{self.environment}] WebSocket 连接已关闭")
            self.is_connected = False
            self.is_authenticated = False
        except Exception as e:
            logger.error(f"[{self.environment}] 消息接收异常: {e}")
            self.is_connected = False
            self.is_authenticated = False

    async def subscribe(self, channel: str, inst_type: str = None, inst_id: str = None):
        """
        订阅频道

        Args:
            channel: 频道名称 (orders/account/positions等)
            inst_type: 产品类型 (SPOT/SWAP等)
            inst_id: 产品ID (可选)
        """
        if not self.is_authenticated:
            raise Exception("WebSocket 未认证，无法订阅")

        args = {}
        if channel:
            args["channel"] = channel
        if inst_type:
            args["instType"] = inst_type
        if inst_id:
            args["instId"] = inst_id

        subscribe_msg = {
            "op": "subscribe",
            "args": [args]
        }

        await self.ws.send(json.dumps(subscribe_msg))
        self.subscriptions.append(args)
        logger.info(f"[{self.environment}] 订阅频道: {subscribe_msg}")

    async def unsubscribe(self, channel: str, inst_type: str = None, inst_id: str = None):
        """
        取消订阅频道

        Args:
            channel: 频道名称
            inst_type: 产品类型
            inst_id: 产品ID
        """
        args = {}
        if channel:
            args["channel"] = channel
        if inst_type:
            args["instType"] = inst_type
        if inst_id:
            args["instId"] = inst_id

        unsubscribe_msg = {
            "op": "unsubscribe",
            "args": [args]
        }

        await self.ws.send(json.dumps(unsubscribe_msg))
        logger.info(f"[{self.environment}] 取消订阅频道: {unsubscribe_msg}")

    async def place_order(
        self,
        inst_id: str,
        side: str,
        order_type: str,
        size: str,
        price: str = None,
        client_order_id: str = None
    ) -> Dict:
        """
        下单 (限价单)

        Args:
            inst_id: 交易对，如 BTC-USDT
            side: 订单方向 (buy/sell)
            order_type: 订单类型 (limit/market)
            size: 委托数量
            price: 委托价格 (限价单必填)
            client_order_id: 客户端订单ID

        Returns:
            下单响应
        """
        if not self.ws_business:
            await self.connect_business()

        order_data = {
            "id": client_order_id or f"order_{int(time.time() * 1000)}",
            "op": "order",
            "args": [{
                "instId": inst_id,
                "tdMode": "cash",  # 现货交易模式
                "side": side,
                "ordType": order_type,
                "sz": size
            }]
        }

        # 限价单需要价格
        if order_type == "limit" and price:
            order_data["args"][0]["px"] = price

        # 客户端订单ID
        if client_order_id:
            order_data["args"][0]["clOrdId"] = client_order_id

        await self.ws_business.send(json.dumps(order_data))
        logger.info(f"[{self.environment}] 下单请求: {order_data}")

        # 等待响应
        response = await self.ws_business.recv()
        resp_data = json.loads(response)
        logger.info(f"[{self.environment}] 下单响应: {resp_data}")

        return resp_data

    async def cancel_order(
        self,
        inst_id: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict:
        """
        撤单

        Args:
            inst_id: 交易对
            order_id: 订单ID (order_id 和 client_order_id 必须提供一个)
            client_order_id: 客户端订单ID

        Returns:
            撤单响应
        """
        if not self.ws_business:
            await self.connect_business()

        cancel_data = {
            "id": f"cancel_{int(time.time() * 1000)}",
            "op": "cancel-order",
            "args": [{
                "instId": inst_id
            }]
        }

        if order_id:
            cancel_data["args"][0]["ordId"] = order_id
        elif client_order_id:
            cancel_data["args"][0]["clOrdId"] = client_order_id
        else:
            raise ValueError("必须提供 order_id 或 client_order_id")

        await self.ws_business.send(json.dumps(cancel_data))
        logger.info(f"[{self.environment}] 撤单请求: {cancel_data}")

        # 等待响应
        response = await self.ws_business.recv()
        resp_data = json.loads(response)
        logger.info(f"[{self.environment}] 撤单响应: {resp_data}")

        return resp_data

    async def amend_order(
        self,
        inst_id: str,
        order_id: str = None,
        client_order_id: str = None,
        new_size: str = None,
        new_price: str = None
    ) -> Dict:
        """
        改单

        Args:
            inst_id: 交易对
            order_id: 订单ID
            client_order_id: 客户端订单ID
            new_size: 新的委托数量
            new_price: 新的委托价格

        Returns:
            改单响应
        """
        if not self.ws_business:
            await self.connect_business()

        amend_data = {
            "id": f"amend_{int(time.time() * 1000)}",
            "op": "amend-order",
            "args": [{
                "instId": inst_id
            }]
        }

        if order_id:
            amend_data["args"][0]["ordId"] = order_id
        elif client_order_id:
            amend_data["args"][0]["clOrdId"] = client_order_id
        else:
            raise ValueError("必须提供 order_id 或 client_order_id")

        if new_size:
            amend_data["args"][0]["newSz"] = new_size
        if new_price:
            amend_data["args"][0]["newPx"] = new_price

        await self.ws_business.send(json.dumps(amend_data))
        logger.info(f"[{self.environment}] 改单请求: {amend_data}")

        # 等待响应
        response = await self.ws_business.recv()
        resp_data = json.loads(response)
        logger.info(f"[{self.environment}] 改单响应: {resp_data}")

        return resp_data

    async def close(self):
        """关闭 WebSocket 连接"""
        if self.ws:
            await self.ws.close()
            logger.info(f"[{self.environment}] WebSocket 私有频道已关闭")

        if self.ws_business:
            await self.ws_business.close()
            logger.info(f"[{self.environment}] WebSocket 业务频道已关闭")

        self.is_connected = False
        self.is_authenticated = False
