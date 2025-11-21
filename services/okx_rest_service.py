"""
OKX REST API 服务
用于获取订单列表等数据
"""
import hmac
import base64
import time
import requests
from datetime import datetime
from typing import Optional, List, Dict
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "okx_rest.log")


class OKXRestService:
    """OKX REST API 服务"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo

        # API Base URL
        if is_demo:
            self.base_url = "https://www.okx.com"
        else:
            self.base_url = "https://www.okx.com"

        # 初始化时获取并缓存代理配置
        if config.PROXY_ENABLED and config.PROXY_URL:
            self.proxies = {
                "http": config.PROXY_URL,
                "https": config.PROXY_URL
            }
            logger.info(f"OKX REST API 使用代理: {config.PROXY_URL}")
        else:
            self.proxies = None
            logger.info("OKX REST API 未启用代理")

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """生成签名"""
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """生成请求头"""
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        sign = self._generate_signature(timestamp, method, request_path, body)

        headers = {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }

        if self.is_demo:
            headers['x-simulated-trading'] = '1'

        return headers

    def get_pending_orders(self, inst_type: str = "SPOT", inst_id: Optional[str] = None) -> List[Dict]:
        """
        获取未完成订单列表

        Args:
            inst_type: 产品类型，SPOT: 币币
            inst_id: 交易对，如 BTC-USDT

        Returns:
            订单列表
        """
        try:
            request_path = "/api/v5/trade/orders-pending"
            params = {"instType": inst_type}

            if inst_id:
                params["instId"] = inst_id

            # 构建查询参数
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_path = f"{request_path}?{query_string}"

            headers = self._get_headers("GET", full_path)
            url = f"{self.base_url}{full_path}"

            response = requests.get(url, headers=headers, proxies=self.proxies, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code') == '0':
                orders = data.get('data', [])
                logger.info(f"获取到 {len(orders)} 个未完成订单")
                return orders
            else:
                logger.error(f"获取订单失败: {data.get('msg')}")
                return []

        except Exception as e:
            logger.error(f"获取订单列表失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_order_history(
        self,
        inst_type: str = "SPOT",
        inst_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取历史订单列表（最近7天）

        Args:
            inst_type: 产品类型
            inst_id: 交易对
            limit: 返回数量限制

        Returns:
            订单列表
        """
        try:
            request_path = "/api/v5/trade/orders-history"
            params = {
                "instType": inst_type,
                "limit": str(limit)
            }

            if inst_id:
                params["instId"] = inst_id

            # 构建查询参数
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_path = f"{request_path}?{query_string}"

            headers = self._get_headers("GET", full_path)
            url = f"{self.base_url}{full_path}"

            response = requests.get(url, headers=headers, proxies=self.proxies, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code') == '0':
                orders = data.get('data', [])
                logger.info(f"获取到 {len(orders)} 个历史订单")
                return orders
            else:
                logger.error(f"获取历史订单失败: {data.get('msg')}")
                return []

        except Exception as e:
            logger.error(f"获取历史订单失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_orders(
        self,
        inst_type: str = "SPOT",
        inst_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        获取所有订单（未完成 + 历史）

        Args:
            inst_type: 产品类型
            inst_id: 交易对
            limit: 历史订单返回数量

        Returns:
            订单列表（未完成订单在前）
        """
        pending_orders = self.get_pending_orders(inst_type, inst_id)
        history_orders = self.get_order_history(inst_type, inst_id, limit)

        # 合并订单，未完成订单在前
        all_orders = pending_orders + history_orders

        return all_orders
