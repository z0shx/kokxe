"""
OKX REST API 客户端
"""
import time
import hmac
import base64
import hashlib
import requests
from typing import Dict, List, Optional
from datetime import datetime, timezone
from config import config
from utils.logger import setup_logger

logger = setup_logger(__name__, "okx_api.log")


class OKXClient:
    """OKX REST API 客户端"""

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        passphrase: str = None,
        is_demo: bool = True
    ):
        """
        初始化 OKX 客户端

        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            is_demo: 是否模拟盘
        """
        self.api_key = api_key or config.OKX_API_KEY
        self.secret_key = secret_key or config.OKX_SECRET_KEY
        self.passphrase = passphrase or config.OKX_PASSPHRASE
        self.is_demo = is_demo

        self.environment = "DEMO" if is_demo else "LIVE"

        # 设置 API 地址
        if is_demo:
            self.base_url = config.OKX_DEMO_REST_URL
        else:
            self.base_url = config.OKX_REST_URL

        self.session = requests.Session()

        # 设置代理
        proxies = config.PROXIES
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"[{self.environment}] 使用代理: {config.PROXY_URL}")
        else:
            logger.info(f"[{self.environment}] 未使用代理")

        logger.info(f"[{self.environment}] OKX REST API 客户端初始化完成")

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        生成签名

        Args:
            timestamp: 时间戳
            method: 请求方法
            request_path: 请求路径
            body: 请求体

        Returns:
            签名字符串
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()

    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict:
        """
        获取请求头

        Args:
            method: 请求方法
            request_path: 请求路径
            body: 请求体

        Returns:
            请求头字典
        """
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'

        headers = {
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self._sign(timestamp, method, request_path, body),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
        }

        # 模拟盘需要添加特殊头
        if self.is_demo:
            headers['x-simulated-trading'] = '1'

        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        data: Dict = None,
        auth: bool = False
    ) -> Dict:
        """
        发送 HTTP 请求

        Args:
            method: 请求方法
            endpoint: API 端点
            params: URL 参数
            data: 请求体数据
            auth: 是否需要签名

        Returns:
            响应数据
        """
        url = self.base_url + endpoint

        # 构建查询字符串
        query_string = ""
        if params:
            query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
            url += query_string

        # 构建请求路径（用于签名）
        request_path = endpoint + query_string

        # 准备请求体
        body = ""
        if data:
            import json
            body = json.dumps(data)

        # 准备请求头
        headers = {}
        if auth and self.api_key:
            headers = self._get_headers(method, request_path, body)
        else:
            headers = {'Content-Type': 'application/json'}
            if self.is_demo:
                headers['x-simulated-trading'] = '1'

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                data=body if data else None,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get('code') != '0':
                logger.error(f"[{self.environment}] API 错误: {result}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.environment}] 请求失败: {e}")
            raise

    def get_history_candles(
        self,
        inst_id: str,
        bar: str = "1H",
        after: str = None,
        before: str = None,
        limit: int = 300
    ) -> List[Dict]:
        """
        获取历史现货K线数据

        Args:
            inst_id: 交易对，如 ETH-USDT
            bar: 时间颗粒度（30m/1H/2H/4H等）
            after: 请求此时间戳之后的数据（毫秒，UTC时区）
            before: 请求此时间戳之前的数据（毫秒，UTC时区）
            limit: 返回结果数量，默认300

        Returns:
            K线数据列表（只返回 confirm=1 的数据）
        """
        endpoint = "/api/v5/market/history-candles"
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }

        # 只使用 after 或 before 其中一个，优先使用 after
        if after:
            params['after'] = str(after)
            logger.info(
                f"[{self.environment}] 获取历史K线: inst_id={inst_id}, "
                f"bar={bar}, after={after}, limit={limit}"
            )
        elif before:
            params['before'] = str(before)
            logger.info(
                f"[{self.environment}] 获取历史K线: inst_id={inst_id}, "
                f"bar={bar}, before={before}, limit={limit}"
            )
        else:
            logger.info(
                f"[{self.environment}] 获取历史K线: inst_id={inst_id}, "
                f"bar={bar}, limit={limit}"
            )

        result = self._request("GET", endpoint, params=params)

        if result.get('code') == '0':
            data = result.get('data', [])
            # 只保留 confirm=1 的数据
            confirmed_data = [candle for candle in data if len(candle) >= 9 and candle[8] == '1']
            logger.info(f"[{self.environment}] 获取到 {len(data)} 条K线数据，其中 {len(confirmed_data)} 条已确认")
            return confirmed_data
        else:
            logger.error(f"[{self.environment}] 获取历史K线失败: {result}")
            return []

    def parse_candle_data(self, candle: List) -> Dict:
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
        if len(candle) < 9:
            return None

        # 只处理已确认的K线数据
        if candle[8] != '1':
            return None

        # ⚠️ 重要: OKX API 返回的是 UTC 时间戳,必须显式指定 UTC 时区
        # 使用 timezone.utc 确保 datetime 对象是 timezone-aware 的
        result = {
            'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5]),  # vol - 交易量（币本位）
            'amount': float(candle[7])   # volCcyQuote - 交易额（USDT）
        }

        return result

    def get_all_spot_instruments(self) -> List[Dict]:
        """
        获取所有现货交易对

        Returns:
            交易对列表
        """
        endpoint = "/api/v5/public/instruments"
        params = {'instType': 'SPOT'}

        logger.info(f"[{self.environment}] 获取所有现货交易对")
        result = self._request("GET", endpoint, params=params)

        if result.get('code') == '0':
            instruments = result.get('data', [])
            logger.info(f"[{self.environment}] 获取到 {len(instruments)} 个现货交易对")
            return instruments
        else:
            logger.error(f"[{self.environment}] 获取交易对失败: {result}")
            return []
