"""
OKX交易工具
负责执行实际的交易操作（下单、调整、取消）
"""
import hmac
import base64
import json
import time
from datetime import datetime
from typing import Dict, Optional, List
import requests
from sqlalchemy import desc
from utils.logger import setup_logger
from config import Config

logger = setup_logger(__name__, "trading_tools.log")


def safe_float(value):
    """安全转换为浮点数"""
    try:
        return float(value) if value and str(value).strip() else 0.0
    except (ValueError, TypeError):
        return 0.0


def safe_int(value):
    """安全转换为整数"""
    try:
        return int(value) if value and str(value).strip() else 0
    except (ValueError, TypeError):
        return 0


def validate_and_generate_client_order_id(client_order_id: str = None) -> str:
    """
    验证并生成符合OKX要求的客户端订单ID

    OKX clOrdId 要求：
    - 字母数字字符，允许连字符(-)
    - 长度：1-32字符
    - 客户端自定义

    Args:
        client_order_id: 用户提供的客户端订单ID

    Returns:
        有效的客户端订单ID
    """
    if client_order_id:
        # 清理ID，只保留字母数字和连字符
        import re
        clean_id = re.sub(r'[^a-zA-Z0-9\-]', '', str(client_order_id))

        # 确保长度在1-32字符之间
        if len(clean_id) > 32:
            clean_id = clean_id[:32]
        elif len(clean_id) == 0:
            clean_id = None
        else:
            return clean_id

    # 生成新的客户端订单ID
    from datetime import datetime
    import random
    import string

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"KOKEX-{timestamp}-{random_str}"


class OKXTradingTools:
    """OKX交易工具类"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        trading_limits: Optional[Dict] = None,
        plan_id: Optional[int] = None
    ):
        """
        初始化交易工具

        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            is_demo: 是否使用模拟盘
            trading_limits: 交易限制配置
            plan_id: 关联的交易计划ID（用于存储订单到数据库）
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.trading_limits = trading_limits or {}
        self.plan_id = plan_id

        # 工具调用上下文信息（用于关联订单和对话）
        self.current_conversation_id = None
        self.current_tool_call_id = None

        # 加载配置
        self.config = Config()

        # 初始化API配置
        self._init_api_config()

    def set_tool_context(self, conversation_id: int = None, tool_call_id: str = None):
        """设置工具调用上下文，用于关联订单和对话记录"""
        self.current_conversation_id = conversation_id
        self.current_tool_call_id = tool_call_id

    def clear_tool_context(self):
        """清除工具调用上下文"""
        self.current_conversation_id = None
        self.current_tool_call_id = None

    def _init_api_config(self):
        """初始化API配置"""
        # API地址
        if self.is_demo:
            self.base_url = "https://www.okx.com"
            self.simulate_flag = "1"  # 模拟盘标志
        else:
            self.base_url = "https://www.okx.com"
            self.simulate_flag = "0"  # 实盘标志

    def _make_request(self, method: str, url: str, headers: Dict[str, str], body: str = "", timeout: int = 10) -> requests.Response:
        """
        发送HTTP请求（支持代理）

        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体
            timeout: 超时时间

        Returns:
            requests.Response对象
        """
        try:
            # 设置代理
            proxies = self.config.PROXIES if self.config.PROXY_ENABLED else None
            if proxies:
                logger.info(f"使用代理: {self.config.PROXY_URL}")

            # 发送请求
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, data=body, timeout=timeout, proxies=proxies)
            else:
                response = requests.request(method, url, headers=headers, data=body, timeout=timeout, proxies=proxies)

            return response

        except Exception as e:
            logger.error(f"HTTP请求失败: {e}")
            raise

    def _sign_request(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """
        生成签名

        Args:
            timestamp: 时间戳
            method: HTTP方法
            request_path: 请求路径
            body: 请求体

        Returns:
            签名字符串
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """
        生成请求头

        Args:
            method: HTTP方法
            request_path: 请求路径
            body: 请求体

        Returns:
            请求头字典
        """
        timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
        sign = self._sign_request(timestamp, method, request_path, body)

        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "x-simulated-trading": self.simulate_flag
        }

    def _save_order_to_database(
        self,
        order_id: str,
        inst_id: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        status: str = "live",
        tool_call_id: Optional[str] = None,
        conversation_id: Optional[int] = None
    ) -> None:
        """
        将订单保存到数据库

        Args:
            order_id: OKX订单ID
            inst_id: 交易对
            side: 买卖方向
            order_type: 订单类型
            size: 数量
            price: 价格
            status: 订单状态
            tool_call_id: 工具调用ID
            conversation_id: 对话会话ID
        """
        if not self.plan_id:
            logger.warning("plan_id未设置，跳过订单数据库存储")
            return

        try:
            from database.db import get_db
            from database.models import TradeOrder
            from database.models import now_beijing

            with get_db() as db:
                # 检查订单是否已存在
                existing_order = db.query(TradeOrder).filter(TradeOrder.order_id == order_id).first()
                if existing_order:
                    logger.info(f"订单 {order_id} 已存在，更新状态")
                    existing_order.status = status
                    existing_order.updated_at = now_beijing()
                else:
                    # 使用上下文信息如果没有显式传递参数
                    final_conversation_id = conversation_id or self.current_conversation_id
                    final_tool_call_id = tool_call_id or self.current_tool_call_id

                    # 创建新订单记录
                    new_order = TradeOrder(
                        plan_id=self.plan_id,
                        order_id=order_id,
                        inst_id=inst_id,
                        side=side,
                        order_type=order_type,
                        price=price,
                        size=size,
                        status=status,
                        is_demo=self.is_demo,
                        is_from_agent=True,  # 标记为Agent操作的订单
                        agent_message_id=None,  # 可以在工具调用时更新
                        conversation_id=final_conversation_id,
                        tool_call_id=final_tool_call_id
                    )
                    db.add(new_order)
                    logger.info(f"保存新订单到数据库: order_id={order_id}, plan_id={self.plan_id}")

                db.commit()

        except Exception as e:
            logger.error(f"保存订单到数据库失败: {e}")
            import traceback
            traceback.print_exc()

    def place_order(
        self,
        inst_id: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        td_mode: str = "cash"
    ) -> Dict:
        """
        下单

        Args:
            inst_id: 交易对（如 ETH-USDT）
            side: 交易方向（buy/sell）
            order_type: 订单类型（market/limit）
            size: 数量
            price: 价格（限价单必填）
            td_mode: 交易模式（cash=现货, cross=全仓, isolated=逐仓）

        Returns:
            订单结果
        """
        try:
            # 构建订单参数
            order_data = {
                "instId": inst_id,
                "tdMode": td_mode,
                "side": side,
                "ordType": order_type,
                "sz": str(size)
            }

            # 限价单需要指定价格
            if order_type == "limit":
                if price is None:
                    return {
                        'success': False,
                        'error': '限价单必须指定价格'
                    }
                order_data["px"] = str(price)

            # 请求路径和方法
            request_path = "/api/v5/trade/order"
            method = "POST"
            body = json.dumps(order_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"下单请求: {order_data}")
            response = self._make_request(method, url, headers, body, timeout=10)
            result = response.json()

            logger.info(f"下单响应: {result}")

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                order_id = result['data'][0].get('ordId')

                # 保存订单到数据库（包含上下文信息）
                self._save_order_to_database(
                    order_id=order_id,
                    inst_id=inst_id,
                    side=side,
                    order_type=order_type,
                    size=size,
                    price=price,
                    status="live",
                    tool_call_id=self._current_tool_call_id,
                    conversation_id=self._current_conversation_id
                )

                return {
                    'success': True,
                    'order_id': order_id,
                    'side': side,
                    'size': size,
                    'price': price,
                    'order_type': order_type
                }
            else:
                return {
                    'success': False,
                    'error': result.get('msg', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"下单失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def place_limit_order(
        self,
        inst_id: str,
        side: str,
        price: float,
        size: Optional[float] = None,
        total_amount: Optional[float] = None,
        client_order_id: Optional[str] = None,
        td_mode: str = "cash"
    ) -> Dict:
        """
        下限价单（强制限价单，防止市价单风险）

        Args:
            inst_id: 交易对（如 ETH-USDT）
            side: 交易方向（buy/sell）
            price: 价格（限价单必填）
            size: 数量（可选，与total_amount二选一）
            total_amount: 总金额（可选，与size二选一）
            client_order_id: 客户端订单ID（可选）
            td_mode: 交易模式（cash=现货, cross=全仓, isolated=逐仓）

        Returns:
            订单结果
        """
        try:
            # 强制使用限价单，确保价格安全
            if not price:
                return {
                    'success': False,
                    'error': '限价单必须指定价格'
                }

            # 计算数量
            if not size and not total_amount:
                return {
                    'success': False,
                    'error': '必须指定 size 或 total_amount 之一'
                }

            if total_amount and not size:
                size = safe_float(total_amount) / safe_float(price)

            if not size or size <= 0:
                return {
                    'success': False,
                    'error': f'无效的数量: {size}'
                }

            # 构建订单参数
            order_data = {
                "instId": inst_id,
                "tdMode": td_mode,
                "side": side,
                "ordType": "limit",  # 强制限价单
                "sz": str(size),
                "px": str(price)
            }

            # 验证并添加客户端订单ID
            validated_client_order_id = validate_and_generate_client_order_id(client_order_id)
            if validated_client_order_id:
                order_data["clOrdId"] = validated_client_order_id

            # 请求路径和方法
            request_path = "/api/v5/trade/order"
            method = "POST"
            body = json.dumps(order_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"限价单请求: {order_data}")
            response = self._make_request(method, url, headers, body, timeout=10)
            result = response.json()

            logger.info(f"限价单响应: {result}")

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                order_id = result['data'][0].get('ordId')
                return {
                    'success': True,
                    'order_id': order_id,
                    'side': side,
                    'order_type': 'limit',
                    'size': size,
                    'price': price,
                    'message': '限价单下单成功'
                }
            else:
                error_msg = result.get('msg', '未知错误')
                return {
                    'success': False,
                    'error': error_msg,
                    'details': result
                }

        except Exception as e:
            logger.error(f"限价单下单失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def cancel_order(
        self,
        inst_id: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict:
        """
        取消订单

        Args:
            inst_id: 交易对
            order_id: 订单ID (可选)
            client_order_id: 客户端订单ID (可选)

        Returns:
            取消结果
        """
        try:
            if not order_id and not client_order_id:
                return {
                    'success': False,
                    'error': '必须提供order_id或client_order_id之一'
                }

            # 构建参数
            cancel_data = {
                "instId": inst_id,
            }
            if order_id:
                cancel_data["ordId"] = order_id
            if client_order_id:
                validated_client_order_id = validate_and_generate_client_order_id(client_order_id)
                cancel_data["clOrdId"] = validated_client_order_id

            # 请求路径和方法
            request_path = "/api/v5/trade/cancel-order"
            method = "POST"
            body = json.dumps(cancel_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"取消订单请求: {cancel_data}")
            response = self._make_request("POST", url, headers=headers, body=body, timeout=10)
            result = response.json()

            logger.info(f"取消订单响应: {result}")

            # 解析结果
            if result.get('code') == '0':
                return {
                    'success': True,
                    'order_id': order_id
                }
            else:
                return {
                    'success': False,
                    'error': result.get('msg', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def amend_order(
        self,
        inst_id: str,
        cl_ord_id: str = None,
        new_sz: str = None,
        new_px: str = None,
        req_id: str = None,
        order_id: str = None,
        new_size: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Dict:
        """
        修改订单

        Args:
            inst_id: 交易对，如 'ETH-USDT'
            cl_ord_id: 客户端订单ID（新参数名，兼容OKX API）
            new_sz: 修改的新数量（新参数名）
            new_px: 修改后的新价格（新参数名）
            req_id: 用户自定义修改事件ID
            order_id: 订单ID（兼容旧参数）
            new_size: 新数量（兼容旧参数）
            new_price: 新价格（兼容旧参数）

        Returns:
            修改结果
        """
        # 参数兼容处理
        if cl_ord_id is None and client_order_id is not None:
            cl_ord_id = client_order_id
        if new_sz is None and new_size is not None:
            new_sz = str(new_size)
        if new_px is None and new_price is not None:
            new_px = str(new_price)
        try:
            if not cl_ord_id and not order_id and not client_order_id:
                return {
                    'success': False,
                    'error': '必须提供cl_ord_id或order_id之一'
                }

            if not new_sz and not new_px and not new_size and not new_price:
                return {
                    'success': False,
                    'error': '必须提供new_size或new_price之一'
                }

            # 构建参数
            amend_data = {
                "instId": inst_id,
            }

            # 使用新的参数名，兼容旧的参数名
            if cl_ord_id:
                validated_cl_ord_id = validate_and_generate_client_order_id(cl_ord_id)
                amend_data["clOrdId"] = validated_cl_ord_id
            elif client_order_id:
                validated_client_order_id = validate_and_generate_client_order_id(client_order_id)
                amend_data["clOrdId"] = validated_client_order_id
            elif order_id:
                amend_data["ordId"] = order_id

            if new_sz is not None:
                amend_data["newSz"] = new_sz
            elif new_size is not None:
                amend_data["newSz"] = str(new_size)

            if new_px is not None:
                amend_data["newPx"] = new_px
            elif new_price is not None:
                amend_data["newPx"] = str(new_price)

            if req_id is not None:
                amend_data["reqId"] = req_id

            # 请求路径和方法
            request_path = "/api/v5/trade/amend-order"
            method = "POST"
            body = json.dumps(amend_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"修改订单请求: {amend_data}")
            response = self._make_request("POST", url, headers=headers, body=body, timeout=10)
            result = response.json()

            logger.info(f"修改订单响应: {result}")

            # 解析结果
            if result.get('code') == '0':
                return {
                    'success': True,
                    'order_id': order_id,
                    'new_size': new_size,
                    'new_price': new_price
                }
            else:
                return {
                    'success': False,
                    'error': result.get('msg', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"修改订单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_order(
        self,
        inst_id: str,
        order_id: str = None,
        client_order_id: str = None
    ) -> Dict:
        """
        查询订单

        Args:
            inst_id: 交易对
            order_id: 订单ID (可选)
            client_order_id: 客户端订单ID (可选)

        Returns:
            订单信息
        """
        try:
            if not order_id and not client_order_id:
                return {
                    'success': False,
                    'error': '必须提供order_id或client_order_id之一'
                }

            # 构建查询参数
            params = f"instId={inst_id}"
            if order_id:
                params += f"&ordId={order_id}"
            if client_order_id:
                validated_client_order_id = validate_and_generate_client_order_id(client_order_id)
                params += f"&clOrdId={validated_client_order_id}"

            request_path = f"/api/v5/trade/order?{params}"
            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询订单: inst_id={inst_id}, order_id={order_id}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                order_data = result['data'][0]
                return {
                    'success': True,
                    'order': order_data
                }
            else:
                return {
                    'success': False,
                    'error': result.get('msg', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"查询订单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_account_balance(self, ccy: str = None) -> Dict:
        """
        查询账户余额

        Args:
            ccy: 币种，如 BTC、USDT。不填则返回所有币种余额

        Returns:
            账户余额信息
        """
        try:
            # 请求路径和方法
            request_path = "/api/v5/account/balance"
            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询账户余额: {ccy if ccy else '所有币种'}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                balance_data = result['data'][0]
                details = balance_data.get('details', [])

                # 如果指定了币种，则筛选
                if ccy:
                    filtered_details = []
                    for detail in details:
                        if detail.get('ccy') == ccy:
                            filtered_details.append(detail)
                    details = filtered_details

                # 格式化余额信息
                balances = {}
                for detail in details:
                    currency = detail.get('ccy')
                    if currency:
                        balances[currency] = {
                            'available': float(detail.get('availBal', '0')),
                            'frozen': float(detail.get('frozenBal', '0')),
                            'total': float(detail.get('bal', '0'))
                        }

                return {
                    'success': True,
                    'balances': balances,
                    'total_equity': float(balance_data.get('totalEq', '0')),
                    'message': f"查询成功: {len(balances)} 种币种"
                }
            else:
                return {
                    'success': False,
                    'error': result.get('msg', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"查询账户余额失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def run_latest_model_inference(
        self,
        lookback_window: Optional[int] = None,
        predict_window: Optional[int] = None,
        force_rerun: Optional[bool] = None,
        plan_id: Optional[int] = None
    ) -> Dict:
        """
        执行最新模型推理

        Args:
            lookback_window: 回溯窗口大小
            predict_window: 预测窗口大小
            force_rerun: 是否强制重新推理
            plan_id: 计划ID（用于查询最新训练记录）

        Returns:
            推理结果
        """
        try:
            from services.inference_service import InferenceService
            from database.db import get_db
            from database.models import TradingPlan, TrainingRecord

            # 查找最新训练记录
            with get_db() as db:
                if plan_id:
                    plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                else:
                    # 获取第一个运行中的计划
                    plan = db.query(TradingPlan).filter(
                        TradingPlan.status == 'running'
                    ).first()

                if not plan:
                    return {
                        'success': False,
                        'error': '未找到可用的交易计划'
                    }

                # 查找最新完成的训练记录
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan.id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    return {
                        'success': False,
                        'error': '未找到可用的训练模型'
                    }

            logger.info(f"执行推理: training_id={latest_training.id}, plan_id={plan.id}")

            # 执行推理 (使用正确的异步方法)
            import asyncio

            # 对于同步调用，创建新的事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # 执行异步推理
            result = loop.run_until_complete(
                InferenceService.start_inference(
                    training_id=latest_training.id
                )
            )

            if result.get('success'):
                return {
                    'success': True,
                    'inference_id': result.get('inference_id'),
                    'training_id': latest_training.id,
                    'model_version': latest_training.model_version,
                    'prediction_count': result.get('prediction_count', 0),
                    'time_range': result.get('time_range', {}),
                    'summary': result.get('summary', {}),
                    'message': f"推理成功完成，生成 {result.get('prediction_count', 0)} 条预测数据"
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', '推理失败')
                }

        except Exception as e:
            logger.error(f"执行推理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def delete_prediction_data_by_batch(
        self,
        batch_id: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        confirm_delete: bool = False
    ) -> Dict:
        """
        删除预测数据

        Args:
            batch_id: 批次ID
            start_time: 开始时间
            end_time: 结束时间
            confirm_delete: 确认删除

        Returns:
            删除结果
        """
        try:
            from database.db import get_db
            from database.models import PredictionData
            from datetime import datetime

            if not confirm_delete:
                return {
                    'success': False,
                    'error': '请设置 confirm_delete=true 来确认删除操作'
                }

            with get_db() as db:
                query = db.query(PredictionData)

                deleted_count = 0
                deleted_batch_ids = []

                if batch_id is not None:
                    # 按批次ID删除
                    predictions = query.filter(PredictionData.training_record_id == batch_id).all()

                    if predictions:
                        deleted_count = len(predictions)
                        deleted_batch_ids = [batch_id]

                        # 删除数据
                        query.filter(PredictionData.training_record_id == batch_id).delete()
                        db.commit()

                elif start_time and end_time:
                    # 按时间范围删除
                    try:
                        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

                        predictions = query.filter(
                            PredictionData.timestamp >= start_dt,
                            PredictionData.timestamp <= end_dt
                        ).all()

                        if predictions:
                            deleted_count = len(predictions)
                            deleted_batch_ids = list(set([p.training_record_id for p in predictions]))

                            # 删除数据
                            query.filter(
                                PredictionData.timestamp >= start_dt,
                                PredictionData.timestamp <= end_dt
                            ).delete()
                            db.commit()

                    except ValueError as e:
                        return {
                            'success': False,
                            'error': f'时间格式错误: {str(e)}'
                        }

                else:
                    return {
                        'success': False,
                        'error': '请提供 batch_id 或 (start_time, end_time) 参数'
                    }

                logger.warning(f"删除预测数据: {deleted_count} 条记录，批次ID: {deleted_batch_ids}")

                return {
                    'success': True,
                    'deleted_count': deleted_count,
                    'deleted_batch_ids': deleted_batch_ids,
                    'time_range': {
                        'start_time': start_time,
                        'end_time': end_time
                    },
                    'message': f"成功删除 {deleted_count} 条预测数据"
                }

        except Exception as e:
            logger.error(f"删除预测数据失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_current_utc_time(self) -> Dict:
        """
        获取当前UTC+0时间

        Returns:
            当前时间信息
        """
        try:
            from datetime import datetime

            now = datetime.utcnow()

            return {
                'success': True,
                'timestamp': int(now.timestamp() * 1000),  # 毫秒时间戳
                'formatted_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                'iso_time': now.isoformat(),
                'timezone': 'UTC+0',
                'message': f"当前UTC+0时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            }

        except Exception as e:
            logger.error(f"获取UTC时间失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_current_price(self, inst_id: str) -> Dict:
        """
        获取交易对当前市场价格

        Args:
            inst_id: 交易对，如 ETH-USDT

        Returns:
            当前价格信息
        """
        try:
            # 请求路径和方法
            request_path = f"/api/v5/market/ticker?instId={inst_id}"
            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"获取价格信息: {inst_id}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            if result.get("code") == "0":
                data = result.get("data", [])
                if data:
                    ticker = data[0]
                    return {
                        'success': True,
                        'inst_id': inst_id,
                        'last_price': float(ticker.get("last", 0)),
                        'best_bid': float(ticker.get("bidPx", 0)),
                        'best_ask': float(ticker.get("askPx", 0)),
                        'volume_24h': float(ticker.get("vol24h", 0)),
                        'high_24h': float(ticker.get("high24h", 0)),
                        'low_24h': float(ticker.get("low24h", 0)),
                        'timestamp': int(ticker.get("ts", 0)),
                        'message': f"获取价格成功: {inst_id}"
                    }
                else:
                    return {
                        'success': False,
                        'error': '没有找到价格数据'
                    }
            else:
                return {
                    'success': False,
                    'error': result.get("msg", "获取价格失败")
                }

        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_account_positions(self, inst_id: str = None) -> Dict:
        """
        查询账户持仓信息

        Args:
            inst_id: 交易对，如 BTC-USDT。不填则返回所有持仓

        Returns:
            持仓信息
        """
        try:
            # 请求路径和方法
            request_path = "/api/v5/account/positions"
            if inst_id:
                request_path += f"?instId={inst_id}"
            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询持仓信息: {inst_id if inst_id else '所有'}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            if result.get("code") == "0":
                data = result.get("data", [])
                positions = []
                for pos in data:
                    if float(pos.get("pos", 0)) != 0:  # 只返回有持仓的
                        positions.append({
                            'inst_id': pos.get("instId"),
                            'position_side': pos.get("posSide"),
                            'position_size': float(pos.get("pos", 0)),
                            'available_position': float(pos.get("availPos", 0)),
                            'average_price': float(pos.get("avgPx", 0)),
                            'unrealized_pnl': float(pos.get("upl", 0)),
                            'unrealized_pnl_ratio': float(pos.get("uplRatio", 0)),
                            'margin': float(pos.get("margin", 0)),
                            'last_price': float(pos.get("last", 0)),
                            'mark_price': float(pos.get("markPx", 0))
                        })

                return {
                    'success': True,
                    'positions': positions,
                    'count': len(positions),
                    'message': f"找到 {len(positions)} 个持仓"
                }
            else:
                return {
                    'success': False,
                    'error': result.get("msg", "查询持仓失败")
                }

        except Exception as e:
            logger.error(f"查询持仓失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_pending_orders(self, inst_id: str = None, state: str = "live", limit: int = 300) -> Dict:
        """
        查询当前未成交订单

        Args:
            inst_id: 交易对，如 BTC-USDT。不填则返回所有交易对的挂单
            state: 订单状态，'live': 等待成交, 'partially_filled': 部分成交，默认'live'
            limit: 返回数量限制，默认300

        Returns:
            未成交订单信息
        """
        try:
            # 请求路径和方法
            request_path = "/api/v5/trade/orders-pending"
            params = []

            # 固定参数
            params.append("instType=SPOT")
            params.append("ordType=limit")

            if inst_id:
                params.append(f"instId={inst_id}")

            if state:
                params.append(f"state={state}")

            if limit:
                params.append(f"limit={limit}")

            if params:
                request_path += "?" + "&".join(params)

            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询挂单: {inst_id if inst_id else '所有'}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            if result.get("code") == "0":
                data = result.get("data", [])
                orders = []
                for order in data:
                    orders.append({
                        'order_id': order.get("ordId"),
                        'client_order_id': order.get("clOrdId"),
                        'inst_id': order.get("instId"),
                        'order_type': order.get("ordType"),
                        'side': order.get("side"),
                        'order_size': safe_float(order.get("sz")),
                        'order_amount': safe_float(order.get("sz")) * safe_float(order.get("px")),
                        'filled_size': safe_float(order.get("fillSz")),
                        'filled_amount': safe_float(order.get("fillSz")) * safe_float(order.get("px")),
                        'average_price': safe_float(order.get("avgPx")),
                        'order_price': safe_float(order.get("px")),
                        'status': order.get("state"),
                        'created_time': safe_int(order.get("cTime")),
                        'updated_time': safe_int(order.get("uTime"))
                    })

                return {
                    'success': True,
                    'orders': orders,
                    'count': len(orders),
                    'message': f"找到 {len(orders)} 个挂单"
                }
            else:
                return {
                    'success': False,
                    'error': result.get("msg", "查询挂单失败")
                }

        except Exception as e:
            logger.error(f"查询挂单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cancel_all_orders(self, inst_id: str = None) -> Dict:
        """
        批量撤销所有未成交订单

        Args:
            inst_id: 交易对，如 BTC-USDT。不填则撤销所有交易对的订单

        Returns:
            撤单结果
        """
        try:
            # 请求路径和方法
            request_path = "/api/v5/trade/cancel-all-orders"
            method = "POST"

            # 构建请求数据
            body_data = {}
            if inst_id:
                body_data["instId"] = inst_id

            body = json.dumps(body_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"批量撤单: {inst_id if inst_id else '所有'}")
            response = self._make_request("POST", url, headers=headers, body=body, timeout=10)
            result = response.json()

            if result.get("code") == "0":
                return {
                    'success': True,
                    'message': f"批量撤单成功: {inst_id if inst_id else '所有交易对'}",
                    'data': result.get("data", [])
                }
            else:
                return {
                    'success': False,
                    'error': result.get("msg", "批量撤单失败")
                }

        except Exception as e:
            logger.error(f"批量撤单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_fills(self, inst_id: str = None, order_id: str = None, limit: int = 100) -> Dict:
        """
        查询成交明细

        Args:
            inst_id: 交易对，如 BTC-USDT
            order_id: 订单ID
            limit: 返回数量，默认100，最大100

        Returns:
            成交明细
        """
        try:
            # 请求路径和方法
            request_path = "/api/v5/trade/fills-history"
            params = []
            if inst_id:
                params.append(f"instId={inst_id}")
            if order_id:
                params.append(f"ordId={order_id}")
            if limit:
                params.append(f"limit={min(limit, 100)}")

            if params:
                request_path += "?" + "&".join(params)

            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询成交明细: instId={inst_id}, ordId={order_id}")
            response = self._make_request("GET", url, headers=headers, timeout=10)
            result = response.json()

            if result.get("code") == "0":
                data = result.get("data", [])
                fills = []
                for fill in data:
                    fills.append({
                        'trade_id': fill.get("tradeId"),
                        'order_id': fill.get("ordId"),
                        'client_order_id': fill.get("clOrdId"),
                        'inst_id': fill.get("instId"),
                        'trade_type': fill.get("tdMode"),
                        'side': fill.get("side"),
                        'fill_size': float(fill.get("sz", 0)),
                        'fill_price': float(fill.get("px", 0)),
                        'quote_currency': fill.get("ccy"),
                        'exec_type': fill.get("execType"),
                        'fee': float(fill.get("fee", 0)),
                        'fee_currency': fill.get("feeCcy"),
                        'timestamp': int(fill.get("ts", 0))
                    })

                return {
                    'success': True,
                    'fills': fills,
                    'count': len(fills),
                    'message': f"找到 {len(fills)} 条成交记录"
                }
            else:
                return {
                    'success': False,
                    'error': result.get("msg", "查询成交明细失败")
                }

        except Exception as e:
            logger.error(f"查询成交明细失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def query_historical_kline_data(
        self,
        inst_id: str,
        interval: str = '1H',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> Dict:
        """
        查询历史K线数据

        Args:
            inst_id: 交易对，如 BTC-USDT
            interval: 时间间隔，如 1H, 4H, 1D
            start_time: 开始时间，格式如 "2025-01-01 00:00:00"
            end_time: 结束时间，格式如 "2025-01-31 23:59:59"
            limit: 限制条数
            order_by: 排序方式，'time_asc'(时间升序) 或 'time_desc'(时间降序)

        Returns:
            K线数据
        """
        try:
            from database.db import get_db
            from database.models import KlineData
            from datetime import datetime
            from sqlalchemy import asc, desc

            with get_db() as db:
                # 基础查询条件，加入interval过滤
                query = db.query(KlineData).filter(
                    KlineData.inst_id == inst_id,
                    KlineData.interval == interval
                )

                # 时间范围过滤，支持多种格式
                if start_time:
                    try:
                        if len(start_time) == 19 and ' ' in start_time:
                            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                        elif len(start_time) == 10:
                            start_dt = datetime.strptime(start_time + " 00:00:00", "%Y-%m-%d %H:%M:%S")
                        else:
                            start_dt = datetime.fromisoformat(start_time)
                        query = query.filter(KlineData.timestamp >= start_dt)
                    except ValueError:
                        return {
                            'success': False,
                            'error': f'开始时间格式错误: {start_time}'
                        }

                if end_time:
                    try:
                        if len(end_time) == 19 and ' ' in end_time:
                            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                        elif len(end_time) == 10:
                            end_dt = datetime.strptime(end_time + " 23:59:59", "%Y-%m-%d %H:%M:%S")
                        else:
                            end_dt = datetime.fromisoformat(end_time)
                        query = query.filter(KlineData.timestamp <= end_dt)
                    except ValueError:
                        return {
                            'success': False,
                            'error': f'结束时间格式错误: {end_time}'
                        }

                # 按时间排序和限制
                if order_by == "time_asc":
                    query = query.order_by(asc(KlineData.timestamp))
                elif order_by == "time_desc":
                    query = query.order_by(desc(KlineData.timestamp))
                else:
                    # 默认按时间降序排列（最新的在前）
                    query = query.order_by(desc(KlineData.timestamp))

                if limit:
                    query = query.limit(min(limit, 500))

                data = query.all()

                if not data:
                    return {
                        'success': False,
                        'error': f'未找到 {inst_id} 的历史K线数据'
                    }

                klines = []
                for kline in data:
                    klines.append({
                        'timestamp': kline.timestamp.isoformat(),
                        'open': float(kline.open),
                        'high': float(kline.high),
                        'low': float(kline.low),
                        'close': float(kline.close),
                        'volume': float(kline.volume),
                        'amount': float(kline.amount),
                        'interval': kline.interval,
                        'inst_id': kline.inst_id
                    })

                return {
                    'success': True,
                    'data': klines,
                    'count': len(klines),
                    'inst_id': inst_id,
                    'interval': interval,
                    'time_range': {
                        'start': start_time,
                        'end': end_time,
                        'actual_start': data[-1].timestamp.isoformat() if data else None,
                        'actual_end': data[0].timestamp.isoformat() if data else None
                    },
                    'message': f"找到 {len(klines)} 条 {inst_id} {interval} 历史K线数据"
                }

        except Exception as e:
            logger.error(f"查询历史K线数据失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def delete_prediction_data_by_batch(
        self,
        batch_id: Optional[int] = None,
        confirm_delete: bool = False
    ) -> Dict:
        """
        删除预测数据（按批次）

        Args:
            batch_id: 批次ID
            confirm_delete: 确认删除标志

        Returns:
            删除结果
        """
        try:
            if not confirm_delete:
                return {
                    'success': False,
                    'error': '需要确认删除操作'
                }

            from database.db import get_db
            from database.models import PredictionData

            with get_db() as db:
                query = db.query(PredictionData)

                if batch_id:
                    query = query.filter(PredictionData.inference_batch_id == batch_id)

                count = query.count()

                if count == 0:
                    return {
                        'success': True,
                        'message': '没有找到要删除的预测数据',
                        'deleted_count': 0
                    }

                query.delete()
                db.commit()

                return {
                    'success': True,
                    'message': f'成功删除 {count} 条预测数据',
                    'deleted_count': count
                }

        except Exception as e:
            logger.error(f"删除预测数据失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
