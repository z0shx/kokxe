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
from utils.logger import setup_logger

logger = setup_logger(__name__, "trading_tools.log")


class OKXTradingTools:
    """OKX交易工具类"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        is_demo: bool = True,
        trading_limits: Optional[Dict] = None
    ):
        """
        初始化交易工具

        Args:
            api_key: API Key
            secret_key: Secret Key
            passphrase: Passphrase
            is_demo: 是否使用模拟盘
            trading_limits: 交易限制配置
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_demo = is_demo
        self.trading_limits = trading_limits or {}

        # API地址
        if is_demo:
            self.base_url = "https://www.okx.com"
            self.simulate_flag = "1"  # 模拟盘标志
        else:
            self.base_url = "https://www.okx.com"
            self.simulate_flag = "0"  # 实盘标志

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
            response = requests.post(url, headers=headers, data=body, timeout=10)
            result = response.json()

            logger.info(f"下单响应: {result}")

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                order_id = result['data'][0].get('ordId')
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

    def cancel_order(
        self,
        inst_id: str,
        order_id: str
    ) -> Dict:
        """
        取消订单

        Args:
            inst_id: 交易对
            order_id: 订单ID

        Returns:
            取消结果
        """
        try:
            # 构建参数
            cancel_data = {
                "instId": inst_id,
                "ordId": order_id
            }

            # 请求路径和方法
            request_path = "/api/v5/trade/cancel-order"
            method = "POST"
            body = json.dumps(cancel_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"取消订单请求: {cancel_data}")
            response = requests.post(url, headers=headers, data=body, timeout=10)
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
        order_id: str,
        new_size: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Dict:
        """
        修改订单

        Args:
            inst_id: 交易对
            order_id: 订单ID
            new_size: 新数量
            new_price: 新价格

        Returns:
            修改结果
        """
        try:
            # 构建参数
            amend_data = {
                "instId": inst_id,
                "ordId": order_id
            }

            if new_size is not None:
                amend_data["newSz"] = str(new_size)
            if new_price is not None:
                amend_data["newPx"] = str(new_price)

            # 请求路径和方法
            request_path = "/api/v5/trade/amend-order"
            method = "POST"
            body = json.dumps(amend_data)

            # 发送请求
            headers = self._get_headers(method, request_path, body)
            url = self.base_url + request_path

            logger.info(f"修改订单请求: {amend_data}")
            response = requests.post(url, headers=headers, data=body, timeout=10)
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
        order_id: str
    ) -> Dict:
        """
        查询订单

        Args:
            inst_id: 交易对
            order_id: 订单ID

        Returns:
            订单信息
        """
        try:
            # 请求路径和方法
            request_path = f"/api/v5/trade/order?instId={inst_id}&ordId={order_id}"
            method = "GET"

            # 发送请求
            headers = self._get_headers(method, request_path)
            url = self.base_url + request_path

            logger.info(f"查询订单: inst_id={inst_id}, order_id={order_id}")
            response = requests.get(url, headers=headers, timeout=10)
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

    def get_account_balance(self) -> Dict:
        """
        查询账户余额

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

            logger.info("查询账户余额")
            response = requests.get(url, headers=headers, timeout=10)
            result = response.json()

            # 解析结果
            if result.get('code') == '0' and result.get('data'):
                balance_data = result['data'][0]
                return {
                    'success': True,
                    'balance': balance_data
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

            # 创建推理服务实例
            inference_service = InferenceService()

            # 执行推理
            result = inference_service.run_inference(
                training_record_id=latest_training.id,
                lookback_window=lookback_window or 512,
                predict_window=predict_window or 48,
                force_rerun=force_rerun or False
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
