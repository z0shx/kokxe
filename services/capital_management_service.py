"""
资金管理服务 - 实现动态平摊下单策略
"""
import logging
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
from database.db import get_db
from database.models import TradingPlan, TradeOrder
from services.trading_tools import OKXTradingTools
from utils.time_utils import now_beijing
from utils.logger import setup_logger

logger = setup_logger(__name__, "capital_management.log")

class CapitalManagementService:
    """资金管理服务"""

    def __init__(self, plan_id: int):
        self.plan_id = plan_id
        self.trading_tools = None

    def _get_trading_tools(self) -> OKXTradingTools:
        """获取交易工具实例"""
        if not self.trading_tools:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
                if plan:
                    self.trading_tools = OKXTradingTools(
                        api_key=plan.okx_api_key,
                        secret_key=plan.okx_secret_key,
                        passphrase=plan.okx_passphrase,
                        is_demo=plan.is_demo
                    )
        return self.trading_tools

    async def get_current_capital_info(self) -> Dict:
        """
        获取当前资金信息

        Returns:
            {
                'total_usdt': float,  # 总USDT余额
                'available_usdt': float,  # 可用USDT余额
                'initial_capital': float,  # 初始本金
                'current_capital': float,  # 当前总资金（USDT + 持仓价值）
                'profit_loss': float,  # 盈亏金额
                'profit_loss_percentage': float,  # 盈亏百分比
                'completed_orders': int,  # 已完成交易对数
                'avg_orders_per_batch': int,  # 平均每批订单数
                'next_order_amount': float,  # 下次下单金额
            }
        """
        try:
            trading_tools = self._get_trading_tools()
            if not trading_tools:
                return {'error': '无法获取交易工具实例'}

            # 获取账户余额
            balance_result = trading_tools.get_account_balance('USDT')
            if not balance_result.get('success'):
                return {'error': f'获取账户余额失败: {balance_result.get("error")}'}

            usdt_balance = balance_result.get('balance', {})
            available_usdt = float(usdt_balance.get('available', 0))
            frozen_usdt = float(usdt_balance.get('frozen', 0))
            total_usdt = float(usdt_balance.get('total', 0))

            # 获取持仓信息
            positions_result = trading_tools.get_account_positions()
            total_position_value = 0.0

            if positions_result.get('success'):
                positions = positions_result.get('positions', [])
                for position in positions:
                    if position.get('inst_id') and float(position.get('notionalUsd', 0)) > 0:
                        total_position_value += float(position.get('notionalUsd', 0))

            # 当前总资金
            current_capital = total_usdt + total_position_value

            # 获取交易配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
                if not plan:
                    return {'error': '交易计划不存在'}

                # 从交易配置中获取参数，如果没有则使用默认值
                initial_capital = getattr(plan, 'initial_capital', 1000.0)  # 默认1000 USDT
                avg_orders_per_batch = getattr(plan, 'avg_orders_per_batch', 10)  # 默认10单

                # 查询已完成交易对数
                completed_orders = db.query(TradeOrder).filter(
                    TradeOrder.plan_id == self.plan_id,
                    TradeOrder.status.in_(['filled', 'partially_filled'])
                ).count()

            # 计算盈亏
            profit_loss = current_capital - initial_capital
            profit_loss_percentage = (profit_loss / initial_capital) * 100 if initial_capital > 0 else 0

            # 计算下次下单金额（动态平摊策略）
            next_order_amount = current_capital / avg_orders_per_batch

            return {
                'total_usdt': total_usdt,
                'available_usdt': available_usdt,
                'frozen_usdt': frozen_usdt,
                'initial_capital': initial_capital,
                'current_capital': current_capital,
                'total_position_value': total_position_value,
                'profit_loss': profit_loss,
                'profit_loss_percentage': profit_loss_percentage,
                'completed_orders': completed_orders,
                'avg_orders_per_batch': avg_orders_per_batch,
                'next_order_amount': next_order_amount,
                'currency': 'USDT'
            }

        except Exception as e:
            logger.error(f"获取资金信息失败: {e}")
            return {'error': f'获取资金信息失败: {str(e)}'}

    async def calculate_order_parameters(
        self,
        side: str,
        price: float,
        custom_amount: Optional[float] = None,
        custom_size: Optional[float] = None
    ) -> Dict:
        """
        计算订单参数（动态平摊策略）

        Args:
            side: 交易方向 (buy/sell)
            price: 价格
            custom_amount: 自定义金额（优先使用）
            custom_size: 自定义数量

        Returns:
            {
                'success': bool,
                'amount': float,  # 订单金额
                'size': float,    # 订单数量
                'capital_info': dict,  # 资金信息
                'risk_warning': str,  # 风险提示
            }
        """
        try:
            # 获取资金信息
            capital_info = await self.get_current_capital_info()
            if 'error' in capital_info:
                return {
                    'success': False,
                    'error': capital_info['error']
                }

            available_usdt = capital_info['available_usdt']
            next_order_amount = capital_info['next_order_amount']

            # 风险检查
            risk_warnings = []

            # 如果使用自定义金额
            if custom_amount is not None:
                # 转换为float类型
                try:
                    custom_amount_float = float(custom_amount)
                except (ValueError, TypeError):
                    return {
                        'success': False,
                        'error': f'无效的自定义金额格式: {custom_amount}'
                    }

                if custom_amount_float > available_usdt:
                    risk_warnings.append(f"自定义金额 ${custom_amount_float:.2f} 超过可用余额 ${available_usdt:.2f}")
                    return {
                        'success': False,
                        'error': '余额不足',
                        'available_usdt': available_usdt,
                        'requested_amount': custom_amount_float
                    }

                order_amount = custom_amount_float
            else:
                # 使用动态平摊金额
                order_amount = next_order_amount

                # 检查是否超过可用余额
                if order_amount > available_usdt:
                    risk_warnings.append(f"动态平摊金额 ${order_amount:.2f} 超过可用余额 ${available_usdt:.2f}")
                    order_amount = available_usdt * 0.95  # 使用95%的可用余额
                    risk_warnings.append(f"调整为可用余额的95%: ${order_amount:.2f}")

                # 检查是否超过总资金的某个比例（比如20%）
                max_single_order_ratio = 0.2  # 单次订单不超过总资金的20%
                max_single_order = capital_info['current_capital'] * max_single_order_ratio
                if order_amount > max_single_order:
                    risk_warnings.append(f"订单金额 ${order_amount:.2f} 超过总资金{max_single_order_ratio*100}%的限制 ${max_single_order:.2f}")
                    order_amount = max_single_order
                    risk_warnings.append(f"调整为最大限制金额: ${order_amount:.2f}")

            # 确保price是float类型
            try:
                price_float = float(price)
            except (ValueError, TypeError):
                return {
                    'success': False,
                    'error': f'无效的价格格式: {price}'
                }

            # 计算数量
            if custom_size is not None:
                # 转换为float类型
                try:
                    custom_size_float = float(custom_size)
                except (ValueError, TypeError):
                    return {
                        'success': False,
                        'error': f'无效的自定义数量格式: {custom_size}'
                    }

                order_size = custom_size_float
                calculated_amount = order_size * price_float
                if side == 'buy' and calculated_amount > available_usdt:
                    return {
                        'success': False,
                        'error': f'自定义数量 {order_size} 需要金额 ${calculated_amount:.2f}，超过可用余额 ${available_usdt:.2f}'
                    }
            else:
                # 根据金额计算数量
                if side == 'buy':
                    order_size = order_amount / price_float
                else:
                    # 卖出时需要检查是否有足够的持仓
                    trading_tools = self._get_trading_tools()
                    if trading_tools:
                        positions_result = trading_tools.get_account_positions()
                        if positions_result.get('success'):
                            positions = positions_result.get('positions', [])
                            available_size = 0.0

                            with get_db() as db:
                                plan = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
                                if plan:
                                    for position in positions:
                                        if position.get('instId') == plan.inst_id:
                                            available_size = float(position.get('availPos', 0))
                                            break

                            if available_size <= 0:
                                return {
                                    'success': False,
                                    'error': '没有足够的持仓可卖出',
                                    'available_size': available_size
                                }

                            order_size = min(order_amount / price_float, available_size)
                        else:
                            order_size = order_amount / price_float
                    else:
                        order_size = order_amount / price_float

            # 精度处理
            order_size = self._round_to_precision(order_size, 6)  # 保留6位小数

            return {
                'success': True,
                'amount': order_amount,
                'size': order_size,
                'capital_info': capital_info,
                'risk_warnings': risk_warnings,
                'risk_warning': '; '.join(risk_warnings) if risk_warnings else None,
                'side': side,
                'price': price
            }

        except Exception as e:
            logger.error(f"计算订单参数失败: {e}")
            # 添加调试信息
            print(f"DEBUG: price type: {type(price)}, value: {price}")
            print(f"DEBUG: custom_amount type: {type(custom_amount)}, value: {custom_amount}")
            print(f"DEBUG: custom_size type: {type(custom_size)}, value: {custom_size}")
            print(f"DEBUG: available_usdt type: {type(available_usdt)}, value: {available_usdt}")
            logger.error(f"计算订单参数失败: {e}")
            return {
                'success': False,
                'error': f'计算订单参数失败: {str(e)}'
            }

    def _round_to_precision(self, value: float, precision: int) -> float:
        """精度处理"""
        try:
            decimal_value = Decimal(str(value))
            rounded_value = decimal_value.quantize(Decimal(f'1e-{precision}'), rounding=ROUND_DOWN)
            return float(rounded_value)
        except:
            return round(value, precision)

    async def place_order_with_capital_management(
        self,
        inst_id: str,
        side: str,
        price: float,
        custom_amount: Optional[float] = None,
        custom_size: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict:
        """
        使用资金管理策略下单

        Args:
            inst_id: 交易对
            side: 交易方向
            price: 价格
            custom_amount: 自定义金额
            custom_size: 自定义数量
            client_order_id: 客户端订单ID

        Returns:
            订单结果
        """
        try:
            # 计算订单参数
            order_params = await self.calculate_order_parameters(
                side=side,
                price=price,
                custom_amount=custom_amount,
                custom_size=custom_size
            )

            if not order_params.get('success'):
                return order_params

            # 记录资金管理日志
            capital_info = order_params['capital_info']
            logger.info(f"资金管理下单 - 计划ID: {self.plan_id}")
            logger.info(f"当前总资金: ${capital_info['current_capital']:.2f} USDT")
            logger.info(f"下次平摊金额: ${capital_info['next_order_amount']:.2f} USDT")
            logger.info(f"实际下单金额: ${order_params['amount']:.2f} USDT")
            logger.info(f"下单数量: {order_params['size']:.6f}")

            if order_params.get('risk_warnings'):
                logger.warning(f"风险提示: {'; '.join(order_params['risk_warnings'])}")

            # 执行下单
            trading_tools = self._get_trading_tools()
            order_result = trading_tools.place_limit_order(
                inst_id=inst_id,
                side=side,
                price=price,
                size=order_params['size'],
                client_order_id=client_order_id
            )

            # 记录下单结果
            if order_result.get('success'):
                logger.info(f"资金管理下单成功: 订单ID {order_result.get('order_id')}")

                # 保存订单记录到数据库
                await self._save_order_record(
                    order_result=order_result,
                    capital_info=capital_info,
                    order_params=order_params
                )
            else:
                logger.error(f"资金管理下单失败: {order_result.get('error')}")

            # 合并返回结果
            return {
                **order_result,
                'capital_management': {
                    'strategy': 'dynamic_averaging',
                    'capital_info': capital_info,
                    'order_amount': order_params['amount'],
                    'risk_warnings': order_params.get('risk_warnings', [])
                }
            }

        except Exception as e:
            logger.error(f"资金管理下单失败: {e}")
            return {
                'success': False,
                'error': f'资金管理下单失败: {str(e)}'
            }

    async def _save_order_record(self, order_result: Dict, capital_info: Dict, order_params: Dict):
        """保存订单记录到数据库"""
        try:
            with get_db() as db:
                order = TradeOrder(
                    plan_id=self.plan_id,
                    order_id=order_result.get('order_id'),
                    inst_id=order_result.get('inst_id'),
                    side=order_result.get('side'),
                    order_type=order_result.get('order_type'),
                    size=order_result.get('size'),
                    price=order_result.get('price'),
                    status='pending',  # 初始状态为待成交
                    is_demo=True,  # 暂时硬编码为模拟盘
                    is_from_agent=True,  # 标记为Agent操作的订单
                    # 额外的资金管理信息可以存储在JSONB字段中（如果需要的话）
                )

                db.add(order)
                db.commit()
                logger.info(f"订单记录已保存: {order_result.get('order_id')}")

        except Exception as e:
            logger.error(f"保存订单记录失败: {e}")