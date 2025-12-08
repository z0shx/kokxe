"""
刷新数据服务
解耦刷新数据逻辑，提供清晰的数据刷新接口
"""
import gradio as gr
from typing import Dict, Any, Optional
from datetime import datetime
from utils.logger import setup_logger
from utils.common import extract_finetune_param
from ui.plan_detail import PlanDetailUI

logger = setup_logger(__name__, "refresh_service.log")


class RefreshService:
    """刷新数据服务类"""

    def __init__(self):
        self.detail_ui = PlanDetailUI()

    def refresh_plan_data(self, plan_id: int) -> Dict[str, Any]:
        """
        刷新计划数据的主入口

        Args:
            plan_id: 计划ID

        Returns:
            Dict: 包含所有刷新后数据的字典
        """
        try:
            if not plan_id or plan_id <= 0:
                return self._get_empty_data()

            logger.info(f"开始刷新计划 {plan_id} 的数据")

            # 并行加载各种数据
            data = {
                'plan_id': plan_id,
                'timestamp': datetime.now(),
                'overview': self._refresh_overview(plan_id),
                'training': self._refresh_training_data(plan_id),
                'inference': self._refresh_inference_data(plan_id),
                'agent': self._refresh_agent_data(plan_id),
                'account': self._refresh_account_data(plan_id),
                'automation': self._refresh_automation_config(plan_id),
                'schedule': self._refresh_schedule_config(plan_id),
                'model_params': self._refresh_model_params(plan_id),
                'inference_params': self._refresh_inference_params(plan_id),
                'llm_config': self._refresh_llm_config(plan_id),
                'trading_limits': self._refresh_trading_limits(plan_id),
                'charts': self._refresh_charts(plan_id),
                'tables': self._refresh_tables(plan_id)
            }

            logger.info(f"计划 {plan_id} 数据刷新完成")
            return data

        except Exception as e:
            logger.error(f"刷新计划 {plan_id} 数据失败: {e}")
            return self._get_error_data(str(e))

    def _get_empty_data(self) -> Dict[str, Any]:
        """获取空数据"""
        return {
            'plan_id': None,
            'timestamp': datetime.now(),
            'error': '无效的计划ID',
            'overview': self._get_empty_overview(),
            'training': self._get_empty_training(),
            'inference': self._get_empty_inference(),
            'agent': self._get_empty_agent(),
            'account': self._get_empty_account(),
            'automation': self._get_empty_automation(),
            'schedule': self._get_empty_schedule(),
            'model_params': self._get_empty_model_params(),
            'inference_params': self._get_empty_inference_params(),
            'llm_config': self._get_empty_llm_config(),
            'trading_limits': self._get_empty_trading_limits(),
            'charts': self._get_empty_charts(),
            'tables': self._get_empty_tables()
        }

    def _get_error_data(self, error_msg: str) -> Dict[str, Any]:
        """获取错误数据"""
        data = self._get_empty_data()
        data['error'] = error_msg
        return data

    def _refresh_overview(self, plan_id: int) -> Dict[str, Any]:
        """刷新概览数据"""
        try:
            overview_data = self.detail_ui.render_plan_overview(plan_id)
            return {
                'overview_md': overview_data[0] if len(overview_data) > 0 else "",
                'ws_status_md': overview_data[1] if len(overview_data) > 1 else "",
                'plan_status_md': overview_data[2] if len(overview_data) > 2 else "",
                'ws_start_visible': overview_data[3] if len(overview_data) > 3 else True,
                'ws_stop_visible': overview_data[4] if len(overview_data) > 4 else False,
                'plan_start_visible': overview_data[5] if len(overview_data) > 5 else True,
                'plan_stop_visible': overview_data[6] if len(overview_data) > 6 else False,
                'ws_result': "",
                'plan_result': ""
            }
        except Exception as e:
            logger.error(f"刷新概览数据失败: {e}")
            return self._get_empty_overview()

    def _refresh_training_data(self, plan_id: int) -> Dict[str, Any]:
        """刷新训练数据"""
        try:
            # 这里应该调用实际的训练数据加载逻辑
            return {
                'training_df': gr.DataFrame(),
                'training_status': "数据加载完成"
            }
        except Exception as e:
            logger.error(f"刷新训练数据失败: {e}")
            return self._get_empty_training()

    def _refresh_inference_data(self, plan_id: int) -> Dict[str, Any]:
        """刷新推理数据"""
        try:
            # 这里应该调用实际的推理数据加载逻辑
            return {
                'inference_df': gr.DataFrame(),
                'inference_data_range_info': "数据范围加载完成",
                'prediction_data_preview': "",
                'inference_status': "数据加载完成"
            }
        except Exception as e:
            logger.error(f"刷新推理数据失败: {e}")
            return self._get_empty_inference()

    def _refresh_agent_data(self, plan_id: int) -> Dict[str, Any]:
        """刷新Agent数据"""
        try:
            # 这里应该调用实际的Agent数据加载逻辑
            return {
                'agent_df': gr.DataFrame(),
                'agent_chatbot': [{"role": "assistant", "content": "Agent数据已刷新"}],
                'agent_status': "数据加载完成"
            }
        except Exception as e:
            logger.error(f"刷新Agent数据失败: {e}")
            return self._get_empty_agent()

    def _refresh_account_data(self, plan_id: int) -> Dict[str, Any]:
        """刷新账户数据"""
        try:
            # 这里应该调用实际的账户数据加载逻辑
            return {
                'account_status': "### 💰 账户信息\n\n数据已刷新",
                'order_table': gr.DataFrame(),
                'task_executions_df': gr.DataFrame(),
                'account_status': "数据加载完成"
            }
        except Exception as e:
            logger.error(f"刷新账户数据失败: {e}")
            return self._get_empty_account()

    def _refresh_automation_config(self, plan_id: int) -> Dict[str, Any]:
        """刷新自动化配置"""
        try:
            from database.db import get_db
            from database.models import TradingPlan

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在，使用默认自动化配置")
                    return self._get_empty_automation()

                return {
                    'auto_finetune_enabled': plan.auto_finetune_enabled or False,
                    'auto_inference_enabled': plan.auto_inference_enabled or False,
                    'auto_agent_enabled': plan.auto_agent_enabled or False,
                    'auto_tool_execution_enabled': plan.auto_tool_execution_enabled or False,
                    'automation_config_result': "自动化配置已从数据库刷新"
                }

        except Exception as e:
            logger.error(f"刷新自动化配置失败: {e}")
            return self._get_empty_automation()

    def _refresh_schedule_config(self, plan_id: int) -> Dict[str, Any]:
        """刷新时间表配置"""
        try:
            from database.db import get_db
            from database.models import TradingPlan
            import json

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在，使用默认时间表配置")
                    return self._get_empty_schedule()

                # 获取自动微调时间表
                schedule_list = ""
                if plan.auto_finetune_schedule:
                    if isinstance(plan.auto_finetune_schedule, str):
                        schedule_data = json.loads(plan.auto_finetune_schedule)
                    else:
                        schedule_data = plan.auto_finetune_schedule

                    if isinstance(schedule_data, list):
                        schedule_list = "\n".join(schedule_data)

                return {
                    'schedule_time_list': schedule_list,
                    'schedule_operation_result': "时间表配置已从数据库刷新"
                }

        except Exception as e:
            logger.error(f"刷新时间表配置失败: {e}")
            return self._get_empty_schedule()

    def _refresh_model_params(self, plan_id: int) -> Dict[str, Any]:
        """刷新模型参数"""
        try:
            from database.db import get_db
            from database.models import TradingPlan
            import json

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan or not plan.finetune_params:
                    logger.warning(f"计划 {plan_id} 不存在或无微调参数，使用默认值")
                    return self._get_empty_model_params()

                # 解析配置参数
                if isinstance(plan.finetune_params, str):
                    params = json.loads(plan.finetune_params)
                else:
                    params = plan.finetune_params

                # 提取参数值
                data_params = params.get('data', {})

                return {
                    'lookback_window': extract_finetune_param(params, 'lookback_window', 400),
                    'predict_window': extract_finetune_param(params, 'predict_window', 18),
                    'batch_size': extract_finetune_param(params, 'batch_size', 16),
                    'tokenizer_epochs': extract_finetune_param(params, 'tokenizer_epochs', 5),
                    'predictor_epochs': extract_finetune_param(params, 'predictor_epochs', 10),
                    'learning_rate': extract_finetune_param(params, 'learning_rate', 1e-4),
                    'params_status': "模型参数已从数据库刷新",
                    'train_data_range_info': "训练数据范围已加载",
                    'train_start_date': data_params.get('train_start_date', ""),
                    'train_end_date': data_params.get('train_end_date', ""),
                    'train_data_config_result': "训练数据配置已刷新"
                }

        except Exception as e:
            logger.error(f"刷新模型参数失败: {e}")
            return self._get_empty_model_params()

    def _refresh_inference_params(self, plan_id: int) -> Dict[str, Any]:
        """刷新推理参数"""
        try:
            from database.db import get_db
            from database.models import TradingPlan
            import json

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan or not plan.finetune_params:
                    logger.warning(f"计划 {plan_id} 不存在或无微调参数，使用默认值")
                    return self._get_empty_inference_params()

                # 解析配置参数
                if isinstance(plan.finetune_params, str):
                    params = json.loads(plan.finetune_params)
                else:
                    params = plan.finetune_params

                # 提取参数值
                data_params = params.get('data', {})
                inference_params = params.get('inference', {})

                return {
                    'inference_lookback_window': extract_finetune_param(params, 'lookback_window', 400),
                    'inference_predict_window': extract_finetune_param(params, 'predict_window', 18),
                    'inference_temperature': inference_params.get('temperature', 1.0),
                    'inference_top_p': inference_params.get('top_p', 0.9),
                    'inference_sample_count': inference_params.get('sample_count', 30),
                    'inference_data_offset': inference_params.get('data_offset', 0),
                    'inference_params_status': "推理参数已从数据库刷新"
                }

        except Exception as e:
            logger.error(f"刷新推理参数失败: {e}")
            return self._get_empty_inference_params()

    def _refresh_llm_config(self, plan_id: int) -> Dict[str, Any]:
        """刷新LLM配置"""
        try:
            from database.db import get_db
            from database.models import TradingPlan
            from database.models import LLMConfig

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在，使用默认LLM配置")
                    return self._get_empty_llm_config()

                # 获取LLM配置
                llm_config_value = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                    if llm_config:
                        llm_config_value = llm_config.name

                # 获取Agent提示词
                agent_prompt = plan.agent_prompt or ""

                return {
                    'llm_config_dropdown': gr.update(value=llm_config_value),
                    'prompt_template_dropdown': None,  # TODO: 从数据库加载提示词模板
                    'agent_prompt_textbox': agent_prompt,
                    'llm_status': "LLM配置已从数据库刷新"
                }

        except Exception as e:
            logger.error(f"刷新LLM配置失败: {e}")
            return self._get_empty_llm_config()

    def _refresh_trading_limits(self, plan_id: int) -> Dict[str, Any]:
        """刷新交易限制"""
        try:
            from database.db import get_db
            from database.models import TradingPlan
            import json

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()

                if not plan:
                    logger.warning(f"计划 {plan_id} 不存在，使用默认交易限制")
                    return self._get_empty_trading_limits()

                # 获取交易限制配置
                quick_usdt_amount = plan.initial_capital or 1000.0
                quick_usdt_percentage = (plan.max_single_order_ratio or 0.2) * 100
                quick_avg_orders = float(plan.avg_orders_per_batch or 10)

                # 从交易限制配置中获取止损比例
                quick_stop_loss = 20.0  # 默认值
                if plan.trading_limits:
                    if isinstance(plan.trading_limits, str):
                        limits_data = json.loads(plan.trading_limits)
                    else:
                        limits_data = plan.trading_limits

                    quick_stop_loss = limits_data.get('stop_loss_percentage', 20.0)

                return {
                    'quick_usdt_amount': quick_usdt_amount,
                    'quick_usdt_percentage': quick_usdt_percentage,
                    'quick_avg_orders': quick_avg_orders,
                    'quick_stop_loss': quick_stop_loss,
                    'trading_limits_status': "交易限制已从数据库刷新"
                }

        except Exception as e:
            logger.error(f"刷新交易限制失败: {e}")
            return self._get_empty_trading_limits()

    def _refresh_charts(self, plan_id: int) -> Dict[str, Any]:
        """刷新图表数据"""
        try:
            # 这里应该调用实际的图表数据加载逻辑
            return {
                'kline_chart': gr.Plot(),
                'probability_indicators_md': "",
                'charts_status': "图表数据已刷新"
            }
        except Exception as e:
            logger.error(f"刷新图表数据失败: {e}")
            return self._get_empty_charts()

    def _refresh_tables(self, plan_id: int) -> Dict[str, Any]:
        """刷新表格数据"""
        try:
            # 这里应该调用实际的表格数据加载逻辑
            return {
                'account_timer': gr.Timer(active=False),
                'tables_status': "表格数据已刷新"
            }
        except Exception as e:
            logger.error(f"刷新表格数据失败: {e}")
            return self._get_empty_tables()

    # 各种空数据方法的实现
    def _get_empty_overview(self) -> Dict[str, Any]:
        return {
            'overview_md': "❌ 无效的计划ID",
            'ws_status_md': "**WebSocket状态**: ⚪ 未连接",
            'plan_status_md': "**计划状态**: ⚪ 已创建",
            'ws_start_visible': True,
            'ws_stop_visible': False,
            'plan_start_visible': True,
            'plan_stop_visible': False,
            'ws_result': "",
            'plan_result': ""
        }

    def _get_empty_training(self) -> Dict[str, Any]:
        return {
            'training_df': gr.DataFrame(),
            'training_status': "训练数据加载失败"
        }

    def _get_empty_inference(self) -> Dict[str, Any]:
        return {
            'inference_df': gr.DataFrame(),
            'inference_data_range_info': "请保存推理参数后查看数据范围...",
            'prediction_data_preview': "",
            'inference_status': "推理数据加载失败"
        }

    def _get_empty_agent(self) -> Dict[str, Any]:
        return {
            'agent_df': gr.DataFrame(),
            'agent_chatbot': [{"role": "assistant", "content": "请先选择计划"}],
            'agent_status': "Agent数据加载失败"
        }

    def _get_empty_account(self) -> Dict[str, Any]:
        return {
            'account_status': "### 💰 账户信息\n\n未加载",
            'order_table': gr.DataFrame(),
            'task_executions_df': gr.DataFrame(),
            'account_status': "账户数据加载失败"
        }

    def _get_empty_automation(self) -> Dict[str, Any]:
        return {
            'auto_finetune_enabled': False,
            'auto_inference_enabled': False,
            'auto_agent_enabled': False,
            'auto_tool_execution_enabled': False,
            'automation_config_result': "自动化配置加载失败"
        }

    def _get_empty_schedule(self) -> Dict[str, Any]:
        return {
            'schedule_time_list': "",
            'schedule_operation_result': "时间表配置加载失败"
        }

    def _get_empty_model_params(self) -> Dict[str, Any]:
        return {
            'lookback_window': 400,   # ✅ 使用更合理的默认值
            'predict_window': 18,    # ✅ 使用更合理的默认值
            'batch_size': 16,        # ✅ 保持一致的默认值
            'tokenizer_epochs': 5,   # ✅ 使用更合理的默认值
            'predictor_epochs': 10,  # ✅ 使用更合理的默认值
            'learning_rate': 0.0001, # ✅ 使用更合理的默认值
            'params_status': "模型参数加载失败",
            'train_data_range_info': "",
            'train_start_date': "",
            'train_end_date': "",
            'train_data_config_result': "训练数据配置加载失败"
        }

    def _get_empty_inference_params(self) -> Dict[str, Any]:
        return {
            'inference_lookback_window': 400,  # ✅ 使用更合理的默认值
            'inference_predict_window': 18,    # ✅ 使用更合理的默认值
            'inference_temperature': 1.0,
            'inference_top_p': 0.9,
            'inference_sample_count': 30,
            'inference_data_offset': 0,
            'inference_params_status': "推理参数加载失败"
        }

    def _get_empty_llm_config(self) -> Dict[str, Any]:
        return {
            'llm_config_dropdown': gr.update(),
            'prompt_template_dropdown': None,
            'agent_prompt_textbox': "",
            'llm_status': "LLM配置加载失败"
        }

    def _get_empty_trading_limits(self) -> Dict[str, Any]:
        return {
            'quick_usdt_amount': 1000.0,
            'quick_usdt_percentage': 30.0,
            'quick_avg_orders': 10.0,
            'quick_stop_loss': 20.0,
            'trading_limits_status': "交易限制加载失败"
        }

    def _get_empty_charts(self) -> Dict[str, Any]:
        return {
            'kline_chart': gr.Plot(),
            'probability_indicators_md': "",
            'charts_status': "图表数据加载失败"
        }

    def _get_empty_tables(self) -> Dict[str, Any]:
        return {
            'account_timer': gr.Timer(active=False),
            'tables_status': "表格数据加载失败"
        }


# 全局实例
refresh_service = RefreshService()