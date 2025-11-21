"""
UI基础组件类
提供统一的UI组件创建和事件绑定模式
"""
import gradio as gr
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable
from utils.logger import setup_logger

logger = setup_logger(__name__, "base_ui.log")


class BaseUIComponent(ABC):
    """UI组件基类"""

    def __init__(self, name: str):
        self.name = name
        self.logger = setup_logger(f"{__name__}.{name}", f"{name}_ui.log")
        self.components = {}
        self.event_handlers = {}

    @abstractmethod
    def build_ui(self) -> Any:
        """构建UI界面，子类必须实现"""
        pass

    def add_component(self, name: str, component: Any) -> None:
        """添加UI组件"""
        self.components[name] = component

    def get_component(self, name: str) -> Any:
        """获取UI组件"""
        return self.components.get(name)

    def add_event_handler(self, event_name: str, handler: Callable) -> None:
        """添加事件处理器"""
        self.event_handlers[event_name] = handler

    def bind_events(self) -> None:
        """绑定所有事件，子类可以重写"""
        for event_name, handler in self.event_handlers.items():
            if hasattr(self, event_name):
                component = getattr(self, event_name)
                if hasattr(component, 'click'):
                    component.click(handler)


class DatabaseMixin:
    """数据库操作混入类"""

    @staticmethod
    def safe_db_operation(operation_func, max_retries: int = 3, *args, **kwargs):
        """
        安全的数据库操作，带重试机制

        Args:
            operation_func: 数据库操作函数
            max_retries: 最大重试次数
            *args, **kwargs: 传递给操作函数的参数
        """
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"数据库操作失败，已重试{max_retries}次: error={e}")
                    raise
                else:
                    logger.warning(f"数据库操作失败，正在重试({attempt + 1}/{max_retries}): error={e}")
                    import time
                    time.sleep(0.1 * (attempt + 1))


class UIHelper:
    """UI辅助类，提供通用的UI创建方法"""

    @staticmethod
    def create_info_box(title: str, content: str, variant: str = "secondary") -> gr.Markdown:
        """创建信息框"""
        return gr.Markdown(f"### {title}\n\n{content}")

    @staticmethod
    def create_status_display(initial_status: str = "准备就绪") -> gr.Markdown:
        """创建状态显示组件"""
        return gr.Markdown(initial_status)

    @staticmethod
    def create_action_button(label: str, variant: str = "primary", size: str = "lg") -> gr.Button:
        """创建操作按钮"""
        return gr.Button(label, variant=variant, size=size)

    @staticmethod
    def create_data_table(headers: List[str], data: List[List] = None,
                         datatype: List[str] = None) -> gr.Dataframe:
        """创建数据表格"""
        return gr.Dataframe(
            value=data or [],
            headers=headers,
            datatype=datatype or ["str"] * len(headers),
            interactive=False,
            wrap=True
        )

    @staticmethod
    def create_input_group(label: str, components: Dict[str, Any]) -> gr.Group:
        """创建输入组件组"""
        with gr.Group():
            gr.Markdown(f"### {label}")
            return {name: comp for name, comp in components.items()}

    @staticmethod
    def bind_component_events(components: Dict[str, Any],
                            event_bindings: Dict[str, Tuple[str, Callable, List, Any]]) -> None:
        """
        批量绑定组件事件

        Args:
            components: 组件字典
            event_bindings: 事件绑定配置 {
                'component_name': ('event_type', handler_func, [inputs], outputs)
            }
        """
        for comp_name, (event_type, handler, inputs, outputs) in event_bindings.items():
            if comp_name in components:
                component = components[comp_name]
                if hasattr(component, event_type):
                    getattr(component, event_type)(
                        fn=handler,
                        inputs=inputs,
                        outputs=outputs
                    )


class ConfigManager:
    """配置管理器，统一管理配置相关操作"""

    @staticmethod
    def load_llm_configs():
        """加载LLM配置"""
        try:
            from services.config_service import ConfigService
            configs = ConfigService.get_all_llm_configs(active_only=True)

            if not configs:
                return [], None, "⚠️ 暂无可用的 LLM 配置，请先在配置中心创建"

            choices = [(f"{cfg.name} ({cfg.provider})", cfg.id) for cfg in configs]
            default_config = ConfigService.get_default_llm_config()
            default_value = default_config.id if default_config else (choices[0][1] if choices else None)

            return choices, default_value, f"✅ 已加载 {len(choices)} 个 LLM 配置"

        except Exception as e:
            logger.error(f"加载 LLM 配置失败: {e}")
            return [], None, f"❌ 加载失败: {str(e)}"

    @staticmethod
    def load_prompt_templates():
        """加载提示词模板"""
        try:
            from services.config_service import ConfigService
            templates = ConfigService.get_all_prompt_templates(active_only=True)

            if not templates:
                return [], None, "⚠️ 暂无可用的提示词模版，请先在配置中心创建"

            choices = [(tpl.name, tpl.id) for tpl in templates]
            return choices, None, f"✅ 已加载 {len(choices)} 个提示词模版"

        except Exception as e:
            logger.error(f"加载提示词模版失败: {e}")
            return [], None, f"❌ 加载失败: {str(e)}"

    @staticmethod
    def get_trading_instruments(is_demo: bool = True):
        """获取交易工具列表"""
        try:
            from api.okx_client import OKXClient
            client = OKXClient(is_demo=is_demo)
            instruments = client.get_all_spot_instruments()

            if not instruments:
                default_pairs = [
                    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT",
                    "XRP-USDT", "ADA-USDT", "DOGE-USDT", "AVAX-USDT"
                ]
                return default_pairs, "ETH-USDT", "⚠️ 无法从OKX获取，显示常用交易对"

            inst_ids = [inst.get('instId', '') for inst in instruments if inst.get('instId')]
            inst_ids = sorted(set(inst_ids))
            default_value = "ETH-USDT" if "ETH-USDT" in inst_ids else (inst_ids[0] if inst_ids else "ETH-USDT")

            return inst_ids, default_value, f"✅ 成功获取 {len(inst_ids)} 个交易对"

        except Exception as e:
            logger.error(f"获取交易对失败: {e}")
            default_pairs = [
                "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT",
                "XRP-USDT", "ADA-USDT", "DOGE-USDT", "AVAX-USDT"
            ]
            return default_pairs, "ETH-USDT", f"⚠️ 获取失败: {str(e)[:50]}... 显示常用交易对"


class ValidationHelper:
    """验证辅助类"""

    @staticmethod
    def validate_plan_id(plan_id) -> Tuple[bool, str]:
        """验证计划ID"""
        if not plan_id or plan_id <= 0:
            return False, "❌ 请输入有效的计划ID"
        return True, ""

    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
        """验证日期范围"""
        try:
            from datetime import datetime
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            if start >= end:
                return False, "❌ 开始时间必须小于结束时间"
            return True, ""
        except ValueError:
            return False, "❌ 时间格式错误，请使用 YYYY-MM-DD 格式"

    @staticmethod
    def validate_time_format(time_str: str) -> Tuple[bool, str]:
        """验证时间格式 HH:MM"""
        import re
        if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', time_str.strip()):
            return False, "❌ 时间格式错误，请使用HH:MM格式（例如：00:00, 12:30）"
        return True, ""

    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
        """验证必填字段"""
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f"❌ 请填写 {field}"
        return True, ""