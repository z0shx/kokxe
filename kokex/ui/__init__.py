"""
UI模块包
提供统一的UI组件创建接口
"""

# 导入基础UI类
from .base_ui import BaseUIComponent, DatabaseMixin, UIHelper, ConfigManager, ValidationHelper

# 导入具体的UI组件类
from .plan_create import PlanCreateUI
from .plan_list import PlanListUI

# 保持向后兼容的函数接口
from .plan_create import create_plan_ui
from .plan_list import create_plan_list_ui
from .config_center import create_config_center_ui

# 版本信息
__version__ = "1.0.0"
__author__ = "KOKEX Team"

# UI组件注册表
UI_COMPONENTS = {
    "plan_create": PlanCreateUI,
    "plan_list": PlanListUI,
}


def create_ui_component(component_name: str, *args, **kwargs):
    """
    创建UI组件的工厂函数

    Args:
        component_name: 组件名称
        *args, **kwargs: 传递给组件构造函数的参数

    Returns:
        UI组件实例
    """
    if component_name not in UI_COMPONENTS:
        raise ValueError(f"Unknown UI component: {component_name}")

    component_class = UI_COMPONENTS[component_name]
    return component_class(*args, **kwargs)


def get_available_components():
    """获取所有可用的UI组件"""
    return list(UI_COMPONENTS.keys())


# 导出所有公开的类和函数
__all__ = [
    # 基础类
    "BaseUIComponent",
    "DatabaseMixin",
    "UIHelper",
    "ConfigManager",
    "ValidationHelper",

    # 具体UI类
    "PlanCreateUI",
    "PlanListUI",

    # 工厂函数
    "create_ui_component",
    "get_available_components",

    # 向后兼容的函数
    "create_plan_ui",
    "create_plan_list_ui",
    "create_config_center_ui",
]
