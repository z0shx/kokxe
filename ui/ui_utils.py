"""
UI工具函数
"""
import functools
import traceback
from typing import Callable, Any, List, Tuple
import gradio as gr
import pandas as pd
from utils.logger import setup_logger
from ui.constants import StatusEmoji

class UIHelper:
    """UI组件辅助类"""

    @staticmethod
    def create_error_handler(operation_name: str) -> Callable:
        """创建统一的错误处理装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger = setup_logger(func.__module__)
                    logger.error(f"{operation_name}失败: {e}")
                    traceback.print_exc()
                    return f"❌ {operation_name}失败: {str(e)}"
            return wrapper
        return decorator

    @staticmethod
    def async_error_handler(operation_name: str) -> Callable:
        """创建异步操作的错误处理装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger = setup_logger(func.__module__)
                    logger.error(f"{operation_name}失败: {e}")
                    traceback.print_exc()
                    return f"❌ {operation_name}失败: {str(e)}"
            return wrapper
        return decorator

    @staticmethod
    def get_status_emoji(status: str, detailed: bool = False) -> str:
        """获取状态对应的emoji"""
        if detailed:
            return StatusEmoji.DETAILED.get(status, StatusEmoji.DETAILED['unknown'])
        return StatusEmoji.BASIC.get(status, '❓')

    @staticmethod
    def bind_event_chain(btn: gr.Button, primary_fn: Callable, secondary_fn: Callable,
                        inputs: List, outputs: List):
        """绑定事件链（主函数 + 刷新函数）"""
        return btn.click(
            fn=primary_fn,
            inputs=inputs,
            outputs=outputs
        ).then(
            fn=secondary_fn,
            outputs=outputs
        )

    @staticmethod
    def create_data_table(headers: List[str], label: str = "",
                         datatypes: List[str] = None,
                         interactive: bool = False) -> gr.DataFrame:
        """创建数据表格组件"""
        return gr.DataFrame(
            value=[],
            headers=headers,
            datatype=datatypes or ["str"] * len(headers),
            interactive=interactive,
            wrap=True,
            label=label
        )

    @staticmethod
    def create_button_group(buttons: List[str], variants: dict = None) -> dict:
        """创建按钮组"""
        btn_dict = {}
        variant_map = {
            "refresh": ("🔄 刷新", "secondary"),
            "create": ("➕ 创建", "primary"),
            "delete": ("🗑️ 删除", "stop"),
            "start": ("🚀 启动", "primary"),
            "stop": ("⏹️ 停止", "stop"),
            "edit": ("✏️ 编辑", "secondary"),
            "view": ("📊 查看", "primary")
        }

        for btn_name in buttons:
            if btn_name in variant_map:
                text, variant = variant_map[btn_name]
                size = "sm" if btn_name in ["refresh", "delete", "edit"] else "md"
                btn_dict[btn_name] = gr.Button(
                    text,
                    variant=variants.get(btn_name, variant) if variants else variant,
                    size=size
                )

        return btn_dict