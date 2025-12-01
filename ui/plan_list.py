"""
计划列表界面
"""
import gradio as gr
import asyncio
from services.plan_service import PlanService
from ui.base_ui import BaseUIComponent, DatabaseMixin, UIHelper, ValidationHelper
from ui.constants import DataFrameHeaders, DataTypes
from ui.ui_utils import UIHelper as NewUIHelper
from utils.logger import setup_logger
from utils.timezone_helper import format_datetime_full_beijing

logger = setup_logger(__name__, "plan_list_ui.log")


class PlanListUI(BaseUIComponent, DatabaseMixin):
    """计划列表界面"""

    def __init__(self):
        super().__init__("plan_list")

    def get_plans_data(self, force_refresh=False):
        """
        获取计划列表数据

        Args:
            force_refresh: 是否强制刷新数据（绕过缓存）
        """
        try:
            plans = PlanService.get_all_plans()

            if not plans:
                return []

            # 如果强制刷新，则重新从数据库获取最新的数据
            if force_refresh:
                from database.db import get_db
                from database.models import TradingPlan
                with get_db() as db:
                    # 重新查询数据库以获取最新数据
                    plan_ids = [plan.id for plan in plans]
                    fresh_plans = db.query(TradingPlan).filter(
                        TradingPlan.id.in_(plan_ids)
                    ).all()
                    # 按ID顺序重新组织数据
                    plans_dict = {plan.id: plan for plan in fresh_plans}
                    plans = [plans_dict[pid] for pid in plan_ids if pid in plans_dict]

            data = []
            for plan in plans:
                # 状态emoji
                status_emoji = NewUIHelper.get_status_emoji(plan.status)

                data.append([
                    plan.id,
                    plan.plan_name,
                    plan.inst_id,
                    plan.interval,
                    f"{status_emoji} {plan.status}",
                    "✅" if plan.ws_connected else "❌",
                    "模拟盘" if plan.is_demo else "实盘",
                    format_datetime_full_beijing(plan.created_at) if plan.created_at else ""
                ])

            return data

        except Exception as e:
            logger.error(f"获取计划列表失败: {e}")
            return []

    def _validate_plan_id(self, plan_id):
        """验证计划ID"""
        is_valid, message = ValidationHelper.validate_plan_id(plan_id)
        if not is_valid:
            return False, message
        return True, ""

    async def start_plan(self, plan_id: int) -> str:
        """启动计划"""
        is_valid, message = self._validate_plan_id(plan_id)
        if not is_valid:
            return message

        try:
            result = await PlanService.start_plan_async(int(plan_id))
            return result['message']
        except Exception as e:
            logger.error(f"启动计划失败: {e}")
            return f"❌ 启动失败: {str(e)}"

    async def stop_plan(self, plan_id: int) -> str:
        """停止计划"""
        is_valid, message = self._validate_plan_id(plan_id)
        if not is_valid:
            return message

        try:
            result = await PlanService.stop_plan_async(int(plan_id))
            return result['message']
        except Exception as e:
            logger.error(f"停止计划失败: {e}")
            return f"❌ 停止失败: {str(e)}"

    def delete_plan(self, plan_id: int) -> str:
        """删除计划"""
        is_valid, message = self._validate_plan_id(plan_id)
        if not is_valid:
            return message

        try:
            result = PlanService.delete_plan(int(plan_id))
            return result['message']
        except Exception as e:
            logger.error(f"删除计划失败: {e}")
            return f"❌ 删除失败: {str(e)}"

    def build_ui(self):
        """构建UI界面（基类要求实现）"""
        # 由于这个类主要保持向后兼容，直接调用原有的create_plan_list_ui逻辑
        components = {}
        gr.Markdown("## 交易计划列表")

        # 刷新按钮
        refresh_btn = gr.Button("🔄 刷新列表")

        # 获取初始数据
        initial_data = self.get_plans_data()

        # 计划列表表格
        plans_table = gr.DataFrame(
            value=initial_data,
            headers=[
                "ID", "计划名称", "交易对", "时间颗粒度",
                "状态", "WebSocket", "环境", "创建时间"
            ],
            datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )

        gr.Markdown("💡 输入计划ID进行操作")

        gr.Markdown("---")
        gr.Markdown("### 计划操作")

        # 操作区域
        with gr.Row():
            plan_id_input = gr.Number(label="计划ID", precision=0, value=None)

        with gr.Row():
            view_detail_btn = gr.Button("📊 查看详情", variant="primary")
            start_btn = gr.Button("🚀 启动计划")
            stop_btn = gr.Button("⏹️ 停止计划")
            delete_btn = gr.Button("🗑️ 删除计划", variant="stop")

        operation_result = gr.Markdown("")

        gr.Markdown("""
        **操作说明**:
        - 📊 **查看详情**: 跳转到计划详情页面
        - 🚀 **启动计划**: 启动定时任务调度，计划将自动执行训练
        - ⏹️ **停止计划**: 停止所有定时任务
        - 🗑️ **删除计划**: 删除计划及其关联数据（训练记录、预测数据、Agent决策）
        - ⚠️ **注意**: 只能删除已停止的计划，运行中的计划需先停止
        """)

        # 保存组件引用
        components.update({
            'refresh_btn': refresh_btn,
            'plans_table': plans_table,
            'plan_id_input': plan_id_input,
            'view_detail_btn': view_detail_btn,
            'start_btn': start_btn,
            'stop_btn': stop_btn,
            'delete_btn': delete_btn,
            'operation_result': operation_result
        })

        # 绑定事件
        refresh_btn.click(
            fn=self.get_plans_data,
            outputs=[plans_table]
        )

        start_btn.click(
            fn=self.start_plan,
            inputs=[plan_id_input],
            outputs=[operation_result]
        ).then(
            fn=self.get_plans_data,
            outputs=[plans_table]
        )

        stop_btn.click(
            fn=self.stop_plan,
            inputs=[plan_id_input],
            outputs=[operation_result]
        ).then(
            fn=self.get_plans_data,
            outputs=[plans_table]
        )

        delete_btn.click(
            fn=self.delete_plan,
            inputs=[plan_id_input],
            outputs=[operation_result]
        ).then(
            fn=self.get_plans_data,
            outputs=[plans_table]
        )

        self.components = components
        return components

    def get_components(self):
        """获取UI组件字典（保持向后兼容）"""
        if not hasattr(self, 'components') or not self.components:
            self.build_ui()
        return self.components


def create_plan_list_ui():
    """创建计划列表界面（供外部调用）"""
    ui = PlanListUI()
    components = ui.build_ui()

    # 返回需要被外部使用的组件（保持向后兼容）
    return {
        'plan_id_input': components['plan_id_input'],
        'view_detail_btn': components['view_detail_btn'],
        'plans_table': components['plans_table']
    }
