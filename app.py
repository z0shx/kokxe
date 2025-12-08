"""
KOKEX 主应用入口
"""
import gradio as gr
import asyncio
import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Callable, Any
from functools import wraps

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from database.db import init_db, export_schema, get_db
from database.migrate import migrate_database
from database.models import TradingPlan, now_beijing
from ui.plan_create import create_plan_ui
from ui.plan_list import create_plan_list_ui
from ui.config_center import create_config_center_ui
from utils.logger import setup_logger
from utils.common import safe_plan_id, validate_plan_exists, extract_finetune_param
from services.langchain_agent import agent_service

logger = setup_logger(__name__, "app.log")


# 安全转换和验证函数已移至 utils/common


def safe_plan_id_wrapper(error_return_value=None):
    """
    装饰器：自动处理plan_id验证和错误处理

    Args:
        error_return_value: 发生错误时的返回值

    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 假设第一个参数是pid
            if args:
                pid = args[0]
                is_valid, plan_id, error_msg = validate_plan_exists(pid)
                if not is_valid:
                    return error_return_value if error_return_value is not None else f"❌ {error_msg}"

                # 用安全的plan_id替换原参数
                new_args = (plan_id,) + args[1:]
                return func(*new_args, **kwargs)
            else:
                # 如果没有参数，直接调用原函数
                return func(*args, **kwargs)
        return wrapper
    return decorator


def initialize_app():
    """初始化应用"""
    logger.info("=" * 60)
    logger.info("KOKEX 应用启动中...")
    logger.info("=" * 60)

    # 初始化数据库
    try:
        logger.info("正在初始化数据库...")
        init_db()
        logger.info("✅ 数据库初始化成功")

        # 执行数据库迁移
        logger.info("正在执行数据库迁移...")
        migrate_database()
        logger.info("✅ 数据库迁移完成")

        # 导出数据库 schema
        logger.info("正在导出数据库 schema...")
        export_schema()
        logger.info("✅ 数据库 schema 导出成功")

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        raise

    # 恢复卡住的训练记录
    try:
        logger.info("正在检查并恢复卡住的训练记录...")
        from services.training_service import TrainingService
        TrainingService.recover_stuck_training_records()
        logger.info("✅ 训练记录恢复检查完成")
    except Exception as e:
        logger.error(f"❌ 训练记录恢复失败: {e}")

    # 初始化调度器并重新加载定时任务
    try:
        logger.info("正在初始化定时任务调度器...")
        from services.schedule_service import ScheduleService

        # 初始化调度器
        ScheduleService.init_scheduler()
        logger.info("✅ 调度器初始化成功")

        # 重新加载定时任务（同步调用）
        ScheduleService.reload_all_schedules()
        logger.info("✅ 定时任务重新加载完成")

    except Exception as e:
        logger.error(f"⚠️ 调度器初始化失败: {e}")
        # 调度器失败不影响应用启动
        import traceback
        traceback.print_exc()

    # 恢复运行中计划的状态（WebSocket连接等）
    try:
        logger.info("正在恢复运行中计划的状态...")
        from services.ws_connection_manager import ws_connection_manager
        from services.account_ws_manager import account_ws_manager
        from database.db import get_db
        from database.models import TradingPlan

        with get_db() as db:
            # 查询所有运行中的计划
            running_plans = db.query(TradingPlan).filter(
                TradingPlan.status == 'running'
            ).all()

            logger.info(f"找到 {len(running_plans)} 个运行中的计划")

            for plan in running_plans:
                try:
                    # 注意：WebSocket连接将按需创建，不在这里强制恢复
                    # 移除重复的连接创建逻辑，避免与应用启动后的按需连接冲突
                    logger.info(f"计划 {plan.id} ({plan.plan_name}) 将在需要时自动恢复WebSocket连接")

                    # 仅重置WebSocket状态，让后续按需连接时正确显示状态
                    try:
                        from database.db import SessionLocal
                        update_db = SessionLocal()
                        try:
                            update_db.query(TradingPlan).filter(TradingPlan.id == plan.id).update({
                                'ws_connected': False  # 初始状态为未连接，将在实际连接时更新
                            })
                            update_db.commit()
                            logger.debug(f"✅ 计划 {plan.id} WebSocket状态已重置")
                        except Exception as db_error:
                            update_db.rollback()
                            logger.error(f"❌ 重置计划 {plan.id} WebSocket状态失败: {db_error}")
                        finally:
                            update_db.close()
                    except Exception as e:
                        logger.error(f"❌ 创建数据库连接重置计划 {plan.id} 状态失败: {e}")

                    # 恢复账户WebSocket连接（如果配置了API Key）
                    if plan.okx_api_key and plan.okx_secret_key and plan.okx_passphrase:
                        logger.info(f"恢复计划 {plan.id} ({plan.plan_name}) 的账户WebSocket连接")
                        account_ws_manager.get_or_create_connection(
                            api_key=plan.okx_api_key,
                            secret_key=plan.okx_secret_key,
                            passphrase=plan.okx_passphrase,
                            is_demo=plan.is_demo,
                            plan_id=plan.id
                        )
                        logger.info(f"✅ 计划 {plan.id} 账户WebSocket连接已恢复")

                    # 记录自动化配置状态
                    automation_status = []
                    if plan.auto_finetune_enabled:
                        automation_status.append("自动微调")
                    if plan.auto_inference_enabled:
                        automation_status.append("自动推理")
                    if plan.auto_agent_enabled:
                        automation_status.append("自动Agent")

                    if automation_status:
                        logger.info(f"计划 {plan.id} 启用的自动化功能: {', '.join(automation_status)}")

                except Exception as e:
                    logger.error(f"⚠️ 恢复计划 {plan.id} 失败: {e}")
                    continue

        logger.info("✅ 运行中计划状态恢复完成")

    except Exception as e:
        logger.error(f"⚠️ 恢复计划状态失败: {e}")
        import traceback
        traceback.print_exc()

    # 启动数据完整性验证服务
    try:
        from services.data_validation_service import data_validation_service
        import asyncio

        # 初始化数据验证服务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(data_validation_service.initialize())
        loop.close()

        # 启动定时验证调度器
        data_validation_service.start_validation_scheduler()
        logger.info("✅ 数据完整性验证服务已启动")

    except Exception as e:
        logger.error(f"⚠️ 启动数据验证服务失败: {e}")

    # 启动调度器健康检查
    try:
        from services.schedule_service import ScheduleService
        import threading
        import time

        def scheduler_health_check():
            """调度器健康检查线程"""
            while True:
                try:
                    time.sleep(300)  # 每5分钟检查一次

                    # 检查调度器是否有任务
                    scheduler = ScheduleService.get_scheduler()
                    jobs = scheduler.get_jobs()

                    # 检查是否有运行中的计划但没有对应调度任务
                    from database.db import get_db
                    from database.models import TradingPlan

                    with get_db() as db:
                        running_plans = db.query(TradingPlan).filter(
                            TradingPlan.status == 'running',
                            TradingPlan.auto_inference_enabled == True
                        ).all()

                        for plan in running_plans:
                            plan_jobs = ScheduleService.get_plan_jobs(plan.id)
                            inference_job = None

                            for job in plan_jobs:
                                if 'inference' in job.id:
                                    inference_job = job
                                    break

                            # 如果没有找到预测任务，重新加载
                            if not inference_job:
                                logger.warning(f"计划 {plan.id} 缺少预测调度任务，重新加载...")
                                try:
                                    loop = __import__('asyncio').new_event_loop()
                                    __import__('asyncio').set_event_loop(loop)
                                    success = loop.run_until_complete(ScheduleService.start_schedule(plan.id))
                                    if success:
                                        logger.info(f"✅ 计划 {plan.id} 预测任务重新加载成功")
                                    else:
                                        logger.error(f"❌ 计划 {plan.id} 预测任务重新加载失败")
                                except Exception as reload_error:
                                    logger.error(f"重新加载计划 {plan.id} 预测任务失败: {reload_error}")
                                finally:
                                    loop.close()

                except Exception as e:
                    logger.error(f"调度器健康检查失败: {e}")
                    time.sleep(60)  # 出错后等待1分钟再继续

        # 创建可停止的健康检查线程类
        class StoppableHealthCheckThread(threading.Thread):
            def __init__(self, target, name):
                super().__init__(target=target, name=name)
                self._stop_event = threading.Event()
                self.daemon = True

            def stop(self):
                self._stop_event.set()

            def stopped(self):
                return self._stop_event.is_set()

        # 修改健康检查函数以支持停止
        def stoppable_scheduler_health_check():
            """可停止的调度器健康检查线程"""
            while not health_check_thread.stopped():
                try:
                    # 使用可中断的sleep
                    if health_check_thread._stop_event.wait(timeout=300):  # 5分钟
                        break

                    # 检查调度器是否有任务
                    scheduler = ScheduleService.get_scheduler()
                    jobs = scheduler.get_jobs()

                    # 检查是否有运行中的计划但没有对应调度任务
                    from database.db import get_db
                    from database.models import TradingPlan

                    with get_db() as db:
                        running_plans = db.query(TradingPlan).filter(
                            TradingPlan.status == 'running',
                            TradingPlan.auto_inference_enabled == True
                        ).all()

                        for plan in running_plans:
                            plan_jobs = ScheduleService.get_plan_jobs(plan.id)
                            inference_job = None

                            for job in plan_jobs:
                                if 'inference' in job.id:
                                    inference_job = job
                                    break

                            # 如果没有找到预测任务，重新加载
                            if not inference_job:
                                logger.warning(f"计划 {plan.id} 缺少预测调度任务，重新加载...")
                                try:
                                    loop = __import__('asyncio').new_event_loop()
                                    __import__('asyncio').set_event_loop(loop)
                                    success = loop.run_until_complete(ScheduleService.start_schedule(plan.id))
                                    if success:
                                        logger.info(f"✅ 计划 {plan.id} 预测任务重新加载成功")
                                    else:
                                        logger.error(f"❌ 计划 {plan.id} 预测任务重新加载失败")
                                except Exception as reload_error:
                                    logger.error(f"重新加载计划 {plan.id} 预测任务失败: {reload_error}")
                                finally:
                                    loop.close()

                except Exception as e:
                    logger.error(f"调度器健康检查失败: {e}")
                    # 出错后等待一段时间，但要检查停止信号
                    if not health_check_thread._stop_event.wait(timeout=60):
                        break

        # 启动可停止的健康检查线程
        health_check_thread = StoppableHealthCheckThread(
            target=stoppable_scheduler_health_check,
            name="SchedulerHealthCheck"
        )
        health_check_thread.start()

  
        logger.info("✅ 调度器健康检查已启动")

    except Exception as e:
        logger.error(f"⚠️ 启动调度器健康检查失败: {e}")

    logger.info("=" * 60)
    logger.info("✅ KOKEX 应用初始化完成")
    logger.info("=" * 60)


def create_app():
    """创建 Gradio 应用"""

    # 初始化应用
    initialize_app()

    # 创建 Gradio 界面
    with gr.Blocks(
        title="KOKEX - AI 智投平台",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1600px !important;
        }

        /* 悬浮时间指示器样式 */
        .floating-time-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            font-weight: bold;
            z-index: 9999;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .floating-time-indicator .time-label {
            font-size: 12px;
            opacity: 0.8;
            margin-bottom: 2px;
        }

        .floating-time-indicator .current-time {
            font-size: 16px;
            color: #00ff88;
            text-shadow: 0 0 5px rgba(0, 255, 136, 0.5);
        }

        .floating-time-indicator .timezone {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 2px;
        }

        .floating-refresh-btn {
            position: fixed;
            top: 120px;
            right: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            min-width: 90px;
        }

        .floating-refresh-btn:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }

        .floating-refresh-btn:active {
            transform: translateY(0px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        /* 隐藏原来的刷新按钮 */
        .original-refresh-btn {
            display: none !important;
        }
        """
    ) as app:
        gr.Markdown(
            """
            # 🚀 KOKEX - AI 智投平台

            基于 Kronos 模型的持续微调与预测 + AI Agent 自动决策投资平台
            """
        )

        # 使用State保存当前选中的计划ID
        selected_plan_id = gr.State(value=None)

        with gr.Tabs() as tabs:
            with gr.Tab("📝 新增计划", id=0):
                create_plan_ui()

            with gr.Tab("📋 计划列表", id=1) as list_tab:
                from ui.plan_list import create_plan_list_ui
                list_components = create_plan_list_ui()
                plan_id_input = list_components['plan_id_input']
                view_detail_btn = list_components['view_detail_btn']

            with gr.Tab("📊 计划详情", id=2) as detail_tab:
                from ui.plan_detail import PlanDetailUI

                detail_ui = PlanDetailUI()

                # 详情页容器
                detail_container = gr.Column(visible=False)

                with detail_container:
                    # 返回按钮
                    back_to_list_btn = gr.Button("← 返回列表", size="sm")

                    # 刷新按钮
                    detail_refresh_btn = gr.Button("🔄 刷新数据", size="sm", elem_classes=["original-refresh-btn"])

                    # 上部：计划概览
                    overview_md = gr.Markdown("")

                    # 控制面板
                    gr.Markdown("### 🎛️ 控制面板")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ws_status_md = gr.Markdown("**WebSocket状态**: ⚪ 未连接")
                            with gr.Row():
                                ws_start_btn = gr.Button("▶️ 启动WebSocket", size="sm", variant="primary")
                                ws_stop_btn = gr.Button("⏸️ 停止WebSocket", size="sm", variant="stop", visible=False)
                            ws_result = gr.Markdown("")

                        with gr.Column(scale=1):
                            plan_status_md = gr.Markdown("**计划状态**: ⚪ 已创建")
                            with gr.Row():
                                plan_start_btn = gr.Button("🚀 启动计划", size="sm", variant="primary")
                                plan_stop_btn = gr.Button("⏹️ 停止计划", size="sm", variant="stop", visible=False)
                            plan_result = gr.Markdown("")

                    # 自动化配置开关
                    with gr.Accordion("⚙️ 自动化配置", open=False):
                        with gr.Row():
                            auto_finetune_switch = gr.Checkbox(
                                label="🔄 自动微调训练（按时间表自动训练模型）",
                                value=False
                            )
                            auto_inference_switch = gr.Checkbox(
                                label="🔮 自动预测推理（训练完成后自动推理）",
                                value=False
                            )
                        with gr.Row():
                            auto_agent_switch = gr.Checkbox(
                                label="🤖 自动Agent决策（推理完成后自动触发Agent）",
                                value=False
                            )
                        with gr.Row():
                            save_automation_btn = gr.Button("💾 保存自动化配置", size="sm", variant="primary")
                            automation_config_result = gr.Markdown("")

                        # 自动微调时间表管理
                        gr.Markdown("**⏰ 自动微调时间表**")
                        with gr.Row():
                            schedule_time_input = gr.Textbox(
                                label="",
                                placeholder="HH:MM (如: 08:00)",
                                scale=2
                            )
                            add_schedule_time_btn = gr.Button("➕ 添加", size="sm", scale=1)
                            remove_schedule_time_btn = gr.Button("➖ 删除", size="sm", scale=1)
                            manual_finetune_btn = gr.Button("🚀 手动触发", size="sm", variant="secondary", scale=1)

                        schedule_time_list = gr.Textbox(
                            label="当前时间表",
                            placeholder="暂无时间点",
                            interactive=False,
                            lines=2
                        )
                        schedule_operation_result = gr.Markdown("")

                        # 自动预测间隔时间管理
                        gr.Markdown("**🔮 自动预测间隔时间**")
                        with gr.Row():
                            inference_interval_input = gr.Number(
                                label="预测间隔时间（小时）",
                                value=4,
                                minimum=1,
                                maximum=168,
                                step=1,
                                scale=2
                            )
                            set_inference_interval_btn = gr.Button("💾 设置间隔", size="sm", scale=1)
                            manual_prediction_trigger_btn = gr.Button("🔮 手动触发", size="sm", variant="secondary", scale=1)

                        inference_schedule_display = gr.Textbox(
                            label="当前预测间隔",
                            placeholder="暂无间隔设置",
                            interactive=False,
                            lines=1
                        )
                        inference_schedule_operation_result = gr.Markdown("")

                    # === 模型训练区域 ===
                    with gr.Accordion("🎯 模型训练记录", open=True):
                        # 微调参数配置
                        with gr.Accordion("⚙️ 微调参数配置", open=False):
                            with gr.Row():
                                lookback_window = gr.Number(
                                    label="Lookback Window",
                                    value=512,
                                    precision=0
                                )
                                predict_window = gr.Number(
                                    label="Predict Window",
                                    value=48,
                                    precision=0
                                )
                                batch_size = gr.Number(
                                    label="Batch Size",
                                    value=16,
                                    precision=0
                                )
                            with gr.Row():
                                tokenizer_epochs = gr.Number(
                                    label="Tokenizer Epochs",
                                    value=25,
                                    precision=0
                                )
                                predictor_epochs = gr.Number(
                                    label="Predictor Epochs",
                                    value=50,
                                    precision=0
                                )
                                learning_rate = gr.Number(
                                    label="Learning Rate",
                                    value=1e-4
                                )
                            save_params_btn = gr.Button("💾 保存参数", size="sm")
                            params_status = gr.Markdown("")

                        # 训练数据范围配置
                        with gr.Accordion("📅 训练数据范围", open=False):
                            train_data_range_info = gr.Markdown(
                                "**数据范围**: 请先加载计划后选择训练范围"
                            )

                            # 快捷选择按钮
                            gr.Markdown("**快捷选择**")
                            with gr.Row():
                                train_days_30_btn = gr.Button("最近30天", size="sm")
                                train_days_60_btn = gr.Button("最近60天", size="sm")
                                train_days_90_btn = gr.Button("最近90天", size="sm")

                            # 日期范围
                            gr.Markdown("**自定义范围**")
                            with gr.Row():
                                train_start_date = gr.Textbox(
                                    label="训练开始日期",
                                    placeholder="YYYY-MM-DD",
                                    scale=1
                                )
                                train_end_date = gr.Textbox(
                                    label="训练结束日期",
                                    placeholder="YYYY-MM-DD",
                                    scale=1
                                )
                                save_train_data_config_btn = gr.Button("💾 保存配置", size="sm", variant="primary")

                            train_data_config_result = gr.Markdown("")

                        training_df = gr.DataFrame(
                            interactive=False,
                            wrap=True,
                            label="训练记录列表"
                        )

                        with gr.Row():
                            start_training_btn = gr.Button("▶️ 开始训练", variant="primary")
                            # auto_train_toggle = gr.Checkbox(label="自动训练", value=False)

                        training_status = gr.Markdown("等待操作...")

                        # 训练记录操作
                        with gr.Accordion("🛠️ 训练记录操作", open=False):
                            training_record_id = gr.Number(
                                label="训练记录ID",
                                precision=0,
                                value=None
                            )
                            with gr.Row():
                                cancel_training_btn = gr.Button("⏸️ 取消训练", size="sm")
                                delete_training_btn = gr.Button("🗑️ 删除记录", size="sm", variant="stop")
                            training_operation_result = gr.Markdown("")

                    # === K线图区域 ===
                    with gr.Accordion("📈 K线图 & 预测数据", open=True):
                        with gr.Row():
                            show_pred_toggle = gr.Checkbox(label="显示预测", value=True)
                            days_slider = gr.Slider(
                                minimum=3, maximum=30, value=10, step=1,
                                label="显示天数"
                            )

                        kline_chart = gr.Plot(
                            label="K线预览图",
                            show_label=True
                        )

                        # 概率指标展示（紧跟在K线图下方）
                        probability_indicators_md = gr.Markdown("")

                    # === Kronos推理区域 ===
                    with gr.Accordion("🔮 Kronos 推理记录", open=True):
                        # 推理参数配置
                        with gr.Accordion("⚙️ 推理参数配置", open=False):
                            with gr.Row():
                                inference_lookback_window = gr.Number(
                                    label="Lookback Window (回看窗口)",
                                    value=512,
                                    minimum=64,
                                    maximum=2048,
                                    precision=0,
                                    info="使用多少历史数据点进行预测"
                                )
                                inference_predict_window = gr.Number(
                                    label="Predict Window (预测窗口)",
                                    value=48,
                                    minimum=1,
                                    maximum=512,
                                    precision=0,
                                    info="预测未来多少个数据点"
                                )
                            with gr.Row():
                                inference_temperature = gr.Number(
                                    label="Temperature (温度)",
                                    value=1.0,
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.1
                                )
                                inference_top_p = gr.Number(
                                    label="Top-p (核采样)",
                                    value=0.9,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.05
                                )
                                inference_sample_count = gr.Number(
                                    label="Sample Count (蒙特卡罗路径数)",
                                    value=30,
                                    minimum=1,
                                    maximum=100,
                                    precision=0
                                )
                            with gr.Row():
                                inference_data_offset = gr.Number(
                                    label="Data Offset (数据偏移)",
                                    value=0,
                                    minimum=0,
                                    maximum=1000,
                                    precision=0,
                                    info="向时间早偏移多少个数据点进行预测"
                                )
                            # 推理数据点时间戳范围显示
                            inference_data_range_info = gr.Markdown("请保存推理参数后查看数据范围...")
                            with gr.Row():
                                save_inference_params_btn = gr.Button("💾 保存推理参数", size="sm", variant="primary")
                                inference_params_status = gr.Markdown("")

                        inference_df = gr.DataFrame(
                            interactive=False,
                            wrap=True,
                            label="推理记录列表"
                        )

                        gr.Markdown("**推理操作**")
                        with gr.Row():
                            inference_record_id = gr.Number(
                                label="训练记录ID",
                                precision=0,
                                value=None
                            )
                            execute_inference_btn = gr.Button("📈 预测交易数据", size="sm", variant="primary")
                            mock_prediction_btn = gr.Button("🎲 Mock预测", size="sm")

                        inference_operation_result = gr.Markdown("")

                        gr.Markdown("**预测数据预览**")
                        prediction_data_preview = gr.Textbox(
                            label="预测数据 (供AI Agent使用)",
                            lines=8,
                            max_lines=12,
                            interactive=False,
                            placeholder="执行预测后将显示预测数据..."
                        )

                    # === AI Agent区域 ===
                    with gr.Accordion("🤖 AI Agent 决策记录", open=True):
                        # Agent配置
                        with gr.Accordion("⚙️ Agent配置", open=False):
                            # LLM配置选择
                            llm_config_dropdown = gr.Dropdown(
                                label="LLM配置",
                                choices=[],
                                value=None
                            )

                            # 提示词模版选择
                            prompt_template_dropdown = gr.Dropdown(
                                label="提示词模版",
                                choices=[],
                                value=None
                            )

                            # 提示词编辑
                            agent_prompt_textbox = gr.Textbox(
                                label="提示词内容",
                                lines=5,
                                placeholder="输入Agent提示词..."
                            )

                            # 工具配置
                            gr.Markdown("**可用工具** (勾选启用工具)")
                            with gr.Row():
                                tool_query_prediction = gr.Checkbox(label="🔮 query_prediction_data", value=True, info="按时间范围和批次ID查询预测数据")
                                tool_prediction_history = gr.Checkbox(label="📈 get_prediction_history", value=True, info="查询历史预测批次列表（最多30批次）")
                                tool_query_historical_kline = gr.Checkbox(label="📈 query_historical_kline_data", value=True, info="查询历史K线数据（UTC+8时间戳）")
                            with gr.Row():
                                tool_get_utc_time = gr.Checkbox(label="🕒 get_current_utc_time", value=True, info="获取当前UTC+8时间")
                                tool_run_inference = gr.Checkbox(label="🤖 run_latest_model_inference", value=False, info="触发最新模型推理")
                                tool_get_account = gr.Checkbox(label="🔍 get_account_balance", value=True, info="查询账户余额")
                            with gr.Row():
                                tool_get_pending_orders = gr.Checkbox(label="📋 get_pending_orders", value=True, info="查询挂单")
                                tool_place_order = gr.Checkbox(label="💰 place_order", value=True, info="下限价单")
                                tool_cancel_order = gr.Checkbox(label="❌ cancel_order", value=True, info="撤单")
                                tool_amend_order = gr.Checkbox(label="✏️ amend_order", value=True, info="改单")

                              
                         # 保存按钮
                            with gr.Row():
                                save_agent_config_btn = gr.Button("💾 保存配置", size="sm")
                                load_template_btn = gr.Button("📥 加载模版", size="sm")

                            agent_config_status = gr.Markdown("")

                        # 交易限制配置
                        with gr.Accordion("💰 交易限制配置", open=False):
                            gr.Markdown("AI Agent将严格遵守以下交易限制进行工具调用：")

                            with gr.Row():
                                quick_usdt_amount = gr.Number(
                                    label="可用资金 (USDT)",
                                    value=1000.0,
                                    minimum=0.0,
                                    maximum=1000000.0,
                                    step=10.0,
                                    info="固定USDT资金数量"
                                )

                                quick_usdt_percentage = gr.Slider(
                                    label="资金比例 (%)",
                                    minimum=1.0,
                                    maximum=100.0,
                                    value=30.0,
                                    step=1.0,
                                    info="资金使用比例，固定USDT不足时使用百分比"
                                )

                            with gr.Row():
                                quick_avg_orders = gr.Number(
                                    label="平摊单量",
                                    value=10.0,
                                    minimum=1.0,
                                    maximum=100.0,
                                    step=1.0,
                                    info="将交易金额平分成多少笔订单"
                                )

                                quick_stop_loss = gr.Slider(
                                    label="止损比例 (%)",
                                    minimum=1.0,
                                    maximum=50.0,
                                    value=20.0,
                                    step=1.0,
                                    info="亏损超过多少百分比时止损卖出"
                                )

                            gr.Markdown("""
                            💡 **说明**:
                            - 当固定USDT资金不足时，AI Agent会自动查询账户余额并使用百分比限制
                            - 平摊操作可以降低市场冲击和风险
                            - 止损机制有助于控制风险
                            """)

                            with gr.Row():
                                save_trading_limits_btn = gr.Button("💾 保存交易限制", size="sm", variant="primary")
                                reset_trading_limits_btn = gr.Button("🔄 重置默认", size="sm")

                            trading_limits_status = gr.Markdown("")

                        with gr.Row():
                            refresh_agent_btn = gr.Button("🔄 刷新对话记录", size="sm", variant="secondary")
                            clear_agent_records_btn = gr.Button("🗑️ 清除记录", size="sm", variant="secondary")

                        agent_df = gr.DataFrame(
                            interactive=True,  # 改为可交互以支持点击事件
                            wrap=True,
                            label="Agent对话记录"
                        )

                        # 移除agent_detail，因为详情将显示在chatbot中

                        # AI Agent 对话界面
                        chat_ui = detail_ui.get_chat_ui_components()
                        chat_components = chat_ui.build_ui()

                        # 从 chat_components 中提取主要组件
                        agent_chatbot = chat_components['agent_chatbot']
                        agent_user_input = chat_components['agent_user_input']
                        agent_send_btn = chat_components['agent_send_btn']
                        agent_execute_inference_btn = chat_components['agent_execute_inference_btn']
                        agent_clear_btn = chat_components['agent_clear_btn']
                        agent_status = chat_components['agent_status']

                        # 工具确认功能已废弃 - AI Agent现在可以直接使用启用的工具

                    # === 账户信息区域 ===
                    with gr.Accordion("💰 账户信息", open=True):
                        with gr.Row():
                            with gr.Column(scale=9):
                                account_status = gr.Markdown("### 💰 账户信息\n\n账户信息加载中...")
                            with gr.Column(scale=1):
                                account_refresh_btn = gr.Button("🔄 刷新", size="sm")

                        # 账户信息自动刷新定时器（每2秒）
                        account_timer = gr.Timer(value=2.0, active=False)

                    # === 订单记录区域 ===
                    with gr.Accordion("📋 订单记录", open=True):
                        with gr.Row():
                            with gr.Column(scale=1):
                                order_refresh_btn = gr.Button("🔄 刷新", size="sm")
                        with gr.Row():
                            with gr.Column():
                                order_table = gr.DataFrame(
                                    label="订单记录",
                                    interactive=False
                                )
                    # === 任务执行记录区域 ===
                    with gr.Accordion("📋 任务执行记录", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                task_refresh_btn = gr.Button("🔄 刷新", size="sm")
                        with gr.Row():
                                task_executions_df = gr.DataFrame(
                                    label="任务执行历史",
                                    interactive=False
                                )

                # 无计划时的提示
                no_plan_msg = gr.Markdown(
                    "### 请先从计划列表中选择一个计划",
                    visible=True
                )

                # 加载详情函数
                def load_plan_detail(plan_id):
                    if not plan_id or plan_id <= 0:
                        return (
                            gr.update(visible=False),  # detail_container
                            gr.update(visible=True),   # no_plan_msg
                            "",  # overview_md
                            "", "",  # ws_result, plan_result
                            "**WebSocket状态**: ⚪ 未连接", "**计划状态**: ⚪ 已创建",  # ws_status_md, plan_status_md
                            gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),  # ws_start_btn, ws_stop_btn, plan_start_btn, plan_stop_btn
                            False, False, False, "",  # automation switches & result
                            "", "", "", "",  # schedule_time_list, schedule_operation_result, inference_schedule_display, inference_schedule_operation_result
                            512, 48, 16, 25, 50, 1e-4, "",  # 微调参数
                            "", "", "", "",  # train_data_range_info, train_start_date, train_end_date, train_data_config_result
                            512, 48,  # inference_lookback_window, inference_predict_window
                            1.0, 0.9, 30, 0, "",  # inference_temperature, inference_top_p, inference_sample_count, inference_data_offset, inference_params_status
                            gr.update(), None, "",  # llm_config, prompt_template, agent_prompt
                            True, True, True, True, True, True, True, True, True, True, True, True, True,  # 工具选择
                            1000.0, 30.0, 10.0, 20.0,  # 交易限制默认值：quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss
                            gr.DataFrame(), gr.Plot(), "", gr.DataFrame(), "请保存推理参数后查看数据范围...", "", gr.DataFrame(), [{"role": "assistant", "content": "请先选择计划"}], "", "", "",  # training_df, kline_chart, probability_indicators_md, inference_df, inference_data_range_info, prediction_data_preview, agent_df, agent_chatbot, agent_user_input, agent_status
                            "### 💰 账户信息\n\n未加载",  # account_status
                            gr.DataFrame(),  # order_table
                            gr.DataFrame(),  # task_executions_df  # task_executions
                            gr.Timer(active=False),  # account_timer
                            None  # inference_record_id
                        )

                    def safe_int(value, default=0):
                        """安全转换为整数"""
                        try:
                            if value is None:
                                return default
                            if isinstance(value, str):
                                return int(float(value))
                            return int(value)
                        except (ValueError, TypeError):
                            return default

                    def safe_float(value, default=0.0):
                        """安全转换为浮点数"""
                        try:
                            if value is None:
                                return default
                            if isinstance(value, str):
                                return float(value)
                            return float(value)
                        except (ValueError, TypeError):
                            return default

                    # 调用render_plan_overview获取概览文本和状态信息
                    overview_data = detail_ui.render_plan_overview(int(plan_id))
                    overview_text = overview_data[0]

                    # 获取微调参数
                    params = detail_ui.get_finetune_params(int(plan_id))

                    # 获取计划信息和数据范围
                    from database.db import get_db
                    from database.models import TradingPlan
                    with get_db() as db:
                        plan = db.query(TradingPlan).filter(TradingPlan.id == int(plan_id)).first()
                        if plan:
                            # 获取训练数据统计信息
                            range_info = detail_ui.get_training_data_stats(int(plan_id))

                            # 获取数据库中的最新数据范围
                            min_date, max_date, total_count = detail_ui.get_data_date_range(plan.inst_id, plan.interval)

                            # 从 finetune_params 中获取已配置的日期范围
                            finetune_params = plan.finetune_params or {}
                            data_config = finetune_params.get('data', {})
                            start_date = data_config.get('train_start_date', '')
                            end_date = data_config.get('train_end_date', '')

                            # 自动更新训练数据范围到最新数据
                            if min_date and max_date:
                                from datetime import datetime, timedelta

                                # 如果没有配置训练范围，或者配置的结束日期早于最新数据日期
                                if not start_date or not end_date:
                                    # 使用最近30天作为默认值
                                    start_date_default = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                                    end_date_default = max_date.strftime('%Y-%m-%d')
                                    start_date = start_date or start_date_default
                                    end_date = end_date or end_date_default
                                else:
                                    # 检查配置的结束日期是否需要更新到最新数据日期
                                    try:
                                        configured_end = datetime.strptime(end_date, '%Y-%m-%d')
                                        if configured_end.date() < max_date.date():
                                            # 自动更新结束日期到最新数据日期
                                            end_date = max_date.strftime('%Y-%m-%d')

                                            # 同时确保开始日期不会过晚
                                            configured_start = datetime.strptime(start_date, '%Y-%m-%d')
                                            if configured_start.date() >= max_date.date():
                                                # 如果开始日期晚于或等于最新数据日期，重新设置为30天范围
                                                start_date = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                                    except ValueError:
                                        # 如果日期解析失败，使用默认值
                                        start_date = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                                        end_date = max_date.strftime('%Y-%m-%d')
                        else:
                            range_info, start_date, end_date = "", "", ""

                    # 获取Agent配置
                    agent_config = detail_ui.get_agent_config(int(plan_id))
                    tools_config = agent_config.get('agent_tools_config', {})

                    # ReAct 配置已移除，使用默认值
                    max_iterations = 3
                    enable_thinking = False
                    thinking_style = "详细"

                    # 获取推理参数配置
                    inference_params = detail_ui.get_inference_params(int(plan_id))

                    # 获取自动化配置
                    automation_config = detail_ui.get_automation_config(int(plan_id))

                    # 获取自动微调时间表
                    schedule_list = detail_ui.get_finetune_schedule(int(plan_id))
                    schedule_text = ', '.join(schedule_list) if schedule_list else '暂无时间点'

                    # 获取自动预测时间表
                    inference_schedule_list = detail_ui.get_inference_schedule(int(plan_id))
                    inference_schedule_text = ', '.join(str(x) for x in inference_schedule_list) + '小时间隔' if inference_schedule_list else '暂无预测时间点'

                    # 获取LLM配置和提示词模板列表
                    try:
                        llm_configs = detail_ui.get_llm_configs()
                        logger.info(f"获取到 {len(llm_configs)} 个LLM配置: {llm_configs}")
                    except Exception as e:
                        logger.error(f"获取LLM配置失败: {e}")
                        llm_configs = []

                    try:
                        prompt_templates = detail_ui.get_prompt_templates()
                        logger.info(f"获取到 {len(prompt_templates)} 个提示词模板: {prompt_templates}")
                    except Exception as e:
                        logger.error(f"获取提示词模板失败: {e}")
                        prompt_templates = []

                    # 获取交易限制配置
                    trading_limits = detail_ui.get_trading_limits_config(int(plan_id))
                    # 确保交易限制的类型转换正确，避免从数据库读取的字符串导致的错误
                    quick_usdt_amount = float(trading_limits.get('available_usdt_amount', 1000.0))
                    quick_usdt_percentage = float(trading_limits.get('available_usdt_percentage', 30.0))
                    quick_avg_orders = int(trading_limits.get('avg_order_count', 10))
                    quick_stop_loss = float(trading_limits.get('stop_loss_percentage', 20.0))

                    # 获取最新的对话消息
                    latest_agent_output = detail_ui.get_latest_conversation_messages(int(plan_id))

                    # 获取账户信息和订单记录
                    account_info = detail_ui.get_account_info(int(plan_id))
                    orders_df = detail_ui.get_orders_info(int(plan_id))

                    # 确保 orders_df 是有效的 DataFrame
                    if not isinstance(orders_df, pd.DataFrame):
                        logger.warning(f"orders_df 不是 DataFrame 类型: {type(orders_df)}")
                        orders_df = pd.DataFrame()

                    # 获取概率指标
                    probability_indicators = detail_ui.get_probability_indicators(int(plan_id))

                    # 获取任务执行记录
                    task_executions_df = detail_ui.load_task_executions(int(plan_id))
                    # 确保 task_executions_df 是有效的 DataFrame
                    if not isinstance(task_executions_df, pd.DataFrame):
                        logger.warning(f"task_executions_df 不是 DataFrame 类型: {type(task_executions_df)}")
                        task_executions_df = pd.DataFrame()

                    return (
                        gr.update(visible=True),   # detail_container
                        gr.update(visible=False),  # no_plan_msg
                        overview_text,  # overview_md - 只需要一个字符串
                        "", "",  # ws_result, plan_result
                        overview_data[1], overview_data[4],  # ws_status_md, plan_status_md
                        gr.update(visible=not overview_data[2]), gr.update(visible=overview_data[3]), gr.update(visible=not overview_data[5]), gr.update(visible=overview_data[6]),  # ws_start_btn, ws_stop_btn, plan_start_btn, plan_stop_btn
                        automation_config.get('auto_finetune_enabled', False),  # auto_finetune_switch
                        automation_config.get('auto_inference_enabled', False),  # auto_inference_switch
                        automation_config.get('auto_agent_enabled', False),  # auto_agent_switch
                        "",  # automation_config_result
                        schedule_text,  # schedule_time_list
                        "",  # schedule_operation_result
                        inference_schedule_text,  # inference_schedule_display
                        "",  # inference_schedule_operation_result
                        safe_int(extract_finetune_param(params, 'lookback_window'), 400),
                        safe_int(extract_finetune_param(params, 'predict_window'), 18),
                        safe_int(extract_finetune_param(params, 'batch_size'), 16),
                        safe_int(extract_finetune_param(params, 'tokenizer_epochs'), 5),
                        safe_int(extract_finetune_param(params, 'predictor_epochs'), 10),
                        safe_float(extract_finetune_param(params, 'learning_rate'), 1e-4),
                        "",  # params_status
                        range_info,  # train_data_range_info
                        start_date,  # train_start_date
                        end_date,    # train_end_date
                        "",  # train_data_config_result
                        safe_int(inference_params.get('lookback_window'), 512),  # inference_lookback_window
                        safe_int(inference_params.get('predict_window'), 48),  # inference_predict_window
                        safe_float(inference_params.get('temperature'), 1.0),  # inference_temperature
                        safe_float(inference_params.get('top_p'), 0.9),  # inference_top_p
                        safe_int(inference_params.get('sample_count'), 30),  # inference_sample_count
                        safe_int(inference_params.get('data_offset'), 0),  # inference_data_offset
                        "",  # inference_params_status
                        gr.update(choices=llm_configs if isinstance(llm_configs, list) else [], value=int(agent_config.get('llm_config_id')) if agent_config.get('llm_config_id') is not None else None),  # llm_config_dropdown
                        gr.update(choices=prompt_templates if isinstance(prompt_templates, list) else [], value=None),  # prompt_template_dropdown
                        agent_config.get('agent_prompt', ''),  # agent_prompt_textbox
                        tools_config.get('query_prediction_data', True),  # tool_query_prediction
                        tools_config.get('get_prediction_history', True),  # tool_prediction_history
                        tools_config.get('query_historical_kline_data', True),  # tool_query_historical_kline
                        tools_config.get('get_current_utc_time', True),  # tool_get_utc_time
                        tools_config.get('run_latest_model_inference', False),  # tool_run_inference
                        tools_config.get('get_account_balance', True),  # tool_get_account
                        tools_config.get('get_pending_orders', True),  # tool_get_pending_orders
                        tools_config.get('place_order', True),  # tool_place_order
                        tools_config.get('cancel_order', True),  # tool_cancel_order
                        tools_config.get('amend_order', True),  # tool_amend_order
                        safe_float(quick_usdt_amount, 1000.0),  # quick_usdt_amount
                        safe_float(quick_usdt_percentage, 30.0),  # quick_usdt_percentage
                        safe_int(quick_avg_orders, 10),  # quick_avg_orders
                        safe_float(quick_stop_loss, 20.0),  # quick_stop_loss
                        detail_ui.load_training_records(int(plan_id)),  # training_df
                        detail_ui.generate_kline_chart(int(plan_id)),  # kline_chart
                        probability_indicators,  # probability_indicators_md
                        detail_ui.load_inference_records(int(plan_id)),  # inference_df
                        detail_ui.get_inference_data_timestamp_range(int(plan_id)),  # inference_data_range_info
                        "",  # prediction_data_preview (空字符串)
                        detail_ui.load_agent_decisions(int(plan_id)),  # agent_df
                        latest_agent_output, "", "",  # agent_chatbot, agent_user_input, agent_status
                        account_info,  # account_status
                        orders_df,  # order_table
                        task_executions_df,  # task_executions_df
                        gr.Timer(active=True),  # account_timer - 启动账户定时器
                        get_latest_training_id(int(plan_id))  # 自动填充最新的训练记录ID
                    )

                # 保存参数函数
                def save_params(plan_id, lw, pw, bs, te, pe, lr):
                    if not plan_id:
                        return "❌ 请先选择计划"

                    # 构建嵌套格式的参数
                    params = {
                        'data': {
                            'lookback_window': int(lw) if lw else 400,
                            'predict_window': int(pw) if pw else 18
                        },
                        'batch_size': int(bs) if bs else 32,
                        'tokenizer_epochs': int(te) if te else 5,
                        'predictor_epochs': int(pe) if pe else 10,
                        'learning_rate': float(lr) if lr else 0.0001
                    }

                    # 获取现有配置以保留其他字段（如inference配置）
                    try:
                        from database.db import get_db
                        from database.models import TradingPlan
                        import json

                        with get_db() as db:
                            plan = db.query(TradingPlan).filter(TradingPlan.id == int(plan_id)).first()
                            if plan and plan.finetune_params:
                                if isinstance(plan.finetune_params, str):
                                    existing_params = json.loads(plan.finetune_params)
                                else:
                                    existing_params = plan.finetune_params

                                # 保留inference配置和其他字段
                                if 'inference' in existing_params:
                                    params['inference'] = existing_params['inference']
                                if 'auto_finetune_schedule' in existing_params:
                                    params['auto_finetune_schedule'] = existing_params['auto_finetune_schedule']
                    except Exception as e:
                        logger.error(f"获取现有配置失败: {e}")
                        # 继续执行，不阻止保存

                    return detail_ui.save_finetune_params(int(plan_id), params)

                # 自动化配置保存函数
                def save_automation_wrapper(pid, auto_ft, auto_inf, auto_ag):
                    if not pid:
                        return "❌ 请先选择计划"
                    # 获取当前时间表
                    current_schedule = detail_ui.get_finetune_schedule(int(pid))
                    schedule_times_str = ",".join(current_schedule) if current_schedule else ""
                    return detail_ui.save_automation_config(
                        int(pid), auto_ft, auto_inf, auto_ag, False, schedule_times_str  # auto_tool固定为False
                    )

                save_automation_btn.click(
                    fn=save_automation_wrapper,
                    inputs=[
                        plan_id_input,
                        auto_finetune_switch,
                        auto_inference_switch,
                        auto_agent_switch
                    ],
                    outputs=[automation_config_result]
                )

                # 时间表管理事件
                def add_schedule_time_wrapper(pid, time_str):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    message, schedule_list = detail_ui.add_finetune_schedule_time(int(pid), time_str)
                    schedule_text = ', '.join(schedule_list) if schedule_list else '暂无时间点'
                    return message, schedule_text

                def remove_schedule_time_wrapper(pid, time_str):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    message, schedule_list = detail_ui.remove_finetune_schedule_time(int(pid), time_str)
                    schedule_text = ', '.join(schedule_list) if schedule_list else '暂无时间点'
                    return message, schedule_text

                def set_inference_interval_wrapper(pid, interval_hours):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    message, interval_list = detail_ui.add_inference_schedule_time(int(pid), f"{interval_hours}:00")  # 兼容性调用
                    interval_text = f"{interval_list[0]}小时间隔" if interval_list else '4小时间隔'
                    return message, interval_text

                def manual_finetune_wrapper(pid):
                    """手动触发微调训练"""
                    if not pid:
                        return "❌ 请先选择计划"

                    try:
                        from services.schedule_service import ScheduleService
                        result = ScheduleService.trigger_finetune(int(pid))
                        if result['success']:
                            return f"✅ 手动微调训练已启动: {result['message']}"
                        else:
                            return f"❌ 手动微调训练失败: {result['error']}"
                    except Exception as e:
                        logger.error(f"手动触发微调失败: {e}")
                        return f"❌ 手动微调训练失败: {str(e)}"

                def manual_inference_wrapper(pid):
                    """手动触发预测推理"""
                    if not pid:
                        return "❌ 请先选择计划"

                    try:
                        from services.schedule_service import ScheduleService
                        result = ScheduleService.trigger_inference(int(pid))
                        if result['success']:
                            return f"✅ 手动预测推理已启动: {result['message']}"
                        else:
                            return f"❌ 手动预测推理失败: {result['error']}"
                    except Exception as e:
                        logger.error(f"手动触发预测失败: {e}")
                        return f"❌ 手动预测推理失败: {str(e)}"

                def add_inference_schedule_time_wrapper(pid, time_str):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    # 为了兼容性，将时间点转换为间隔时间设置
                    message, interval_list = detail_ui.add_inference_schedule_time(int(pid), time_str)
                    interval_text = f"{interval_list[0]}小时间隔" if interval_list else '4小时间隔'
                    return message, interval_text

                def remove_inference_schedule_time_wrapper(pid, time_str):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    # 兼容性调用，实际上会重置为默认间隔
                    message, interval_list = detail_ui.remove_inference_schedule_time(int(pid), time_str)
                    interval_text = f"{interval_list[0]}小时间隔" if interval_list else '4小时间隔'
                    return message, interval_text

                add_schedule_time_btn.click(
                    fn=add_schedule_time_wrapper,
                    inputs=[plan_id_input, schedule_time_input],
                    outputs=[schedule_operation_result, schedule_time_list]
                )

                remove_schedule_time_btn.click(
                    fn=remove_schedule_time_wrapper,
                    inputs=[plan_id_input, schedule_time_input],
                    outputs=[schedule_operation_result, schedule_time_list]
                )

                # 自动预测间隔时间事件
                set_inference_interval_btn.click(
                    fn=set_inference_interval_wrapper,
                    inputs=[plan_id_input, inference_interval_input],
                    outputs=[inference_schedule_operation_result, inference_schedule_display]
                )

                # 手动触发事件
                manual_finetune_btn.click(
                    fn=manual_finetune_wrapper,
                    inputs=[plan_id_input],
                    outputs=[schedule_operation_result]
                )

                manual_prediction_trigger_btn.click(
                    fn=manual_inference_wrapper,
                    inputs=[plan_id_input],
                    outputs=[inference_schedule_operation_result]
                )

                # 保留兼容性事件（如果还有其他地方使用的话）
                # add_inference_schedule_time_btn.click(
                #     fn=add_inference_schedule_time_wrapper,
                #     inputs=[plan_id_input, "08:00"],  # 默认时间点
                #     outputs=[inference_schedule_operation_result, inference_schedule_display]
                # )

                # remove_inference_schedule_time_btn.click(
                #     fn=remove_inference_schedule_time_wrapper,
                #     inputs=[plan_id_input, "08:00"],  # 默认时间点
                #     outputs=[inference_schedule_operation_result, inference_schedule_display]
                # )

                # WebSocket控制事件
                async def ws_start_wrapper(pid):
                    if not pid:
                        return "❌ 请先选择计划", gr.update(), gr.update()
                    result = await detail_ui.start_websocket_async(int(pid))
                    # 重新获取状态
                    overview_data = detail_ui.render_plan_overview(int(pid))
                    return result, overview_data[1], gr.update(visible=not overview_data[2]), gr.update(visible=overview_data[3])

                async def ws_stop_wrapper(pid):
                    if not pid:
                        return "❌ 请先选择计划", gr.update(), gr.update()
                    result = await detail_ui.stop_websocket_async(int(pid))
                    # 重新获取状态
                    overview_data = detail_ui.render_plan_overview(int(pid))
                    return result, overview_data[1], gr.update(visible=overview_data[2]), gr.update(visible=not overview_data[3])

                ws_start_btn.click(
                    fn=ws_start_wrapper,
                    inputs=[plan_id_input],
                    outputs=[ws_result, ws_status_md, ws_start_btn, ws_stop_btn]
                )

                ws_stop_btn.click(
                    fn=ws_stop_wrapper,
                    inputs=[plan_id_input],
                    outputs=[ws_result, ws_status_md, ws_start_btn, ws_stop_btn]
                )

                # 计划控制事件
                async def plan_start_wrapper(pid):
                    if not pid:
                        return "❌ 请先选择计划", gr.update(), gr.update()
                    result = await detail_ui.start_plan_async(int(pid))
                    # 重新获取状态
                    overview_data = detail_ui.render_plan_overview(int(pid))
                    return result, overview_data[4], gr.update(visible=not overview_data[5]), gr.update(visible=overview_data[6])

                async def plan_stop_wrapper(pid):
                    if not pid:
                        return "❌ 请先选择计划", gr.update(), gr.update()
                    result = await detail_ui.stop_plan_async(int(pid))
                    # 重新获取状态
                    overview_data = detail_ui.render_plan_overview(int(pid))
                    return result, overview_data[4], gr.update(visible=overview_data[5]), gr.update(visible=not overview_data[6])

                plan_start_btn.click(
                    fn=plan_start_wrapper,
                    inputs=[plan_id_input],
                    outputs=[plan_result, plan_status_md, plan_start_btn, plan_stop_btn]
                )

                plan_stop_btn.click(
                    fn=plan_stop_wrapper,
                    inputs=[plan_id_input],
                    outputs=[plan_result, plan_status_md, plan_start_btn, plan_stop_btn]
                )

                # 查看详情按钮
                view_detail_btn.click(
                    fn=lambda pid: load_plan_detail(pid),
                    inputs=[plan_id_input],
                    outputs=[
                        detail_container, no_plan_msg,
                        overview_md,  # 只需要一个overview
                        ws_result, plan_result,  # 控制面板结果
                        ws_status_md, plan_status_md,  # 状态显示
                        ws_start_btn, ws_stop_btn, plan_start_btn, plan_stop_btn,  # 按钮状态
                        auto_finetune_switch, auto_inference_switch,  # 自动化开关
                        auto_agent_switch,  # auto_tool_execution_switch已移除
                        automation_config_result,  # 自动化配置结果
                        schedule_time_list, schedule_operation_result, inference_schedule_display, inference_schedule_operation_result,  # 时间表管理
                        lookback_window, predict_window, batch_size,
                        tokenizer_epochs, predictor_epochs, learning_rate, params_status,
                        train_data_range_info, train_start_date, train_end_date, train_data_config_result,  # 训练数据范围
                        inference_lookback_window, inference_predict_window,  # 推理数据窗口
                        inference_temperature, inference_top_p, inference_sample_count, inference_data_offset, inference_params_status,  # 推理参数
                        llm_config_dropdown, prompt_template_dropdown, agent_prompt_textbox,  # Agent配置
                        tool_query_prediction, tool_prediction_history, tool_query_historical_kline,  # 数据查询工具
                        tool_get_utc_time, tool_run_inference, tool_get_account,  # 系统和账户工具
                        tool_get_pending_orders, tool_place_order, tool_cancel_order, tool_amend_order,  # 交易工具
                        # ReAct配置已移除
                        quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss,  # 交易限制配置
                        training_df, kline_chart, probability_indicators_md,  # K线图和概率指标
                        inference_df, inference_data_range_info, prediction_data_preview, agent_df,
                        agent_chatbot, agent_user_input, agent_status,  # agent_chatbot, agent_user_input, agent_status
                        account_status, order_table, task_executions_df,  # 账户信息、订单记录和任务记录
                        account_timer,  # 定时器
                        inference_record_id  # 自动填充训练记录ID
                    ]
                ).then(
                    fn=lambda: gr.Tabs(selected=2),  # 切换到详情Tab
                    outputs=[tabs]
                )

                # 保存参数按钮
                save_params_btn.click(
                    fn=save_params,
                    inputs=[plan_id_input, lookback_window, predict_window, batch_size,
                            tokenizer_epochs, predictor_epochs, learning_rate],
                    outputs=[params_status]
                )

                # 控制面板功能已移至 ui/plan_detail.py 中的专用Tab

                # 详情页刷新（注意：控制面板组件已移至ui/plan_detail.py，变量定义已在前面添加）
                def safe_int(value, default=0):
                    """安全转换为整数"""
                    try:
                        if value is None:
                            return default
                        if isinstance(value, str):
                            return int(float(value))
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                def safe_float(value, default=0.0):
                    """安全转换为浮点数"""
                    try:
                        if value is None:
                            return default
                        if isinstance(value, str):
                            return float(value)
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                def refresh_plan_detail_wrapper(pid):
                    """刷新计划详情的包装函数，使用原有的load_plan_detail逻辑"""
                    # 直接调用原有的load_plan_detail函数
                    result = load_plan_detail(pid)
                    # 返回除了detail_container和no_plan_msg之外的所有值，只取前66个
                    return result[2:68]

                detail_refresh_btn.click(
                    fn=refresh_plan_detail_wrapper,
                    inputs=[plan_id_input],
                    outputs=[
                        overview_md, ws_result, plan_result,  # 概览和结果
                        ws_status_md, plan_status_md,  # 状态显示
                        ws_start_btn, ws_stop_btn, plan_start_btn, plan_stop_btn,  # 按钮状态
                        auto_finetune_switch, auto_inference_switch,  # 自动化开关
                        auto_agent_switch,  # auto_tool_execution_switch已移除
                        automation_config_result,  # 自动化配置结果
                        schedule_time_list, schedule_operation_result, inference_schedule_display, inference_schedule_operation_result,  # 时间表管理
                        lookback_window, predict_window, batch_size,  # 模型参数
                        tokenizer_epochs, predictor_epochs, learning_rate, params_status,
                        train_data_range_info, train_start_date, train_end_date, train_data_config_result,
                        inference_lookback_window, inference_predict_window,  # 推理数据窗口
                        inference_temperature, inference_top_p, inference_sample_count, inference_data_offset, inference_params_status,
                        llm_config_dropdown, prompt_template_dropdown, agent_prompt_textbox,  # Agent配置
                        tool_query_prediction, tool_prediction_history, tool_query_historical_kline,  # 数据查询工具
                        tool_get_utc_time, tool_run_inference, tool_get_account,  # 系统和账户工具
                        tool_get_pending_orders, tool_place_order, tool_cancel_order, tool_amend_order,  # 交易工具
                        # ReAct配置已移除
                        quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss,  # 交易限制配置
                        training_df, kline_chart, probability_indicators_md,  # K线图和概率指标
                        inference_df, inference_data_range_info, prediction_data_preview, agent_df,
                        agent_chatbot, agent_user_input, agent_status,  # agent_chatbot, agent_user_input, agent_status
                        account_status, order_table, task_executions_df,  # 账户信息、订单记录和任务记录
                        account_timer  # 定时器
                    ]
                )

                # WebSocket和计划控制事件已移至 ui/plan_detail.py 中的专用控制面板Tab

                # 开始训练
                async def start_training_wrapper(pid, start_date, end_date):
                    """训练包装函数 - 异步生成器"""
                    if not pid:
                        yield "❌ 请先选择计划"
                        return

                    # 迭代异步生成器，逐个yield结果
                    async for message in detail_ui.start_training_async(int(pid), start_date, end_date):
                        yield message

                start_training_btn.click(
                    fn=start_training_wrapper,
                    inputs=[plan_id_input, train_start_date, train_end_date],
                    outputs=[training_status]
                )

                # 训练数据范围快捷按钮
                def set_train_range_wrapper(pid, days):
                    if not pid:
                        return "", "", ""
                    from database.db import get_db
                    from database.models import TradingPlan
                    with get_db() as db:
                        plan = db.query(TradingPlan).filter(TradingPlan.id == int(pid)).first()
                        if plan:
                            return detail_ui.set_training_date_range(plan.inst_id, plan.interval, days)
                        else:
                            return "⚠️ **计划不存在**", "", ""

                train_days_30_btn.click(
                    fn=lambda pid: set_train_range_wrapper(pid, 30),
                    inputs=[plan_id_input],
                    outputs=[train_data_range_info, train_start_date, train_end_date]
                )

                train_days_60_btn.click(
                    fn=lambda pid: set_train_range_wrapper(pid, 60),
                    inputs=[plan_id_input],
                    outputs=[train_data_range_info, train_start_date, train_end_date]
                )

                train_days_90_btn.click(
                    fn=lambda pid: set_train_range_wrapper(pid, 90),
                    inputs=[plan_id_input],
                    outputs=[train_data_range_info, train_start_date, train_end_date]
                )

                # 保存训练数据配置
                def save_train_data_config_wrapper(pid, start_date, end_date):
                    if not pid:
                        return "❌ 请先选择计划", ""
                    if not start_date or not end_date:
                        return "❌ 请输入开始和结束日期", ""
                    message, stats_info = detail_ui.save_training_data_config(int(pid), start_date, end_date)
                    # 同时更新 train_data_range_info
                    if stats_info:
                        return message, stats_info
                    else:
                        # 如果保存失败，返回当前统计信息
                        return message, detail_ui.get_training_data_stats(int(pid))

                save_train_data_config_btn.click(
                    fn=save_train_data_config_wrapper,
                    inputs=[plan_id_input, train_start_date, train_end_date],
                    outputs=[train_data_config_result, train_data_range_info]
                )

                # 保存推理参数
                def save_inference_params_wrapper(pid, lookback, predict, temp, top_p, sample_count, data_offset):
                    if not pid:
                        return "❌ 请先选择计划", "请先选择计划"

                    # 保存推理参数
                    status_msg = detail_ui.save_inference_params(
                        int(pid),
                        int(lookback),
                        int(predict),
                        temp,
                        top_p,
                        sample_count,
                        int(data_offset)
                    )

                    # 获取推理数据点时间戳范围
                    data_range_info = detail_ui.get_inference_data_timestamp_range(
                        int(pid),
                        int(lookback),
                        int(data_offset)
                    )

                    return status_msg, data_range_info

                save_inference_params_btn.click(
                    fn=save_inference_params_wrapper,
                    inputs=[
                        plan_id_input,
                        inference_lookback_window,
                        inference_predict_window,
                        inference_temperature,
                        inference_top_p,
                        inference_sample_count,
                        inference_data_offset
                    ],
                    outputs=[inference_params_status, inference_data_range_info]
                )

                # 取消训练
                def cancel_training_wrapper(training_id):
                    if not training_id:
                        return "❌ 请输入训练记录ID"
                    from services.training_service import TrainingService
                    result = TrainingService.cancel_training(int(training_id))
                    return result['message']

                cancel_training_btn.click(
                    fn=cancel_training_wrapper,
                    inputs=[training_record_id],
                    outputs=[training_operation_result]
                ).then(
                    fn=lambda pid: detail_ui.load_training_records(safe_plan_id(pid)) if pid else gr.DataFrame(),
                    inputs=[plan_id_input],
                    outputs=[training_df]
                )

                # 删除训练记录
                def delete_training_wrapper(training_id):
                    if not training_id:
                        return "❌ 请输入训练记录ID"
                    from services.training_service import TrainingService
                    result = TrainingService.delete_training_record(int(training_id))
                    return result['message']

                delete_training_btn.click(
                    fn=delete_training_wrapper,
                    inputs=[training_record_id],
                    outputs=[training_operation_result]
                ).then(
                    fn=lambda pid: detail_ui.load_training_records(safe_plan_id(pid)) if pid else gr.DataFrame(),
                    inputs=[plan_id_input],
                    outputs=[training_df]
                )

                # K线图更新
                days_slider.change(
                    fn=lambda days, show_pred, pid: detail_ui.generate_kline_chart(
                        int(pid), show_pred, None, days
                    ) if pid else gr.Plot(),
                    inputs=[days_slider, show_pred_toggle, plan_id_input],
                    outputs=[kline_chart]
                )

                show_pred_toggle.change(
                    fn=lambda show_pred, days, pid: detail_ui.generate_kline_chart(
                        int(pid), show_pred, None, days
                    ) if pid else gr.Plot(),
                    inputs=[show_pred_toggle, days_slider, plan_id_input],
                    outputs=[kline_chart]
                )

                # Agent配置事件
                def save_agent_config_wrapper(pid, llm_id, prompt, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10):
                    if not pid:
                        return "❌ 请先选择计划"

                    # 验证LLM配置ID是否有效
                    if llm_id:
                        try:
                            # 动态获取LLM配置列表
                            llm_configs = detail_ui.get_llm_configs()
                            valid_llm_ids = [config_id for _, config_id in llm_configs]
                            if llm_id not in valid_llm_ids:
                                return f"❌ 选择的LLM配置ID {llm_id} 无效，请重新选择LLM配置"
                        except Exception as e:
                            logger.error(f"验证LLM配置失败: {e}")
                            return "❌ 获取LLM配置列表失败，请重试"

                    tools_config = {
                        'query_prediction_data': t1,
                        'get_prediction_history': t2,
                        'query_historical_kline_data': t3,
                        'get_current_utc_time': t4,
                        'run_latest_model_inference': t5,
                        'get_account_balance': t6,
                        'get_pending_orders': t7,
                        'place_order': t8,
                        'cancel_order': t9,
                        'amend_order': t10
                    }
                    # 保存Agent配置
                    agent_result = detail_ui.save_agent_config(int(pid), llm_id, prompt, tools_config)

                    # ReAct 配置已移除
                    return f"{agent_result}"

                save_agent_config_btn.click(
                    fn=save_agent_config_wrapper,
                    inputs=[
                        plan_id_input, llm_config_dropdown, agent_prompt_textbox,
                        tool_query_prediction, tool_prediction_history, tool_query_historical_kline,
                        tool_get_utc_time, tool_run_inference, tool_get_account,
                        tool_get_pending_orders, tool_place_order, tool_cancel_order, tool_amend_order
                    ],
                    outputs=[agent_config_status]
                )

                # 加载提示词模板
                def load_template_wrapper(template_id):
                    if not template_id:
                        return ""
                    return detail_ui.load_prompt_template(int(template_id))

                load_template_btn.click(
                    fn=load_template_wrapper,
                    inputs=[prompt_template_dropdown],
                    outputs=[agent_prompt_textbox]
                )

                # 交易限制配置保存
                def save_trading_limits_wrapper(pid, usdt_amount, usdt_percentage, avg_orders, stop_loss):
                    if not pid:
                        return "❌ 请先选择计划"
                    return detail_ui.save_trading_limits_config(
                        int(pid), usdt_amount, usdt_percentage, int(avg_orders), stop_loss
                    )

                save_trading_limits_btn.click(
                    fn=save_trading_limits_wrapper,
                    inputs=[
                        plan_id_input, quick_usdt_amount, quick_usdt_percentage,
                        quick_avg_orders, quick_stop_loss
                    ],
                    outputs=[trading_limits_status]
                )

                # 重置交易限制到默认值
                def reset_trading_limits_wrapper(pid):
                    if not pid:
                        return "❌ 请先选择计划", 1000.0, 30.0, 10.0, 20.0

                    # 保存默认配置
                    result = detail_ui.save_trading_limits_config(
                        int(pid), 1000.0, 30.0, 10, 20.0
                    )
                    return result, 1000.0, 30.0, 10.0, 20.0

                reset_trading_limits_btn.click(
                    fn=reset_trading_limits_wrapper,
                    inputs=[plan_id_input],
                    outputs=[trading_limits_status, quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss]
                )

                # 加载交易限制配置当计划改变时
                def load_trading_limits_wrapper(pid):
                    if not pid:
                        return 1000.0, 30.0, 10.0, 20.0
                    limits = detail_ui.get_trading_limits_config(int(pid))
                    return (
                        limits['available_usdt_amount'],
                        limits['available_usdt_percentage'],
                        limits['avg_order_count'],
                        limits['stop_loss_percentage']
                    )

                # 当计划ID改变时，加载交易限制配置
                def update_trading_limits_on_plan_change(pid):
                    return load_trading_limits_wrapper(pid)

                # 这个会在计划加载时调用，我们稍后添加到计划选择事件中

                # 获取最新训练记录ID
                def get_latest_training_id(pid):
                    from database.db import get_db
                    from database.models import TrainingRecord

                    if not pid:
                        return None

                    with get_db() as db:
                        latest_training = db.query(TrainingRecord).filter(
                            TrainingRecord.plan_id == int(pid),
                            TrainingRecord.status == 'completed'
                        ).order_by(TrainingRecord.created_at.desc()).first()

                        return latest_training.id if latest_training else None

                # 执行推理（预测交易数据）
                async def execute_inference_wrapper(training_id, pid):
                    from database.db import get_db
                    from database.models import TrainingRecord

                    # 如果没有提供训练ID，尝试获取最新的已完成训练记录
                    if not training_id:
                        training_id = get_latest_training_id(pid)
                        if not training_id:
                            return "❌ 未找到可用的训练记录，请先完成模型训练", "", gr.Plot(), ""

                    result = await detail_ui.execute_inference_async(int(training_id))
                    # 更新预测数据预览和K线图
                    prediction_text = detail_ui.get_prediction_text(int(training_id))
                    # 获取计划ID和概率指标
                    with get_db() as db:
                        record = db.query(TrainingRecord).filter(TrainingRecord.id == int(training_id)).first()
                        if record:
                            kline_chart = detail_ui.generate_kline_chart(record.plan_id, show_predictions=True, training_id=int(training_id))
                            probability_indicators = detail_ui.get_probability_indicators(record.plan_id)
                        else:
                            kline_chart = detail_ui._empty_chart("训练记录不存在")
                            probability_indicators = ""
                    return result, prediction_text, kline_chart, probability_indicators

                execute_inference_btn.click(
                    fn=execute_inference_wrapper,
                    inputs=[inference_record_id, plan_id_input],
                    outputs=[inference_operation_result, prediction_data_preview, kline_chart, probability_indicators_md]
                ).then(
                    fn=lambda training_id, pid: detail_ui.load_inference_records(int(training_id)) if training_id else detail_ui.load_inference_records(int(pid)) if pid else gr.DataFrame(),
                    inputs=[inference_record_id, plan_id_input],
                    outputs=[inference_df]
                )

                # Mock预测数据
                async def mock_prediction_wrapper(training_id, pid):
                    from database.db import get_db
                    from database.models import TrainingRecord

                    # 如果没有提供训练ID，尝试获取最新的已完成训练记录
                    if not training_id:
                        training_id = get_latest_training_id(pid)
                        if not training_id:
                            return "❌ 未找到可用的训练记录，请先完成模型训练", "", gr.Plot(), ""

                    result = await detail_ui.mock_predictions_async(int(training_id))
                    # 更新预测数据预览和K线图
                    prediction_text = detail_ui.get_prediction_text(int(training_id))
                    # 获取计划ID和概率指标
                    with get_db() as db:
                        record = db.query(TrainingRecord).filter(TrainingRecord.id == int(training_id)).first()
                        if record:
                            kline_chart = detail_ui.generate_kline_chart(record.plan_id, show_predictions=True, training_id=int(training_id))
                            probability_indicators = detail_ui.get_probability_indicators(record.plan_id)
                        else:
                            kline_chart = detail_ui._empty_chart("训练记录不存在")
                            probability_indicators = ""
                    return result, prediction_text, kline_chart, probability_indicators

                mock_prediction_btn.click(
                    fn=mock_prediction_wrapper,
                    inputs=[inference_record_id, plan_id_input],
                    outputs=[inference_operation_result, prediction_data_preview, kline_chart, probability_indicators_md]
                )

                # 聊天功能已移动到计划详情页面

  
  
                # 聊天功能已移动到计划详情页面

              
                # 聊天功能已移动到计划详情页面

                # 刷新决策记录和聊天上下文
                def refresh_agent_wrapper(pid):
                    # 使用安全的plan_id处理函数
                    is_valid, plan_id, error_msg = validate_plan_exists(pid)

                    if not is_valid:
                        return gr.DataFrame(), [{"role": "assistant", "content": f"❌ {error_msg}"}]

                    try:
                        # 刷新决策列表
                        agent_df_updated = detail_ui.load_agent_decisions(plan_id)

                        # 刷新最新的聊天上下文
                        latest_messages = detail_ui.get_latest_conversation_messages(plan_id)

                        return agent_df_updated, latest_messages
                    except Exception as e:
                        logger.error(f"刷新Agent记录失败: {e}")
                        return gr.DataFrame(), [{"role": "assistant", "content": f"❌ 刷新失败: {str(e)}"}]

                # 清除推理记录
                def clear_agent_records_wrapper(pid):
                    # 使用安全的plan_id处理函数
                    is_valid, plan_id, error_msg = validate_plan_exists(pid)

                    if not is_valid:
                        return gr.DataFrame(), [{"role": "assistant", "content": f"❌ {error_msg}"}]

                    try:
                        result = detail_ui.clear_agent_records(plan_id)
                        # 刷新推理记录列表
                        agent_df_updated = detail_ui.load_agent_decisions(plan_id)
                        # 将结果显示在聊天中
                        status_message = f"✅ {result}"
                        return agent_df_updated, [{"role": "assistant", "content": status_message}]
                    except Exception as e:
                        logger.error(f"清除Agent记录失败: {e}")
                        return gr.DataFrame(), [{"role": "assistant", "content": f"❌ 清除失败: {str(e)}"}]

                refresh_agent_btn.click(
                    fn=refresh_agent_wrapper,
                    inputs=[plan_id_input],
                    outputs=[agent_df, agent_chatbot]
                )

                clear_agent_records_btn.click(
                    fn=clear_agent_records_wrapper,
                    inputs=[plan_id_input],
                    outputs=[agent_df, agent_chatbot]
                )

                # AI Agent 事件绑定现在通过 chat_ui.bind_events() 处理
                chat_ui.bind_events(chat_components, plan_id_input)

                # 刷新账户信息
                def refresh_account_wrapper(pid):
                    # 使用安全的plan_id处理函数
                    is_valid, plan_id, error_msg = validate_plan_exists(pid)

                    if not is_valid:
                        return f"### 💰 账户信息\n\n❌ {error_msg}"

                    try:
                        return detail_ui.get_account_info(plan_id)
                    except Exception as e:
                        logger.error(f"刷新账户信息失败: {e}")
                        return f"### 💰 账户信息\n\n❌ 刷新失败: {str(e)}"

                account_refresh_btn.click(
                    fn=refresh_account_wrapper,
                    inputs=[plan_id_input],
                    outputs=[account_status]
                )

                # 刷新订单记录
                def refresh_orders_wrapper(pid):
                    # 使用安全的plan_id处理函数
                    is_valid, plan_id, error_msg = validate_plan_exists(pid)

                    if not is_valid:
                        return gr.DataFrame()

                    try:
                        return detail_ui.get_orders_info(plan_id)
                    except Exception as e:
                        logger.error(f"刷新订单记录失败: {e}")
                        return gr.DataFrame()

                order_refresh_btn.click(
                    fn=refresh_orders_wrapper,
                    inputs=[plan_id_input],
                    outputs=[order_table]
                )

                # 任务执行记录刷新
                def refresh_tasks_wrapper(pid):
                    # 使用安全的plan_id处理函数
                    is_valid, plan_id, error_msg = validate_plan_exists(pid)

                    if not is_valid:
                        return pd.DataFrame()

                    try:
                        return detail_ui.load_task_executions(plan_id)
                    except Exception as e:
                        logger.error(f"刷新任务执行记录失败: {e}")
                        return pd.DataFrame()

                task_refresh_btn.click(
                    fn=refresh_tasks_wrapper,
                    inputs=[plan_id_input],
                    outputs=[task_executions_df]
                )

                # 定时器事件：自动刷新账户信息
                account_timer.tick(
                    fn=refresh_account_wrapper,
                    inputs=[plan_id_input],
                    outputs=[account_status]
                )

                # Agent决策记录点击事件 - 在chatbot中显示详情
                def show_agent_decision_detail(evt: gr.SelectData, plan_id):
                    """显示Agent决策详情到chatbot"""
                    try:
                        if evt is None or not hasattr(evt, 'index') or not evt.index:
                            return [{"role": "assistant", "content": "请点击决策记录查看详情"}]

                        if not plan_id:
                            return [{"role": "assistant", "content": "请先选择计划"}]

                        # 获取点击的行索引
                        row_index = evt.index[0]

                        # 从数据库重新获取Agent决策数据
                        try:
                            agent_decisions = detail_ui.load_agent_decisions(int(plan_id))
                            if agent_decisions.empty or row_index >= len(agent_decisions):
                                return [{"role": "assistant", "content": "决策记录不存在或已被更新"}]

                            # 获取点击行的ID
                            clicked_row = agent_decisions.iloc[row_index]
                            if 'ID' in clicked_row:
                                decision_id = int(clicked_row['ID'])
                            else:
                                # 假设第一列是ID
                                decision_id = int(clicked_row.iloc[0])

                        except Exception as load_error:
                            logger.error(f"加载决策数据失败: {load_error}")
                            return [{"role": "assistant", "content": "无法加载决策数据"}]

                        # 获取决策详情
                        detail_content = detail_ui.get_agent_decision_detail(decision_id)

                        # 格式化为chatbot消息
                        chat_messages = [
                            {"role": "user", "content": f"查看决策记录 ID: {decision_id} 的详情"},
                            {"role": "assistant", "content": detail_content}
                        ]

                        return chat_messages

                    except Exception as e:
                        logger.error(f"获取Agent决策详情失败: {e}")
                        import traceback
                        traceback.print_exc()
                        return [{"role": "assistant", "content": f"获取决策详情失败: {str(e)}"}]

                # 绑定Agent决策列表点击事件
                agent_df.select(
                    fn=show_agent_decision_detail,
                    inputs=[plan_id_input],
                    outputs=[agent_chatbot]
                )

                # 返回列表 - 停止定时器
                def back_to_list_wrapper():
                    return (
                        gr.Tabs(selected=1),
                        gr.Timer(active=False)  # 停止账户定时器
                    )

                back_to_list_btn.click(
                    fn=back_to_list_wrapper,
                    outputs=[tabs, account_timer]
                )

            # 列表页面的查看详情按钮事件已在上方（lines 695-723）绑定，此处不再重复绑定

            with gr.Tab("⚙️ 配置中心", id=3):
                create_config_center_ui()

        gr.Markdown(
            """
            ---
            💡 **使用提示**:
            - 新增计划前，请先在"配置中心"确认基础配置
            - 模拟盘交易不会影响实际资金，建议先使用模拟盘测试
            - WebSocket 会在后台自动运行，持续同步K线数据
            - 系统日志保存在 `logs/` 目录
            """
        )

        # 添加悬浮时间指示器的JavaScript
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function() {
                // 创建时间指示器元素
                const timeIndicator = document.createElement('div');
                timeIndicator.className = 'floating-time-indicator';
                timeIndicator.innerHTML = `
                    <div class="time-label">系统时间</div>
                    <div class="current-time" id="current-time">--:--:--</div>
                    <div class="timezone">UTC+8 (北京时间)</div>
                `;

                // 创建悬浮刷新按钮
                const floatingRefreshBtn = document.createElement('button');
                floatingRefreshBtn.className = 'floating-refresh-btn';
                floatingRefreshBtn.innerHTML = '🔄 刷新数据';
                floatingRefreshBtn.title = '刷新计划详情数据';

                // 添加点击事件
                floatingRefreshBtn.addEventListener('click', function() {
                    // 触发Gradio的刷新按钮点击事件
                    const refreshButtons = document.querySelectorAll('button');
                    for (let btn of refreshButtons) {
                        if (btn.textContent.includes('刷新数据') && !btn.textContent.includes('刷新对话记录')) {
                            btn.click();
                            break;
                        }
                    }
                });

                // 将元素添加到页面
                document.body.appendChild(timeIndicator);
                document.body.appendChild(floatingRefreshBtn);

                // 更新时间的函数
                function updateTime() {
                    const now = new Date();

                    // 获取UTC+8时间
                    const utc8Time = new Date(now.getTime() + (8 * 60 * 60 * 1000) + (now.getTimezoneOffset() * 60 * 1000));

                    // 格式化时间
                    const hours = utc8Time.getHours().toString().padStart(2, '0');
                    const minutes = utc8Time.getMinutes().toString().padStart(2, '0');
                    const seconds = utc8Time.getSeconds().toString().padStart(2, '0');

                    // 更新显示
                    const timeElement = document.getElementById('current-time');
                    if (timeElement) {
                        timeElement.textContent = `${hours}:${minutes}:${seconds}`;
                    }
                }

                // 立即更新一次时间
                updateTime();

                // 每秒更新时间
                setInterval(updateTime, 1000);

                // 监听页面切换，确保时间显示准确
                document.addEventListener('visibilitychange', function() {
                    if (!document.hidden) {
                        updateTime();
                    }
                });

                // 添加一些样式效果
                timeIndicator.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.05)';
                    this.style.transition = 'transform 0.2s ease';
                });

                timeIndicator.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                });

                console.log('悬浮时间指示器已启动 (UTC+8)');
            }
            """
        )

    return app


def main():
    """主函数"""
    try:
        # 创建应用
        app = create_app()

        # 恢复WebSocket连接
        logger.info("恢复WebSocket连接...")
        from services.connection_recovery_service import connection_recovery_service
        recovery_success = connection_recovery_service.recover_all_connections()

        if recovery_success:
            logger.info("✅ WebSocket连接恢复成功")
        else:
            logger.warning("⚠️ WebSocket连接恢复失败，请手动检查")

        # 启动应用
        logger.info(f"启动 Gradio 服务: {config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
        logger.info("使用 Ctrl+C 或发送 SIGTERM 信号可以优雅关闭程序")

        # Gradio 启动时需要访问 localhost 进行自检，临时禁用代理
        import os
        original_http_proxy = os.environ.get('http_proxy')
        original_https_proxy = os.environ.get('https_proxy')
        original_HTTP_PROXY = os.environ.get('HTTP_PROXY')
        original_HTTPS_PROXY = os.environ.get('HTTPS_PROXY')

        try:
            # 设置 NO_PROXY 确保 localhost 不走代理
            os.environ['NO_PROXY'] = 'localhost,127.0.0.1,0.0.0.0'
            os.environ['no_proxy'] = 'localhost,127.0.0.1,0.0.0.0'
            # logger.info("已设置 NO_PROXY，确保 Gradio 自检不走代理")

            app.launch(
                server_name=config.GRADIO_SERVER_NAME,
                server_port=config.GRADIO_SERVER_PORT,
                share=False,
                show_api=False
            )
        finally:
            # 恢复原始代理设置（虽然 app.launch 会阻塞，但为了完整性还是写上）
            if original_http_proxy is not None:
                os.environ['http_proxy'] = original_http_proxy
            elif 'http_proxy' in os.environ:
                del os.environ['http_proxy']

            if original_https_proxy is not None:
                os.environ['https_proxy'] = original_https_proxy
            elif 'https_proxy' in os.environ:
                del os.environ['https_proxy']

            if original_HTTP_PROXY is not None:
                os.environ['HTTP_PROXY'] = original_HTTP_PROXY
            elif 'HTTP_PROXY' in os.environ:
                del os.environ['HTTP_PROXY']

            if original_HTTPS_PROXY is not None:
                os.environ['HTTPS_PROXY'] = original_HTTPS_PROXY
            elif 'HTTPS_PROXY' in os.environ:
                del os.environ['HTTPS_PROXY']

    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭应用...")
    except Exception as e:
        logger.error(f"应用运行错误: {e}")
        raise


if __name__ == "__main__":
    main()
