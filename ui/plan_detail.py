"""
计划详情页UI
包含：上部概览、左侧训练列表、中间K线图、右侧Agent记录、下方账户订单
"""
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from services.plan_service import PlanService
from services.training_service import TrainingService
from services.inference_service import InferenceService
from services.conversation_service import ConversationService
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, AgentDecision, KlineData, AgentConversation, AgentMessage
from sqlalchemy import and_, desc, func
from utils.logger import setup_logger
from ui.constants import DataFrameHeaders, DataTypes, create_empty_dataframe
from ui.ui_utils import UIHelper
from ui.custom_chatbot import create_custom_chatbot, process_streaming_messages, format_conversation_history
from database.models import now_beijing
from utils.timezone_helper import (format_datetime_full_beijing, format_datetime_short_beijing,
                                   format_datetime_beijing, format_time_range_utc8)

logger = setup_logger(__name__, "plan_detail_ui.log")


def safe_dataframe_from_data(data: List[Dict]) -> pd.DataFrame:
    """
    安全创建DataFrame，确保所有数据类型都适合Gradio显示

    Args:
        data: 字典列表数据

    Returns:
        pd.DataFrame: 安全的DataFrame
    """
    if not data:
        return pd.DataFrame()

    safe_data = []
    for row in data:
        safe_row = {}
        for key, value in row.items():
            if value is None or pd.isna(value):
                safe_row[key] = 'N/A'
            elif isinstance(value, (int, float)):
                # 保持数字类型，但确保不是NaN
                if pd.isna(value):
                    safe_row[key] = 0
                else:
                    safe_row[key] = value
            else:
                # 转换为字符串
                safe_row[key] = str(value)
        safe_data.append(safe_row)

    return pd.DataFrame(safe_data)


class PlanDetailUI:
    """计划详情页UI"""

    def __init__(self):
        self.current_plan_id = None

    def _safe_db_update(self, update_func, plan_id: int, max_retries: int = 3):
        """
        安全的数据库更新操作，带重试机制

        Args:
            update_func: 更新函数，接收db会话作为参数
            plan_id: 计划ID
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                from database.db import SessionLocal
                db = SessionLocal()
                try:
                    update_func(db)
                    db.commit()
                    return True
                finally:
                    db.close()
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"数据库更新失败，已重试{max_retries}次: plan_id={plan_id}, error={e}")
                    return False
                else:
                    logger.warning(f"数据库更新失败，正在重试({attempt + 1}/{max_retries}): plan_id={plan_id}, error={e}")
                    import time
                    time.sleep(0.1 * (attempt + 1))
        return False

    def load_plan_data(self, plan_id: int) -> Dict:
        """加载计划数据"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'error': '计划不存在'}

                # 获取最新训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                # 获取最新Agent决策
                latest_agent = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).order_by(desc(AgentDecision.decision_time)).first()

                return {
                    'plan': plan,
                    'latest_training': latest_training,
                    'latest_agent': latest_agent
                }
        except Exception as e:
            logger.error(f"加载计划数据失败: {e}")
            return {'error': str(e)}

    def render_plan_overview(self, plan_id: int) -> tuple:
        """
        渲染计划概览（上部）

        Returns:
            tuple: (overview_text, ws_status_text, ws_start_visible, ws_stop_visible, plan_status_text, plan_start_visible, plan_stop_visible)
        """
        data = self.load_plan_data(plan_id)
        if 'error' in data:
            return (
                f"❌ 错误: {data['error']}",  # overview_text
                "**WebSocket状态**: ⚪ 未连接",  # ws_status_text
                True,  # ws_start_visible
                False, # ws_stop_visible
                "**计划状态**: ⚪ 已创建",  # plan_status_text
                True,  # plan_start_visible
                False  # plan_stop_visible
            )

        plan = data['plan']
        latest_training = data['latest_training']
        latest_agent = data['latest_agent']

        training_version = latest_training.version if latest_training else "未训练"
        agent_time = format_datetime_full_beijing(latest_agent.decision_time) if latest_agent else "无记录"

        # 自动微调时间表
        schedule = plan.auto_finetune_schedule or []
        schedule_str = ', '.join(schedule) if schedule else '未配置'

        # 自动预测间隔时间
        inference_interval_hours = getattr(plan, 'auto_inference_interval_hours', 4)
        if inference_interval_hours and inference_interval_hours > 0:
            inference_schedule_str = f'{inference_interval_hours}小时间隔'
        else:
            inference_schedule_str = '未配置'

        # 计划状态
        plan_status_emoji = UIHelper.get_status_emoji(plan.status, detailed=True)

        overview = f"""
# 📊 {plan.plan_name}

---

**交易对**: `{plan.inst_id}` | **时间颗粒度**: `{plan.interval}` | **环境**: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}

**计划状态**: {plan_status_emoji}

**最新模型版本**: `{training_version}` | **AI Agent最后运行**: {agent_time}

**自动微调时间**: {schedule_str}

**自动预测时间**: {inference_schedule_str}

**创建时间**: {format_datetime_full_beijing(plan.created_at)}

---
"""

        # 获取控制面板状态
        control_status = self.get_control_panel_status(plan_id)

        return (
            overview,  # overview_text
            control_status[0],  # ws_status_text
            control_status[1],  # ws_start_visible
            control_status[2],  # ws_stop_visible
            control_status[3],  # plan_status_text
            control_status[4],  # plan_start_visible
            control_status[5]   # plan_stop_visible
        )

    def get_control_panel_status(self, plan_id: int) -> tuple:
        """
        获取控制面板状态

        Returns:
            tuple: (ws_status_text, ws_start_visible, ws_stop_visible, plan_status_text, plan_start_visible, plan_stop_visible)
        """
        data = self.load_plan_data(plan_id)
        if 'error' in data:
            return (
                "**WebSocket状态**: ⚪ 未连接",  # ws_status_text
                True,  # ws_start_visible
                False, # ws_stop_visible
                "**计划状态**: ⚪ 已创建",  # plan_status_text
                True,  # plan_start_visible
                False  # plan_stop_visible
            )

        plan = data['plan']

        # 多重检查WebSocket连接状态
        try:
            # 1. 首先检查WebSocket订阅表的状态（最可靠的数据源）
            from database.db import get_db
            from database.models import WebSocketSubscription

            ws_connected = False

            with get_db() as db:
                subscription = db.query(WebSocketSubscription).filter(
                    WebSocketSubscription.inst_id == plan.inst_id,
                    WebSocketSubscription.interval == plan.interval,
                    WebSocketSubscription.is_demo == plan.is_demo
                ).first()

                if subscription:
                    # 如果订阅表中有记录，检查是否真的在运行
                    # 主要指标：状态为running且有数据接收
                    if (subscription.status == 'running' and
                        subscription.total_received > 0 and
                        subscription.last_data_time):
                        ws_connected = True
                        logger.debug(f"WebSocket运行中 (来自订阅表): plan_id={plan_id}, received={subscription.total_received}")

            # 2. 如果订阅表显示未连接，尝试连接管理器
            if not ws_connected:
                try:
                    from services.ws_connection_manager import ws_connection_manager
                    ws_status = ws_connection_manager.get_connection_status(
                        inst_id=plan.inst_id,
                        interval=plan.interval,
                        is_demo=plan.is_demo
                    )

                    if ws_status['exists'] and ws_status['thread_alive']:
                        ws_connected = ws_status['connected'] and ws_status['running']
                        logger.debug(f"WebSocket状态 (来自连接管理器): plan_id={plan_id}, connected={ws_connected}")

                except Exception as conn_error:
                    logger.debug(f"连接管理器状态获取失败: {conn_error}")

            # 3. 最后回退到数据库中的状态
            if not ws_connected and plan.ws_connected:
                ws_connected = plan.ws_connected
                logger.debug(f"WebSocket状态 (来自数据库): plan_id={plan_id}, connected={ws_connected}")

            # 4. 异步更新数据库状态（如果发现不一致）
            if ws_connected != plan.ws_connected:
                def update_plan_ws_status(db):
                    current_plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                    if current_plan and current_plan.ws_connected != ws_connected:
                        db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                            'ws_connected': ws_connected,
                            'last_sync_time': now_beijing()
                        })
                        logger.info(f"计划WebSocket状态已更新: plan_id={plan_id}, ws_connected={ws_connected}")

                self._safe_db_update(update_plan_ws_status, plan_id)

        except Exception as e:
            logger.error(f"获取WebSocket状态失败: {e}")
            ws_connected = plan.ws_connected

        ws_status_text = "🟢 已连接" if ws_connected else "⚪ 未连接"
        ws_status_display = f"**WebSocket状态**: {ws_status_text}"

        # 计划状态
        plan_status_emoji = UIHelper.get_status_emoji(plan.status, detailed=True)
        plan_status_display = f"**计划状态**: {plan_status_emoji}"

        # WebSocket状态控制
        ws_start_visible = not ws_connected
        ws_stop_visible = ws_connected

        # 计划状态控制
        plan_start_visible = plan.status != 'running'
        plan_stop_visible = plan.status == 'running'

        return (
            ws_status_display,  # ws_status_text
            ws_start_visible,   # ws_start_visible
            ws_stop_visible,    # ws_stop_visible
            plan_status_display,  # plan_status_text
            plan_start_visible,  # plan_start_visible
            plan_stop_visible   # plan_stop_visible
        )

    def load_training_records(self, plan_id: int) -> pd.DataFrame:
        """加载训练记录列表（左侧）"""
        try:
            records = TrainingService.list_training_records(plan_id)
            if not records:
                return pd.DataFrame()

            df_data = []
            for record in records:
                status_emoji = {
                    'waiting': '⏳',
                    'training': '🔄',
                    'completed': '✅',
                    'failed': '❌'
                }.get(record['status'], '❓')

                df_data.append({
                    'ID': record['id'],
                    '版本': record['version'],
                    '状态': f"{status_emoji} {record['status']}",
                    '启用': '✓' if record['is_active'] else '✗',
                    '数据量': record['data_count'] or 0,
                    '训练时长(秒)': record['train_duration'] or 0,  # 改为纯数字
                    '创建时间': format_datetime_short_beijing(record['created_at'])
                })

            return safe_dataframe_from_data(df_data)

        except Exception as e:
            logger.error(f"加载训练记录失败: {e}")
            return pd.DataFrame()

    def get_current_training_status(self, plan_id: int) -> str:
        """
        获取当前计划的训练状态和进度

        Args:
            plan_id: 计划ID

        Returns:
            格式化的训练状态字符串
        """
        try:
            from services.training_service import TrainingService

            with get_db() as db:
                # 查找当前计划正在训练的记录
                training_record = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status.in_(['waiting', 'training'])
                    )
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not training_record:
                    return None

                # 获取训练进度
                progress_info = TrainingService.get_training_progress(training_record.id)

                status_emoji = {
                    'waiting': '⏳',
                    'training': '🔄'
                }.get(training_record.status, '❓')

                if progress_info:
                    progress_percent = int(progress_info['progress'] * 100)
                    stage = progress_info['stage']
                    message = progress_info['message']
                    progress_bar = '█' * (progress_percent // 5) + '░' * (20 - progress_percent // 5)

                    return f"""
**{status_emoji} 当前训练状态**

- **记录ID**: {training_record.id}
- **版本**: {training_record.version}
- **进度**: {progress_percent}%
- **阶段**: {stage}
- **状态**: {message}

`{progress_bar}` ({progress_percent}%)
                    """
                else:
                    return f"""
**{status_emoji} 当前训练状态**

- **记录ID**: {training_record.id}
- **版本**: {training_record.version}
- **状态**: {training_record.status}
- **等待进度更新...**
                    """

        except Exception as e:
            logger.error(f"获取训练状态失败: {e}")
            return f"❌ 获取训练状态失败: {str(e)}"

    def get_training_options(self, plan_id: int) -> List[tuple]:
        """
        获取训练 ID 选项列表（用于下拉选择器）

        Returns:
            List[tuple]: [(显示文本, training_id), ...]
        """
        try:
            with get_db() as db:
                # 获取所有已完成且有预测数据的训练记录
                training_records = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed'
                    )
                ).order_by(desc(TrainingRecord.created_at)).all()

                options = [("全部", None)]  # 第一项是"全部"

                for record in training_records:
                    # 检查是否有预测数据
                    pred_count = db.query(func.count(PredictionData.id)).filter(
                        PredictionData.training_record_id == record.id
                    ).scalar()

                    if pred_count > 0:
                        # 获取推理时间（最早的预测数据创建时间）
                        first_pred = db.query(PredictionData).filter(
                            PredictionData.training_record_id == record.id
                        ).order_by(PredictionData.created_at.asc()).first()

                        inference_time_str = format_datetime_short_beijing(first_pred.created_at) if first_pred else ''

                        # 格式：v1 (推理: 12-20 10:30)
                        display_text = f"{record.version} (推理: {inference_time_str})"
                        options.append((display_text, record.id))

                return options

        except Exception as e:
            logger.error(f"获取训练选项失败: {e}")
            return [("全部", None)]

    def generate_kline_chart(
        self,
        plan_id: int,
        show_predictions: bool = True,
        training_id: Optional[int] = None,
        last_days: int = 30
    ) -> go.Figure:
        """
        生成K线图（中间，含预测数据）

        Args:
            plan_id: 计划ID
            show_predictions: 是否显示预测
            training_id: 训练记录ID，None表示显示全部
            last_days: 显示最近多少天
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return self._empty_chart("计划不存在")

                # 查询历史K线数据
                end_time = datetime.now()
                start_time = end_time - pd.Timedelta(days=last_days)

                klines = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= start_time,
                        KlineData.timestamp <= end_time
                    )
                ).order_by(KlineData.timestamp).all()

                if not klines:
                    return self._empty_chart("无历史数据")

                # 创建K线图
                fig = go.Figure()

                # K线数据时间戳已经是北京时间，直接使用
                timestamps_beijing = [k.timestamp for k in klines]

                # 添加真实K线
                fig.add_trace(go.Candlestick(
                    x=timestamps_beijing,
                    open=[k.open for k in klines],
                    high=[k.high for k in klines],
                    low=[k.low for k in klines],
                    close=[k.close for k in klines],
                    name='实际K线',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ))

                # 添加预测数据
                if show_predictions:
                    # 如果 training_id 为 None，显示所有历史预测
                    if training_id is None:
                        # 获取所有已完成的训练记录
                        all_training_records = db.query(TrainingRecord).filter(
                            and_(
                                TrainingRecord.plan_id == plan_id,
                                TrainingRecord.status == 'completed'
                            )
                        ).order_by(TrainingRecord.created_at.asc()).all()  # 从旧到新

                        # 为每个训练记录分配颜色（相同训练版本的所有批次使用相同颜色）
                        colors = ['#ff9800', '#2196f3', '#4caf50', '#9c27b0', '#f44336', '#00bcd4']

                        for record_idx, record in enumerate(all_training_records):
                            # 为当前训练版本分配颜色
                            record_color = colors[record_idx % len(colors)]

                            # 获取该训练记录的所有推理批次
                            batches = db.query(
                                PredictionData.inference_batch_id,
                                func.min(PredictionData.created_at).label('inference_time')
                            ).filter(
                                PredictionData.training_record_id == record.id
                            ).group_by(
                                PredictionData.inference_batch_id
                            ).order_by(
                                func.min(PredictionData.created_at).asc()
                            ).all()

                            for batch_idx, batch in enumerate(batches):
                                predictions = db.query(PredictionData).filter(
                                    and_(
                                        PredictionData.training_record_id == record.id,
                                        PredictionData.inference_batch_id == batch.inference_batch_id
                                    )
                                ).order_by(PredictionData.timestamp).all()

                                if predictions:
                                    # 只在第一个批次时显示图例
                                    show_in_legend = (batch_idx == 0)
                                    self._add_prediction_trace(
                                        fig,
                                        predictions,
                                        record.id,
                                        f"{record.version} (推理: {format_datetime_short_beijing(batch.inference_time)})",
                                        batch.inference_time,
                                        record_color,  # 相同训练版本使用相同颜色
                                        show_in_legend=show_in_legend
                                    )
                    else:
                        # 显示指定训练记录的所有批次预测
                        record = db.query(TrainingRecord).filter(
                            TrainingRecord.id == training_id
                        ).first()

                        if record:
                            # 为该训练版本分配一个固定颜色（所有批次使用相同颜色）
                            colors = ['#ff9800', '#2196f3', '#4caf50', '#9c27b0', '#f44336', '#00bcd4']
                            record_color = colors[record.id % len(colors)]

                            # 获取该训练记录的所有推理批次
                            batches = db.query(
                                PredictionData.inference_batch_id,
                                func.min(PredictionData.created_at).label('inference_time')
                            ).filter(
                                PredictionData.training_record_id == training_id
                            ).group_by(
                                PredictionData.inference_batch_id
                            ).order_by(
                                func.min(PredictionData.created_at).asc()
                            ).all()

                            for batch_idx, batch in enumerate(batches):
                                predictions = db.query(PredictionData).filter(
                                    and_(
                                        PredictionData.training_record_id == training_id,
                                        PredictionData.inference_batch_id == batch.inference_batch_id
                                    )
                                ).order_by(PredictionData.timestamp).all()

                                if predictions:
                                    # 只在第一个批次时显示图例
                                    show_in_legend = (batch_idx == 0)
                                    self._add_prediction_trace(
                                        fig,
                                        predictions,
                                        record.id,
                                        f"{record.version} (推理: {format_datetime_short_beijing(batch.inference_time)})",
                                        batch.inference_time,
                                        record_color,  # 相同训练版本使用相同颜色
                                        show_in_legend=show_in_legend
                                    )

                fig.update_layout(
                    title=f"{plan.inst_id} {plan.interval} K线图 (最近{last_days}天)",
                    xaxis_title="时间 (UTC+8)",
                    yaxis_title="价格",
                    height=600,
                    template="plotly_white",
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False
                )

                return fig

        except Exception as e:
            logger.error(f"生成K线图失败: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_chart(f"生成失败: {str(e)}")

    def _add_prediction_trace(
        self,
        fig: go.Figure,
        predictions: List[PredictionData],
        training_id: int,
        version: str,
        inference_time: datetime,
        color: str,
        show_in_legend: bool = True
    ):
        """
        添加预测轨迹到图表

        Args:
            fig: Plotly Figure对象
            predictions: 预测数据列表
            training_id: 训练记录ID
            version: 版本号（已包含推理时间）
            inference_time: 推理时间
            color: 线条颜色
            show_in_legend: 是否在图例中显示（用于控制同一训练版本只显示一次）
        """
        # 预测数据时间戳已经是北京时间，直接使用
        pred_timestamps_beijing = [p.timestamp for p in predictions]

        # 检查是否有不确定性数据
        has_uncertainty = any(p.close_min is not None and p.close_max is not None for p in predictions)

        if has_uncertainty:
            # 绘制不确定性阴影区域
            # 1. 上边界
            fig.add_trace(go.Scatter(
                x=pred_timestamps_beijing,
                y=[p.close_max if p.close_max is not None else p.close for p in predictions],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=f'group_{training_id}'
            ))

            # 2. 下边界（填充阴影）
            fig.add_trace(go.Scatter(
                x=pred_timestamps_beijing,
                y=[p.close_min if p.close_min is not None else p.close for p in predictions],
                mode='lines',
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                line=dict(width=0),
                name=f'{version.split(" ")[0]} 不确定性' if show_in_legend else '',
                showlegend=show_in_legend and has_uncertainty,
                hovertemplate='<b>不确定性范围</b><br>最高: %{customdata[0]:.2f}<br>最低: %{y:.2f}<extra></extra>',
                customdata=[[p.close_max if p.close_max is not None else p.close] for p in predictions],
                legendgroup=f'group_{training_id}'
            ))

        # 格式化推理时间
        inference_time_str = format_datetime_short_beijing(inference_time)

        # 3. 平均值线条
        fig.add_trace(go.Scatter(
            x=pred_timestamps_beijing,
            y=[p.close for p in predictions],
            mode='lines+markers',
            name=version,
            showlegend=show_in_legend,
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate=f'<b>{version}</b><br>时间: %{{x}}<br>收盘价: %{{y:.2f}}<extra></extra>',
            legendgroup=f'group_{training_id}'
        ))

    def _empty_chart(self, message: str) -> go.Figure:
        """空图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=600)
        return fig

    def load_agent_decisions(self, plan_id: int) -> pd.DataFrame:
        """加载Agent对话记录（右侧）"""
        try:
            with get_db() as db:
                # 查询该计划相关的对话会话
                conversations = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(desc(AgentConversation.last_message_at)).limit(20).all()

                if not conversations:
                    return pd.DataFrame()

                df_data = []
                for conv in conversations:
                    # 获取该会话的消息数量
                    message_count = db.query(func.count(AgentMessage.id)).filter(
                        AgentMessage.conversation_id == conv.id
                    ).scalar()

                    # 统计工具调用次数
                    tool_call_count = db.query(func.count(AgentMessage.id)).filter(
                        and_(
                            AgentMessage.conversation_id == conv.id,
                            AgentMessage.message_type.in_(['tool_call', 'tool_result'])
                        )
                    ).scalar()

                    status_emoji = {
                        'active': '💬',
                        'completed': '✅',
                        'error': '❌',
                        'paused': '⏸️'
                    }.get(conv.status, '💬')

                    # 获取最新消息的内容预览
                    latest_message = db.query(AgentMessage).filter(
                        AgentMessage.conversation_id == conv.id
                    ).order_by(desc(AgentMessage.created_at)).first()

                    content_preview = ""
                    if latest_message:
                        # 截取内容前50个字符作为预览
                        content = latest_message.content or ""
                        if len(content) > 50:
                            content_preview = content[:47] + "..."
                        else:
                            content_preview = content

                    df_data.append({
                        'ID': conv.id,
                        '时间': format_datetime_full_beijing(conv.last_message_at) if conv.last_message_at else 'N/A',
                        '会话类型': conv.conversation_type or 'analysis',
                        '状态': f"{status_emoji} {conv.status}",
                        '消息数': message_count,
                        '工具调用': tool_call_count,
                        '预览': content_preview
                    })

                return safe_dataframe_from_data(df_data)

        except Exception as e:
            logger.error(f"加载Agent对话记录失败: {e}")
            return pd.DataFrame()

    def load_inference_records(self, plan_id: int) -> pd.DataFrame:
        """加载Kronos推理记录列表"""
        try:
            records = InferenceService.list_inference_records(plan_id)
            if not records:
                return pd.DataFrame()

            df_data = []
            for record in records:
                has_pred_emoji = '✅' if record['has_predictions'] else '⚪'
                inference_time_str = format_datetime_short_beijing(record['inference_time']) if record['inference_time'] else 'N/A'

                df_data.append({
                    'ID': record['training_record_id'],
                    '版本': record['version'],
                    '推理时间': inference_time_str,
                    '预测数据': f"{has_pred_emoji} {record['predictions_count']}条",
                    '数据范围': record.get('date_range', 'N/A'),
                    '训练完成': format_datetime_short_beijing(record['train_end_time']) if record['train_end_time'] else 'N/A'
                })

            return safe_dataframe_from_data(df_data)

        except Exception as e:
            logger.error(f"加载推理记录失败: {e}")
            return pd.DataFrame()

    def get_inference_data_timestamp_range(self, plan_id: int, lookback_window: int = None, data_offset: int = None) -> str:
        """基于最新K线数据和回看窗口动态计算推理数据点时间戳范围"""
        try:
            with get_db() as db:
                # 获取交易计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "**📊 推理数据点范围**\n\n计划不存在"

                # 获取最新的K线数据（最后的数据点）
                latest_kline = db.query(KlineData).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == plan.interval
                ).order_by(KlineData.timestamp.desc()).first()

                if not latest_kline:
                    return "**📊 推理数据点范围**\n\n暂无K线数据"

                # 使用传入的参数或默认值
                if lookback_window is None:
                    # 从计划的微调参数中获取回看窗口，如果没有则使用默认值
                    lookback_window = 200  # 默认200个数据点
                    if plan.finetune_params and isinstance(plan.finetune_params, dict):
                        lookback_window = plan.finetune_params.get('lookback_window', 200)

                if data_offset is None:
                    data_offset = 0  # 默认无偏移

                # 计算推理数据的起始时间戳
                # 推理使用的数据点范围：从最新数据点向前推 lookback_window + data_offset 个数据点
                total_data_points = lookback_window + data_offset

                # 获取用于推理的数据起始点
                start_kline = db.query(KlineData).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == plan.interval,
                    KlineData.timestamp <= latest_kline.timestamp
                ).order_by(KlineData.timestamp.desc()).offset(total_data_points - 1).first()

                if not start_kline:
                    # 如果数据点不够，获取最早的数据点
                    start_kline = db.query(KlineData).filter(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval
                    ).order_by(KlineData.timestamp.asc()).first()

                # 计算数据点总数
                total_count = db.query(KlineData).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == plan.interval,
                    KlineData.timestamp >= start_kline.timestamp,
                    KlineData.timestamp <= latest_kline.timestamp
                ).count()

                # 格式化时间范围显示（K线数据时间戳已经是北京时间，直接格式化）
                start_time = start_kline.timestamp.strftime('%Y-%m-%d %H:%M')
                end_time = latest_kline.timestamp.strftime('%Y-%m-%d %H:%M')

                # 计算时间跨度
                time_diff = latest_kline.timestamp - start_kline.timestamp
                days = time_diff.days
                hours = time_diff.seconds // 3600

                time_span = ""
                if days > 0:
                    time_span = f"{days}天"
                if hours > 0:
                    time_span += f"{hours}小时" if time_span else f"{hours}小时"

                range_info = f"""**📊 推理数据点范围**

**📅 时间范围**: {start_time} ~ {end_time}
**📈 数据点数量**: {total_count}条
**⏱️ 时间跨度**: {time_span or '不足1小时'}
**🔧 回看窗口**: {lookback_window}个数据点
**📍 数据偏移**: {data_offset}个数据点
**💡 最新数据**: {latest_kline.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""

                return range_info

        except Exception as e:
            logger.error(f"获取推理数据点时间戳范围失败: {e}")
            return "**📊 推理数据点范围**\n\n获取失败: " + str(e)

    def get_data_range_info(self, plan_id: int) -> str:
        """获取数据输入的回看数据日期时间范围信息"""
        try:
            # 获取最新的推理记录中的数据范围
            records = InferenceService.list_inference_records(plan_id)
            if not records:
                return "**📊 Data输入信息**\n\n暂无推理记录"

            # 使用最新的推理记录的数据范围
            latest_record = records[0]  # records是按创建时间降序排列的

            if latest_record.get('datetime_range') and latest_record.get('datetime_range') != 'N/A':
                data_range_info = f"""**📊 Data输入信息**

**数据时间范围**: {latest_record.get('datetime_range')}

**训练版本**: {latest_record.get('version', 'N/A')}
**训练完成时间**: {format_datetime_beijing(latest_record.get('train_end_time'), '%Y-%m-%d %H:%M') if latest_record.get('train_end_time') else 'N/A'}
**预测数据条数**: {latest_record.get('predictions_count', 0)}条"""
            else:
                data_range_info = f"""**📊 Data输入信息**

**数据时间范围**: 暂无数据
**训练版本**: {latest_record.get('version', 'N/A')}
**训练完成时间**: {format_datetime_beijing(latest_record.get('train_end_time'), '%Y-%m-%d %H:%M') if latest_record.get('train_end_time') else 'N/A'}
**预测数据条数**: {latest_record.get('predictions_count', 0)}条"""

            return data_range_info

        except Exception as e:
            logger.error(f"获取数据范围信息失败: {e}")
            return "**📊 Data输入信息**\n\n获取数据范围信息失败"

    def get_agent_decision_detail(self, decision_id: int) -> str:
        """获取Agent决策详情"""
        try:
            with get_db() as db:
                decision = db.query(AgentDecision).filter(
                    AgentDecision.id == decision_id
                ).first()

                if not decision:
                    return "决策记录不存在"

                # 获取交易计划信息以显示交易限制
                plan = db.query(TradingPlan).filter(
                    TradingPlan.id == decision.plan_id
                ).first()

                trading_limits_info = ""
                if plan and plan.trading_limits:
                    limits = plan.trading_limits
                    trading_limits_info = f"""
### 💰 交易限制配置

- **可用账户资金**: {limits.get('available_usdt_amount', 'N/A')} USDT
- **可用资金比例**: {limits.get('available_usdt_percentage', 'N/A')}%
- **平摊操作单量**: {limits.get('avg_order_count', 'N/A')} 笔
- **止损比例**: {limits.get('stop_loss_percentage', 'N/A')}%
- **最大持仓**: {limits.get('max_position_size', 'N/A')}
- **最大订单金额**: {limits.get('max_order_amount', 'N/A')} USDT

"""

                detail = f"""
## 📋 Agent决策详情 (ID: {decision.id})

**决策时间**: {format_datetime_full_beijing(decision.decision_time)}
**决策类型**: `{decision.decision_type}`
**状态**: `{decision.status}`
**使用模型**: v{decision.training_record_id} | **LLM**: {decision.llm_model or 'N/A'}

{trading_limits_info}

---

### 💭 决策理由
{decision.reasoning or '无'}

---

### 🛠️ 工具调用
{self._format_tool_calls(decision.tool_calls, decision.tool_results)}

---

### 📦 关联订单
{self._format_order_ids(decision.order_ids)}
"""
                return detail

        except Exception as e:
            logger.error(f"获取决策详情失败: {e}")
            return f"获取失败: {str(e)}"

    def _format_tool_calls(self, tool_calls, tool_results) -> str:
        """格式化工具调用 - 显示完整的ReAct过程"""
        if not tool_calls:
            return "无工具调用"

        lines = []
        lines.append("### 🔄 ReAct 工具调用过程")
        lines.append("")

        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get('name', 'unknown')
            tool_args = call.get('arguments', {})
            result = tool_results[i-1] if tool_results and len(tool_results) >= i else {}

            # 工具调用步骤
            lines.append(f"#### **步骤 {i}: 调用工具 `{tool_name}`**")

            # 显示工具参数
            lines.append("**📝 调用参数:**")
            if tool_args:
                for key, value in tool_args.items():
                    lines.append(f"   - `{key}`: `{value}`")
            else:
                lines.append("   - 无参数")

            # 显示执行结果
            lines.append("**⚡ 执行结果:**")
            if result:
                success = result.get('success', False)
                status_emoji = '✅ 成功' if success else '❌ 失败'
                lines.append(f"   - **状态**: {status_emoji}")

                if success:
                    # 成功时显示数据摘要
                    data = result.get('data')
                    if data:
                        if isinstance(data, dict):
                            if 'message' in data:
                                lines.append(f"   - **消息**: {data['message']}")
                            if 'count' in data:
                                lines.append(f"   - **数据量**: {data['count']} 条")
                            if 'total' in data:
                                lines.append(f"   - **总数**: {data['total']}")
                            if 'records' in data and isinstance(data['records'], list):
                                lines.append(f"   - **记录数**: {len(data['records'])} 条")
                        elif isinstance(data, list):
                            lines.append(f"   - **数组长度**: {len(data)} 项")

                    if result.get('message'):
                        lines.append(f"   - **说明**: {result['message']}")
                else:
                    # 失败时显示错误信息
                    error_msg = result.get('error', result.get('message', '未知错误'))
                    lines.append(f"   - **错误**: {error_msg}")
            else:
                lines.append("   - **状态**: ⚠️ 无结果数据")

            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _format_order_ids(self, order_ids) -> str:
        """格式化订单ID"""
        if not order_ids:
            return "无关联订单"

        return ", ".join([f"`{oid}`" for oid in order_ids])

    def get_latest_agent_decision_output(self, plan_id: int):
        """
        获取最新的Agent决策输出（Chatbot格式），包含最新的预测数据预览

        Returns:
            List[Dict]: Chatbot messages 格式 [{"role": "assistant", "content": ...}]
        """
        try:
            with get_db() as db:
                # 获取最新的Agent决策
                decision = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).order_by(desc(AgentDecision.decision_time)).first()

                # 获取最新的已完成训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                # 构建输出内容
                output_parts = []

                # 添加预测数据预览部分
                if latest_training:
                    # 获取最后一次推理的预测数据
                    from sqlalchemy import func
                    latest_batch = db.query(
                        PredictionData.inference_batch_id,
                        func.max(PredictionData.created_at).label('max_time')
                    ).filter(
                        PredictionData.training_record_id == latest_training.id
                    ).group_by(
                        PredictionData.inference_batch_id
                    ).order_by(
                        func.max(PredictionData.created_at).desc()
                    ).first()

                    if latest_batch:
                        # 获取该批次的所有预测数据
                        predictions_query = db.query(PredictionData).filter(
                            and_(
                                PredictionData.training_record_id == latest_training.id,
                                PredictionData.inference_batch_id == latest_batch.inference_batch_id
                            )
                        ).order_by(PredictionData.timestamp).all()

                        if predictions_query:
                            predictions = []
                            for pred in predictions_query:
                                predictions.append({
                                    'timestamp': pred.timestamp,
                                    'open': pred.open,
                                    'high': pred.high,
                                    'low': pred.low,
                                    'close': pred.close,
                                    'volume': pred.volume
                                })

                            # 格式化预测数据预览
                            pred_output = self._format_prediction_preview(predictions, latest_batch.inference_batch_id, latest_training.version)
                            output_parts.append(pred_output)

                # 添加AI Agent决策结果部分
                if decision:
                    # 创建完整的ReAct对话格式
                    messages = []

                    # 1. 系统和初始提示
                    messages.append({
                        "role": "system",
                        "content": f"AI Agent 交易分析系统\n使用模型: v{decision.training_record_id} | LLM: {decision.llm_model or 'N/A'}\n决策时间: {format_datetime_full_beijing(decision.decision_time)}"
                    })

                    # 2. 用户输入（预测数据）
                    if output_parts:
                        messages.append({
                            "role": "user",
                            "content": output_parts[-1] if output_parts else "请分析最新的市场预测数据并提供交易建议。"
                        })

                    # 3. AI思考过程（推理）
                    reasoning_content = f"""## 🧠 AI思考过程

**📊 分析阶段**:
- 正在分析最新的Kronos预测数据
- 评估市场趋势和价格波动
- 结合交易限制和风险控制要求

**💡 决策逻辑**:
{decision.reasoning or '无详细推理过程'}

**🎯 决策结论**:
- **决策类型**: {decision.decision_type or 'N/A'}
- **执行状态**: {decision.status}
- **建议操作**: 基于分析结果决定是否执行交易

"""
                    messages.append({
                        "role": "assistant",
                        "content": reasoning_content
                    })

                    # 4. 工具调用过程
                    if decision.tool_calls:
                        tool_process_content = "## 🔧 工具执行阶段\n\n"

                        for i, call in enumerate(decision.tool_calls, 1):
                            tool_name = call.get('name', 'unknown')
                            tool_args = call.get('arguments', {})
                            result = decision.tool_results[i-1] if decision.tool_results and len(decision.tool_results) >= i else {}

                            # 工具调用步骤
                            tool_process_content += f"### **步骤 {i}: 执行 `{tool_name}`**\n\n"

                            # 调用参数
                            tool_process_content += "**📋 调用参数:**\n"
                            if tool_args:
                                for key, value in tool_args.items():
                                    tool_process_content += f"- `{key}`: `{value}`\n"
                            else:
                                tool_process_content += "- 无参数\n"

                            tool_process_content += "\n**⚡ 执行结果:**\n"

                            if result:
                                success = result.get('success', False)
                                status_emoji = '✅ 成功' if success else '❌ 失败'
                                tool_process_content += f"- **状态**: {status_emoji}\n"

                                if success:
                                    data = result.get('data')
                                    if data:
                                        if isinstance(data, dict):
                                            if 'message' in data:
                                                tool_process_content += f"- **返回信息**: {data['message']}\n"
                                            if 'count' in data:
                                                tool_process_content += f"- **数据量**: {data['count']} 条\n"
                                            if 'total' in data:
                                                tool_process_content += f"- **总数**: {data['total']}\n"
                                            if 'records' in data and isinstance(data['records'], list):
                                                tool_process_content += f"- **记录数**: {len(data['records'])} 条\n"
                                                # 显示前几条数据示例
                                                if len(data['records']) > 0:
                                                    sample_record = data['records'][0]
                                                    tool_process_content += f"- **数据示例**: {str(sample_record)[:100]}...\n"
                                        elif isinstance(data, list):
                                            tool_process_content += f"- **数组长度**: {len(data)} 项\n"
                                            if len(data) > 0:
                                                tool_process_content += f"- **示例数据**: {str(data[0])[:100]}...\n"

                                if result.get('message'):
                                    tool_process_content += f"- **说明**: {result['message']}\n"
                            else:
                                tool_process_content += "- **状态**: ⚠️ 无结果数据\n"

                            tool_process_content += "\n---\n\n"

                        messages.append({
                            "role": "assistant",
                            "content": tool_process_content
                        })

                    # 5. 最终总结
                    summary_content = f"""## 📋 执行总结

**🎯 决策结果**: {decision.status}
**🔧 工具调用次数**: {len(decision.tool_calls) if decision.tool_calls else 0}
**📦 关联订单**: {len(decision.order_ids) if decision.order_ids else 0}
**⏰ 完成时间**: {format_datetime_full_beijing(decision.decision_time)}

**💰 交易执行情况**:
{self._format_order_ids(decision.order_ids) if decision.order_ids else "无订单生成"}

---
*此决策由AI Agent基于Kronos预测数据自动分析生成*
"""
                    messages.append({
                        "role": "assistant",
                        "content": summary_content
                    })

                    return messages
                else:
                    # 没有决策记录时的默认响应
                    if output_parts:
                        return [{"role": "assistant", "content": output_parts[-1]}]
                    else:
                        return [{"role": "assistant", "content": "暂无AI Agent决策记录。请先运行推理并让AI Agent进行分析。"}]

        except Exception as e:
            logger.error(f"获取最新Agent决策输出失败: {e}")
            import traceback
            traceback.print_exc()
            return [{"role": "assistant", "content": f"等待推理...\n\n❌ 获取失败: {str(e)}"}]

    def get_latest_conversation_messages(self, plan_id: int) -> List[Dict]:
        """
        获取最新的对话消息，支持新的消息格式（think模式、tool调用、play结果）

        Returns:
            List[Dict]: Chatbot messages 格式 [{"role": "assistant", "content": ...}]
        """
        try:
            from database.models import AgentConversation, AgentMessage, TradingPlan
            from database.db import get_db

            with get_db() as db:
                # 获取最新的对话（任何类型）
                latest_conversation = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).order_by(desc(AgentConversation.started_at)).first()

                if latest_conversation:
                    # 获取该对话的所有消息
                    messages = db.query(AgentMessage).filter(
                        AgentMessage.conversation_id == latest_conversation.id
                    ).order_by(AgentMessage.timestamp.asc()).all()

                    if messages:
                        # 使用新的格式化方法
                        return format_conversation_history(messages)

                # 如果没有任何对话，返回欢迎消息并检查配置
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if plan and plan.llm_config_id:
                    welcome_msg = """👋 欢迎使用AI Agent对话系统

✅ **配置已完成**
- LLM配置已就绪
- Agent工具已配置
- 准备就绪

💡 **使用方式**
- 点击「执行推理」开始市场分析
- 在对话框中输入消息进行对话
- 支持思考模式、工具调用、投资决策展示

🚀 开始您的智能交易之旅吧！"""
                else:
                    welcome_msg = """👋 欢迎使用AI Agent对话系统

⚠️ **需要配置**
- 请先在「⚙️ Agent配置」中设置LLM提供商
- 配置可用工具
- 设置交易限制

💡 **配置完成后即可开始**:
- 智能市场分析
- 自动交易决策
- 对话式交互"""

                return [{"role": "assistant", "content": welcome_msg}]

        except Exception as e:
            logger.error(f"获取最新对话消息失败: {e}")
            return [{"role": "assistant", "content": f"等待对话...\n\n❌ 获取失败: {str(e)}"}]

    def get_conversation_tool_calls_summary(self, plan_id: int) -> List[Dict]:
        """
        获取对话的工具调用摘要

        Returns:
            List[Dict]: 工具调用列表
        """
        try:
            latest_conversation = ConversationService.get_latest_conversation(
                plan_id=plan_id,
                conversation_type="auto_inference"
            )

            if latest_conversation:
                return ConversationService.get_tool_calls_summary(latest_conversation.id)

            return []

        except Exception as e:
            logger.error(f"获取工具调用摘要失败: {e}")
            return []

    def get_plan_conversations_list(self, plan_id: int, limit: int = 10) -> List[Dict]:
        """
        获取计划的对话会话列表

        Returns:
            List[Dict]: 对话会话列表
        """
        try:
            conversations = ConversationService.get_plan_conversations(plan_id, limit)

            conversation_list = []
            for conv in conversations:
                conversation_list.append({
                    'id': conv.id,
                    'session_name': conv.session_name,
                    'conversation_type': conv.conversation_type,
                    'status': conv.status,
                    'total_messages': conv.total_messages,
                    'total_tool_calls': conv.total_tool_calls,
                    'started_at': format_datetime_full_beijing(conv.started_at),
                    'last_message_at': format_datetime_full_beijing(conv.last_message_at) if conv.last_message_at else 'N/A',
                    'completed_at': format_datetime_full_beijing(conv.completed_at) if conv.completed_at else '进行中'
                })

            return conversation_list

        except Exception as e:
            logger.error(f"获取对话会话列表失败: {e}")
            return []

    async def enhanced_inference_stream(self, plan_id: int, training_record_id: int = None, progress=None):
        """
        增强版推理流式输出，支持React循环展示和工具调用记录

        Args:
            plan_id: 计划ID
            training_record_id: 训练记录ID
            progress: Gradio进度条

        Yields:
            Dict: 包含对话状态、消息等的流式数据
        """
        try:
            from services.agent_decision_service import AgentDecisionService

            async for chunk in AgentDecisionService.enhanced_react_tool_use_stream(
                plan_id=plan_id,
                training_id=training_record_id,
                progress=progress,
                session_name=f"推理会话_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                yield chunk

        except Exception as e:
            logger.error(f"增强版推理流失败: {e}")
            yield {
                'type': 'error',
                'content': f'❌ 推理失败: {str(e)}',
                'messages': [{"role": "assistant", "content": f"❌ 推理失败: {str(e)}"}]
            }

    def _format_prediction_preview(self, predictions: list, batch_id: str, version: str) -> str:
        """
        格式化预测数据预览

        Args:
            predictions: 预测数据列表
            batch_id: 推理批次ID
            version: 训练版本

        Returns:
            格式化后的预测数据预览字符串
        """
        try:
            if not predictions or len(predictions) == 0:
                return "## 📊 最新预测数据\n\n暂无预测数据"

            # 计算趋势
            first_pred = predictions[0]
            last_pred = predictions[-1]
            first_close = first_pred['close']
            last_close = last_pred['close']
            change_pct = ((last_close - first_close) / first_close) * 100

            # 获取概率指标（从数据库查询）
            upward_prob = None
            volatility_amp_prob = None
            sample_count = 1

            try:
                with get_db() as db:
                    # 使用第一个预测数据获取相关信息
                    pred_record = db.query(PredictionData).filter(
                        PredictionData.inference_batch_id == batch_id
                    ).first()

                    if pred_record:
                        upward_prob = pred_record.upward_probability
                        volatility_amp_prob = pred_record.volatility_amplification_probability
                        if pred_record.inference_params:
                            sample_count = pred_record.inference_params.get('sample_count', 1)
            except Exception as e:
                logger.error(f"获取概率指标失败: {e}")

            # 时间范围
            first_time = format_datetime_beijing(first_pred['timestamp'], '%Y-%m-%d %H:%M') if hasattr(first_pred['timestamp'], 'strftime') else str(first_pred['timestamp'])[:16]
            last_time = format_datetime_beijing(last_pred['timestamp'], '%Y-%m-%d %H:%M') if hasattr(last_pred['timestamp'], 'strftime') else str(last_pred['timestamp'])[:16]

            # 价格统计
            close_prices = [p['close'] for p in predictions]
            min_close = min(close_prices)
            max_close = max(close_prices)

            # 趋势判断
            trend = "📈 上涨趋势" if change_pct > 0 else "📉 下跌趋势" if change_pct < 0 else "➡️ 横盘"
            trend_emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"

            # 构建输出
            output = f"""## 📊 最新预测数据预览

**批次ID**: {batch_id} | **训练版本**: {version}
**预测周期数**: {len(predictions)} | **时间范围**: {first_time} ~ {last_time}

---

### 📈 价格预测

**当前价格**: ${first_close:.4f}
**预测价格**: ${last_close:.4f}
**价格区间**: ${min_close:.4f} ~ ${max_close:.4f}
**预测涨跌**: {trend_emoji} {change_pct:+.2f}%
**趋势判断**: {trend}"""

            # 添加概率指标（如果有）
            if upward_prob is not None and volatility_amp_prob is not None:
                upward_percent = upward_prob * 100
                volatility_percent = volatility_amp_prob * 100

                # 根据概率值选择表情和颜色
                if upward_percent >= 60:
                    upward_emoji = "📈"
                    upward_color = "green"
                elif upward_percent >= 40:
                    upward_emoji = "➡️"
                    upward_color = "orange"
                else:
                    upward_emoji = "📉"
                    upward_color = "red"

                if volatility_percent >= 60:
                    volatility_emoji = "⚡"
                    volatility_color = "red"
                elif volatility_percent >= 40:
                    volatility_emoji = "〰️"
                    volatility_color = "orange"
                else:
                    volatility_emoji = "😴"
                    volatility_color = "green"

                output += f"""

---

### 🎯 概率指标

<div style="display: flex; gap: 20px; margin: 10px 0;">
  <div style="flex: 1; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <div style="font-size: 14px; opacity: 0.9;">上涨概率（未来预测期）</div>
    <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">{upward_emoji} {upward_percent:.1f}%</div>
    <div style="font-size: 12px; opacity: 0.8;">模型对价格上涨的置信度</div>
  </div>
  <div style="flex: 1; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white;">
    <div style="font-size: 14px; opacity: 0.9;">波动性放大</div>
    <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">{volatility_emoji} {volatility_percent:.1f}%</div>
    <div style="font-size: 12px; opacity: 0.8;">未来波动率超过历史的概率</div>
  </div>
</div>

**数据来源**: 基于 {sample_count} 条蒙特卡罗路径"""

            # 添加详细预测数据预览
            output += f"""

---

### 📋 预测数据详情 (前10条)

| 序号 | 时间 | 开盘 | 最高 | 最低 | 收盘 |
|------|------|------|------|------|------|"""

            # 显示前10条详细数据
            for i, pred in enumerate(predictions[:10], 1):
                timestamp_str = format_datetime_short_beijing(pred['timestamp']) if hasattr(pred['timestamp'], 'strftime') else str(pred['timestamp'])[:16]
                output += f"\n| {i} | {timestamp_str} | ${pred['open']:.2f} | ${pred['high']:.2f} | ${pred['low']:.2f} | ${pred['close']:.2f} |"

            if len(predictions) > 10:
                output += f"\n... (共{len(predictions)}条，仅显示前10条)"

            output += f"""

---

**💡 提示**: 此预测数据可用于 AI Agent 分析和决策。点击「手动推理」按钮可以让 AI Agent 基于这些数据进行交易分析。"""

            return output

        except Exception as e:
            logger.error(f"格式化预测数据预览失败: {e}")
            return f"## 📊 最新预测数据\n\n❌ 格式化失败: {str(e)}"

    def get_finetune_params(self, plan_id: int) -> dict:
        """获取微调参数"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {}
                return plan.finetune_params or {}
        except Exception as e:
            logger.error(f"获取微调参数失败: {e}")
            return {}

    def save_finetune_params(self, plan_id: int, params: dict) -> str:
        """保存微调参数"""
        try:
            with get_db() as db:
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'finetune_params': params
                })
                db.commit()
                logger.info(f"微调参数已保存: plan_id={plan_id}")
                return "✅ 参数已保存"
        except Exception as e:
            logger.error(f"保存微调参数失败: {e}")
            return f"❌ 保存失败: {str(e)}"

    def get_llm_configs(self) -> list:
        """获取所有LLM配置"""
        try:
            with get_db() as db:
                from database.models import LLMConfig
                configs = db.query(LLMConfig).filter(LLMConfig.is_active == True).all()
                return [(f"{c.provider} - {c.model_name}", c.id) for c in configs]
        except Exception as e:
            logger.error(f"获取LLM配置失败: {e}")
            return []

    def get_prompt_templates(self) -> list:
        """获取所有提示词模版"""
        try:
            with get_db() as db:
                from database.models import AgentPromptTemplate
                templates = db.query(AgentPromptTemplate).filter(
                    AgentPromptTemplate.is_active == True
                ).all()
                return [(f"{t.name} ({t.category})", t.id) for t in templates]
        except Exception as e:
            logger.error(f"获取提示词模版失败: {e}")
            return []

    def get_inference_params(self, plan_id: int) -> dict:
        """获取推理参数"""
        try:
            # 首先确保参数结构正确
            self.ensure_finetune_params_structure(plan_id)

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {}

                finetune_params = plan.finetune_params or {}
                inference_config = finetune_params.get('inference', {})
                data_config = finetune_params.get('data', {})

                # 确保类型转换：从数据库JSONB读取的数值可能是字符串，需要转换为数字
                return {
                    'lookback_window': int(data_config.get('lookback_window', 512)),
                    'predict_window': int(data_config.get('predict_window', 48)),
                    'temperature': float(inference_config.get('temperature', 1.0)),
                    'top_p': float(inference_config.get('top_p', 0.9)),
                    'sample_count': int(inference_config.get('sample_count', 30)),
                    'data_offset': int(inference_config.get('data_offset', 0))
                }
        except Exception as e:
            logger.error(f"获取推理参数失败: {e}")
            return {
                'lookback_window': 512,
                'predict_window': 48,
                'temperature': 1.0,
                'top_p': 0.9,
                'sample_count': 30,
                'data_offset': 0
            }

    def ensure_finetune_params_structure(self, plan_id: int) -> bool:
        """
        确保finetune_params结构正确，如果不正确则修复

        Args:
            plan_id: 计划ID

        Returns:
            是否进行了修复
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return False

                finetune_params = plan.finetune_params or {}
                needs_fix = False

                # 确保嵌套结构存在
                if 'data' not in finetune_params:
                    finetune_params['data'] = {}
                    needs_fix = True

                if 'inference' not in finetune_params:
                    finetune_params['inference'] = {}
                    needs_fix = True

                # 处理扁平结构参数（兼容性）
                # 如果参数在顶层，移动到对应的嵌套结构中
                if 'lookback_window' in finetune_params and 'lookback_window' not in finetune_params['data']:
                    finetune_params['data']['lookback_window'] = finetune_params['lookback_window']
                    del finetune_params['lookback_window']  # 移除顶层参数
                    needs_fix = True

                if 'predict_window' in finetune_params and 'predict_window' not in finetune_params['data']:
                    finetune_params['data']['predict_window'] = finetune_params['predict_window']
                    del finetune_params['predict_window']  # 移除顶层参数
                    needs_fix = True

                # 设置默认推理参数（如果不存在）
                inference_defaults = {
                    'temperature': 1.0,
                    'top_p': 0.9,
                    'sample_count': 30,
                    'data_offset': 0
                }

                for key, default_value in inference_defaults.items():
                    if key not in finetune_params['inference']:
                        finetune_params['inference'][key] = default_value
                        needs_fix = True

                # 如果需要修复，更新数据库
                if needs_fix:
                    db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                        'finetune_params': finetune_params
                    })
                    db.commit()
                    logger.info(f"已修复计划{plan_id}的参数结构")

                return needs_fix

        except Exception as e:
            logger.error(f"确保参数结构失败: plan_id={plan_id}, error={e}")
            return False

    def save_inference_params(
        self,
        plan_id: int,
        lookback_window: int,
        predict_window: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        data_offset: int = 0
    ) -> str:
        """保存推理参数"""
        try:
            # 验证参数范围
            if not (64 <= lookback_window <= 2048):
                return "❌ Lookback Window 必须在 64 到 2048 之间"

            if not (1 <= predict_window <= 512):
                return "❌ Predict Window 必须在 1 到 512 之间"

            if not (0.0 <= temperature <= 2.0):
                return "❌ Temperature 必须在 0.0 到 2.0 之间"

            if not (0.0 <= top_p <= 1.0):
                return "❌ Top-p 必须在 0.0 到 1.0 之间"

            if not (1 <= sample_count <= 100):
                return "❌ Sample Count 必须在 1 到 100 之间"

            if not (0 <= data_offset <= 1000):
                return "❌ 数据偏移必须在 0 到 1000 之间"

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 获取现有的 finetune_params
                finetune_params = plan.finetune_params or {}
                if 'inference' not in finetune_params:
                    finetune_params['inference'] = {}
                if 'data' not in finetune_params:
                    finetune_params['data'] = {}

                # 更新数据窗口参数（保存到 data 配置中）
                finetune_params['data']['lookback_window'] = int(lookback_window)
                finetune_params['data']['predict_window'] = int(predict_window)

                # 更新推理参数
                finetune_params['inference']['temperature'] = float(temperature)
                finetune_params['inference']['top_p'] = float(top_p)
                finetune_params['inference']['sample_count'] = int(sample_count)
                finetune_params['inference']['data_offset'] = int(data_offset)

                # 保存到数据库
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'finetune_params': finetune_params
                })
                db.commit()

                logger.info(
                    f"推理参数已保存: plan_id={plan_id}, "
                    f"lookback={lookback_window}, predict={predict_window}, "
                    f"temperature={temperature}, top_p={top_p}, sample_count={sample_count}, data_offset={data_offset}"
                )
                return "✅ 推理参数已保存"
        except Exception as e:
            logger.error(f"保存推理参数失败: {e}")
            return f"❌ 保存失败: {str(e)}"

    def get_agent_config(self, plan_id: int) -> dict:
        """获取Agent配置"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {}
                return {
                    'llm_config_id': plan.llm_config_id,
                    'agent_prompt': plan.agent_prompt or '',
                    'agent_tools_config': plan.agent_tools_config or {}
                }
        except Exception as e:
            logger.error(f"获取Agent配置失败: {e}")
            return {}

    def save_agent_config(self, plan_id: int, llm_config_id: int,
                         agent_prompt: str, tools_config: dict) -> str:
        """保存Agent配置"""
        try:
            with get_db() as db:
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'llm_config_id': llm_config_id,
                    'agent_prompt': agent_prompt,
                    'agent_tools_config': tools_config
                })
                db.commit()
                logger.info(f"Agent配置已保存: plan_id={plan_id}")
                return "✅ Agent配置已保存"
        except Exception as e:
            logger.error(f"保存Agent配置失败: {e}")
            return f"❌ 保存失败: {str(e)}"

    
    def get_trading_limits_config(self, plan_id: int) -> dict:
        """获取交易限制配置"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {
                        'available_usdt_amount': 1000.0,
                        'available_usdt_percentage': 30.0,
                        'avg_order_count': 10,
                        'stop_loss_percentage': 20.0
                    }

                trading_limits = plan.trading_limits or {}
                # 确保类型转换：从数据库JSONB读取的数值可能是字符串，需要转换为数字
                return {
                    'available_usdt_amount': float(trading_limits.get('available_usdt_amount', 1000.0)),
                    'available_usdt_percentage': float(trading_limits.get('available_usdt_percentage', 30.0)),
                    'avg_order_count': int(trading_limits.get('avg_order_count', 10)),
                    'stop_loss_percentage': float(trading_limits.get('stop_loss_percentage', 20.0))
                }
        except Exception as e:
            logger.error(f"获取交易限制配置失败: {e}")
            return {
                'available_usdt_amount': 1000.0,
                'available_usdt_percentage': 30.0,
                'avg_order_count': 10,
                'stop_loss_percentage': 20.0
            }

    def save_trading_limits_config(
        self,
        plan_id: int,
        available_usdt_amount: float,
        available_usdt_percentage: float,
        avg_order_count: int,
        stop_loss_percentage: float
    ) -> str:
        """保存交易限制配置"""
        try:
            # 验证参数
            if available_usdt_amount < 0:
                return "❌ 可用USDT金额不能为负数"
            if not (0 <= available_usdt_percentage <= 100):
                return "❌ 可用资金比例必须在 0% 到 100% 之间"
            if avg_order_count < 1:
                return "❌ 平摊单量必须大于0"
            if not (0.1 <= stop_loss_percentage <= 100):
                return "❌ 止损比例必须在 0.1% 到 100% 之间"

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 获取现有的trading_limits配置
                trading_limits = plan.trading_limits or {}

                # 更新交易限制配置
                trading_limits.update({
                    'available_usdt_amount': float(available_usdt_amount),
                    'available_usdt_percentage': float(available_usdt_percentage),
                    'avg_order_count': int(avg_order_count),
                    'stop_loss_percentage': float(stop_loss_percentage),
                    # 保留原有的兼容字段
                    'max_position_size': trading_limits.get('max_position_size', 1.0),
                    'max_order_amount': float(available_usdt_amount)
                })

                # 保存到数据库
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'trading_limits': trading_limits
                })
                db.commit()

                logger.info(
                    f"交易限制配置已保存: plan_id={plan_id}, "
                    f"available_usdt={available_usdt_amount}, percentage={available_usdt_percentage}%, "
                    f"avg_orders={avg_order_count}, stop_loss={stop_loss_percentage}%"
                )
                return "✅ 交易限制配置已保存"

        except Exception as e:
            logger.error(f"保存交易限制配置失败: {e}")
            return f"❌ 保存失败: {str(e)}"

    def load_prompt_template(self, template_id: int) -> str:
        """加载提示词模版内容"""
        try:
            with get_db() as db:
                from database.models import AgentPromptTemplate
                template = db.query(AgentPromptTemplate).filter(
                    AgentPromptTemplate.id == template_id
                ).first()
                if template:
                    return template.content
                return ""
        except Exception as e:
            logger.error(f"加载提示词模版失败: {e}")
            return ""

    def get_automation_config(self, plan_id: int) -> dict:
        """获取自动化配置（四个开关和时间表）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {
                        'auto_finetune_enabled': False,
                        'auto_inference_enabled': False,
                        'auto_agent_enabled': False,
                        'auto_tool_execution_enabled': False,
                        'schedule': []
                    }
                return {
                    'auto_finetune_enabled': plan.auto_finetune_enabled or False,
                    'auto_inference_enabled': plan.auto_inference_enabled or False,
                    'auto_agent_enabled': plan.auto_agent_enabled or False,
                    'auto_tool_execution_enabled': plan.auto_tool_execution_enabled or False,
                    'schedule': plan.auto_finetune_schedule or []
                }
        except Exception as e:
            logger.error(f"获取自动化配置失败: {e}")
            return {
                'auto_finetune_enabled': False,
                'auto_inference_enabled': False,
                'auto_agent_enabled': False,
                'auto_tool_execution_enabled': False,
                'schedule': []
            }

    def save_automation_config(self, plan_id: int, auto_finetune: bool, auto_inference: bool,
                               auto_agent: bool, auto_tool_execution: bool, schedule_times: str) -> str:
        """保存自动化配置"""
        try:
            # 解析时间表
            schedule_list = []
            if schedule_times and schedule_times.strip():
                time_parts = [t.strip() for t in schedule_times.split(',')]
                for time_str in time_parts:
                    # 验证时间格式 HH:MM
                    if len(time_str) == 5 and time_str.count(':') == 1:
                        try:
                            hour, minute = time_str.split(':')
                            if 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59:
                                schedule_list.append(time_str)
                        except ValueError:
                            continue

            with get_db() as db:
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'auto_finetune_enabled': auto_finetune,
                    'auto_inference_enabled': auto_inference,
                    'auto_agent_enabled': auto_agent,
                    'auto_tool_execution_enabled': auto_tool_execution,
                    'auto_finetune_schedule': schedule_list
                })
                db.commit()
                logger.info(
                    f"自动化配置已保存: plan_id={plan_id}, "
                    f"finetune={auto_finetune}, inference={auto_inference}, "
                    f"agent={auto_agent}, tool_exec={auto_tool_execution}, "
                    f"schedule={schedule_list}"
                )
                return f"✅ 自动化配置已保存\n📅 训练时间表: {', '.join(schedule_list) if schedule_list else '未配置'}"
        except Exception as e:
            logger.error(f"保存自动化配置失败: {e}")
            return f"❌ 保存失败: {str(e)}"

    def get_automation_status_display(self, plan_id: int) -> str:
        """获取自动化状态显示"""
        try:
            from services.automation_service import automation_service
            status = automation_service.get_automation_status(plan_id)

            if not status:
                return "### 📊 自动化状态\n\n❌ 无法获取状态信息"

            # 构建状态显示
            lines = ["### 📊 自动化状态"]
            lines.append("━━━━━━━━━━━━━━━━━━━━━━")

            # 自动化配置状态
            lines.append("#### 🎯 自动化配置")
            finetune_status = "✅" if status.get('auto_finetune_enabled') else "❌"
            inference_status = "✅" if status.get('auto_inference_enabled') else "❌"
            agent_status = "✅" if status.get('auto_agent_enabled') else "❌"
            tool_status = "🚫"  # 工具确认功能已废弃

            lines.append(f"- 🧠 自动微调训练: {finetune_status}")
            lines.append(f"- 🔮 自动预测推理: {inference_status}")
            lines.append(f"- 🤖 自动Agent决策: {agent_status}")
            lines.append(f"- ⚡ 自动工具执行: {tool_status} (已废弃 - AI Agent直接调用工具)")

            # 调度器状态
            lines.append("")
            lines.append("#### 🔄 调度器状态")
            scheduler_emoji = "✅" if status.get('scheduler_running') else "❌"
            lines.append(f"- 调度器运行状态: {scheduler_emoji}")

            if status.get('last_check_time'):
                last_check = format_datetime_full_beijing(status['last_check_time'])
                lines.append(f"- 最后检查时间: {last_check}")

            # 当前任务状态
            current_task = status.get('current_task', {})
            if current_task:
                lines.append("")
                lines.append("#### ⚡ 当前任务")
                task_stage = current_task.get('stage', 'unknown')
                task_start = current_task.get('start_time')
                if task_start:
                    start_str = format_datetime_beijing(task_start, "%H:%M:%S")
                    lines.append(f"- 任务阶段: {task_stage}")
                    lines.append(f"- 开始时间: {start_str}")
                    lines.append(f"- 任务ID: {current_task.get('task_id', 'N/A')}")
            else:
                lines.append("")
                lines.append("#### ⚡ 当前任务: 无活跃任务")

            # 最新训练记录
            latest_training = status.get('latest_auto_training')
            if latest_training:
                lines.append("")
                lines.append("#### 📈 最新自动训练")
                lines.append(f"- 训练ID: {latest_training['id']}")
                lines.append(f"- 状态: {latest_training['status']}")
                if latest_training['created_at']:
                    training_time = format_datetime_beijing(latest_training['created_at'], "%Y-%m-%d %H:%M")
                    lines.append(f"- 训练时间: {training_time}")

            # 时间表配置
            schedule = status.get('auto_finetune_schedule', [])
            if schedule:
                lines.append("")
                lines.append("#### ⏰ 训练时间表")
                for time_point in schedule:
                    lines.append(f"- {time_point}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"获取自动化状态失败: {e}")
            return f"### 📊 自动化状态\n\n❌ 获取失败: {str(e)}"

    def get_scheduler_status_display(self) -> str:
        """获取调度器状态显示"""
        try:
            from services.automation_service import automation_service
            running = automation_service.scheduler_running
            last_check = automation_service.last_check_time

            if running:
                status_emoji = "✅ 运行中"
                status_text = "调度器正在运行，会每分钟检查自动化任务"
            else:
                status_emoji = "❌ 已停止"
                status_text = "调度器已停止，不会自动执行任务"

            lines = [f"🔄 调度器状态: {status_emoji}"]
            lines.append(status_text)

            if last_check:
                last_check_str = format_datetime_full_beijing(last_check)
                lines.append(f"最后检查: {last_check_str}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"获取调度器状态失败: {e}")
            return "🔄 调度器状态: ❌ 获取失败"

    def start_automation_scheduler(self) -> str:
        """启动自动化调度器"""
        try:
            from services.automation_service import automation_service
            automation_service.start_scheduler()
            return "✅ 自动化调度器已启动"
        except Exception as e:
            logger.error(f"启动自动化调度器失败: {e}")
            return f"❌ 启动失败: {str(e)}"

    def stop_automation_scheduler(self) -> str:
        """停止自动化调度器"""
        try:
            from services.automation_service import automation_service
            automation_service.stop_scheduler()
            return "✅ 自动化调度器已停止"
        except Exception as e:
            logger.error(f"停止自动化调度器失败: {e}")
            return f"❌ 停止失败: {str(e)}"

    # 工具确认相关方法已移除 - AI Agent现在直接使用启用的工具，无需确认

    def get_finetune_schedule(self, plan_id: int) -> list:
        """获取自动微调时间表"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return []
                return plan.auto_finetune_schedule or []
        except Exception as e:
            logger.error(f"获取自动微调时间表失败: {e}")
            return []

    def add_finetune_schedule_time(self, plan_id: int, time_str: str) -> tuple:
        """
        添加自动微调时间点

        Args:
            plan_id: 计划ID
            time_str: 时间字符串，格式 HH:MM

        Returns:
            (结果消息, 更新后的时间表列表)
        """
        try:
            # 验证时间格式
            import re
            if not re.match(r'^\d{2}:\d{2}$', time_str):
                return "❌ 时间格式错误，请使用 HH:MM 格式", []

            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return "❌ 时间范围错误，小时应为 00-23，分钟应为 00-59", []

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在", []

                schedule = plan.auto_finetune_schedule or []

                # 检查是否已存在
                if time_str in schedule:
                    return f"⚠️ 时间点 {time_str} 已存在", schedule

                # 添加并排序
                schedule.append(time_str)
                schedule.sort()

                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'auto_finetune_schedule': schedule
                })
                db.commit()

                logger.info(f"已添加自动微调时间点: plan_id={plan_id}, time={time_str}")
                return f"✅ 已添加时间点 {time_str}", schedule

        except Exception as e:
            logger.error(f"添加自动微调时间点失败: {e}")
            return f"❌ 添加失败: {str(e)}", []

    def remove_finetune_schedule_time(self, plan_id: int, time_str: str) -> tuple:
        """
        删除自动微调时间点

        Args:
            plan_id: 计划ID
            time_str: 时间字符串，格式 HH:MM

        Returns:
            (结果消息, 更新后的时间表列表)
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在", []

                schedule = plan.auto_finetune_schedule or []

                if time_str not in schedule:
                    return f"⚠️ 时间点 {time_str} 不存在", schedule

                schedule.remove(time_str)

                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'auto_finetune_schedule': schedule
                })
                db.commit()

                logger.info(f"已删除自动微调时间点: plan_id={plan_id}, time={time_str}")
                return f"✅ 已删除时间点 {time_str}", schedule

        except Exception as e:
            logger.error(f"删除自动微调时间点失败: {e}")
            return f"❌ 删除失败: {str(e)}", []

    def get_inference_schedule(self, plan_id: int) -> list:
        """获取自动预测间隔时间"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return [4]  # 默认4小时
                interval_hours = getattr(plan, 'auto_inference_interval_hours', 4)
                return [interval_hours] if interval_hours else [4]  # 默认4小时
        except Exception as e:
            logger.error(f"获取自动预测间隔时间失败: {e}")
            return [4]

    def add_inference_schedule_time(self, plan_id: int, time_str: str) -> tuple:
        """
        添加自动预测时间点

        Args:
            plan_id: 计划ID
            time_str: 时间字符串，格式 HH:MM

        Returns:
            (结果消息, 更新后的时间表列表)
        """
        try:
            # 验证时间格式
            import re
            if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', time_str.strip()):
                return "❌ 时间格式错误，请使用HH:MM格式（例如：06:00, 18:30）", []

            time_str = time_str.strip()

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在", []

                # 为了兼容性，将时间点转换为间隔时间
                # 解析HH:MM，转换为小时数作为间隔
                try:
                    if ':' in time_str:
                        hour, minute = time_str.split(':')
                        interval_hours = int(hour) + 1  # 简单转换：时间点+1小时作为间隔
                    else:
                        interval_hours = 4  # 默认4小时
                except:
                    interval_hours = 4  # 出错时使用默认值

                # 验证间隔时间
                if interval_hours <= 0:
                    interval_hours = 4
                elif interval_hours > 168:  # 7天
                    interval_hours = 168

                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'auto_inference_interval_hours': interval_hours,
                    'auto_inference_enabled': True  # 设置间隔时间时自动启用
                })
                db.commit()

                logger.info(f"已设置自动预测间隔时间: plan_id={plan_id}, interval={interval_hours}小时")
                return f"✅ 已设置为{interval_hours}小时间隔", [interval_hours]

        except Exception as e:
            logger.error(f"添加自动预测时间点失败: {e}")
            return f"❌ 添加失败: {str(e)}", []

    def remove_inference_schedule_time(self, plan_id: int, time_str: str) -> tuple:
        """
        删除自动预测时间点

        Args:
            plan_id: 计划ID
            time_str: 时间字符串，格式 HH:MM

        Returns:
            (结果消息, 更新后的时间表列表)
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在", []

                # 兼容性方法：重置为默认4小时间隔
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'auto_inference_interval_hours': 4,
                    'auto_inference_enabled': False  # 重置时禁用
                })
                db.commit()

                logger.info(f"已重置自动预测间隔时间: plan_id={plan_id}, interval=4小时")
                return f"✅ 已重置为4小时间隔", [4]

        except Exception as e:
            logger.error(f"删除自动预测时间点失败: {e}")
            return f"❌ 删除失败: {str(e)}", []

    def get_data_date_range(self, inst_id: str, interval: str):
        """
        获取K线数据的日期范围

        Returns:
            (min_date, max_date, count)
        """
        try:
            with get_db() as db:
                result = db.query(
                    func.min(KlineData.timestamp).label('min_date'),
                    func.max(KlineData.timestamp).label('max_date'),
                    func.count(KlineData.id).label('count')
                ).filter(
                    and_(
                        KlineData.inst_id == inst_id,
                        KlineData.interval == interval
                    )
                ).first()

                if result and result.min_date and result.max_date:
                    return result.min_date, result.max_date, result.count
                else:
                    return None, None, 0

        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            return None, None, 0

    def save_training_data_config(self, plan_id: int, start_date_str: str, end_date_str: str) -> tuple:
        """
        保存训练数据范围配置

        Args:
            plan_id: 计划ID
            start_date_str: 开始日期字符串 (YYYY-MM-DD)
            end_date_str: 结束日期字符串 (YYYY-MM-DD)

        Returns:
            (结果消息, 数据统计信息)
        """
        try:
            from datetime import datetime

            # 解析日期
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            if start_date >= end_date:
                return "❌ 开始日期必须早于结束日期", ""

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在", ""

                # 统计数据量
                data_count = db.query(func.count(KlineData.id)).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= start_date,
                        KlineData.timestamp <= end_date
                    )
                ).scalar()

                if data_count == 0:
                    return f"⚠️ 该时间范围内没有数据", ""

                # 获取现有的 finetune_params
                finetune_params = plan.finetune_params or {}
                if 'data' not in finetune_params:
                    finetune_params['data'] = {}

                # 更新数据范围配置
                finetune_params['data']['train_start_date'] = start_date_str
                finetune_params['data']['train_end_date'] = end_date_str

                # 保存到数据库
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'finetune_params': finetune_params,
                    'data_start_time': start_date,
                    'data_end_time': end_date
                })
                db.commit()

                logger.info(f"训练数据配置已保存: plan_id={plan_id}, range={start_date_str} to {end_date_str}, count={data_count}")

                # 构建统计信息
                stats_info = f"""**已配置训练数据范围**

📅 **日期**: {start_date_str} 至 {end_date_str}
📊 **数据点**: {data_count} 条
✅ **状态**: 配置已保存
"""
                return "✅ 训练数据配置已保存", stats_info

        except ValueError as e:
            return f"❌ 日期格式错误: {str(e)}", ""
        except Exception as e:
            logger.error(f"保存训练数据配置失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 保存失败: {str(e)}", ""

    def get_training_data_stats(self, plan_id: int) -> str:
        """
        获取训练数据统计信息

        Args:
            plan_id: 计划ID

        Returns:
            str - Markdown 格式的统计信息
        """
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "**数据统计**: 计划不存在"

                # 获取数据库中的实际范围
                min_date, max_date, total_count = self.get_data_date_range(plan.inst_id, plan.interval)

                if min_date is None or max_date is None:
                    return "**数据统计**: 暂无数据"

                # 从 finetune_params 中获取配置的范围
                finetune_params = plan.finetune_params or {}
                data_config = finetune_params.get('data', {})
                train_start_date_str = data_config.get('train_start_date')
                train_end_date_str = data_config.get('train_end_date')

                # 自动更新训练数据范围到最新数据
                from datetime import datetime, timedelta
                if train_start_date_str and train_end_date_str:
                    try:
                        configured_start = datetime.strptime(train_start_date_str, '%Y-%m-%d')
                        configured_end = datetime.strptime(train_end_date_str, '%Y-%m-%d')

                        # 如果配置的结束日期早于最新数据日期，自动更新
                        if configured_end.date() < max_date.date():
                            train_end_date_str = max_date.strftime('%Y-%m-%d')
                            # 确保开始日期不会过晚
                            if configured_start.date() >= max_date.date():
                                train_start_date_str = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                    except ValueError:
                        # 如果日期解析失败，使用默认的最近30天
                        train_start_date_str = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                        train_end_date_str = max_date.strftime('%Y-%m-%d')
                else:
                    # 如果没有配置，使用最近30天
                    train_start_date_str = (max_date - timedelta(days=30)).strftime('%Y-%m-%d')
                    train_end_date_str = max_date.strftime('%Y-%m-%d')

                # 统计更新后的训练数据量
                train_start = datetime.strptime(train_start_date_str, '%Y-%m-%d')
                train_end = datetime.strptime(train_end_date_str, '%Y-%m-%d')

                train_data_count = db.query(func.count(KlineData.id)).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= train_start,
                        KlineData.timestamp <= train_end
                    )
                ).scalar()

                return f"""**训练数据统计**

📅 **当前训练范围**: {train_start_date_str} ~ {train_end_date_str}
📊 **训练数据点**: {train_data_count} 条

---

📅 **全部可用数据**: {format_datetime_beijing(min_date, '%Y-%m-%d')} ~ {format_datetime_beijing(max_date, '%Y-%m-%d')}
📊 **总数据点**: {total_count} 条

✅ **自动更新至最新数据范围**
"""

        except Exception as e:
            logger.error(f"获取训练数据统计失败: {e}")
            import traceback
            traceback.print_exc()
            return f"**数据统计**: 获取失败 - {str(e)}"

    def get_probability_indicators(self, plan_id: int) -> str:
        """
        获取最新预测的概率指标（上涨概率、波动性放大概率）

        Args:
            plan_id: 计划ID

        Returns:
            str - Markdown 格式的概率指标展示
        """
        try:
            with get_db() as db:
                # 获取最新的已完成训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    return """
### 📊 概率指标

暂无数据（尚未完成推理）
"""

                # 获取该训练记录的最新一条预测数据（概率指标对所有时间点相同）
                prediction = db.query(PredictionData).filter(
                    PredictionData.training_record_id == latest_training.id
                ).order_by(PredictionData.timestamp.desc()).first()

                if not prediction:
                    return """
### 📊 概率指标

暂无数据（尚未完成推理）
"""

                # 获取概率指标
                upward_prob = prediction.upward_probability
                volatility_amp_prob = prediction.volatility_amplification_probability

                # 如果没有概率数据（旧版本推理结果）
                if upward_prob is None or volatility_amp_prob is None:
                    return """
### 📊 概率指标

⚠️ **当前预测数据不包含概率指标**

请重新执行推理以获取最新的概率指标（需要多路径蒙特卡罗采样）
"""

                # 格式化显示
                upward_percent = upward_prob * 100
                volatility_percent = volatility_amp_prob * 100

                # 根据概率值选择表情和颜色
                if upward_percent >= 60:
                    upward_emoji = "📈"
                    upward_color = "green"
                elif upward_percent >= 40:
                    upward_emoji = "➡️"
                    upward_color = "orange"
                else:
                    upward_emoji = "📉"
                    upward_color = "red"

                if volatility_percent >= 60:
                    volatility_emoji = "⚡"
                    volatility_color = "red"
                elif volatility_percent >= 40:
                    volatility_emoji = "〰️"
                    volatility_color = "orange"
                else:
                    volatility_emoji = "😴"
                    volatility_color = "green"

                return f"""
### 📊 概率指标

<div style="display: flex; gap: 20px; margin: 10px 0;">
  <div style="flex: 1; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <div style="font-size: 14px; opacity: 0.9;">上涨概率（未来预测期）</div>
    <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">{upward_emoji} {upward_percent:.1f}%</div>
    <div style="font-size: 12px; opacity: 0.8;">模型对价格上涨的置信度</div>
  </div>
  <div style="flex: 1; padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white;">
    <div style="font-size: 14px; opacity: 0.9;">波动性放大</div>
    <div style="font-size: 36px; font-weight: bold; margin: 10px 0;">{volatility_emoji} {volatility_percent:.1f}%</div>
    <div style="font-size: 12px; opacity: 0.8;">未来波动率超过历史的概率</div>
  </div>
</div>

**数据来源**: 训练版本 v{latest_training.id} | 基于 {prediction.inference_params.get('sample_count', 1)} 条蒙特卡罗路径
"""

        except Exception as e:
            logger.error(f"获取概率指标失败: {e}")
            import traceback
            traceback.print_exc()
            return f"""
### 📊 概率指标

❌ 获取失败: {str(e)}
"""

    def set_training_date_range(self, inst_id: str, interval: str, days: int):
        """
        设置训练数据范围（快捷按钮）

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            days: 最近多少天

        Returns:
            (数据范围信息, 开始日期, 结束日期)
        """
        try:
            from datetime import timedelta
            min_date, max_date, count = self.get_data_date_range(inst_id, interval)

            if min_date is None or max_date is None:
                return (
                    "⚠️ **数据范围**: 未找到数据",
                    "",
                    ""
                )

            # 计算开始日期（从最新日期往前推N天）
            start_date = max_date - timedelta(days=days)

            # 确保开始日期不早于数据最早日期
            if start_date < min_date:
                start_date = min_date

            info = f"""
**数据范围**: {format_datetime_beijing(min_date, '%Y-%m-%d')} 至 {format_datetime_beijing(max_date, '%Y-%m-%d')} (共 {count} 条)

**已选择**: 最近 {days} 天 ({format_datetime_beijing(start_date, '%Y-%m-%d')} 至 {format_datetime_beijing(max_date, '%Y-%m-%d')})
"""

            return (
                info,
                format_datetime_beijing(start_date, '%Y-%m-%d'),
                format_datetime_beijing(max_date, '%Y-%m-%d')
            )

        except Exception as e:
            logger.error(f"设置训练日期范围失败: {e}")
            return (
                f"❌ 设置失败: {str(e)}",
                "",
                ""
            )

    async def manual_inference_async(self, plan_id: int) -> str:
        """手动执行AI Agent推理"""
        try:
            from services.agent_decision_service import AgentDecisionService
            from database.models import LLMConfig, AgentDecision
            import json

            # 获取计划配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 检查LLM配置
                if not plan.llm_config_id:
                    return "❌ 未配置LLM，请先在Agent配置中选择LLM"

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    return "❌ LLM配置不存在"

            # 获取最新的训练记录
            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

                if not latest_training:
                    return "❌ 没有可用的训练记录，请先完成模型训练"

            # 触发AI Agent决策
            decision_id = await AgentDecisionService.trigger_decision(plan_id, latest_training.id)

            if not decision_id:
                return "❌ AI Agent决策触发失败，请查看日志"

            # 获取决策结果
            with get_db() as db:
                decision = db.query(AgentDecision).filter(
                    AgentDecision.id == decision_id
                ).first()

                if not decision:
                    return "❌ 无法获取决策结果"

            # 构建返回结果
            result_md = f"""## ✅ AI Agent 推理完成

**决策时间**: {format_datetime_full_beijing(decision.decision_time)}

**LLM**: {llm_config.provider} / {llm_config.model_name}

**训练版本**: v{latest_training.version}

---

### 💭 AI分析与推理

{decision.reasoning or '无'}

---

### 🛠️ 工具调用

"""
            tool_calls = decision.tool_calls or []
            if tool_calls:
                for i, call in enumerate(tool_calls, 1):
                    result_md += f"**{i}. {call.get('name', 'unknown')}**\n"
                    result_md += f"   - 参数: `{call.get('arguments', {})}`\n"

                    # 显示执行结果
                    if decision.tool_results and len(decision.tool_results) >= i:
                        result = decision.tool_results[i-1]
                        success = result.get('success', False)
                        status_emoji = '✅' if success else '❌'
                        result_md += f"   - 结果: {status_emoji} {result.get('message', result.get('error', 'N/A'))}\n"
                    result_md += "\n"
            else:
                result_md += "无工具调用\n"

            return result_md

        except Exception as e:
            logger.error(f"手动推理失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 推理失败: {str(e)}"

    async def manual_inference_stream(self, plan_id: int):
        """手动执行AI Agent推理（流式输出），使用新的LangChain Agent v2服务"""
        try:
            from services.langchain_agent import agent_service

            # 使用Agent服务进行流式推理
            async for message_batch in agent_service.stream_conversation(
                plan_id=plan_id,
                user_message="请根据最新数据进行分析和决策",
                conversation_type="auto_inference"
            ):
                yield message_batch

        except Exception as e:
            logger.error(f"手动推理失败: {e}")
            import traceback
            traceback.print_exc()
            yield [{"role": "assistant", "content": f"❌ 推理失败: {str(e)}"}]

    async def chat_with_agent_stream(self, plan_id: int, user_message: str):
        """与AI Agent进行流式对话"""
        try:
            from services.langchain_agent import agent_service

            # 使用Agent服务进行流式对话
            async for message_batch in agent_service.stream_conversation(
                plan_id=plan_id,
                user_message=user_message
            ):
                yield message_batch

        except Exception as e:
            logger.error(f"Agent对话失败: {e}")
            import traceback
            traceback.print_exc()
            yield [{"role": "assistant", "content": f"❌ 对话失败: {str(e)}"}]

    def _build_inference_system_prompt(self, plan, latest_training) -> str:
        """构建推理系统提示词"""
        try:
            # 导入工具获取函数
            from services.agent_tools import get_all_tools

            # 获取预测数据
            prediction_data = []
            with get_db() as db:
                if latest_training:
                    prediction_data = db.query(PredictionData).filter(
                        PredictionData.training_record_id == latest_training.id
                    ).order_by(PredictionData.timestamp.desc()).limit(10).all()

            # 基础系统提示
            system_prompt = f"""你是一个专业的加密货币交易AI助手，负责分析市场数据并做出交易决策。

**交易计划信息**:
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 环境: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}
- 计划状态: {plan.status}

**推理任务**:
基于Kronos模型的预测数据，使用ReAct模式进行思考和决策：
1. **思考** (Thought): 分析市场状况和预测数据
2. **行动** (Action): 调用工具获取更多信息或执行交易
3. **观察** (Observation): 分析工具返回的结果
4. **重复** 直到得出最终结论

**可用工具**:
你可以调用以下工具来获取信息和执行操作："""

            # 添加工具说明
            tools_config = plan.agent_tools_config or {}
            for tool_name, tool_obj in get_all_tools().items():
                if tools_config.get(tool_name, False):
                    description = tool_obj.description
                    system_prompt += f"- {tool_name}: {description}\n"

            # 添加预测数据信息
            if prediction_data:
                latest_prediction = prediction_data[0]
                system_prompt += f"""

**最新预测数据概览**:
- 当前价格: ${latest_prediction.close or 0:.4f}
- 预测区间: ${latest_prediction.close_min or 0:.4f} ~ ${latest_prediction.close_max or 0:.4f}
- 上涨概率: {latest_prediction.upward_probability or 0:.2%}
- 波动放大概率: {latest_prediction.volatility_amplification_probability or 0:.2%}
- 模型版本: {latest_training.version if latest_training else 'N/A'}"""

            # 添加自定义提示词
            if plan.agent_prompt:
                system_prompt += f"""

**额外指示**:
{plan.agent_prompt}"""

            system_prompt += """

现在请基于用户提供的最新预测数据进行详细分析。"""

            return system_prompt

        except Exception as e:
            logger.error(f"构建系统提示词失败: {e}")
            return "系统提示词构建失败，请检查配置。"

    def _get_latest_prediction_data_text(self, training_record_id: int) -> str:
        """获取最新预测数据的文本格式"""
        try:
            with get_db() as db:
                # 获取最新的预测数据
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_record_id
                ).order_by(PredictionData.timestamp.desc()).first()

                if not latest_prediction:
                    return None

                # 安全处理数据
                current_price = latest_prediction.close or 0
                upward_prob = latest_prediction.upward_probability or 0
                volatility_prob = latest_prediction.volatility_amplification_probability or 0
                close_min = latest_prediction.close_min or 0
                close_max = latest_prediction.close_max or 0

                # 预测趋势判断
                trend = '未知'
                if close_max > current_price:
                    trend = '📈 上涨'
                elif close_max < current_price:
                    trend = '📉 下跌'
                else:
                    trend = '➡️ 横盘'

                return f"""**当前价格**: ${current_price:.4f}
**预测趋势**: {trend}
**价格区间**: ${close_min:.4f} ~ ${close_max:.4f}
**上涨概率**: {upward_prob:.2%}
**波动放大概率**: {volatility_prob:.2%}
**预测时间**: {latest_prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""

        except Exception as e:
            logger.error(f"获取预测数据文本失败: {e}")
            return None

    # continue_inference_stream方法已移除 - 工具确认功能已废弃，AI Agent现在直接使用启用的工具

    async def _get_prediction_preview(self, plan_id: int, training_id: int) -> str:
        """获取预测数据预览"""
        try:
            from database.models import PredictionData
            from database.db import get_db
            from sqlalchemy import desc

            with get_db() as db:
                # 获取最新的预测数据
                latest_prediction = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                ).order_by(desc(PredictionData.timestamp)).first()

                if not latest_prediction:
                    return "⚠️ 暂无预测数据"

                # 安全处理概率值的格式化
                upward_prob = latest_prediction.upward_probability
                volatility_prob = latest_prediction.volatility_amplification_probability

                upward_prob_str = f"{upward_prob:.2%}" if upward_prob is not None else "N/A"
                volatility_prob_str = f"{volatility_prob:.2%}" if volatility_prob is not None else "N/A"

                # 安全处理价格区间的格式化
                close_min = latest_prediction.close_min
                close_max = latest_prediction.close_max
                current_price = latest_prediction.close

                close_min_str = f"${close_min:.4f}" if close_min is not None else "N/A"
                close_max_str = f"${close_max:.4f}" if close_max is not None else "N/A"
                current_price_str = f"${current_price:.4f}" if current_price is not None else "N/A"

                # 预测趋势判断
                trend = '未知'
                if close_max is not None and current_price is not None:
                    trend = '📈 上涨' if close_max > current_price else '📉 下跌' if close_max < current_price else '➡️ 横盘'

                return f"""📊 **最新预测数据**:
- **当前价格**: {current_price_str}
- **预测趋势**: {trend}
- **价格区间**: {close_min_str} ~ {close_max_str}
- **上涨概率**: {upward_prob_str}
- **波动放大概率**: {volatility_prob_str}
- **预测时间**: {latest_prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""

        except Exception as e:
            logger.error(f"获取预测预览失败: {e}")
            return "⚠️ 无法获取预测数据"

    def _format_tool_arguments(self, tool_args: dict) -> str:
        """格式化工具参数"""
        if not tool_args:
            return "- 无参数"

        formatted_lines = []
        for key, value in tool_args.items():
            if isinstance(value, dict):
                value_str = json.dumps(value, ensure_ascii=False, indent=2)
                formatted_lines.append(f"- `{key}`: {value_str}")
            else:
                formatted_lines.append(f"- `{key}`: `{value}`")

        return "\n".join(formatted_lines)

    def _format_tool_result(self, tool_result: dict) -> str:
        """格式化工具执行结果"""
        if not tool_result:
            return "- 无返回数据"

        # 检查是否成功
        success = tool_result.get('success', False)
        status_emoji = "✅ 成功" if success else "❌ 失败"

        result_lines = [f"**状态**: {status_emoji}"]

        # 添加消息
        if tool_result.get('message'):
            result_lines.append(f"**说明**: {tool_result['message']}")

        # 添加数据统计
        data = tool_result.get('data')
        if data:
            if isinstance(data, dict):
                if 'count' in data:
                    result_lines.append(f"**数据量**: {data['count']} 条")
                if 'total' in data:
                    result_lines.append(f"**总数**: {data['total']}")
                if 'records' in data and isinstance(data['records'], list):
                    result_lines.append(f"**记录数**: {len(data['records'])} 条")
                    # 显示前几条数据示例
                    if len(data['records']) > 0:
                        sample_record = data['records'][0]
                        result_lines.append(f"**数据示例**: {str(sample_record)[:150]}...")
            elif isinstance(data, list):
                result_lines.append(f"**数组长度**: {len(data)} 项")
                if len(data) > 0:
                    result_lines.append(f"**示例数据**: {str(data[0])[:150]}...")

        # 添加错误信息
        if tool_result.get('error'):
            result_lines.append(f"**错误信息**: {tool_result['error']}")

        return "\n".join(result_lines) if len(result_lines) > 1 else result_lines[0]

    def _build_system_message(self, plan):
        """构建系统消息"""
        if plan.agent_prompt:
            return plan.agent_prompt

        return f"""你是一个专业的加密货币交易AI助手，负责分析市场数据并做出交易决策。

**交易对**: {plan.inst_id}
**时间周期**: {plan.interval}
**环境**: {'模拟盘' if plan.is_demo else '实盘'}

你的任务是：
1. 分析Kronos模型的价格预测数据
2. 结合当前账户状态和持仓情况
3. 做出合理的交易决策（买入/卖出/持有）
4. 必要时调用工具执行交易操作

请始终谨慎决策，控制风险。
"""

    def _build_user_message_with_prediction(self, plan, predictions):
        """构建包含预测数据的用户消息"""
        if not predictions or len(predictions) == 0:
            return "⚠️ 无预测数据可用，请先完成模型训练和推理。"

        # 计算趋势
        first_close = predictions[0]['close']
        last_close = predictions[-1]['close']
        change_pct = ((last_close - first_close) / first_close) * 100

        # 提取关键数据点
        first_5 = predictions[:5] if len(predictions) >= 5 else predictions
        last_5 = predictions[-5:] if len(predictions) >= 5 else []

        # 时间范围
        first_time = format_datetime_beijing(predictions[0]['timestamp'], '%Y-%m-%d %H:%M') if hasattr(predictions[0]['timestamp'], 'strftime') else str(predictions[0]['timestamp'])[:16]
        last_time = format_datetime_beijing(predictions[-1]['timestamp'], '%Y-%m-%d %H:%M') if hasattr(predictions[-1]['timestamp'], 'strftime') else str(predictions[-1]['timestamp'])[:16]

        # 价格统计
        close_prices = [p['close'] for p in predictions]
        min_close = min(close_prices)
        max_close = max(close_prices)

        pred_summary = f"""**预测时长**: 未来 {len(predictions)} 个周期

**时间范围**: {first_time} ~ {last_time}

**当前价格**: ${first_close:.4f}

**预测价格**: ${last_close:.4f}

**价格区间**: ${min_close:.4f} ~ ${max_close:.4f}

**预测涨跌**: {change_pct:+.2f}%

**趋势判断**: {'📈 上涨趋势' if change_pct > 0 else '📉 下跌趋势' if change_pct < 0 else '➡️ 横盘'}
"""

        message = f"""## 📊 Kronos模型预测分析

{pred_summary}

### 预测数据（前5个周期）

"""
        for i, p in enumerate(first_5, 1):
            timestamp_str = format_datetime_beijing(p['timestamp'], '%Y-%m-%d %H:%M') if hasattr(p['timestamp'], 'strftime') else str(p['timestamp'])[:19]
            message += f"{i}. **{timestamp_str}** - 开: ${p['open']:.2f}, 高: ${p['high']:.2f}, 低: ${p['low']:.2f}, 收: ${p['close']:.2f}\n"

        if last_5:
            message += "\n...\n\n### 预测数据（后5个周期）\n\n"

            for i, p in enumerate(last_5, 1):
                timestamp_str = format_datetime_beijing(p['timestamp'], '%Y-%m-%d %H:%M') if hasattr(p['timestamp'], 'strftime') else str(p['timestamp'])[:19]
                message += f"{i}. **{timestamp_str}** - 开: ${p['open']:.2f}, 高: ${p['high']:.2f}, 低: ${p['low']:.2f}, 收: ${p['close']:.2f}\n"

        message += f"""

## 🎯 请执行以下任务

1. 分析预测趋势和关键价格水平
2. 查询当前账户余额和持仓情况
3. 基于预测和账户状态，制定交易策略
4. 如果合适，执行交易操作

请提供详细的分析和决策理由。
"""

        return message

    def _get_tool_definitions(self, plan):
        """获取工具定义"""
        tools_config = plan.agent_tools_config or {}
        tools = []

        # 1. 🔮 query_prediction_data - 按时间范围和批次ID查询预测数据
        if tools_config.get('query_prediction_data', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "query_prediction_data",
                    "description": "按时间范围和批次ID查询预测数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_time": {"type": "string", "description": "开始时间 (YYYY-MM-DD HH:MM:SS)"},
                            "end_time": {"type": "string", "description": "结束时间 (YYYY-MM-DD HH:MM:SS)"},
                            "batch_id": {"type": "string", "description": "推理批次ID"}
                        },
                        "required": []
                    }
                }
            })

        # 2. 📈 get_prediction_history - 查询历史预测批次列表（最多30批次）
        if tools_config.get('get_prediction_history', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_prediction_history",
                    "description": "查询历史预测批次列表（最多30批次）",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        # 3. 📈 query_historical_kline_data - 查询历史K线数据（UTC+8时间戳）
        if tools_config.get('query_historical_kline_data', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "query_historical_kline_data",
                    "description": "查询历史K线数据（UTC+8时间戳）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_time": {"type": "string", "description": "开始时间 (YYYY-MM-DD HH:MM:SS)"},
                            "end_time": {"type": "string", "description": "结束时间 (YYYY-MM-DD HH:MM:SS)"},
                            "limit": {"type": "integer", "description": "数据条数限制"}
                        },
                        "required": []
                    }
                }
            })

        # 4. 🕒 get_current_utc_time - 获取当前UTC+8时间
        if tools_config.get('get_current_utc_time', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_current_utc_time",
                    "description": "获取当前UTC+8时间",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        # 5. 🤖 run_latest_model_inference - 触发最新模型推理
        if tools_config.get('run_latest_model_inference', False):
            tools.append({
                "type": "function",
                "function": {
                    "name": "run_latest_model_inference",
                    "description": "触发最新模型推理",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        # 6. 🔍 get_account_balance - 查询账户余额
        if tools_config.get('get_account_balance', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "description": "查询账户余额",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        # 7. 📋 get_pending_orders - 查询挂单
        if tools_config.get('get_pending_orders', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_pending_orders",
                    "description": "查询挂单",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        # 8. 💰 place_order - 下限价单
        if tools_config.get('place_order', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "place_order",
                    "description": "下限价单",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                            "size": {"type": "number"},
                            "price": {"type": "number"}
                        },
                        "required": ["side", "size"]
                    }
                }
            })

        # 9. ❌ cancel_order - 撤单
        if tools_config.get('cancel_order', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "cancel_order",
                    "description": "撤单",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "订单ID"}
                        },
                        "required": ["order_id"]
                    }
                }
            })

        # 10. ✏️ amend_order - 改单
        if tools_config.get('amend_order', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "amend_order",
                    "description": "改单",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "订单ID"},
                            "new_size": {"type": "number", "description": "新数量"},
                            "new_price": {"type": "number", "description": "新价格"}
                        },
                        "required": ["order_id"]
                    }
                }
            })

        return tools if tools else None

    def get_prediction_text(self, training_id: int) -> str:
        """
        获取预测数据的文本格式（供AI Agent使用）

        Args:
            training_id: 训练记录ID

        Returns:
            预测数据的文本描述
        """
        try:
            predictions = InferenceService.get_prediction_data(training_id)

            if not predictions or len(predictions) == 0:
                return "⚠️ 暂无预测数据，请先执行\"预测交易数据\"或\"Mock预测\""

            # 获取概率指标（从数据库）
            upward_prob = None
            volatility_amp_prob = None
            sample_count = 1

            with get_db() as db:
                pred = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                ).order_by(PredictionData.timestamp.desc()).first()

                if pred:
                    upward_prob = pred.upward_probability
                    volatility_amp_prob = pred.volatility_amplification_probability
                    if pred.inference_params:
                        sample_count = pred.inference_params.get('sample_count', 1)

            # 构建文本格式
            text_lines = []
            text_lines.append(f"📊 预测数据统计")
            text_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            text_lines.append(f"预测数据条数: {len(predictions)}条")

            # 显示概率指标（如果有）
            if upward_prob is not None and volatility_amp_prob is not None:
                text_lines.append(f"蒙特卡罗路径: {sample_count}条")
                text_lines.append(f"")
                text_lines.append(f"📊 概率指标")
                text_lines.append(f"  • 上涨概率: {upward_prob*100:.1f}%")
                text_lines.append(f"  • 波动性放大概率: {volatility_amp_prob*100:.1f}%")

            first_pred = predictions[0]
            last_pred = predictions[-1]

            # 时间范围
            first_time = format_datetime_beijing(first_pred['timestamp'], '%Y-%m-%d %H:%M') if hasattr(first_pred['timestamp'], 'strftime') else str(first_pred['timestamp'])[:16]
            last_time = format_datetime_beijing(last_pred['timestamp'], '%Y-%m-%d %H:%M') if hasattr(last_pred['timestamp'], 'strftime') else str(last_pred['timestamp'])[:16]
            text_lines.append(f"")
            text_lines.append(f"时间范围: {first_time} ~ {last_time}")

            # 价格统计
            close_prices = [p['close'] for p in predictions]
            min_close = min(close_prices)
            max_close = max(close_prices)
            first_close = close_prices[0]
            last_close = close_prices[-1]

            text_lines.append(f"")
            text_lines.append(f"价格区间: ${min_close:.2f} ~ ${max_close:.2f}")
            text_lines.append(f"起始价格: ${first_close:.2f}")
            text_lines.append(f"结束价格: ${last_close:.2f}")

            # 趋势判断
            change_pct = ((last_close - first_close) / first_close) * 100
            trend = "📈 上涨趋势" if change_pct > 0 else "📉 下跌趋势" if change_pct < 0 else "➡️ 横盘"
            text_lines.append(f"预测涨跌: {change_pct:+.2f}%")
            text_lines.append(f"趋势判断: {trend}")

            text_lines.append(f"")
            text_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
            text_lines.append(f"📋 详细预测数据 (前10条)")
            text_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")

            # 显示前10条详细数据
            for i, pred in enumerate(predictions[:10], 1):
                timestamp_str = format_datetime_short_beijing(pred['timestamp']) if hasattr(pred['timestamp'], 'strftime') else str(pred['timestamp'])[:16]
                text_lines.append(
                    f"{i:2d}. {timestamp_str} | "
                    f"开: ${pred['open']:7.2f} | 高: ${pred['high']:7.2f} | "
                    f"低: ${pred['low']:7.2f} | 收: ${pred['close']:7.2f}"
                )

            if len(predictions) > 10:
                text_lines.append(f"... (共{len(predictions)}条，仅显示前10条)")

            return "\n".join(text_lines)

        except Exception as e:
            logger.error(f"获取预测数据文本失败: {e}")
            return f"❌ 获取预测数据失败: {str(e)}"

    async def execute_inference_async(self, training_id: int) -> str:
        """执行Kronos模型推理"""
        try:
            success = await InferenceService.start_inference(training_id)
            if success:
                return f"✅ 推理已完成，训练记录ID: {training_id}"
            else:
                return f"❌ 推理失败，请查看日志"
        except Exception as e:
            logger.error(f"执行推理失败: {e}")
            return f"❌ 推理失败: {str(e)}"

    async def mock_predictions_async(self, training_id: int) -> str:
        """生成Mock预测数据"""
        try:
            result = InferenceService.mock_prediction_data(training_id, predict_window=48)
            if result['success']:
                return f"✅ Mock数据已生成，共 {result['predictions_count']} 条"
            else:
                return f"❌ 生成失败: {result.get('error', '未知错误')}"
        except Exception as e:
            logger.error(f"生成Mock数据失败: {e}")
            return f"❌ 生成失败: {str(e)}"

    async def start_training_async(self, plan_id: int, train_start_date: str = None, train_end_date: str = None):
        """
        开始训练（异步）- 使用生成器实时返回进度

        Args:
            plan_id: 计划ID
            train_start_date: 训练开始日期 (YYYY-MM-DD)，为空则使用计划配置
            train_end_date: 训练结束日期 (YYYY-MM-DD)，为空则使用计划配置

        Yields:
            训练进度消息
        """
        try:
            from datetime import datetime
            import time

            # 如果指定了日期范围，临时更新计划的数据时间范围
            if train_start_date and train_end_date:
                try:
                    start_dt = datetime.strptime(train_start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(train_end_date, '%Y-%m-%d')

                    # 临时更新计划的数据时间范围
                    with get_db() as db:
                        db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                            'data_start_time': start_dt,
                            'data_end_time': end_dt
                        })
                        db.commit()

                    logger.info(f"已更新训练数据范围: {train_start_date} 至 {train_end_date}")

                except ValueError as e:
                    yield f"❌ 日期格式错误: {str(e)}"
                    return

            # 启动训练
            training_id = await TrainingService.start_training(plan_id, manual=True)
            if not training_id:
                yield "❌ 训练启动失败"
                return

            yield f"✅ 训练已启动，记录ID: {training_id}\n\n开始训练..."

            # 轮询训练进度
            last_progress = -1
            max_wait_time = 3600  # 最多等待1小时
            start_time = time.time()

            while True:
                # 检查超时
                if time.time() - start_time > max_wait_time:
                    yield "\n\n⚠️ 训练超时（超过1小时）"
                    break

                # 获取进度
                progress_info = TrainingService.get_training_progress(training_id)

                if progress_info:
                    current_progress = progress_info['progress']
                    stage = progress_info['stage']
                    message = progress_info['message']

                    # 只在进度变化时更新
                    if abs(current_progress - last_progress) > 0.01:
                        progress_percent = int(current_progress * 100)
                        progress_bar = '█' * (progress_percent // 2) + '░' * (50 - progress_percent // 2)
                        yield f"\n\n**训练进度**: {progress_percent}%\n\n`{progress_bar}`\n\n**阶段**: {stage}\n\n**状态**: {message}"
                        last_progress = current_progress

                    # 检查是否完成
                    if stage == 'completed':
                        yield f"\n\n✅ 训练完成！记录ID: {training_id}"
                        break
                    elif stage == 'failed':
                        yield f"\n\n❌ 训练失败: {message}"
                        break

                # 检查训练记录状态
                with get_db() as db:
                    record = db.query(TrainingRecord).filter(
                        TrainingRecord.id == training_id
                    ).first()

                    if record:
                        if record.status == 'completed':
                            yield f"\n\n✅ 训练完成！\n\n- Tokenizer损失: {record.train_metrics.get('tokenizer_loss', 'N/A')}\n- Predictor损失: {record.train_metrics.get('predictor_loss', 'N/A')}"
                            break
                        elif record.status == 'failed':
                            yield f"\n\n❌ 训练失败: {record.error_message or '未知错误'}"
                            break

                # 等待一段时间再查询
                await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"训练监控失败: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n\n❌ 错误: {str(e)}"

    async def start_websocket_async(self, plan_id: int) -> str:
        """启动WebSocket（异步）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 使用全局连接管理器
                from services.ws_connection_manager import ws_connection_manager

                # 获取或创建WebSocket连接
                ws_service = ws_connection_manager.get_or_create_connection(
                    inst_id=plan.inst_id,
                    interval=plan.interval,
                    is_demo=plan.is_demo,
                    ui_callback=None
                )

                if ws_service:
                    # 订阅K线事件
                    from services.kline_event_service import get_kline_event_service
                    get_kline_event_service().subscribe_plan(plan_id)

                    # 更新计划的ws_connected状态
                    db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                        'ws_connected': True
                    })
                    db.commit()
                    return "✅ WebSocket已启动，已订阅K线事件"
                else:
                    return "❌ WebSocket启动失败"

        except Exception as e:
            logger.error(f"启动WebSocket失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 启动失败: {str(e)}"

    async def stop_websocket_async(self, plan_id: int) -> str:
        """停止WebSocket（异步）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 使用全局连接管理器
                from services.ws_connection_manager import ws_connection_manager

                # 停止WebSocket (stop_connection是同步方法)
                ws_connection_manager.stop_connection(
                    inst_id=plan.inst_id,
                    interval=plan.interval,
                    is_demo=plan.is_demo
                )

                # 取消订阅K线事件
                from services.kline_event_service import get_kline_event_service
                get_kline_event_service().unsubscribe_plan(plan_id)

                # 更新计划的ws_connected状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'ws_connected': False
                })
                db.commit()
                return "✅ WebSocket已停止，已取消订阅K线事件"

        except Exception as e:
            logger.error(f"停止WebSocket失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 停止失败: {str(e)}"

    async def start_plan_async(self, plan_id: int) -> str:
        """启动计划（启动定时任务）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 更新计划状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'status': 'running'
                })
                db.commit()

                # 启动定时任务调度器
                from services.schedule_service import ScheduleService
                success = await ScheduleService.start_schedule(plan_id)

                # 订阅K线事件
                from services.kline_event_service import get_kline_event_service
                get_kline_event_service().subscribe_plan(plan_id)

                # 启动自动化调度器（如果配置了自动化）
                automation_status = ""
                if plan.auto_finetune_enabled or plan.auto_inference_enabled or plan.auto_agent_enabled:  # auto_tool_execution_enabled已废弃
                    try:
                        from services.automation_service import automation_service
                        automation_service.start_scheduler()
                        automation_status = "\n🤖 自动化调度器已启动"
                        logger.info(f"自动化调度器已启动: plan_id={plan_id}")
                    except Exception as e:
                        logger.error(f"启动自动化调度器失败: {e}")
                        automation_status = f"\n⚠️ 自动化调度器启动失败: {str(e)}"

                # 启动账户WebSocket连接
                if plan.okx_api_key and plan.okx_secret_key and plan.okx_passphrase:
                    from services.account_ws_manager import account_ws_manager
                    account_ws_manager.get_or_create_connection(
                        api_key=plan.okx_api_key,
                        secret_key=plan.okx_secret_key,
                        passphrase=plan.okx_passphrase,
                        is_demo=plan.is_demo,
                        plan_id=plan_id
                    )
                    logger.info(f"账户WebSocket已启动: plan_id={plan_id}")

                logger.info(f"计划已启动: plan_id={plan_id}, schedule_success={success}")

                result_msg = "✅ 计划已启动"
                if success:
                    # 获取已创建的任务信息
                    jobs = ScheduleService.get_plan_jobs(plan_id)
                    job_info = f"已创建 {len(jobs)} 个定时任务"
                    result_msg += f"\n✅ {job_info}"
                else:
                    result_msg += "\n⚠️ 定时任务创建失败（可能未配置时间表）"

                result_msg += automation_status

                return result_msg

        except Exception as e:
            logger.error(f"启动计划失败: {e}")
            return f"❌ 启动失败: {str(e)}"

    async def stop_plan_async(self, plan_id: int) -> str:
        """停止计划（停止定时任务）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return "❌ 计划不存在"

                # 更新计划状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'status': 'stopped'
                })
                db.commit()

                # 停止账户WebSocket连接
                if plan.okx_api_key:
                    from services.account_ws_manager import account_ws_manager
                    account_ws_manager.stop_connection(
                        api_key=plan.okx_api_key,
                        is_demo=plan.is_demo,
                        plan_id=plan_id
                    )
                    logger.info(f"账户WebSocket已停止: plan_id={plan_id}")

                # 取消订阅K线事件
                from services.kline_event_service import get_kline_event_service
                get_kline_event_service().unsubscribe_plan(plan_id)

                # 停止定时任务调度器
                from services.schedule_service import ScheduleService
                success = await ScheduleService.stop_schedule(plan_id)

                logger.info(f"计划已停止: plan_id={plan_id}, schedule_success={success}")
                return "✅ 计划已停止\n✅ 所有定时任务已移除"

        except Exception as e:
            logger.error(f"停止计划失败: {e}")
            return f"❌ 停止失败: {str(e)}"

    def get_account_info(self, plan_id: int) -> str:
        """获取账户信息（Markdown格式）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.okx_api_key:
                    return "### 💰 账户信息\n\n未配置OKX API Key"

                from services.account_ws_manager import account_ws_manager

                # 获取连接状态
                status = account_ws_manager.get_connection_status(
                    api_key=plan.okx_api_key,
                    is_demo=plan.is_demo
                )

                if not status['connected']:
                    return f"### 💰 账户信息\n\n⚪ 未连接\n\n{'模拟盘' if plan.is_demo else '真实盘'}"

                # 获取账户数据
                account_info = account_ws_manager.get_account_info(
                    api_key=plan.okx_api_key,
                    is_demo=plan.is_demo
                )

                if not account_info:
                    return "### 💰 账户信息\n\n⚪ 暂无数据"

                balances = account_info.get('balances', {})
                positions = account_info.get('positions', [])
                last_update = account_info.get('last_update')

                # 构建Markdown
                lines = ["### 💰 账户信息\n"]
                lines.append(f"**环境**: {'🧪 模拟盘' if plan.is_demo else '💰 真实盘'}")
                lines.append(f"**状态**: 🟢 已连接")

                if last_update:
                    lines.append(f"**更新时间**: {format_datetime_beijing(last_update, '%H:%M:%S')}")

                lines.append("\n---\n")

                # 余额信息
                if balances:
                    lines.append("**账户余额**:\n")
                    for ccy, data in balances.items():
                        available = data.get('available', 0)
                        balance = data.get('balance', 0)
                        equity = data.get('equity', 0)
                        lines.append(f"- **{ccy}**: 可用 {available:.4f} | 余额 {balance:.4f} | 权益 {equity:.4f}")
                else:
                    lines.append("**账户余额**: 暂无数据")

                lines.append("\n---\n")

                # 持仓信息
                if positions:
                    lines.append(f"**持仓** ({len(positions)}个):\n")
                    for pos in positions[:5]:  # 只显示前5个
                        inst_id = pos.get('inst_id', 'N/A')
                        pos_qty = pos.get('pos', 0)
                        avg_price = pos.get('avg_price', 0)
                        upl = pos.get('upl', 0)
                        upl_ratio = pos.get('upl_ratio', 0)

                        upl_emoji = '📈' if upl >= 0 else '📉'
                        lines.append(
                            f"- {upl_emoji} **{inst_id}**: {pos_qty} @ {avg_price:.4f} | "
                            f"盈亏 {upl:+.2f} ({upl_ratio:+.2%})"
                        )
                else:
                    lines.append("**持仓**: 无")

                return "\n".join(lines)

        except Exception as e:
            logger.error(f"获取账户信息失败: {e}")
            import traceback
            traceback.print_exc()
            return f"### 💰 账户信息\n\n❌ 获取失败: {str(e)}"

    def get_orders_info(self, plan_id: int) -> pd.DataFrame:
        """获取订单记录（仅显示Agent操作的订单）"""
        try:
            with get_db() as db:
                # 从数据库获取仅Agent操作的订单
                from database.models import TradeOrder

                orders = db.query(TradeOrder).filter(
                    TradeOrder.plan_id == plan_id,
                    TradeOrder.is_from_agent == True
                ).order_by(TradeOrder.created_at.desc()).limit(50).all()

                if not orders:
                    return pd.DataFrame()

                # 构建DataFrame
                df_data = []
                for order in orders:
                    side_emoji = '🟢' if order.side == 'buy' else '🔴'
                    state_map = {
                        'live': '⏳ 未成交',
                        'partially_filled': '⏸️ 部分成交',
                        'filled': '✅ 完全成交',
                        'canceled': '❌ 已取消',
                        'mmp_canceled': '❌ MMP取消',
                        'failed': '❌ 失败'
                    }
                    state_emoji = state_map.get(order.status, f"❓ {order.status}")

                    # 转换时间戳
                    create_time = format_datetime_beijing(order.created_at, '%m-%d %H:%M:%S')
                    update_time = format_datetime_beijing(order.updated_at, '%m-%d %H:%M:%S')

                    df_data.append({
                        '订单ID': order.order_id[:10] + '...' if order.order_id else '本地订单',
                        '交易对': order.inst_id,
                        '方向': f"{side_emoji} {order.side}",
                        '类型': order.order_type,
                        '价格': f"{float(order.price):.4f}" if order.price else '市价',
                        '数量': f"{float(order.size):.4f}",
                        '已成交': f"{float(order.filled_size):.4f}",
                        '状态': state_emoji,
                        '创建时间': create_time,
                        '更新时间': update_time
                    })

                return safe_dataframe_from_data(df_data)

        except Exception as e:
            logger.error(f"获取订单信息失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def clear_agent_records(self, plan_id: int) -> str:
        """清除AI Agent对话记录"""
        try:
            if not plan_id:
                return "❌ 请先选择计划"

            with get_db() as db:
                # 先获取要删除的对话会话数量
                conversations = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).all()

                if not conversations:
                    return "✅ 没有找到需要清除的对话记录"

                # 获取会话ID列表
                conversation_ids = [conv.id for conv in conversations]

                # 删除所有相关的消息记录
                deleted_messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id.in_(conversation_ids)
                ).delete(synchronize_session=False)

                # 删除对话会话记录
                deleted_conversations = db.query(AgentConversation).filter(
                    AgentConversation.plan_id == plan_id
                ).delete()

                db.commit()

                logger.info(f"清除计划 {plan_id} 的 {deleted_conversations} 个对话会话和 {deleted_messages} 条消息记录")

                return f"✅ 已清除 {deleted_conversations} 个对话会话和 {deleted_messages} 条消息记录"

        except Exception as e:
            logger.error(f"清除对话记录失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 清除失败: {str(e)}"

    def load_task_executions(self, plan_id: int) -> pd.DataFrame:
        """加载任务执行记录"""
        try:
            from services.scheduler_service import scheduler_service

            # 获取任务历史
            task_history = scheduler_service.get_task_history(plan_id, limit=100)

            if not task_history:
                return pd.DataFrame(columns=[
                    'ID', '任务类型', '任务名称', '状态', '计划时间', '开始时间',
                    '完成时间', '执行时长(秒)', '触发方式', '进度(%)'
                ]).astype({
                    'ID': 'int',
                    '执行时长(秒)': 'int',
                    '进度(%)': 'int'
                })

            # 构建DataFrame
            df_data = []
            for task in task_history:
                df_data.append({
                    'ID': task['id'],
                    '任务类型': task['type_display'],
                    '任务名称': task['task_name'],
                    '状态': task['status_display'],
                    '计划时间': task['scheduled_time'] or '',
                    '开始时间': task['started_at'] or '',
                    '完成时间': task['completed_at'] or '',
                    '执行时长(秒)': task['duration_seconds'] or 0,  # 确保数字类型
                    '触发方式': task['trigger_type'],
                    '进度(%)': task['progress_percentage'] or 0  # 确保数字类型
                })

            # 使用安全的DataFrame创建函数
            return safe_dataframe_from_data(df_data)

        except Exception as e:
            logger.error(f"加载任务执行记录失败: {e}")
            return pd.DataFrame(columns=[
                    'ID', '任务类型', '任务名称', '状态', '计划时间', '开始时间',
                    '完成时间', '执行时长(秒)', '触发方式', '进度(%)'
                ]).astype({
                    'ID': 'int',
                    '执行时长(秒)': 'int',
                    '进度(%)': 'int'
                })
