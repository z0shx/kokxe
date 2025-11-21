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
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, AgentDecision, KlineData
from sqlalchemy import and_, desc, func
from utils.logger import setup_logger

logger = setup_logger(__name__, "plan_detail_ui.log")


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
        agent_time = latest_agent.decision_time.strftime('%Y-%m-%d %H:%M:%S') if latest_agent else "无记录"

        # 自动微调时间表
        schedule = plan.auto_finetune_schedule or []
        schedule_str = ', '.join(schedule) if schedule else '未配置'

        # 计划状态
        plan_status_emoji = {
            'created': '⚪ 已创建',
            'running': '🟢 运行中',
            'paused': '🟡 已暂停',
            'stopped': '🔴 已停止'
        }.get(plan.status, '❓ 未知')

        overview = f"""
# 📊 {plan.plan_name}

---

**交易对**: `{plan.inst_id}` | **时间颗粒度**: `{plan.interval}` | **环境**: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}

**计划状态**: {plan_status_emoji}

**最新模型版本**: `{training_version}` | **AI Agent最后运行**: {agent_time}

**自动微调时间**: {schedule_str}

**创建时间**: {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}

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
                            'last_sync_time': datetime.utcnow()
                        })
                        logger.info(f"计划WebSocket状态已更新: plan_id={plan_id}, ws_connected={ws_connected}")

                self._safe_db_update(update_plan_ws_status, plan_id)

        except Exception as e:
            logger.error(f"获取WebSocket状态失败: {e}")
            ws_connected = plan.ws_connected

        ws_status_text = "🟢 已连接" if ws_connected else "⚪ 未连接"
        ws_status_display = f"**WebSocket状态**: {ws_status_text}"

        # 计划状态
        plan_status_emoji = {
            'created': '⚪ 已创建',
            'running': '🟢 运行中',
            'paused': '🟡 已暂停',
            'stopped': '🔴 已停止'
        }.get(plan.status, '❓ 未知')
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
                    '创建时间': record['created_at'].strftime('%m-%d %H:%M')
                })

            return pd.DataFrame(df_data)

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

                        inference_time_str = first_pred.created_at.strftime('%m-%d %H:%M') if first_pred else ''

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

                # 将UTC时间转换为UTC+8（北京时间）显示
                timestamps_utc8 = []
                for k in klines:
                    ts = k.timestamp
                    # 如果是naive datetime，假设它是UTC
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    # 转换为UTC+8
                    ts_utc8 = ts + timedelta(hours=8)
                    timestamps_utc8.append(ts_utc8)

                # 添加真实K线
                fig.add_trace(go.Candlestick(
                    x=timestamps_utc8,
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
                                        f"{record.version} (推理: {batch.inference_time.strftime('%m-%d %H:%M')})",
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
                                        f"{record.version} (推理: {batch.inference_time.strftime('%m-%d %H:%M')})",
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
        # 预测数据转换为UTC+8
        pred_timestamps_utc8 = []
        for p in predictions:
            ts = p.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_utc8 = ts + timedelta(hours=8)
            pred_timestamps_utc8.append(ts_utc8)

        # 检查是否有不确定性数据
        has_uncertainty = any(p.close_min is not None and p.close_max is not None for p in predictions)

        if has_uncertainty:
            # 绘制不确定性阴影区域
            # 1. 上边界
            fig.add_trace(go.Scatter(
                x=pred_timestamps_utc8,
                y=[p.close_max if p.close_max is not None else p.close for p in predictions],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip',
                legendgroup=f'group_{training_id}'
            ))

            # 2. 下边界（填充阴影）
            fig.add_trace(go.Scatter(
                x=pred_timestamps_utc8,
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
        inference_time_str = inference_time.strftime('%m-%d %H:%M')

        # 3. 平均值线条
        fig.add_trace(go.Scatter(
            x=pred_timestamps_utc8,
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
        """加载Agent决策记录（右侧）"""
        try:
            with get_db() as db:
                decisions = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).order_by(desc(AgentDecision.decision_time)).limit(50).all()

                if not decisions:
                    return pd.DataFrame()

                df_data = []
                for decision in decisions:
                    status_emoji = {
                        'completed': '✅',
                        'failed': '❌',
                        'partial': '⚠️'
                    }.get(decision.status, '❓')

                    df_data.append({
                        'ID': decision.id,
                        '时间': decision.decision_time.strftime('%m-%d %H:%M:%S'),
                        '决策类型': decision.decision_type or 'N/A',
                        '状态': f"{status_emoji} {decision.status}",
                        '模型版本': f"v{decision.training_record_id}" if decision.training_record_id else 'N/A',
                        '工具调用': len(decision.tool_calls) if decision.tool_calls else 0
                    })

                return pd.DataFrame(df_data)

        except Exception as e:
            logger.error(f"加载Agent决策失败: {e}")
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
                inference_time_str = record['inference_time'].strftime('%m-%d %H:%M') if record['inference_time'] else 'N/A'

                df_data.append({
                    'ID': record['training_record_id'],
                    '版本': record['version'],
                    '推理时间': inference_time_str,
                    '预测数据': f"{has_pred_emoji} {record['predictions_count']}条",
                    '数据范围': record.get('date_range', 'N/A'),
                    '训练完成': record['train_end_time'].strftime('%m-%d %H:%M') if record['train_end_time'] else 'N/A'
                })

            return pd.DataFrame(df_data)

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

                # 格式化时间范围显示
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
**训练完成时间**: {latest_record.get('train_end_time').strftime('%Y-%m-%d %H:%M') if latest_record.get('train_end_time') else 'N/A'}
**预测数据条数**: {latest_record.get('predictions_count', 0)}条"""
            else:
                data_range_info = f"""**📊 Data输入信息**

**数据时间范围**: 暂无数据
**训练版本**: {latest_record.get('version', 'N/A')}
**训练完成时间**: {latest_record.get('train_end_time').strftime('%Y-%m-%d %H:%M') if latest_record.get('train_end_time') else 'N/A'}
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

**决策时间**: {decision.decision_time.strftime('%Y-%m-%d %H:%M:%S')}
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
        """格式化工具调用"""
        if not tool_calls:
            return "无工具调用"

        lines = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get('name', 'unknown')
            tool_args = call.get('arguments', {})
            result = tool_results[i-1] if tool_results and len(tool_results) >= i else {}

            lines.append(f"**{i}. {tool_name}**")
            lines.append(f"   - 参数: `{tool_args}`")
            lines.append(f"   - 结果: `{result}`")
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
                    decision_output = f"""## 🤖 AI Agent 最新推理结果

**决策时间**: {decision.decision_time.strftime('%Y-%m-%d %H:%M:%S')}
**决策类型**: {decision.decision_type or 'N/A'}
**状态**: {decision.status}
**使用模型**: v{decision.training_record_id} | **LLM**: {decision.llm_model or 'N/A'}

---

### 💭 AI分析与推理

{decision.reasoning or '无'}

---

### 🛠️ 工具调用

"""
                    # 格式化工具调用
                    if decision.tool_calls:
                        for i, call in enumerate(decision.tool_calls, 1):
                            tool_name = call.get('name', 'unknown')
                            tool_args = call.get('arguments', {})
                            decision_output += f"**{i}. {tool_name}**\n"
                            decision_output += f"   - 参数: `{tool_args}`\n"

                            # 显示执行结果
                            if decision.tool_results and len(decision.tool_results) >= i:
                                result = decision.tool_results[i-1]
                                success = result.get('success', False)
                                status_emoji = '✅' if success else '❌'
                                decision_output += f"   - 结果: {status_emoji} {result.get('message', result.get('error', 'N/A'))}\n"
                            decision_output += "\n"
                    else:
                        decision_output += "无工具调用\n"

                    output_parts.append(decision_output)
                else:
                    # 如果没有决策记录，只显示预测数据
                    if not output_parts:  # 如果也没有预测数据
                        output_parts.append("等待推理...\n\n暂无AI Agent决策记录")

                # 合并所有输出
                combined_output = "\n\n---\n\n".join(output_parts)

                # 返回 messages 格式
                return [{"role": "assistant", "content": combined_output}]

        except Exception as e:
            logger.error(f"获取最新Agent决策输出失败: {e}")
            import traceback
            traceback.print_exc()
            return [{"role": "assistant", "content": f"等待推理...\n\n❌ 获取失败: {str(e)}"}]

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
            first_time = first_pred['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(first_pred['timestamp'], 'strftime') else str(first_pred['timestamp'])[:16]
            last_time = last_pred['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(last_pred['timestamp'], 'strftime') else str(last_pred['timestamp'])[:16]

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
                timestamp_str = pred['timestamp'].strftime('%m-%d %H:%M') if hasattr(pred['timestamp'], 'strftime') else str(pred['timestamp'])[:16]
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

    def get_react_config(self, plan_id: int) -> dict:
        """获取ReAct配置"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    # 返回默认配置
                    return {
                        'max_iterations': 3,
                        'enable_thinking': True,
                        'tool_approval': False,
                        'thinking_style': '详细'
                    }

                react_config = plan.react_config or {}
                # 确保所有必需字段都有默认值
                return {
                    'max_iterations': int(react_config.get('max_iterations', 3)),
                    'enable_thinking': bool(react_config.get('enable_thinking', True)),
                    'tool_approval': bool(react_config.get('tool_approval', False)),
                    'thinking_style': react_config.get('thinking_style', '详细')
                }
        except Exception as e:
            logger.error(f"获取ReAct配置失败: {e}")
            return {
                'max_iterations': 3,
                'enable_thinking': True,
                'tool_approval': False,
                'thinking_style': '详细'
            }

    def save_react_config(self, plan_id: int, max_iterations: int, enable_thinking: bool,
                          tool_approval: bool, thinking_style: str) -> str:
        """保存ReAct配置"""
        try:
            react_config = {
                'max_iterations': max_iterations,
                'enable_thinking': enable_thinking,
                'tool_approval': tool_approval,
                'thinking_style': thinking_style
            }

            with get_db() as db:
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'react_config': react_config
                })
                db.commit()
                logger.info(f"ReAct配置已保存: plan_id={plan_id}, config={react_config}")
                return f"✅ ReAct配置已保存\n- 最大推理轮数: {max_iterations}\n- 思考过程显示: {'启用' if enable_thinking else '禁用'}\n- 工具审批: {'启用' if tool_approval else '禁用'}\n- 思考风格: {thinking_style}"
        except Exception as e:
            logger.error(f"保存ReAct配置失败: {e}")
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
            tool_status = "✅" if status.get('auto_tool_execution_enabled') else "❌"

            lines.append(f"- 🧠 自动微调训练: {finetune_status}")
            lines.append(f"- 🔮 自动预测推理: {inference_status}")
            lines.append(f"- 🤖 自动Agent决策: {agent_status}")
            lines.append(f"- ⚡ 自动工具执行: {tool_status}")

            # 调度器状态
            lines.append("")
            lines.append("#### 🔄 调度器状态")
            scheduler_emoji = "✅" if status.get('scheduler_running') else "❌"
            lines.append(f"- 调度器运行状态: {scheduler_emoji}")

            if status.get('last_check_time'):
                last_check = status['last_check_time'].strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"- 最后检查时间: {last_check}")

            # 当前任务状态
            current_task = status.get('current_task', {})
            if current_task:
                lines.append("")
                lines.append("#### ⚡ 当前任务")
                task_stage = current_task.get('stage', 'unknown')
                task_start = current_task.get('start_time')
                if task_start:
                    start_str = task_start.strftime("%H:%M:%S")
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
                    training_time = latest_training['created_at'].strftime("%Y-%m-%d %H:%M")
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
                last_check_str = last_check.strftime("%Y-%m-%d %H:%M:%S")
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

    def get_pending_tools_data(self, plan_id: int):
        """获取待执行工具数据"""
        try:
            from services.automation_service import automation_service
            pending_tools = automation_service.get_pending_tool_executions(plan_id)

            data = []
            for tool in pending_tools:
                decision_time = tool.get('decision_time', '').strftime("%Y-%m-%d %H:%M:%S") if tool.get('decision_time') else 'N/A'
                tool_name = tool.get('tool_name', 'N/A')
                tool_args = str(tool.get('tool_args', {})) if tool.get('tool_args') else '{}'
                status = tool.get('status', 'pending')

                # 状态映射
                status_map = {
                    'pending': '⏳ 待确认',
                    'approved': '✅ 已批准',
                    'rejected': '❌ 已拒绝',
                    'executed': '✅ 已执行',
                    'failed': '❌ 执行失败'
                }
                status_display = status_map.get(status, status)

                data.append([decision_time, tool_name, tool_args, status_display])

            return data

        except Exception as e:
            logger.error(f"获取待执行工具失败: {e}")
            return []

    def handle_pending_tool_action(self, plan_id: int, action: str, selected_row: dict) -> str:
        """处理待执行工具操作"""
        try:
            if not selected_row or len(selected_row) < 2:
                return "❌ 请选择要操作的工具记录"

            decision_id = selected_row.get('decision_id')
            tool_name = selected_row.get('tool_name')

            if not decision_id or not tool_name:
                return "❌ 无效的工具记录"

            from services.automation_service import automation_service

            if action == "approve":
                result = automation_service.approve_pending_tool(plan_id, decision_id, tool_name)
            elif action == "reject":
                result = automation_service.reject_pending_tool(plan_id, decision_id, tool_name)
            else:
                return "❌ 无效的操作类型"

            return result

        except Exception as e:
            logger.error(f"处理工具操作失败: {e}")
            return f"❌ 操作失败: {str(e)}"

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

                # 从 finetune_params 中获取配置的范围
                finetune_params = plan.finetune_params or {}
                data_config = finetune_params.get('data', {})
                train_start_date_str = data_config.get('train_start_date')
                train_end_date_str = data_config.get('train_end_date')

                # 获取数据库中的实际范围
                min_date, max_date, total_count = self.get_data_date_range(plan.inst_id, plan.interval)

                if min_date is None or max_date is None:
                    return "**数据统计**: 暂无数据"

                # 如果有配置的训练范围，统计该范围内的数据量
                if train_start_date_str and train_end_date_str:
                    from datetime import datetime
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

📅 **配置范围**: {train_start_date_str} ~ {train_end_date_str}
📊 **训练数据点**: {train_data_count} 条

---

📅 **全部数据**: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}
📊 **总数据点**: {total_count} 条
"""
                else:
                    return f"""**数据统计**

📅 **全部数据**: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}
📊 **总数据点**: {total_count} 条

⚠️ **未配置训练范围**
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
**数据范围**: {min_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')} (共 {count} 条)

**已选择**: 最近 {days} 天 ({start_date.strftime('%Y-%m-%d')} 至 {max_date.strftime('%Y-%m-%d')})
"""

            return (
                info,
                start_date.strftime('%Y-%m-%d'),
                max_date.strftime('%Y-%m-%d')
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

**决策时间**: {decision.decision_time.strftime('%Y-%m-%d %H:%M:%S')}

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
        first_time = predictions[0]['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(predictions[0]['timestamp'], 'strftime') else str(predictions[0]['timestamp'])[:16]
        last_time = predictions[-1]['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(predictions[-1]['timestamp'], 'strftime') else str(predictions[-1]['timestamp'])[:16]

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
            timestamp_str = p['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(p['timestamp'], 'strftime') else str(p['timestamp'])[:19]
            message += f"{i}. **{timestamp_str}** - 开: ${p['open']:.2f}, 高: ${p['high']:.2f}, 低: ${p['low']:.2f}, 收: ${p['close']:.2f}\n"

        if last_5:
            message += "\n...\n\n### 预测数据（后5个周期）\n\n"

            for i, p in enumerate(last_5, 1):
                timestamp_str = p['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(p['timestamp'], 'strftime') else str(p['timestamp'])[:19]
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

        if tools_config.get('get_account_balance', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_account_balance",
                    "description": "查询账户余额",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        if tools_config.get('get_positions', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_positions",
                    "description": "查询当前持仓",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        if tools_config.get('get_pending_orders', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_pending_orders",
                    "description": "查询挂单列表",
                    "parameters": {"type": "object", "properties": {}}
                }
            })

        if tools_config.get('place_order', True):
            tools.append({
                "type": "function",
                "function": {
                    "name": "place_order",
                    "description": "下单",
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
            first_time = first_pred['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(first_pred['timestamp'], 'strftime') else str(first_pred['timestamp'])[:16]
            last_time = last_pred['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(last_pred['timestamp'], 'strftime') else str(last_pred['timestamp'])[:16]
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
                timestamp_str = pred['timestamp'].strftime('%m-%d %H:%M') if hasattr(pred['timestamp'], 'strftime') else str(pred['timestamp'])[:16]
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
                    # 更新计划的ws_connected状态
                    db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                        'ws_connected': True
                    })
                    db.commit()
                    return "✅ WebSocket已启动"
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

                # 更新计划的ws_connected状态
                db.query(TradingPlan).filter(TradingPlan.id == plan_id).update({
                    'ws_connected': False
                })
                db.commit()
                return "✅ WebSocket已停止"

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

                # 启动自动化调度器（如果配置了自动化）
                automation_status = ""
                if plan.auto_finetune_enabled or plan.auto_inference_enabled or plan.auto_agent_enabled or plan.auto_tool_execution_enabled:
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
                    lines.append(f"**更新时间**: {last_update.strftime('%H:%M:%S')}")

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
        """获取订单记录（通过 REST API）"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan or not plan.okx_api_key:
                    return pd.DataFrame()

                from services.okx_rest_service import OKXRestService

                # 创建 REST API 服务
                rest_service = OKXRestService(
                    api_key=plan.okx_api_key,
                    secret_key=plan.okx_secret_key,
                    passphrase=plan.okx_passphrase,
                    is_demo=plan.is_demo
                )

                # 获取订单列表（SPOT 类型）
                orders = rest_service.get_all_orders(
                    inst_type="SPOT",
                    inst_id=None,  # 获取所有交易对
                    limit=50
                )

                if not orders:
                    return pd.DataFrame()

                # 构建DataFrame
                df_data = []
                for order in orders:
                    side_emoji = '🟢' if order['side'] == 'buy' else '🔴'
                    state_map = {
                        'live': '⏳ 未成交',
                        'partially_filled': '⏸️ 部分成交',
                        'filled': '✅ 完全成交',
                        'canceled': '❌ 已取消',
                        'mmp_canceled': '❌ MMP取消',
                        'failed': '❌ 失败'
                    }
                    state_emoji = state_map.get(order['state'], f"❓ {order['state']}")

                    # 转换时间戳
                    create_time = datetime.fromtimestamp(int(order['cTime']) / 1000).strftime('%m-%d %H:%M:%S')
                    update_time = datetime.fromtimestamp(int(order['uTime']) / 1000).strftime('%m-%d %H:%M:%S')

                    df_data.append({
                        '订单ID': order['ordId'][:10] + '...',
                        '交易对': order['instId'],
                        '方向': f"{side_emoji} {order['side']}",
                        '类型': order['ordType'],
                        '价格': f"{float(order['px']):.4f}" if order.get('px') else '市价',
                        '数量': f"{float(order['sz']):.4f}",
                        '已成交': f"{float(order.get('accFillSz', 0)):.4f}",
                        '状态': state_emoji,
                        '创建时间': create_time,
                        '更新时间': update_time
                    })

                return pd.DataFrame(df_data)

        except Exception as e:
            logger.error(f"获取订单信息失败: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def clear_agent_records(self, plan_id: int) -> str:
        """清除AI Agent推理记录"""
        try:
            if not plan_id:
                return "❌ 请先选择计划"

            with get_db() as db:
                # 删除该计划的所有Agent决策记录
                deleted_count = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).delete()

                db.commit()

                logger.info(f"清除计划 {plan_id} 的 {deleted_count} 条AI Agent推理记录")

                return f"✅ 已清除 {deleted_count} 条推理记录"

        except Exception as e:
            logger.error(f"清除推理记录失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 清除失败: {str(e)}"

    def build_ui(self, plan_id: int):
        """构建详情页UI"""
        self.current_plan_id = plan_id

        with gr.Column():
            # 上部：计划概览
            overview = gr.Markdown(
                value=self.render_plan_overview(plan_id),
                elem_id="plan-overview"
            )

            # 刷新按钮
            refresh_btn = gr.Button("🔄 刷新数据", size="sm")

            with gr.Row():
                # 左侧：训练列表
                with gr.Column(scale=2):
                    gr.Markdown("### 🎯 模型训练记录")

                    training_df = gr.Dataframe(
                        value=self.load_training_records(plan_id),
                        interactive=False,
                        wrap=True
                    )

                    with gr.Row():
                        start_training_btn = gr.Button("▶️ 开始训练", variant="primary")
                        # auto_train_toggle = gr.Checkbox(label="自动训练", value=False)

                    training_status = gr.Markdown(
                        value=self.get_current_training_status(plan_id) or "等待操作...",
                        elem_id="training-status"
                    )

                # 中间：K线图
                with gr.Column(scale=5):
                    gr.Markdown("### 📈 K线图 & 预测数据")

                    with gr.Row():
                        show_pred_toggle = gr.Checkbox(label="显示预测", value=True)
                        days_slider = gr.Slider(
                            minimum=1, maximum=120, value=3, step=1,
                            label="显示天数"
                        )

                    with gr.Row():
                        # 训练 ID 下拉选择器
                        training_options = self.get_training_options(plan_id)
                        training_selector = gr.Dropdown(
                            choices=training_options,
                            value=None,  # 默认选择"全部"
                            label="选择训练 ID",
                            info="选择特定的训练版本，或选择「全部」显示所有历史预测"
                        )

                    kline_chart = gr.Plot(
                        value=self.generate_kline_chart(plan_id),
                        show_label=False
                    )

                # 右侧：Agent决策
                with gr.Column(scale=2):
                    gr.Markdown("### 🤖 AI Agent 决策记录")

                    agent_df = gr.Dataframe(
                        value=self.load_agent_decisions(plan_id),
                        interactive=False,
                        wrap=True
                    )

                    agent_detail = gr.Markdown("点击记录查看详情")

                    # 添加交易限制配置折叠面板
                    with gr.Accordion("💰 交易限制配置", open=False):
                        gr.Markdown("### 快速调整交易限制")

                        # 加载当前交易限制配置
                        current_limits = self.get_trading_limits_config(plan_id)

                        with gr.Row():
                            quick_usdt_amount = gr.Number(
                                label="可用资金 (USDT)",
                                value=current_limits['available_usdt_amount'],
                                minimum=0.0,
                                maximum=1000000.0,
                                step=10.0,
                                info="固定USDT资金数量"
                            )

                            quick_usdt_percentage = gr.Slider(
                                label="资金比例 (%)",
                                minimum=1.0,
                                maximum=100.0,
                                value=current_limits['available_usdt_percentage'],
                                step=1.0,
                                info="资金使用比例"
                            )

                        with gr.Row():
                            quick_avg_orders = gr.Number(
                                label="平摊单量",
                                value=current_limits['avg_order_count'],
                                minimum=1.0,
                                maximum=100.0,
                                step=1.0,
                                info="订单平摊笔数"
                            )

                            quick_stop_loss = gr.Slider(
                                label="止损比例 (%)",
                                minimum=1.0,
                                maximum=50.0,
                                value=current_limits['stop_loss_percentage'],
                                step=1.0,
                                info="亏损止损百分比"
                            )

                        save_quick_limits_btn = gr.Button("💾 快速保存", size="sm", variant="primary")
                        quick_limits_status = gr.Markdown("")

                        def save_quick_limits_wrapper(usdt_amount, usdt_percentage, avg_orders, stop_loss):
                            """快速保存交易限制"""
                            result = self.save_trading_limits_config(
                                plan_id, usdt_amount, usdt_percentage, int(avg_orders), stop_loss
                            )
                            return f"**{result}**"

                        save_quick_limits_btn.click(
                            fn=save_quick_limits_wrapper,
                            inputs=[quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss],
                            outputs=[quick_limits_status]
                        )

                        # 显示当前配置概览
                        limits_overview = f"""
**当前配置概览**:
- 资金: {current_limits['available_usdt_amount']} USDT 或 {current_limits['available_usdt_percentage']}%
- 平摊: {current_limits['avg_order_count']} 笔
- 止损: {current_limits['stop_loss_percentage']}%
"""
                        gr.Markdown(limits_overview)

            # 下方：账户和订单
            gr.Markdown("### 💰 账户信息 & 订单记录")

            with gr.Row():
                account_status = gr.Markdown("账户信息加载中...")
                order_table = gr.Dataframe(
                    value=pd.DataFrame(),
                    label="订单记录"
                )

        # 添加配置Tab页
        with gr.Tabs():
            # Agent配置Tab
            with gr.TabItem("🤖 Agent配置"):
                with gr.Row():
                    with gr.Column():
                        llm_configs = self.get_llm_configs()
                        llm_selector = gr.Dropdown(
                            choices=llm_configs,
                            label="LLM配置",
                            info="选择用于AI推理的LLM配置"
                        )

                        templates = self.get_prompt_templates()
                        template_selector = gr.Dropdown(
                            choices=templates,
                            label="提示词模版",
                            info="选择预定义的提示词模版"
                        )

                        load_template_btn = gr.Button("加载模版", size="sm")

                        agent_prompt = gr.Textbox(
                            label="Agent提示词",
                            lines=8,
                            placeholder="输入自定义的AI Agent提示词..."
                        )

                        # ReAct配置部分
                        gr.Markdown("### ⚙️ ReAct推理配置")

                        with gr.Row():
                            max_iterations = gr.Number(
                                label="最大推理轮数 (1-5)",
                                value=3,
                                minimum=1,
                                maximum=5,
                                precision=0,
                                info="控制ReAct循环的最大次数，避免无限循环，默认3"
                            )

                            enable_thinking = gr.Checkbox(
                                label="启用思考过程显示",
                                value=True,
                                info="是否显示AI的完整思考过程，适配Qwen3等思考模式LLM"
                            )

                        with gr.Row():
                            tool_approval = gr.Checkbox(
                                label="工具调用审批",
                                value=False,
                                info="是否需要手动确认每次工具调用，启用可增加交易安全性"
                            )

                            thinking_style = gr.Dropdown(
                                choices=["详细", "简洁", "极简"],
                                value="详细",
                                label="思考过程风格",
                                info="控制思考过程的详细程度"
                            )

                        react_config_status = gr.Markdown("ReAct配置将影响AI的推理深度和交互方式")

                    with gr.Column():
                        agent_config_status = gr.Markdown("### 🛠️ Agent工具配置\n\n默认启用所有交易工具")

                        with gr.Row():
                            save_agent_config_btn = gr.Button("💾 保存Agent配置", variant="primary")
                            save_react_config_btn = gr.Button("🔄 保存ReAct配置", variant="secondary")

            # 交易限制配置Tab
            with gr.TabItem("💰 交易限制配置"):
                gr.Markdown("### 交易限制参数设置")

                with gr.Row():
                    available_usdt_amount = gr.Number(
                        label="可用账户资金 (USDT)",
                        value=1000.0,
                        minimum=0.0,
                        maximum=1000000.0,
                        step=10.0,
                        info="固定的USDT资金数量"
                    )

                    available_usdt_percentage = gr.Slider(
                        label="可用账户资金比例 (%)",
                        minimum=1.0,
                        maximum=100.0,
                        value=30.0,
                        step=1.0,
                        info="使用账户总资金的比例，当固定USDT不足时使用百分比"
                    )

                with gr.Row():
                    avg_order_count = gr.Number(
                        label="平摊操作单量 (笔)",
                        value=10.0,
                        minimum=1.0,
                        maximum=100.0,
                        step=1.0,
                        info="将交易金额平分成多少笔订单执行"
                    )

                    stop_loss_percentage = gr.Slider(
                        label="止损比例 (%)",
                        minimum=1.0,
                        maximum=50.0,
                        value=20.0,
                        step=1.0,
                        info="亏损超过多少百分比时止损卖出"
                    )

                gr.Markdown("""
                💡 **交易限制说明**:
                - AI Agent 将严格遵守这些交易限制进行工具调用
                - 当固定USDT资金不足时，将使用账户总资金的百分比
                - 平摊操作可以降低市场冲击和风险
                - 止损机制有助于控制风险
                """)

                trading_limits_status = gr.Markdown("配置加载中...")
                save_trading_limits_btn = gr.Button("💾 保存交易限制配置", variant="primary")

            # 推理参数配置Tab
            with gr.TabItem("🔬 推理参数配置"):
                with gr.Row():
                    with gr.Column():
                        lookback_window = gr.Number(
                            label="Lookback Window",
                            value=512,
                            minimum=64,
                            maximum=2048,
                            step=64
                        )

                        predict_window = gr.Number(
                            label="Predict Window",
                            value=48,
                            minimum=1,
                            maximum=512,
                            step=1
                        )

                    with gr.Column():
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )

                        top_p = gr.Slider(
                            label="Top-p",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05
                        )

                        sample_count = gr.Number(
                            label="Sample Count",
                            value=30,
                            minimum=1,
                            maximum=100,
                            step=1
                        )

                inference_params_status = gr.Markdown("配置加载中...")
                save_inference_params_btn = gr.Button("💾 保存推理参数", variant="primary")

            # 自动化配置Tab
            with gr.TabItem("🔄 自动化配置"):
                gr.Markdown("### 🤖 自动化交易流程配置")

                gr.Markdown("""
                **自动化流程说明**：
                1. **自动微调训练**: 按配置的时间表自动执行模型训练
                2. **自动预测推理**: 训练完成后自动进行价格预测推理
                3. **自动Agent决策**: 推理完成后自动进行AI Agent交易决策
                4. **自动工具执行**: 决策完成后自动执行交易工具

                ⚠️ **注意**: 自动化流程仅在计划启动时生效，人工操作不会触发后续自动步骤
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎯 自动化开关")

                        # 自动化开关配置
                        auto_finetune_enabled = gr.Checkbox(
                            label="🧠 启用自动微调训练",
                            value=False,
                            info="在配置的时间自动执行模型训练"
                        )

                        auto_inference_enabled = gr.Checkbox(
                            label="🔮 启用自动预测推理",
                            value=False,
                            info="训练完成后自动进行价格预测"
                        )

                        auto_agent_enabled = gr.Checkbox(
                            label="🤖 启用自动Agent决策",
                            value=False,
                            info="推理完成后自动进行交易决策分析"
                        )

                        auto_tool_execution_enabled = gr.Checkbox(
                            label="⚡ 启用自动工具执行",
                            value=False,
                            info="决策后自动执行交易工具，无需确认"
                        )

                    with gr.Column():
                        gr.Markdown("#### ⏰ 自动微调时间表")

                        # 时间表配置
                        schedule_times = gr.Textbox(
                            label="训练时间点",
                            value="00:00, 12:00",
                            placeholder="例如: 00:00, 06:00, 12:00, 18:00",
                            info="用逗号分隔多个时间点（24小时制）",
                            lines=2
                        )

                        gr.Markdown("""
                        **时间表配置说明**:
                        - 支持24小时制格式 (HH:MM)
                        - 多个时间点用英文逗号分隔
                        - 系统会在指定时间自动检查并触发训练
                        - 每天每个时间点最多执行一次训练
                        """, variant="secondary")

                # 自动化状态显示
                automation_status_display = gr.Markdown("### 📊 自动化状态\n\n正在加载...")

                with gr.Row():
                    # 调度器控制
                    scheduler_status = gr.Markdown("🔄 调度器状态: 检查中...")
                    start_scheduler_btn = gr.Button("🚀 启动自动化调度器", variant="primary")
                    stop_scheduler_btn = gr.Button("⏹️ 停止自动化调度器", variant="stop")

                automation_config_status = gr.Markdown("配置加载中...")
                save_automation_config_btn = gr.Button("💾 保存自动化配置", variant="primary")

                # 待执行工具管理
                with gr.Accordion("📋 待执行工具管理", open=False):
                    gr.Markdown("### 🔧 待确认的自动化工具执行")

                    pending_tools_display = gr.DataFrame(
                        headers=["决策时间", "工具名称", "工具参数", "状态"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        label="待执行工具列表"
                    )

                    with gr.Row():
                        approve_tool_btn = gr.Button("✅ 批准执行", variant="primary")
                        reject_tool_btn = gr.Button("❌ 拒绝执行", variant="stop")

                    tool_action_result = gr.Markdown("选择工具记录后操作")

            # 控制面板Tab
            with gr.TabItem("🎛️ 控制面板"):
                gr.Markdown("### 🌐 WebSocket 和计划状态控制")

                # 获取控制面板状态（初始值）
                control_status = self.get_control_panel_status(plan_id)

                with gr.Row():
                    with gr.Column():
                        # WebSocket状态显示和控制
                        ws_status_display = gr.Markdown(value=control_status[0])

                        with gr.Row():
                            start_ws_btn = gr.Button("🔌 启动 WebSocket", variant="primary", visible=control_status[1])
                            stop_ws_btn = gr.Button("🔌 停止 WebSocket", variant="stop", visible=control_status[2])

                    with gr.Column():
                        # 计划状态显示和控制
                        plan_status_display = gr.Markdown(value=control_status[3])

                        with gr.Row():
                            start_plan_btn = gr.Button("▶️ 启动计划", variant="primary", visible=control_status[4])
                            stop_plan_btn = gr.Button("⏹️ 停止计划", variant="stop", visible=control_status[5])

                # 控制操作结果显示
                control_result = gr.Markdown("等待操作...")

                # 添加控制面板专用的刷新按钮
                with gr.Row():
                    refresh_control_btn = gr.Button("🔄 刷新状态", size="sm")
                    # 隐藏的状态检查组件，用于页面加载后立即刷新
                    initial_load_check = gr.Number(value=0, visible=False)

                # 页面加载后立即执行状态同步
                def on_page_load():
                    """页面加载后立即同步状态"""
                    # 强制从数据库重新加载最新状态
                    import time
                    time.sleep(0.5)  # 短暂延迟确保页面完全加载

                    control_status = self.get_control_panel_status(plan_id)
                    return (
                        control_status[0],  # ws_status_display
                        control_status[1],  # start_ws_btn visible
                        control_status[2],  # stop_ws_btn visible
                        control_status[3],  # plan_status_display
                        control_status[4],  # start_plan_btn visible
                        control_status[5],  # stop_plan_btn visible
                        "📋 状态已同步"  # control_result
                    )

                # 使用初始加载检查触发状态同步
                initial_load_check.change(
                    fn=on_page_load,
                    inputs=[],
                    outputs=[
                        ws_status_display, start_ws_btn, stop_ws_btn,
                        plan_status_display, start_plan_btn, stop_plan_btn,
                        control_result
                    ]
                )

                # 控制面板专用刷新函数
                def refresh_control_panel():
                    """只刷新控制面板状态"""
                    control_status = self.get_control_panel_status(plan_id)
                    return (
                        control_status[0],  # ws_status_display
                        control_status[1],  # start_ws_btn visible
                        control_status[2],  # stop_ws_btn visible
                        control_status[3],  # plan_status_display
                        control_status[4],  # start_plan_btn visible
                        control_status[5],  # stop_plan_btn visible
                        "✅ 状态已刷新"  # control_result
                    )

                # 绑定控制面板刷新事件
                refresh_control_btn.click(
                    fn=refresh_control_panel,
                    outputs=[
                        ws_status_display, start_ws_btn, stop_ws_btn,
                        plan_status_display, start_plan_btn, stop_plan_btn,
                        control_result
                    ]
                )

                # 自动刷新：当切换到控制面板Tab时自动刷新状态
                # 通过控制面板Tab的标签页选择事件触发
                # 这里使用一个隐藏的定时器来定期检查状态更新

            # 事件绑定
            def refresh_all():
                """刷新所有数据"""
                # 刷新基本数据
                overview_data = self.render_plan_overview(plan_id)
                training_data = self.load_training_records(plan_id)
                chart_data = self.generate_kline_chart(plan_id)
                agent_data = self.load_agent_decisions(plan_id)
                training_options = self.get_training_options(plan_id)

                # 刷新训练状态
                training_status_data = self.get_current_training_status(plan_id) or "等待操作..."

                # 刷新交易限制配置
                current_limits = self.get_trading_limits_config(plan_id)

                # 刷新控制面板状态
                control_status = self.get_control_panel_status(plan_id)

                # 安全的类型转换函数
                def safe_float(value, default=0.0):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                def safe_int(value, default=0):
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                return (
                    overview_data,
                    training_data,
                    chart_data,
                    agent_data,
                    training_options,
                    # 安全类型转换，防止字符串类型导致Gradio错误
                    safe_float(current_limits.get('available_usdt_amount', 1000.0)),
                    safe_float(current_limits.get('available_usdt_percentage', 30.0)),
                    safe_int(current_limits.get('avg_order_count', 10)),
                    safe_float(current_limits.get('stop_loss_percentage', 20.0)),
                    # 控制面板状态
                    control_status[0],  # ws_status_display
                    control_status[1],  # start_ws_btn visible
                    control_status[2],  # stop_ws_btn visible
                    control_status[3],  # plan_status_display
                    control_status[4],  # start_plan_btn visible
                    control_status[5],  # stop_plan_btn visible
                    # 训练状态
                    training_status_data
                )

            refresh_btn.click(
                fn=refresh_all,
                outputs=[
                    overview, training_df, kline_chart, agent_df, training_selector,
                    quick_usdt_amount, quick_usdt_percentage, quick_avg_orders, quick_stop_loss,
                    # 控制面板组件
                    ws_status_display, start_ws_btn, stop_ws_btn,
                    plan_status_display, start_plan_btn, stop_plan_btn,
                    # 训练状态
                    training_status
                ]
            )

            start_training_btn.click(
                fn=lambda: self.start_training_async(plan_id, None, None),
                outputs=[training_status]
            )

            # K线图更新
            days_slider.change(
                fn=lambda days, show_pred, training_id: self.generate_kline_chart(
                    plan_id, show_pred, training_id, days
                ),
                inputs=[days_slider, show_pred_toggle, training_selector],
                outputs=[kline_chart]
            )

            show_pred_toggle.change(
                fn=lambda show_pred, days, training_id: self.generate_kline_chart(
                    plan_id, show_pred, training_id, days
                ),
                inputs=[show_pred_toggle, days_slider, training_selector],
                outputs=[kline_chart]
            )

            # 训练 ID 选择器变更
            training_selector.change(
                fn=lambda training_id, show_pred, days: self.generate_kline_chart(
                    plan_id, show_pred, training_id, days
                ),
                inputs=[training_selector, show_pred_toggle, days_slider],
                outputs=[kline_chart]
            )

            # 配置Tab的事件绑定
            def load_configs():
                """加载所有配置"""
                # 加载Agent配置
                agent_config = self.get_agent_config(plan_id)

                # 加载交易限制配置
                trading_limits = self.get_trading_limits_config(plan_id)

                # 加载推理参数
                inference_params = self.get_inference_params(plan_id)

                # 加载自动化配置
                automation_config = self.get_automation_config(plan_id)
                schedule_str = ', '.join(automation_config.get('schedule', [])) if automation_config.get('schedule') else ''

                # 加载ReAct配置
                react_config = self.get_react_config(plan_id)

                # 安全的类型转换函数
                def safe_float(value, default=0.0):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                def safe_int(value, default=0):
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default

                return (
                    agent_config.get('llm_config_id'),
                    agent_config.get('agent_prompt', ''),
                    # 安全类型转换，防止字符串类型导致Gradio错误
                    safe_float(trading_limits.get('available_usdt_amount', 1000.0)),
                    safe_float(trading_limits.get('available_usdt_percentage', 30.0)),
                    safe_int(trading_limits.get('avg_order_count', 10)),
                    safe_float(trading_limits.get('stop_loss_percentage', 20.0)),
                    safe_int(inference_params.get('lookback_window', 576)),
                    safe_int(inference_params.get('predict_window', 24)),
                    safe_float(inference_params.get('temperature', 1.0)),
                    safe_float(inference_params.get('top_p', 0.9)),
                    safe_int(inference_params.get('sample_count', 30)),
                    # 自动化配置
                    automation_config.get('auto_finetune_enabled', False),
                    # ReAct配置
                    safe_int(react_config.get('max_iterations', 3)),
                    react_config.get('enable_thinking', True),
                    react_config.get('tool_approval', False),
                    react_config.get('thinking_style', '详细'),
                    automation_config.get('auto_inference_enabled', False),
                    automation_config.get('auto_agent_enabled', False),
                    automation_config.get('auto_tool_execution_enabled', False),
                    schedule_str
                )

            # 页面加载时设置配置初始值
            initial_configs = load_configs()
            if initial_configs and len(initial_configs) >= 19:  # 增加了4个ReAct配置参数
                llm_selector.value = initial_configs[0]
                agent_prompt.value = initial_configs[1]
                available_usdt_amount.value = initial_configs[2]
                available_usdt_percentage.value = initial_configs[3]
                avg_order_count.value = initial_configs[4]
                stop_loss_percentage.value = initial_configs[5]
                lookback_window.value = initial_configs[6]
                predict_window.value = initial_configs[7]
                temperature.value = initial_configs[8]
                top_p.value = initial_configs[9]
                sample_count.value = initial_configs[10]

                # 设置自动化配置初始值
                auto_finetune_enabled.value = initial_configs[11]
                auto_inference_enabled.value = initial_configs[12]
                auto_agent_enabled.value = initial_configs[13]
                auto_tool_execution_enabled.value = initial_configs[14]
                schedule_times.value = initial_configs[15] if len(initial_configs) > 15 else "00:00, 12:00"

                # 设置ReAct配置初始值
                max_iterations.value = initial_configs[16] if len(initial_configs) > 16 else 10
                enable_thinking.value = initial_configs[17] if len(initial_configs) > 17 else True
                tool_approval.value = initial_configs[18] if len(initial_configs) > 18 else False
                thinking_style.value = initial_configs[19] if len(initial_configs) > 19 else "详细"

                # 设置自动化状态显示
                automation_status_display.value = self.get_automation_status_display(plan_id)
                scheduler_status.value = self.get_scheduler_status_display()
                pending_tools_display.value = self.get_pending_tools_data(plan_id)

            # 加载提示词模版
            load_template_btn.click(
                fn=self.load_prompt_template,
                inputs=[template_selector],
                outputs=[agent_prompt]
            )

            # 保存Agent配置
            def save_agent_config_wrapper(llm_id, prompt_text):
                return self.save_agent_config(plan_id, llm_id, prompt_text, {})

            save_agent_config_btn.click(
                fn=save_agent_config_wrapper,
                inputs=[llm_selector, agent_prompt],
                outputs=[agent_config_status]
            )

            # 保存ReAct配置
            def save_react_config_wrapper(max_iter, enable_think, tool_appr, think_style):
                return self.save_react_config(plan_id, int(max_iter), enable_think, tool_appr, think_style)

            save_react_config_btn.click(
                fn=save_react_config_wrapper,
                inputs=[max_iterations, enable_thinking, tool_approval, thinking_style],
                outputs=[react_config_status]
            )

            # 保存交易限制配置
            save_trading_limits_btn.click(
                fn=self.save_trading_limits_config,
                inputs=[available_usdt_amount, available_usdt_percentage, avg_order_count, stop_loss_percentage],
                outputs=[trading_limits_status]
            )

            # 保存推理参数
            def save_inference_params_wrapper(
                lookback_window, predict_window, temperature, top_p, sample_count
            ):
                return self.save_inference_params(
                    plan_id, lookback_window, predict_window, temperature, top_p, sample_count
                )

            save_inference_params_btn.click(
                fn=save_inference_params_wrapper,
                inputs=[lookback_window, predict_window, temperature, top_p, sample_count],
                outputs=[inference_params_status]
            )

            # 自动化配置事件绑定
            def save_automation_config_wrapper(
                auto_finetune, auto_inference, auto_agent, auto_tool_execution, schedule_times
            ):
                return self.save_automation_config(
                    plan_id, auto_finetune, auto_inference, auto_agent, auto_tool_execution, schedule_times
                )

            save_automation_config_btn.click(
                fn=save_automation_config_wrapper,
                inputs=[
                    auto_finetune_enabled, auto_inference_enabled,
                    auto_agent_enabled, auto_tool_execution_enabled, schedule_times
                ],
                outputs=[automation_config_status]
            )

            # 调度器控制事件绑定
            start_scheduler_btn.click(
                fn=self.start_automation_scheduler,
                outputs=[scheduler_status]
            )

            stop_scheduler_btn.click(
                fn=self.stop_automation_scheduler,
                outputs=[scheduler_status]
            )

            # 工具操作事件绑定
            def approve_tool_wrapper(selected_row):
                return self.handle_pending_tool_action(plan_id, "approve", selected_row)

            def reject_tool_wrapper(selected_row):
                return self.handle_pending_tool_action(plan_id, "reject", selected_row)

            approve_tool_btn.click(
                fn=approve_tool_wrapper,
                inputs=[pending_tools_display],
                outputs=[tool_action_result]
            ).then(
                fn=self.get_pending_tools_data,
                inputs=[],
                outputs=[pending_tools_display]
            )

            reject_tool_btn.click(
                fn=reject_tool_wrapper,
                inputs=[pending_tools_display],
                outputs=[tool_action_result]
            ).then(
                fn=self.get_pending_tools_data,
                inputs=[],
                outputs=[pending_tools_display]
            )

            # 控制面板事件绑定
            start_ws_btn.click(
                fn=lambda: self.start_websocket_async(plan_id),
                outputs=[control_result]
            )

            stop_ws_btn.click(
                fn=lambda: self.stop_websocket_async(plan_id),
                outputs=[control_result]
            )

            start_plan_btn.click(
                fn=lambda: self.start_plan_async(plan_id),
                outputs=[control_result]
            )

            stop_plan_btn.click(
                fn=lambda: self.stop_plan_async(plan_id),
                outputs=[control_result]
            )


    async def manual_inference_stream(self, plan_id: int):
        """
        流式ReAct+Tool Use手动推理（用于Chatbot）

        Args:
            plan_id: 计划ID

        Yields:
            Chatbot消息历史列表: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        """
        from services.agent_decision_service import AgentDecisionService
        from database.models import TradingPlan, TrainingRecord, LLMConfig
        from sqlalchemy import and_, desc

        try:
            # 初始化对话历史
            history = []

            # 获取计划和配置
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    history.append({"role": "assistant", "content": "❌ 计划不存在"})
                    yield history
                    return

                # 检查LLM配置
                if not plan.llm_config_id:
                    history.append({"role": "assistant", "content": "❌ 未配置LLM，请先在Agent配置中选择LLM"})
                    yield history
                    return

                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()
                if not llm_config:
                    history.append({"role": "assistant", "content": "❌ LLM配置不存在"})
                    yield history
                    return

                # 获取最新的训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(desc(TrainingRecord.created_at)).first()

            # 使用AgentDecisionService的ReAct+Tool Use流式方法
            async for message in AgentDecisionService.react_tool_use_stream(
                plan_id=plan_id,
                training_id=latest_training.id if latest_training else None
            ):
                yield message

        except Exception as e:
            logger.error(f"ReAct推理失败: {e}")
            history.append({"role": "assistant", "content": f"❌ 推理过程出错: {str(e)}"})
            yield history

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

            # 创建DataFrame
            df = pd.DataFrame(df_data)

            # 确保数字列的类型正确
            if '执行时长(秒)' in df.columns:
                df['执行时长(秒)'] = pd.to_numeric(df['执行时长(秒)'], errors='coerce').fillna(0).astype(int)
            if '进度(%)' in df.columns:
                df['进度(%)'] = pd.to_numeric(df['进度(%)'], errors='coerce').fillna(0).astype(int)

            return df

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
