"""
新增计划界面
"""
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from api.okx_client import OKXClient
from services.plan_service import PlanService
from services.data_sync_service import DataSyncService
from services.ws_data_service import WebSocketDataService
from services.config_service import ConfigService
from services.ws_connection_manager import ws_connection_manager
from database.db import get_db
from database.models import KlineData, WebSocketSubscription
from sqlalchemy import and_, func, desc
from utils.logger import setup_logger
from utils.data_checker import DataIntegrityChecker
from utils.data_downloader import DataDownloader
from utils.timezone_helper import format_datetime_full_beijing, format_datetime_short_beijing, format_time_range_utc8
from ui.base_ui import BaseUIComponent, DatabaseMixin, UIHelper, ValidationHelper, ConfigManager

logger = setup_logger(__name__, "plan_create_ui.log")


class PlanCreateUI(BaseUIComponent, DatabaseMixin):
    """新增计划界面"""

    def __init__(self):
        super().__init__("plan_create")
        self.okx_client = OKXClient(is_demo=True)
        # 不再需要 self.ws_service，改用全局管理器
        self.current_inst_id: str = ""
        self.current_interval: str = ""
        self.current_is_demo: bool = True

    def check_websocket_status(self, inst_id: str, interval: str, is_demo: bool) -> Tuple[str, bool, bool]:
        """
        检查 WebSocket 连接状态（使用全局管理器）

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘

        Returns:
            (状态文本, 启动按钮是否可见, 停止按钮是否可见)
        """
        try:
            # 使用全局管理器查询实时状态
            status = ws_connection_manager.get_connection_status(inst_id, interval, is_demo)

            if status['exists'] and status['connected']:
                # WebSocket 正在运行
                last_time = format_datetime_full_beijing(status['last_data_time']) if status['last_data_time'] else '无'
                return (
                    f"🟢 WebSocket 已连接（全局连接复用中）\n"
                    f"接收消息: {status['total_received']} 条\n"
                    f"保存数据: {status['total_saved']} 条\n"
                    f"最后数据: {last_time}",
                    False,  # 隐藏启动按钮
                    True   # 显示停止按钮
                )
            else:
                return (
                    "⚪ WebSocket 未连接",
                    True,   # 显示启动按钮
                    False  # 隐藏停止按钮
                )

        except Exception as e:
            logger.error(f"检查 WebSocket 状态失败: {e}")
            return (
                "⚪ WebSocket 未连接",
                True,
                False
            )

    def generate_plan_name(self, inst_id: str, interval: str) -> str:
        """
        生成默认计划名称

        Args:
            inst_id: 交易对
            interval: 时间颗粒度

        Returns:
            计划名称
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{inst_id}_{interval}_{timestamp}"

    def update_plan_name(self, inst_id: str, interval: str) -> str:
        """更新计划名称（当交易对或时间颗粒度改变时）"""
        return self.generate_plan_name(inst_id, interval)

    def load_llm_config_choices(self) -> gr.Dropdown:
        """加载 LLM 配置选项"""
        choices, default_value, info = ConfigManager.load_llm_configs()
        return gr.Dropdown(
            choices=choices,
            value=default_value,
            info=info
        )

    def load_prompt_template_choices(self) -> gr.Dropdown:
        """加载 Agent 提示词模版选项"""
        choices, _, info = ConfigManager.load_prompt_templates()
        return gr.Dropdown(
            choices=choices,
            value=None,
            info=info
        )

    def fill_prompt_from_template(self, template_id: Optional[int]) -> str:
        """从模版填充提示词内容"""
        try:
            if not template_id or template_id <= 0:
                return ""

            template = ConfigService.get_prompt_template(int(template_id))
            if template:
                logger.info(f"加载提示词模版: {template.name}")
                return template.content
            else:
                return ""

        except Exception as e:
            logger.error(f"加载提示词模版内容失败: {e}")
            return ""

    def refresh_trading_pairs(self, is_demo: bool):
        """刷新交易对列表"""
        try:
            logger.info(f"刷新交易对列表: is_demo={is_demo}")

            # 重新创建客户端
            self.okx_client = OKXClient(is_demo=is_demo)

            # 使用ConfigManager获取交易对
            inst_ids, default_value, info = ConfigManager.get_trading_instruments(is_demo)

            logger.info(f"获取到 {len(inst_ids)} 个交易对")

            return gr.Dropdown(
                choices=inst_ids,
                value=default_value,
                info=info
            )

        except Exception as e:
            logger.error(f"刷新交易对失败: {e}")
            # ConfigManager已经处理了错误，返回默认值
            inst_ids, default_value, info = ConfigManager.get_trading_instruments(is_demo)
            return gr.Dropdown(
                choices=inst_ids,
                value=default_value,
                info=info
            )

    def reset_data(self, inst_id: str, interval: str) -> str:
        """
        重置数据（truncate 表数据）

        Args:
            inst_id: 交易对
            interval: 时间颗粒度

        Returns:
            状态信息
        """
        try:
            logger.info(f"重置数据: {inst_id} {interval}")

            with get_db() as db:
                # 删除指定交易对和时间颗粒度的数据
                deleted_count = db.query(KlineData).filter(
                    KlineData.inst_id == inst_id,
                    KlineData.interval == interval
                ).delete()
                db.commit()

            logger.info(f"成功删除 {deleted_count} 条数据")

            return f"""
✅ **重置完成**

已删除 **{deleted_count}** 条数据

交易对: {inst_id}
时间颗粒度: {interval}

💡 **提示**: 请重新点击"检查数据"按钮下载数据
"""

        except Exception as e:
            logger.error(f"重置数据失败: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 重置失败: {str(e)}"

    def get_data_date_range(self, inst_id: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
        """
        获取数据库中指定交易对的日期范围

        Args:
            inst_id: 交易对
            interval: 时间颗粒度

        Returns:
            (最早日期, 最新日期, 总数据条数)
        """
        try:
            with get_db() as db:
                # 查询最早和最新的时间戳
                result = db.query(
                    func.min(KlineData.timestamp).label('min_date'),
                    func.max(KlineData.timestamp).label('max_date'),
                    func.count(KlineData.id).label('count')
                ).filter(
                    KlineData.inst_id == inst_id,
                    KlineData.interval == interval
                ).first()

                if result and result.count > 0:
                    return result.min_date, result.max_date, result.count
                else:
                    return None, None, 0

        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            return None, None, 0

    def set_training_date_range(
        self,
        inst_id: str,
        interval: str,
        days: int
    ) -> Tuple[str, str, str]:
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
            min_date, max_date, count = self.get_data_date_range(inst_id, interval)

            if min_date is None or max_date is None:
                return (
                    "⚠️ **数据范围**: 未找到数据，请先检查数据",
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

    def update_data_range_info(
        self,
        inst_id: str,
        interval: str
    ) -> str:
        """
        更新数据范围信息显示

        Args:
            inst_id: 交易对
            interval: 时间颗粒度

        Returns:
            数据范围信息文本
        """
        try:
            min_date, max_date, count = self.get_data_date_range(inst_id, interval)

            if min_date is None or max_date is None:
                return "**数据范围**: 暂无数据，请先检查数据"

            return f"""
**数据范围**: {format_datetime_beijing(min_date, '%Y-%m-%d %H:%M')} 至 {format_datetime_beijing(max_date, '%Y-%m-%d %H:%M')}

**总数据量**: {count} 条
"""

        except Exception as e:
            logger.error(f"更新数据范围信息失败: {e}")
            return f"**数据范围**: 获取失败 - {str(e)}"

    def check_and_download_data(
        self,
        inst_id: str,
        interval: str,
        is_demo: bool,
        progress=gr.Progress()
    ) -> Tuple[str, go.Figure, bool, str, bool, bool]:
        """
        智能检查并下载历史数据

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
            progress: 进度对象

        Returns:
            (状态信息, K线图表, 是否可启用WS, WebSocket状态, 启动按钮可见性, 停止按钮可见性)
        """
        try:
            progress(0, desc="检查数据状态...")

            # 创建检查器和下载器
            checker = DataIntegrityChecker(inst_id, interval)
            downloader = DataDownloader(inst_id, interval, is_demo)

            # 获取下载策略
            strategy = checker.get_download_strategy()

            logger.info(
                f"数据策略: {strategy['strategy']}, "
                f"操作: {strategy['action']}"
            )

            # 根据策略执行操作
            if strategy['action'] == 'enable_ws':
                # 数据完整，直接显示预览
                progress(0.5, desc="数据完整，生成预览图...")
                chart = self._generate_kline_chart(inst_id, interval, last_days=30)
                progress(1.0, desc="完成!")

                # 格式化时间范围
                status_data = strategy['status']
                if status_data['start_time'] and status_data['end_time']:
                    time_range = format_time_range_utc8(status_data['start_time'], status_data['end_time'], '%Y-%m-%d')
                else:
                    time_range = "N/A"

                status_msg = f"""
✅ **{strategy['message']}**

📊 **数据统计**:
- 总数据量: {status_data['count']} 条
- 时间范围: {time_range}
- 完整度: {status_data['completeness']:.1f}%

💡 **提示**: 数据已齐全，可以启动 WebSocket 进行实时同步
"""

                # 检查 WebSocket 状态
                ws_status, ws_start_visible, ws_stop_visible = self.check_websocket_status(inst_id, interval, is_demo)

                return status_msg, chart, True, ws_status, ws_start_visible, ws_stop_visible

            # 需要下载数据
            progress(0.1, desc=f"{strategy['message']}")

            def progress_callback(current, total, message):
                pct = 0.1 + (current / total) * 0.7 if total > 0 else 0.1
                progress(pct, desc=message)

            # 执行智能下载
            result = downloader.smart_download(progress_callback)

            # 生成预览图
            progress(0.9, desc="生成K线预览图...")
            chart = self._generate_kline_chart(inst_id, interval, last_days=30)

            progress(1.0, desc="完成!")

            # 生成状态信息
            final_status = result['status']
            can_enable_ws = final_status['is_complete']

            # 格式化时间范围
            if final_status['start_time'] and final_status['end_time']:
                time_range = format_time_range_utc8(final_status['start_time'], final_status['end_time'], '%Y-%m-%d')
            else:
                time_range = "N/A"

            # 生成状态提示
            if can_enable_ws:
                tip_msg = "✅ 数据已齐全，可以启动 WebSocket 进行实时同步"
            else:
                # 分析未完整的原因
                reasons = []
                if not final_status['is_up_to_date']:
                    reasons.append("数据未同步到最新")
                if final_status['completeness'] < 99.0:
                    reasons.append(f"完整度不足(当前{final_status['completeness']:.1f}%)")
                if final_status['missing_segments']:
                    reasons.append(f"有{len(final_status['missing_segments'])}处缺失")

                if reasons:
                    tip_msg = f"⚠️ {', '.join(reasons)}，建议再次点击【检查数据】"
                else:
                    tip_msg = "⚠️ 数据不完整，建议再次点击【检查数据】"

            status_msg = f"""
✅ **{result['message']}**

📊 **数据统计**:
- 下载数据: {result['downloaded_count']} 条
- 总数据量: {final_status['count']} 条
- 时间范围: {time_range}
- 完整度: {final_status['completeness']:.1f}%

💡 **提示**: {tip_msg}
"""

            # 检查 WebSocket 状态
            ws_status, ws_start_visible, ws_stop_visible = self.check_websocket_status(inst_id, interval, is_demo)

            return status_msg, chart, can_enable_ws, ws_status, ws_start_visible, ws_stop_visible

        except Exception as e:
            logger.error(f"检查和下载数据失败: {e}")
            import traceback
            traceback.print_exc()
            empty_fig = go.Figure()
            return f"❌ 错误: {str(e)}", empty_fig, False, "⚪ WebSocket 未连接", True, False

    def _save_candles_to_db(
        self,
        inst_id: str,
        interval: str,
        candles: List[list],
        okx_client: OKXClient
    ) -> int:
        """保存K线数据到数据库"""
        saved_count = 0

        with get_db() as db:
            for candle in candles:
                parsed = okx_client.parse_candle_data(candle)
                if not parsed:
                    continue

                # 检查是否已存在
                exists = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == inst_id,
                        KlineData.interval == interval,
                        KlineData.timestamp == parsed['timestamp']
                    )
                ).first()

                if exists:
                    continue

                # 插入新数据
                kline = KlineData(
                    inst_id=inst_id,
                    interval=interval,
                    **parsed
                )
                db.add(kline)
                saved_count += 1

            db.commit()

        return saved_count

    def _check_data_completeness(
        self,
        inst_id: str,
        interval: str
    ) -> Tuple[bool, str]:
        """
        检查数据完整性

        Returns:
            (是否完整, 缺失信息)
        """
        with get_db() as db:
            # 获取所有数据点
            klines = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == inst_id,
                    KlineData.interval == interval
                )
            ).order_by(KlineData.timestamp).all()

            if len(klines) < 2:
                return False, "⚠️ 数据点不足 2 条"

            # 计算时间间隔（秒）
            interval_mapping = {
                "30m": 30 * 60,
                "1H": 60 * 60,
                "2H": 2 * 60 * 60,
                "4H": 4 * 60 * 60
            }
            interval_seconds = interval_mapping.get(interval, 60 * 60)

            # 检查缺失
            gaps = []
            for i in range(len(klines) - 1):
                current_time = klines[i].timestamp
                next_time = klines[i + 1].timestamp
                expected_time = current_time + timedelta(seconds=interval_seconds)

                time_diff = (next_time - expected_time).total_seconds()
                if time_diff > interval_seconds / 2:  # 容忍半个周期的误差
                    gap_count = int(time_diff / interval_seconds)
                    gaps.append((current_time, next_time, gap_count))

            if not gaps:
                return True, ""

            # 生成缺失信息
            gap_info = f"\n⚠️ **发现 {len(gaps)} 处数据缺失**:\n"
            for i, (start, end, count) in enumerate(gaps[:5], 1):  # 只显示前5个
                gap_info += f"  {i}. {start} → {end} (缺失约 {count} 条)\n"

            if len(gaps) > 5:
                gap_info += f"  ... 还有 {len(gaps) - 5} 处缺失\n"

            return False, gap_info

    def _generate_kline_chart(
        self,
        inst_id: str,
        interval: str,
        last_days: Optional[int] = None
    ) -> go.Figure:
        """
        生成K线预览图

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            last_days: 最后N天的数据，None表示最后10%数据

        Returns:
            Plotly 图表
        """
        try:
            checker = DataIntegrityChecker(inst_id, interval)

            if last_days is not None:
                # 根据天数查询
                klines = checker.query_preview_data(last_days=last_days)
            else:
                # 默认最后10%数据
                klines = checker.query_preview_data()

            if not klines:
                fig = go.Figure()
                fig.add_annotation(
                    text="无数据",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
                return fig

            # 创建K线图
            # 将UTC时间转换为UTC+8（北京时间）显示
            timestamps_utc8 = []
            for k in klines:
                ts = k['timestamp']
                # 如果是naive datetime，假设它是UTC
                if ts.tzinfo is None:
                    from datetime import timezone
                    ts = ts.replace(tzinfo=timezone.utc)
                # 转换为UTC+8
                ts_utc8 = ts + timedelta(hours=8)
                timestamps_utc8.append(ts_utc8)

            fig = go.Figure(data=[go.Candlestick(
                x=timestamps_utc8,
                open=[k['open'] for k in klines],
                high=[k['high'] for k in klines],
                low=[k['low'] for k in klines],
                close=[k['close'] for k in klines],
                name=inst_id
            )])

            days_text = f"最后 {last_days} 天" if last_days else "最后 10%"
            fig.update_layout(
                title=f"{inst_id} {interval} K线图 ({days_text}, 共 {len(klines)} 条数据)",
                xaxis_title="时间 (UTC+8)",
                yaxis_title="价格",
                height=400,
                template="plotly_white"
            )

            return fig

        except Exception as e:
            logger.error(f"生成K线图失败: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"生成图表失败: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

    def update_chart_preview(
        self,
        inst_id: str,
        interval: str,
        last_days: int
    ) -> go.Figure:
        """
        更新K线预览图（供滑块使用）

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            last_days: 最后N天

        Returns:
            Plotly 图表
        """
        return self._generate_kline_chart(inst_id, interval, last_days)

    def start_websocket(self, inst_id: str, interval: str, is_demo: bool):
        """启动 WebSocket 实时同步（使用全局管理器）"""
        try:
            logger.info(f"启动 WebSocket: {inst_id} {interval} demo={is_demo}")

            # 保存当前配置供定时器使用
            self.current_inst_id = inst_id
            self.current_interval = interval
            self.current_is_demo = is_demo

            # 使用全局管理器获取或创建连接（会自动复用已有连接）
            ws_service = ws_connection_manager.get_or_create_connection(
                inst_id=inst_id,
                interval=interval,
                is_demo=is_demo,
                ui_callback=None  # 如果需要UI回调，可以传入
            )

            logger.info(f"WebSocket 连接已启动或复用: {inst_id} {interval}")

            return (
                "🟢 WebSocket 已启动（使用全局连接管理），正在同步数据...",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.Timer(active=True)  # 启动定时器
            )

        except Exception as e:
            logger.error(f"启动 WebSocket 失败: {e}")
            import traceback
            traceback.print_exc()
            return (
                f"❌ 启动失败: {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.Timer(active=False)
            )

    def stop_websocket(self):
        """停止 WebSocket 实时同步（使用全局管理器）"""
        try:
            logger.info(f"停止 WebSocket: {self.current_inst_id} {self.current_interval}")

            # 使用全局管理器停止连接
            ws_connection_manager.stop_connection(
                self.current_inst_id,
                self.current_interval,
                self.current_is_demo
            )

            return (
                "⚪ WebSocket 已停止",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.Timer(active=False)  # 停止定时器
            )

        except Exception as e:
            logger.error(f"停止 WebSocket 失败: {e}")
            return (
                f"❌ 停止失败: {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.Timer(active=False)
            )

    def auto_refresh_chart(self, preview_days: int):
        """定时自动刷新图表（WebSocket 运行时）"""
        # 检查全局管理器中的连接状态
        status = ws_connection_manager.get_connection_status(
            self.current_inst_id,
            self.current_interval,
            self.current_is_demo
        )

        if not status['exists'] or not status['connected']:
            return gr.update()  # 不更新

        # 重新生成图表
        try:
            chart = self._generate_kline_chart(
                self.current_inst_id,
                self.current_interval,
                preview_days
            )
            return chart
        except Exception as e:
            logger.error(f"自动刷新图表失败: {e}")
            return gr.update()

    def create_plan(
        self,
        plan_name: str,
        inst_id: str,
        interval: str,
        train_start_date: str,
        train_end_date: str,
        auto_finetune_times: str,  # 新增：自动微调时间点（逗号分隔）
        # 数据配置参数
        lookback_window: int,
        predict_window: int,
        max_context: int,
        clip_value: float,
        train_ratio: float,
        val_ratio: float,
        # Tokenizer 训练参数
        tokenizer_epochs: int,
        tokenizer_lr: float,
        # Predictor 训练参数
        predictor_epochs: int,
        predictor_lr: float,
        # Adam 优化器参数
        adam_beta1: float,
        adam_beta2: float,
        adam_weight_decay: float,
        # 通用训练参数
        batch_size: int,
        accumulation_steps: int,
        num_workers: int,
        seed: int,
        # 预训练模型选择
        model_size: str,
        # Agent 配置
        llm_config_id: Optional[int],
        agent_prompt: str,
        # 交易限制配置
        available_usdt_amount: float,
        available_usdt_percentage: float,
        avg_order_count: int,
        stop_loss_percentage: float,
        # OKX API 配置
        okx_api_key: str,
        okx_secret_key: str,
        okx_passphrase: str,
        is_demo: bool,
        progress=gr.Progress()
    ):
        """创建交易计划"""
        try:
            progress(0, desc="验证输入参数...")

            # 验证必填字段
            if not plan_name:
                return "❌ 请输入计划名称"

            if not inst_id:
                return "❌ 请选择交易对"

            if not okx_api_key or not okx_secret_key or not okx_passphrase:
                return "❌ 请填写完整的 OKX API 配置"

            progress(0.1, desc="下载预训练模型...")

            # 使用 ModelService 下载预训练模型
            try:
                from services.model_service import ModelService
                pretrained_tokenizer_path, pretrained_predictor_path = ModelService.download_model(model_size)
                logger.info(f"预训练模型路径: Tokenizer={pretrained_tokenizer_path}, Predictor={pretrained_predictor_path}")
            except Exception as e:
                logger.error(f"下载预训练模型失败: {e}")
                return f"❌ 下载预训练模型失败: {str(e)}"

            progress(0.2, desc="解析配置参数...")

            # 解析训练时间范围
            try:
                data_start_time = datetime.strptime(train_start_date, "%Y-%m-%d")
                data_end_time = datetime.strptime(train_end_date, "%Y-%m-%d")
            except ValueError:
                return "❌ 时间格式错误，请使用 YYYY-MM-DD 格式"

            if data_start_time >= data_end_time:
                return "❌ 开始时间必须小于结束时间"

            progress(0.4, desc="构建配置...")

            # 解析自动微调时间点
            auto_schedule = [t.strip() for t in auto_finetune_times.split(',') if t.strip()]
            logger.info(f"自动微调时间表: {auto_schedule}")

            # 获取微调模型保存路径
            model_save_base_path = ModelService.get_finetuned_save_path(inst_id, interval)

            # 构建完整的微调参数（参考 finetune_csv 的配置结构）
            finetune_params = {
                # 数据配置
                "data": {
                    "lookback_window": int(lookback_window),
                    "predict_window": int(predict_window),
                    "max_context": int(max_context),
                    "clip": float(clip_value),
                    "train_ratio": float(train_ratio),
                    "val_ratio": float(val_ratio),
                    "test_ratio": 0.0
                },
                # 训练配置
                "training": {
                    "tokenizer_epochs": int(tokenizer_epochs),
                    "basemodel_epochs": int(predictor_epochs),
                    "batch_size": int(batch_size),
                    "log_interval": 50,
                    "num_workers": int(num_workers),
                    "seed": int(seed),
                    "tokenizer_learning_rate": float(tokenizer_lr),
                    "predictor_learning_rate": float(predictor_lr),
                    "adam_beta1": float(adam_beta1),
                    "adam_beta2": float(adam_beta2),
                    "adam_weight_decay": float(adam_weight_decay),
                    "accumulation_steps": int(accumulation_steps)
                },
                # 模型路径配置
                "model_paths": {
                    "pretrained_tokenizer": pretrained_tokenizer_path,
                    "pretrained_predictor": pretrained_predictor_path,
                    "exp_name": f"{inst_id.replace('-', '_')}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "base_path": model_save_base_path,
                    "base_save_path": model_save_base_path,
                    "finetuned_tokenizer": "",  # 将在训练时自动生成
                    "tokenizer_save_name": "tokenizer",
                    "basemodel_save_name": "basemodel"
                },
                # 实验配置
                "experiment": {
                    "name": "kronos_kokex_finetune",
                    "description": f"KOKEX auto finetune for {inst_id} {interval}",
                    "use_comet": False,
                    "train_tokenizer": True,
                    "train_basemodel": True,
                    "skip_existing": False
                },
                # 设备配置
                "device": {
                    "use_cuda": True,
                    "device_id": 0
                }
            }

            # 构建 Agent 工具配置（使用默认值）
            agent_tools_config = {
                "enable_order": True,
                "enable_adjust": True,
                "enable_cancel": True
            }

            # 构建交易限制（使用用户配置的值）
            trading_limits = {
                "available_usdt_amount": float(available_usdt_amount),
                "available_usdt_percentage": float(available_usdt_percentage),
                "avg_order_count": int(avg_order_count),
                "stop_loss_percentage": float(stop_loss_percentage),
                "max_position_size": 1.0,  # 保留原有字段以兼容旧代码
                "max_order_amount": float(available_usdt_amount)  # 使用用户配置的USDT数量
            }

            progress(0.6, desc="创建计划...")

            # 创建计划（不传递 model_version，使用可选参数默认值 None）
            plan_id = PlanService.create_plan(
                plan_name=plan_name,
                inst_id=inst_id,
                interval=interval,
                data_start_time=data_start_time,
                data_end_time=data_end_time,
                finetune_params=finetune_params,
                auto_finetune_schedule=auto_schedule,  # 使用用户配置的时间点
                llm_config_id=llm_config_id,
                agent_prompt=agent_prompt,
                agent_tools_config=agent_tools_config,
                trading_limits=trading_limits,
                okx_api_key=okx_api_key,
                okx_secret_key=okx_secret_key,
                okx_passphrase=okx_passphrase,
                is_demo=is_demo,
                model_version=None  # 不再需要 model_version
            )

            if not plan_id:
                return "❌ 创建计划失败"

            progress(1.0, desc="完成!")

            logger.info(f"创建计划成功: ID={plan_id}, Name={plan_name}")
            return f"✅ 创建计划成功！计划 ID: {plan_id}"

        except Exception as e:
            logger.error(f"创建计划失败: {e}")
            return f"❌ 创建失败: {str(e)}"

    def build_ui(self):
        """构建界面"""
        with gr.Column():
            gr.Markdown("## 新增交易计划")

            with gr.Group():
                gr.Markdown("### 基本配置")

                plan_name = gr.Textbox(
                    label="计划名称",
                    value=self.generate_plan_name("ETH-USDT", "1H"),
                    placeholder="自动生成，可修改",
                    info="默认根据交易对+时间颗粒度+时间戳生成"
                )

                refresh_btn = gr.Button("🔄 刷新交易对")

                with gr.Row():
                    inst_id = gr.Dropdown(
                        label="交易对",
                        choices=["ETH-USDT"],
                        value="ETH-USDT",
                        allow_custom_value=True
                    )

                    interval = gr.Dropdown(
                        label="时间颗粒度",
                        choices=["30m", "1H", "2H", "4H"],
                        value="1H"
                    )

                gr.Markdown("### 数据状态")

                data_status = gr.Markdown(
                    "ℹ️ 点击下方按钮检查数据状态"
                )

                with gr.Row():
                    check_data_btn = gr.Button(
                        "🔍 检查数据",
                        variant="secondary",
                        size="lg"
                    )
                    reset_data_btn = gr.Button(
                        "🗑️ 重置数据",
                        variant="stop",
                        size="lg"
                    )

                kline_chart = gr.Plot(
                    label="K线预览图",
                    visible=False
                )

                preview_days_slider = gr.Slider(
                    minimum=1,
                    maximum=365,
                    value=30,
                    step=1,
                    label="预览天数",
                    info="调整显示最后N天的K线数据",
                    visible=False
                )

                # WebSocket 实时同步控件
                ws_control_group = gr.Group(visible=False)
                with ws_control_group:
                    gr.Markdown("### 实时数据同步")

                    ws_status = gr.Markdown("⚪ WebSocket 未连接")

                    with gr.Row():
                        ws_start_btn = gr.Button(
                            "▶️ 启动同步",
                            variant="primary"
                        )
                        ws_stop_btn = gr.Button(
                            "⏹️ 停止同步",
                            variant="stop",
                            visible=False
                        )

                    # 自动刷新定时器（WebSocket 运行时每 10 秒刷新图表）
                    auto_refresh_timer = gr.Timer(value=10, active=False)

            with gr.Group():
                gr.Markdown("### Kronos 模型微调配置")

                # 训练数据范围选择
                gr.Markdown("#### 训练数据范围")

                data_range_info = gr.Markdown(
                    "**数据范围**: 请先检查数据后再选择训练范围"
                )

                # 快捷选择按钮
                with gr.Row():
                    days_30_btn = gr.Button("📅 最近30天", size="sm")
                    days_60_btn = gr.Button("📅 最近60天", size="sm")
                    days_90_btn = gr.Button("📅 最近90天", size="sm")

                # 日期范围滑块（使用文本框，因为 Gradio 没有日期范围滑块）
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

                # 自动微调时间配置
                gr.Markdown("#### 自动微调时间配置")

                with gr.Row():
                    auto_finetune_time_input = gr.Textbox(
                        label="添加微调时间点（HH:MM格式）",
                        placeholder="例如: 00:00 或 12:30",
                        scale=2
                    )
                    add_time_btn = gr.Button("➕ 添加", size="sm", scale=1)

                auto_finetune_times = gr.Textbox(
                    label="已配置的微调时间点（逗号分隔）",
                    value="00:00",
                    interactive=True,
                    info="每天会在这些时间点自动触发微调训练"
                )

                with gr.Row():
                    clear_times_btn = gr.Button("🗑️ 清空所有时间点", size="sm")

                # 数据配置参数
                gr.Markdown("#### 数据配置")

                with gr.Row():
                    lookback_window = gr.Number(
                        label="历史窗口长度 (lookback_window)",
                        value=512,
                        minimum=64,
                        maximum=2048,
                        info="模型使用的历史数据点数"
                    )

                    predict_window = gr.Number(
                        label="预测窗口长度 (predict_window)",
                        value=48,
                        minimum=1,
                        maximum=512,
                        info="要预测的未来点数"
                    )

                with gr.Row():
                    max_context = gr.Number(
                        label="最大上下文长度 (max_context)",
                        value=512,
                        minimum=64,
                        maximum=2048,
                        info="最大上下文长度"
                    )

                    clip_value = gr.Number(
                        label="数据裁剪值 (clip)",
                        value=5.0,
                        minimum=1.0,
                        maximum=10.0,
                        info="标准化后的数据裁剪值"
                    )

                with gr.Row():
                    train_ratio = gr.Slider(
                        label="训练集比例",
                        minimum=0.5,
                        maximum=0.95,
                        value=0.9,
                        step=0.05,
                        info="训练集占总数据的比例"
                    )

                    val_ratio = gr.Slider(
                        label="验证集比例",
                        minimum=0.05,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        info="验证集占总数据的比例"
                    )

                # Tokenizer 训练参数
                gr.Markdown("#### Tokenizer 训练参数")

                with gr.Row():
                    tokenizer_epochs = gr.Number(
                        label="Tokenizer 训练轮数",
                        value=25,
                        minimum=1,
                        maximum=200,
                        info="Tokenizer 训练的 epoch 数"
                    )

                    tokenizer_lr = gr.Number(
                        label="Tokenizer 学习率",
                        value=0.0002,
                        minimum=1e-6,
                        maximum=1e-2,
                        info="Tokenizer 的学习率"
                    )

                # Predictor 训练参数
                gr.Markdown("#### Predictor 训练参数")

                with gr.Row():
                    predictor_epochs = gr.Number(
                        label="Predictor 训练轮数",
                        value=50,
                        minimum=1,
                        maximum=200,
                        info="Predictor 训练的 epoch 数"
                    )

                    predictor_lr = gr.Number(
                        label="Predictor 学习率",
                        value=0.000001,
                        minimum=1e-8,
                        maximum=1e-4,
                        info="Predictor 的学习率"
                    )

                # Adam 优化器参数
                gr.Markdown("#### Adam 优化器参数")

                with gr.Row():
                    adam_beta1 = gr.Number(
                        label="Adam Beta1",
                        value=0.9,
                        minimum=0.0,
                        maximum=1.0,
                        info="Adam 优化器的 beta1 参数"
                    )

                    adam_beta2 = gr.Number(
                        label="Adam Beta2",
                        value=0.95,
                        minimum=0.0,
                        maximum=1.0,
                        info="Adam 优化器的 beta2 参数"
                    )

                    adam_weight_decay = gr.Number(
                        label="权重衰减",
                        value=0.1,
                        minimum=0.0,
                        maximum=1.0,
                        info="Adam 优化器的权重衰减"
                    )

                # 通用训练参数
                gr.Markdown("#### 通用训练参数")

                with gr.Row():
                    batch_size = gr.Number(
                        label="批次大小 (batch_size)",
                        value=16,
                        minimum=1,
                        maximum=128,
                        info="训练时的批次大小"
                    )

                    accumulation_steps = gr.Number(
                        label="梯度累积步数",
                        value=1,
                        minimum=1,
                        maximum=32,
                        info="梯度累积步数，用于模拟更大的批次"
                    )

                with gr.Row():
                    num_workers = gr.Number(
                        label="数据加载线程数",
                        value=4,
                        minimum=0,
                        maximum=16,
                        info="数据加载的线程数"
                    )

                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        minimum=0,
                        maximum=9999,
                        info="随机种子，用于结果复现"
                    )

                # 预训练模型选择
                gr.Markdown("#### 预训练模型")

                model_size = gr.Dropdown(
                    label="模型大小",
                    choices=["kronos-mini", "kronos-small", "kronos-base", "kronos-large"],
                    value="kronos-base",
                    info="选择预训练模型大小（将自动从 Hugging Face 下载到 kokex/models/pretrained）"
                )

                gr.Markdown("💡 **说明**: 模型将自动下载并保存到 `kokex/models/pretrained`，微调后的模型将保存到 `kokex/models/train`")

            with gr.Group():
                gr.Markdown("### AI Agent 配置")

                # LLM 配置选择
                gr.Markdown("#### LLM 模型选择")

                llm_config_id = gr.Dropdown(
                    label="LLM 配置",
                    choices=[],
                    value=None,
                    info="选择 AI Agent 使用的 LLM 配置（请先在配置中心创建）"
                )

                llm_refresh_btn = gr.Button("🔄 刷新 LLM 配置列表", size="sm")

                # Agent 提示词模版选择
                gr.Markdown("#### Agent 提示词")

                prompt_template_dropdown = gr.Dropdown(
                    label="选择提示词模版",
                    choices=[],
                    value=None,
                    info="选择预设的提示词模版（可选）"
                )

                prompt_template_refresh_btn = gr.Button("🔄 刷新模版列表", size="sm")

                agent_prompt = gr.Textbox(
                    label="Agent 提示词内容",
                    lines=8,
                    placeholder="请输入 AI Agent 的交易策略提示词，或从上方模版中选择...",
                    value="你是一个专业的加密货币交易员。根据预测的K线数据，分析市场趋势并做出交易决策。"
                )

  
            with gr.Group():
                gr.Markdown("### 交易限制配置")

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

            with gr.Group():
                gr.Markdown("### OKX API 配置")

                is_demo = gr.Checkbox(
                    label="模拟盘",
                    value=True,
                    info="✅ 建议先使用模拟盘测试"
                )

                okx_api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="请输入 OKX API Key"
                )

                okx_secret_key = gr.Textbox(
                    label="Secret Key",
                    type="password",
                    placeholder="请输入 OKX Secret Key"
                )

                okx_passphrase = gr.Textbox(
                    label="Passphrase",
                    type="password",
                    placeholder="请输入 OKX Passphrase"
                )

            submit_btn = gr.Button("创建计划", variant="primary", size="lg")

            result = gr.Textbox(
                label="执行结果",
                interactive=False
            )

            # 事件绑定

            # 刷新交易对
            refresh_btn.click(
                fn=self.refresh_trading_pairs,
                inputs=[is_demo],
                outputs=[inst_id]
            )

            # 交易对或时间颗粒度改变时，自动更新计划名称
            inst_id.change(
                fn=self.update_plan_name,
                inputs=[inst_id, interval],
                outputs=[plan_name]
            )

            interval.change(
                fn=self.update_plan_name,
                inputs=[inst_id, interval],
                outputs=[plan_name]
            )

            # 检查并下载数据
            def download_and_show_chart(inst_id, interval, is_demo):
                status, chart, can_enable_ws, ws_status_text, ws_start_visible, ws_stop_visible = self.check_and_download_data(inst_id, interval, is_demo)

                # 自动更新训练数据范围信息
                data_range_text = self.update_data_range_info(inst_id, interval)

                return (
                    status,
                    chart,
                    gr.update(visible=True),  # kline_chart
                    gr.update(visible=True),  # preview_days_slider
                    gr.update(visible=can_enable_ws),  # ws_control_group
                    ws_status_text,  # ws_status
                    gr.update(visible=ws_start_visible),  # ws_start_btn
                    gr.update(visible=ws_stop_visible),  # ws_stop_btn
                    data_range_text  # data_range_info
                )

            check_data_btn.click(
                fn=download_and_show_chart,
                inputs=[inst_id, interval, is_demo],
                outputs=[data_status, kline_chart, kline_chart, preview_days_slider, ws_control_group, ws_status, ws_start_btn, ws_stop_btn, data_range_info]
            )

            # 训练数据范围快捷按钮
            def set_30_days(inst_id, interval):
                return self.set_training_date_range(inst_id, interval, 30)

            def set_60_days(inst_id, interval):
                return self.set_training_date_range(inst_id, interval, 60)

            def set_90_days(inst_id, interval):
                return self.set_training_date_range(inst_id, interval, 90)

            days_30_btn.click(
                fn=set_30_days,
                inputs=[inst_id, interval],
                outputs=[data_range_info, train_start_date, train_end_date]
            )

            days_60_btn.click(
                fn=set_60_days,
                inputs=[inst_id, interval],
                outputs=[data_range_info, train_start_date, train_end_date]
            )

            days_90_btn.click(
                fn=set_90_days,
                inputs=[inst_id, interval],
                outputs=[data_range_info, train_start_date, train_end_date]
            )

            # 自动微调时间配置事件
            def add_finetune_time(current_times, new_time):
                """添加新的微调时间点"""
                import re
                # 验证时间格式 HH:MM
                if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', new_time.strip()):
                    return current_times, "❌ 时间格式错误，请使用HH:MM格式（例如：00:00, 12:30）"

                # 分割现有时间点
                times = [t.strip() for t in current_times.split(',') if t.strip()]

                # 检查是否已存在
                if new_time.strip() in times:
                    return current_times, "⚠️ 该时间点已存在"

                # 添加新时间点
                times.append(new_time.strip())
                times.sort()  # 按时间排序

                return ', '.join(times), f"✅ 已添加时间点: {new_time}"

            def clear_finetune_times():
                """清空所有时间点"""
                return "00:00", "✅ 已清空，保留默认时间 00:00"

            add_time_btn.click(
                fn=add_finetune_time,
                inputs=[auto_finetune_times, auto_finetune_time_input],
                outputs=[auto_finetune_times, data_status]
            )

            clear_times_btn.click(
                fn=clear_finetune_times,
                outputs=[auto_finetune_times, data_status]
            )

            # 重置数据
            reset_data_btn.click(
                fn=self.reset_data,
                inputs=[inst_id, interval],
                outputs=[data_status]
            )

            # 滑块控制预览
            preview_days_slider.change(
                fn=self.update_chart_preview,
                inputs=[inst_id, interval, preview_days_slider],
                outputs=[kline_chart]
            )

            # WebSocket 控制
            ws_start_btn.click(
                fn=self.start_websocket,
                inputs=[inst_id, interval, is_demo],
                outputs=[ws_status, ws_start_btn, ws_stop_btn, auto_refresh_timer]
            )

            ws_stop_btn.click(
                fn=self.stop_websocket,
                inputs=[],
                outputs=[ws_status, ws_start_btn, ws_stop_btn, auto_refresh_timer]
            )

            # 定时器触发图表刷新
            auto_refresh_timer.tick(
                fn=self.auto_refresh_chart,
                inputs=[preview_days_slider],
                outputs=[kline_chart]
            )

            # LLM 配置刷新
            llm_refresh_btn.click(
                fn=self.load_llm_config_choices,
                inputs=[],
                outputs=[llm_config_id]
            )

            # Agent 提示词模版刷新
            prompt_template_refresh_btn.click(
                fn=self.load_prompt_template_choices,
                inputs=[],
                outputs=[prompt_template_dropdown]
            )

            # 选择模版时自动填充提示词
            prompt_template_dropdown.change(
                fn=self.fill_prompt_from_template,
                inputs=[prompt_template_dropdown],
                outputs=[agent_prompt]
            )

            # 创建计划
            submit_btn.click(
                fn=self.create_plan,
                inputs=[
                    plan_name, inst_id, interval,
                    train_start_date, train_end_date,
                    auto_finetune_times,  # 新增：自动微调时间点
                    # 数据配置参数
                    lookback_window, predict_window, max_context, clip_value,
                    train_ratio, val_ratio,
                    # Tokenizer 训练参数
                    tokenizer_epochs, tokenizer_lr,
                    # Predictor 训练参数
                    predictor_epochs, predictor_lr,
                    # Adam 优化器参数
                    adam_beta1, adam_beta2, adam_weight_decay,
                    # 通用训练参数
                    batch_size, accumulation_steps, num_workers, seed,
                    # 预训练模型选择
                    model_size,
                    # Agent 配置
                    llm_config_id,
                    agent_prompt,
                    # 交易限制配置
                    available_usdt_amount, available_usdt_percentage,
                    avg_order_count, stop_loss_percentage,
                    # OKX API 配置
                    okx_api_key, okx_secret_key, okx_passphrase,
                    is_demo
                ],
                outputs=[result]
            )


def create_plan_ui():
    """创建新增计划界面"""
    ui = PlanCreateUI()
    return ui.build_ui()
