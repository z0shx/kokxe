"""
数据完整性验证和补全服务
定期检查K线数据的完整性，自动补全缺失的数据点
"""
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import func
from database.db import get_db
from database.models import TradingPlan, KlineData
from api.okx_client import OKXClient
from utils.logger import setup_logger
from utils.timezone_helper import get_current_beijing_time
from config import config

logger = setup_logger(__name__, "data_validation.log")


class DataValidationService:
    """数据完整性验证服务"""

    def __init__(self):
        self.running = False
        self.validation_thread = None
        self.okx_client = None

    async def initialize(self):
        """初始化服务"""
        self.okx_client = OKXClient()
        logger.info("数据验证服务初始化完成")

    def start_validation_scheduler(self):
        """启动数据验证调度器（在独立线程中运行）"""
        if self.running:
            logger.warning("数据验证调度器已在运行")
            return

        if not config.DATA_VALIDATION_ENABLED:
            logger.info("数据验证功能已禁用")
            return

        self.running = True
        self.validation_thread = threading.Thread(
            target=self._validation_loop,
            daemon=True,
            name="DataValidationThread"
        )
        self.validation_thread.start()
        logger.info(f"数据验证调度器已启动，验证间隔: {config.DATA_VALIDATION_INTERVAL_HOURS}小时")

    def stop_validation_scheduler(self):
        """停止数据验证调度器"""
        self.running = False
        if self.validation_thread and self.validation_thread.is_alive():
            self.validation_thread.join(timeout=5)
        logger.info("数据验证调度器已停止")

    def _validation_loop(self):
        """数据验证循环（在独立线程中运行）"""
        logger.info("数据验证循环开始")

        # 创建新的事件循环用于异步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.running:
                try:
                    # 执行数据验证
                    loop.run_until_complete(self.validate_all_plans_data())

                    # 等待下一次验证
                    interval_seconds = config.DATA_VALIDATION_INTERVAL_HOURS * 3600
                    logger.info(f"下一次数据验证将在 {config.DATA_VALIDATION_INTERVAL_HOURS} 小时后执行")

                    # 分段等待，以便能够及时响应停止信号
                    wait_start = time.time()
                    while time.time() - wait_start < interval_seconds and self.running:
                        time.sleep(60)  # 每分钟检查一次停止信号

                except Exception as e:
                    logger.error(f"数据验证循环出错: {e}")
                    # 出错后等待1小时再重试
                    for _ in range(60):
                        if not self.running:
                            break
                        time.sleep(60)

        except Exception as e:
            logger.error(f"数据验证循环致命错误: {e}")
        finally:
            loop.close()
            logger.info("数据验证循环结束")

    async def validate_all_plans_data(self):
        """验证所有运行中计划的数据完整性"""
        try:
            with get_db() as db:
                # 获取所有运行中的计划
                running_plans = db.query(TradingPlan).filter(
                    TradingPlan.status == 'running'
                ).all()

                if not running_plans:
                    logger.info("没有运行中的计划，跳过数据验证")
                    return

                logger.info(f"开始验证 {len(running_plans)} 个运行中计划的数据完整性")

                for plan in running_plans:
                    try:
                        await self._validate_plan_data(plan)
                    except Exception as e:
                        logger.error(f"验证计划 {plan.id} 数据失败: {e}")

                logger.info("所有计划数据验证完成")

        except Exception as e:
            logger.error(f"获取运行中计划失败: {e}")

    async def _validate_plan_data(self, plan: TradingPlan):
        """验证单个计划的数据完整性"""
        try:
            logger.info(f"开始验证计划 {plan.id} ({plan.inst_id}) 的数据完整性")

            # 检查数据完整性
            missing_data = await self._detect_missing_data(plan)

            if not missing_data:
                logger.info(f"计划 {plan.id} 数据完整，无需补全")
                return

            logger.warning(f"计划 {plan.id} 检测到 {len(missing_data)} 个缺失数据点，开始补全")

            # 补全缺失数据
            await self._fill_missing_data(plan, missing_data)

            logger.info(f"计划 {plan.id} 数据补全完成")

        except Exception as e:
            logger.error(f"验证计划 {plan.id} 数据失败: {e}")

    async def _detect_missing_data(self, plan: TradingPlan) -> List[datetime]:
        """检测缺失的数据点"""
        try:
            with get_db() as db:
                # 获取计划的时间间隔配置
                interval = plan.interval or config.CANDLE_INTERVAL

                # 计算时间间隔（分钟）
                interval_minutes = self._get_interval_minutes(interval)

                # 获取最新和最早的数据时间
                latest_record = db.query(func.max(KlineData.timestamp)).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == interval
                ).scalar()

                earliest_record = db.query(func.min(KlineData.timestamp)).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == interval
                ).scalar()

                if not latest_record:
                    logger.warning(f"计划 {plan.id} 没有历史数据")
                    return []

                # 计算期望的数据点数量
                now = get_current_beijing_time().replace(tzinfo=None)  # 移除时区信息以避免比较错误

                # 从最早记录开始，计算到现在的所有期望时间点
                expected_times = []
                current_time = earliest_record.replace(second=0, microsecond=0)

                # 确保时间对齐到间隔边界
                current_time = self._align_time_to_interval(current_time, interval_minutes)

                while current_time <= now:
                    expected_times.append(current_time)
                    current_time += timedelta(minutes=interval_minutes)

                # 获取实际存在的数据时间
                existing_records = db.query(KlineData.timestamp).filter(
                    KlineData.inst_id == plan.inst_id,
                    KlineData.interval == interval,
                    KlineData.timestamp >= earliest_record,
                    KlineData.timestamp <= now
                ).all()

                existing_times = {r[0] for r in existing_records}

                # 找出缺失的时间点
                missing_times = [t for t in expected_times if t not in existing_times]

                # 只返回最近的缺失数据（限制数量）
                max_missing = config.DATA_VALIDATION_MAX_MISSING_POINTS
                if len(missing_times) > max_missing:
                    missing_times = missing_times[-max_missing:]
                    logger.warning(f"计划 {plan.id} 缺失数据点过多，只补全最近的 {max_missing} 个")

                logger.info(f"计划 {plan.id} 检测到 {len(missing_times)} 个缺失数据点")
                return missing_times

        except Exception as e:
            logger.error(f"检测计划 {plan.id} 缺失数据失败: {e}")
            return []

    async def _fill_missing_data(self, plan: TradingPlan, missing_times: List[datetime]):
        """补全缺失的数据"""
        try:
            interval = plan.interval or config.CANDLE_INTERVAL
            batch_size = config.DATA_VALIDATION_BATCH_SIZE

            # 分批处理缺失数据
            for i in range(0, len(missing_times), batch_size):
                batch_times = missing_times[i:i + batch_size]

                logger.info(f"补全计划 {plan.id} 第 {i//batch_size + 1} 批数据 ({len(batch_times)} 个点)")

                for missing_time in batch_times:
                    try:
                        # 转换为毫秒时间戳
                        after_timestamp = str(int(missing_time.timestamp() * 1000))
                        before_timestamp = str(int((missing_time + timedelta(minutes=self._get_interval_minutes(interval))).timestamp() * 1000))

                        # 使用OKXClient获取历史K线数据
                        kline_data = await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.okx_client.get_history_candles,
                            plan.inst_id,  # inst_id
                            interval,         # bar
                            after_timestamp, # after
                            before_timestamp,# before
                            1                # limit (只获取一个点的数据)
                        )

                        if kline_data:
                            # 使用OKXClient的parse_candle_data方法正确解析数据
                            formatted_data = []
                            for candle in kline_data:
                                parsed = self.okx_client.parse_candle_data(candle)
                                if parsed:
                                    # 保持UTC时间戳用于数据库存储，UI显示时再转换为北京时间
                                    utc_time = parsed['timestamp'].replace(tzinfo=None)

                                    formatted_data.append({
                                        'instId': plan.inst_id,
                                        'ts': str(int(parsed['timestamp'].timestamp() * 1000)),  # 毫秒时间戳
                                        'o': str(parsed['open']),      # 开盘价
                                        'h': str(parsed['high']),      # 最高价
                                        'l': str(parsed['low']),       # 最低价
                                        'c': str(parsed['close']),     # 收盘价
                                        'vol': str(parsed['volume']),  # 成交量
                                        'ccy': str(parsed['amount'])   # 成交额
                                    })

                            # 保存到数据库
                            await self._save_kline_data(plan.id, formatted_data, interval)
                            logger.debug(f"已补全计划 {plan.id} {missing_time} 的数据")
                        else:
                            logger.warning(f"无法获取计划 {plan.id} {missing_time} 的K线数据")

                    except Exception as e:
                        logger.error(f"补全计划 {plan.id} {missing_time} 数据失败: {e}")
                        continue

                # 批次间稍作延迟，避免API限流
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"补全计划 {plan.id} 缺失数据失败: {e}")

    async def _save_kline_data(self, plan_id: int, kline_data: List[Dict], interval: str):
        """保存K线数据到数据库"""
        try:
            with get_db() as db:
                for data in kline_data:
                    logger.debug(f"处理数据: {data}")

                    # 转换时间戳 - 确保ts是整数而不是字符串
                    ts_value = int(data['ts']) if isinstance(data['ts'], str) else data['ts']
                    timestamp = datetime.fromtimestamp(ts_value / 1000)
                    logger.debug(f"转换后的时间戳: {timestamp}")

                    # 检查是否已存在
                    existing = db.query(KlineData).filter(
                        KlineData.inst_id == data['instId'],
                        KlineData.interval == interval,
                        KlineData.timestamp == timestamp
                    ).first()

                    if existing:
                        logger.debug(f"更新现有数据: {timestamp}")
                        # 更新现有数据 - 确保所有值都是正确的类型
                        existing.open = float(data['o'])
                        existing.high = float(data['h'])
                        existing.low = float(data['l'])
                        existing.close = float(data['c'])
                        existing.volume = float(data['vol'])
                        existing.amount = float(data['ccy'])
                    else:
                        logger.debug(f"创建新数据: {timestamp}")
                        # 创建新数据 - 确保所有值都是正确的类型
                        kline = KlineData(
                            inst_id=data['instId'],
                            interval=interval,
                            timestamp=timestamp,
                            open=float(data['o']),
                            high=float(data['h']),
                            low=float(data['l']),
                            close=float(data['c']),
                            volume=float(data['vol']),
                            amount=float(data['ccy'])
                        )
                        db.add(kline)

                db.commit()
                logger.debug("数据库提交成功")

        except Exception as e:
            logger.error(f"保存K线数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def _get_interval_minutes(self, interval: str) -> int:
        """获取时间间隔的分钟数"""
        interval_mapping = {
            "30m": 30,
            "1H": 60,
            "2H": 120,
            "4H": 240
        }
        return interval_mapping.get(interval, 60)

    def _align_time_to_interval(self, dt: datetime, interval_minutes: int) -> datetime:
        """将时间对齐到间隔边界"""
        minutes = dt.minute
        aligned_minutes = (minutes // interval_minutes) * interval_minutes
        return dt.replace(minute=aligned_minutes, second=0, microsecond=0)

    async def manual_validate_plan(self, plan_id: int) -> Dict:
        """手动验证指定计划的数据完整性"""
        try:
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {"success": False, "message": "计划不存在"}

                await self._validate_plan_data(plan)
                return {"success": True, "message": "数据验证完成"}

        except Exception as e:
            logger.error(f"手动验证计划 {plan_id} 失败: {e}")
            return {"success": False, "message": f"验证失败: {str(e)}"}

    def get_validation_status(self) -> Dict:
        """获取验证服务状态"""
        return {
            "running": self.running,
            "enabled": config.DATA_VALIDATION_ENABLED,
            "interval_hours": config.DATA_VALIDATION_INTERVAL_HOURS,
            "max_missing_points": config.DATA_VALIDATION_MAX_MISSING_POINTS,
            "batch_size": config.DATA_VALIDATION_BATCH_SIZE,
            "thread_alive": self.validation_thread.is_alive() if self.validation_thread else False
        }


# 全局数据验证服务实例
data_validation_service = DataValidationService()