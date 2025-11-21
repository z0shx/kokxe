"""
数据完整性检查工具
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from sqlalchemy import func, and_
from database.models import KlineData
from database.db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__, "data_checker.log")


class DataIntegrityChecker:
    """数据完整性检查器"""

    # 时间间隔映射（分钟）
    INTERVAL_MINUTES = {
        '30m': 30,
        '1H': 60,
        '2H': 120,
        '4H': 240,
    }

    def __init__(self, inst_id: str, interval: str):
        """
        初始化数据检查器

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
        """
        self.inst_id = inst_id
        self.interval = interval
        self.interval_minutes = self.INTERVAL_MINUTES.get(interval, 60)

    def check_data_status(self) -> Dict:
        """
        检查数据状态

        Returns:
            字典包含：
            - has_data: 是否有数据
            - count: 数据总条数
            - start_time: 最早时间
            - end_time: 最新时间
            - expected_count: 预期数据条数
            - missing_count: 缺失数据条数
            - completeness: 完整度百分比
            - is_complete: 是否完整（一年数据且数据同步到今天）
            - missing_segments: 缺失的数据段列表
            - is_up_to_date: 数据是否已同步到今天
        """
        with get_db() as db:
            # 查询数据总数
            count = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                )
            ).count()

            if count == 0:
                logger.info(f"[{self.inst_id}][{self.interval}] 无数据")
                return {
                    'has_data': False,
                    'count': 0,
                    'start_time': None,
                    'end_time': None,
                    'expected_count': 0,
                    'missing_count': 0,
                    'completeness': 0.0,
                    'is_complete': False,
                    'missing_segments': [],
                    'is_up_to_date': False
                }

            # 查询时间范围
            result = db.query(
                func.min(KlineData.timestamp).label('start_time'),
                func.max(KlineData.timestamp).label('end_time')
            ).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval
                )
            ).first()

            start_time = result.start_time
            end_time = result.end_time

            # 计算预期数据条数（从最早到最新）
            time_diff = end_time - start_time
            expected_count = int(time_diff.total_seconds() / 60 / self.interval_minutes) + 1

            # 计算缺失数据条数
            missing_count = expected_count - count

            # 计算完整度
            completeness = (count / expected_count * 100) if expected_count > 0 else 0

            # 检查数据是否同步到今天（容忍一个时间间隔的延迟）
            # 重要: 使用 UTC 时间进行比较，避免时区问题
            now = datetime.now(timezone.utc)
            tolerance = timedelta(minutes=self.interval_minutes * 2)  # 容忍2个时间间隔

            # 确保 end_time 是 timezone-aware 的
            if end_time.tzinfo is None:
                # 如果是 naive datetime，假设它是 UTC
                end_time_utc = end_time.replace(tzinfo=timezone.utc)
            else:
                end_time_utc = end_time

            is_up_to_date = (now - end_time_utc) <= tolerance

            # 检查是否满足一年数据（从最新数据往前推365天）
            one_year_ago_from_latest = end_time_utc - timedelta(days=365)

            # 确保 start_time 也是 timezone-aware 的
            if start_time.tzinfo is None:
                start_time_utc = start_time.replace(tzinfo=timezone.utc)
            else:
                start_time_utc = start_time

            has_one_year = start_time_utc <= one_year_ago_from_latest

            # 检查是否完整（一年数据、完整度 >= 99%、且数据已同步到今天）
            is_complete = has_one_year and completeness >= 99.0 and is_up_to_date

            # 查找缺失的数据段
            missing_segments = self._find_missing_segments(db, start_time, end_time)

            logger.info(
                f"[{self.inst_id}][{self.interval}] "
                f"数据状态: {count}/{expected_count} 条, "
                f"完整度: {completeness:.2f}%, "
                f"缺失段: {len(missing_segments)} 个, "
                f"最新数据: {end_time_utc}, "
                f"是否最新: {is_up_to_date}, "
                f"一年数据: {has_one_year}, "
                f"是否完整: {is_complete}"
            )

            return {
                'has_data': True,
                'count': count,
                'start_time': start_time,
                'end_time': end_time,
                'expected_count': expected_count,
                'missing_count': missing_count,
                'completeness': completeness,
                'is_complete': is_complete,
                'missing_segments': missing_segments,
                'is_up_to_date': is_up_to_date
            }

    def _find_missing_segments(
        self,
        db,
        start_time: datetime,
        end_time: datetime,
        max_segments: int = 10
    ) -> List[Dict]:
        """
        查找缺失的数据段

        Args:
            db: 数据库会话
            start_time: 开始时间
            end_time: 结束时间
            max_segments: 最大返回段数

        Returns:
            缺失数据段列表，每个段包含 start_time 和 end_time
        """
        # 查询所有时间戳，按时间排序
        timestamps = db.query(KlineData.timestamp).filter(
            and_(
                KlineData.inst_id == self.inst_id,
                KlineData.interval == self.interval,
                KlineData.timestamp >= start_time,
                KlineData.timestamp <= end_time
            )
        ).order_by(KlineData.timestamp).all()

        timestamps = [t[0] for t in timestamps]

        if len(timestamps) < 2:
            return []

        # 查找缺失段
        missing_segments = []
        interval_delta = timedelta(minutes=self.interval_minutes)

        for i in range(len(timestamps) - 1):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            expected_next = current_time + interval_delta

            # 如果下一个时间戳不是预期的，说明有缺失
            if next_time > expected_next:
                missing_segments.append({
                    'start_time': expected_next,
                    'end_time': next_time - interval_delta,
                    'missing_count': int((next_time - expected_next).total_seconds() / 60 / self.interval_minutes)
                })

                # 限制返回段数
                if len(missing_segments) >= max_segments:
                    break

        return missing_segments

    def get_download_strategy(self) -> Dict:
        """
        获取数据下载策略

        Returns:
            字典包含：
            - strategy: 策略类型（no_data/partial_data/complete）
            - action: 建议的操作（download_history/fill_gaps/enable_ws）
            - download_params: 下载参数
            - message: 提示消息
        """
        status = self.check_data_status()

        # 情况1：无数据
        if not status['has_data']:
            # 不传时间参数，先获取最新可用数据
            logger.info(
                f"[{self.inst_id}][{self.interval}] "
                f"无数据，将先获取最新数据，然后往前补充到一年"
            )

            return {
                'strategy': 'no_data',
                'action': 'download_history',
                'download_params': [{
                    'get_latest_first': True
                }],
                'message': f'未发现 {self.inst_id} {self.interval} 的数据，将下载最近一年的历史数据'
            }

        # 情况2：数据不完整或未同步到今天
        if not status['is_complete']:
            # 从最早数据时间戳开始，使用 after 参数往前补充到一年
            start_time = status['start_time']
            end_time = status['end_time']
            now = datetime.now()

            # 计算一年前的时间戳（基于当前时间）
            one_year_ago = now - timedelta(days=365)

            download_params = []

            # 如果数据未同步到今天，先同步最新数据
            if not status['is_up_to_date']:
                logger.info(
                    f"[{self.inst_id}][{self.interval}] "
                    f"数据未同步到今天（最新: {end_time}），将先同步最新数据"
                )
                # 先获取最新数据（不传参数获取最近300条）
                download_params.append({
                    'get_latest_first': True
                })

            # 检查是否需要补充历史数据到一年前
            if start_time > one_year_ago:
                # 使用最早数据的时间戳作为 after，往前补充到一年前
                download_params.append({
                    'use_after': True,
                    'after_time': int(start_time.timestamp() * 1000),
                    'one_year_ago_time': int(one_year_ago.timestamp() * 1000),
                    'reason': 'fill_to_one_year'
                })

            # 补充缺失段
            for segment in status['missing_segments']:
                if segment['missing_count'] > 5:
                    # 从缺失段的结束时间往前下载到开始时间
                    download_params.append({
                        'use_after': True,
                        'after_time': int(segment['end_time'].timestamp() * 1000),
                        'one_year_ago_time': int(segment['start_time'].timestamp() * 1000),
                        'reason': 'fill_gap'
                    })

            # 构建消息
            messages = []
            if not status['is_up_to_date']:
                days_behind = (now - end_time).days
                messages.append(f'数据落后 {days_behind} 天，将同步到今天')
            if start_time > one_year_ago:
                messages.append(f'将补充历史数据到 {one_year_ago.strftime("%Y-%m-%d")}')
            if status['missing_segments']:
                messages.append(f'将填补 {len(status["missing_segments"])} 处缺失')

            message = '；'.join(messages) if messages else '数据需要更新'

            return {
                'strategy': 'partial_data',
                'action': 'fill_gaps',
                'download_params': download_params,
                'status': status,
                'message': message
            }

        # 情况3：数据完整
        return {
            'strategy': 'complete',
            'action': 'enable_ws',
            'download_params': [],
            'status': status,
            'message': f'数据完整：{status["count"]} 条数据，完整度 {status["completeness"]:.1f}%，可以启动 WebSocket 实时同步'
        }

    def get_preview_data_params(self, last_days: int = None) -> Dict:
        """
        获取预览数据的查询参数

        Args:
            last_days: 最后N天的数据，如果为None则默认最后10%数据

        Returns:
            查询参数字典
        """
        status = self.check_data_status()

        if not status['has_data']:
            return {
                'has_data': False,
                'limit': 0,
                'offset': 0,
                'start_time': None,
                'end_time': None
            }

        if last_days is not None:
            # 根据天数计算
            end_time = status['end_time']
            start_time = end_time - timedelta(days=last_days)
        else:
            # 默认最后 10% 数据
            total_time = status['end_time'] - status['start_time']
            preview_time = total_time * 0.1
            start_time = status['end_time'] - preview_time
            end_time = status['end_time']

        return {
            'has_data': True,
            'start_time': start_time,
            'end_time': end_time,
            'inst_id': self.inst_id,
            'interval': self.interval
        }

    def query_preview_data(self, last_days: int = None) -> List[Dict]:
        """
        查询预览数据

        Args:
            last_days: 最后N天的数据

        Returns:
            K线数据字典列表
        """
        params = self.get_preview_data_params(last_days)

        if not params['has_data']:
            return []

        with get_db() as db:
            data = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == params['inst_id'],
                    KlineData.interval == params['interval'],
                    KlineData.timestamp >= params['start_time'],
                    KlineData.timestamp <= params['end_time']
                )
            ).order_by(KlineData.timestamp).all()

            # 转换为字典列表，避免session关闭后无法访问
            result = []
            for kline in data:
                result.append({
                    'timestamp': kline.timestamp,
                    'open': kline.open,
                    'high': kline.high,
                    'low': kline.low,
                    'close': kline.close,
                    'volume': kline.volume,
                    'amount': kline.amount
                })

            return result
