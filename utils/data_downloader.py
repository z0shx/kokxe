"""
智能数据下载器
"""
from datetime import datetime, timezone
from typing import List, Dict
from sqlalchemy.dialects.postgresql import insert
from api.okx_client import OKXClient
from database.models import KlineData
from database.db import get_db
from utils.logger import setup_logger
from utils.data_checker import DataIntegrityChecker

logger = setup_logger(__name__, "data_downloader.log")


class DataDownloader:
    """智能数据下载器"""

    def __init__(self, inst_id: str, interval: str, is_demo: bool = True):
        """
        初始化数据下载器

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            is_demo: 是否模拟盘
        """
        self.inst_id = inst_id
        self.interval = interval
        self.is_demo = is_demo
        self.okx_client = OKXClient(is_demo=is_demo)
        self.checker = DataIntegrityChecker(inst_id, interval)
        self.environment = "DEMO" if is_demo else "LIVE"

    def smart_download(self, progress_callback=None) -> Dict:
        """
        智能下载数据

        Args:
            progress_callback: 进度回调函数(current, total, message)

        Returns:
            下载结果字典
        """
        # 获取下载策略
        strategy = self.checker.get_download_strategy()

        logger.info(
            f"[{self.environment}][{self.inst_id}][{self.interval}] "
            f"下载策略: {strategy['strategy']}, 操作: {strategy['action']}"
        )

        if strategy['action'] == 'enable_ws':
            # 数据完整，无需下载
            return {
                'success': True,
                'action': 'enable_ws',
                'message': strategy['message'],
                'downloaded_count': 0,
                'status': strategy['status']
            }

        # 执行下载
        download_params = strategy['download_params']
        total_downloaded = 0
        total_tasks = len(download_params)

        for idx, params in enumerate(download_params, 1):
            if progress_callback:
                progress_callback(
                    idx - 1,
                    total_tasks,
                    f"正在下载数据段 {idx}/{total_tasks}..."
                )

            try:
                logger.info(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"下载段 {idx}/{total_tasks} 参数: {params}"
                )
                count = self._download_data_segment(params)
                total_downloaded += count

                logger.info(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"下载段 {idx}/{total_tasks} 完成，下载 {count} 条数据"
                )

            except Exception as e:
                logger.error(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"下载段 {idx}/{total_tasks} 失败: {e}, 参数类型: {type(params)}"
                )
                import traceback
                traceback.print_exc()

        if progress_callback:
            progress_callback(total_tasks, total_tasks, "下载完成")

        # 重新检查数据状态
        final_status = self.checker.check_data_status()

        return {
            'success': True,
            'action': strategy['action'],
            'message': f"下载完成，共下载 {total_downloaded} 条数据",
            'downloaded_count': total_downloaded,
            'status': final_status
        }

    def _download_data_segment(self, params: Dict) -> int:
        """
        下载单个数据段

        Args:
            params: 下载参数

        Returns:
            下载的数据条数
        """
        downloaded_count = 0

        # 情况1：首次下载，先获取最新可用数据
        if params.get('get_latest_first'):
            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"首次下载：先获取最新可用数据"
            )

            # 不传任何时间参数，获取最新数据
            candles = self.okx_client.get_history_candles(
                inst_id=self.inst_id,
                bar=self.interval,
                limit=300
            )

            if not candles:
                logger.warning(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"无法获取最新数据"
                )
                return 0

            # 保存第一批数据
            saved_count = self._save_candles(candles)
            downloaded_count += saved_count

            # 获取最新数据的时间戳
            newest_time = int(candles[0][0])
            oldest_time = int(candles[-1][0])

            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"首批下载 {len(candles)} 条，保存 {saved_count} 条，"
                f"时间范围：{datetime.fromtimestamp(newest_time/1000)} ~ "
                f"{datetime.fromtimestamp(oldest_time/1000)}"
            )

            # 计算一年前的时间戳（基于最新数据）
            one_year_ms = 365 * 24 * 60 * 60 * 1000
            target_time = newest_time - one_year_ms

            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"从 {datetime.fromtimestamp(oldest_time/1000)} "
                f"继续往前下载到 {datetime.fromtimestamp(target_time/1000)}"
            )

            # 使用 after 参数从最早的数据继续往前下载
            current_after = oldest_time
            iteration = 1

            while current_after > target_time:
                iteration += 1
                candles = self.okx_client.get_history_candles(
                    inst_id=self.inst_id,
                    bar=self.interval,
                    after=str(current_after),
                    limit=300
                )

                if not candles:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"第 {iteration} 次请求无数据，停止下载"
                    )
                    break

                # 保存数据
                saved_count = self._save_candles(candles)
                downloaded_count += saved_count

                # 返回的数据从新到旧排序：candles[0]最新，candles[-1]最旧
                newest_time_batch = int(candles[0][0])
                oldest_time_batch = int(candles[-1][0])

                logger.info(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"第 {iteration} 次下载 {len(candles)} 条，保存 {saved_count} 条，"
                    f"时间范围：{datetime.fromtimestamp(newest_time_batch/1000)} ~ "
                    f"{datetime.fromtimestamp(oldest_time_batch/1000)}"
                )

                # 检查最旧的时间戳是否有进展
                if oldest_time_batch >= current_after:
                    logger.warning(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"时间戳未向前推进（oldest={oldest_time_batch} >= after={current_after}），停止下载"
                    )
                    break

                # 使用最旧的时间戳作为下一次的 after 参数
                current_after = oldest_time_batch

                # 检查是否已经达到目标时间
                if current_after <= target_time:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"已达到目标时间，停止下载"
                    )
                    break

                # 如果返回数据少于300条，说明已经没有更多数据
                if len(candles) < 300:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"返回数据少于300条，可能已到达最早可用数据"
                    )
                    break

            return downloaded_count

        if params.get('use_after'):
            # 使用 after 参数往前补充历史数据
            # OKX API: after 参数实际返回该时间戳之前的历史数据
            after_time = params['after_time']
            one_year_ago = params.get('one_year_ago_time', 0)

            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"从 {datetime.fromtimestamp(after_time/1000)} "
                f"往前补充到一年前 {datetime.fromtimestamp(one_year_ago/1000) if one_year_ago else '或最早可用数据'}"
            )

            current_after = after_time
            iteration = 0

            while True:
                iteration += 1
                candles = self.okx_client.get_history_candles(
                    inst_id=self.inst_id,
                    bar=self.interval,
                    after=str(current_after),
                    limit=300
                )

                if not candles:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"第 {iteration} 次请求无数据，停止下载"
                    )
                    break

                # 保存数据
                saved_count = self._save_candles(candles)
                downloaded_count += saved_count

                # 返回的数据从新到旧排序：candles[0]最新，candles[-1]最旧
                newest_time = int(candles[0][0])
                oldest_time = int(candles[-1][0])

                logger.info(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"第 {iteration} 次下载 {len(candles)} 条，保存 {saved_count} 条，"
                    f"时间范围：{datetime.fromtimestamp(newest_time/1000)} ~ "
                    f"{datetime.fromtimestamp(oldest_time/1000)}"
                )

                # 检查是否已经达到一年前的时间
                # 注意：时间戳越早越小，所以要检查 oldest_time 是否已经早于目标时间
                if one_year_ago and oldest_time <= one_year_ago:
                    # oldest_time 已经早于或等于一年前的目标时间
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"已达到一年前的时间（oldest={datetime.fromtimestamp(oldest_time/1000)}, target={datetime.fromtimestamp(one_year_ago/1000)}），停止下载"
                    )
                    break

                # 检查最旧的时间戳是否有进展
                if oldest_time >= current_after:
                    logger.warning(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"时间戳未向前推进（oldest={oldest_time} >= after={current_after}），停止下载"
                    )
                    break

                # 使用最旧的时间戳作为下一次的 after 参数
                current_after = oldest_time

                # 如果返回数据少于300条，说明已经没有更多数据
                if len(candles) < 300:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"返回数据少于300条，可能已到达最早可用数据"
                    )
                    break

            return downloaded_count

        elif params.get('use_before'):
            # 使用 before 参数，从新到旧下载
            before_time = params['before_time']
            target_time = params.get('target_time', 0)

            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"开始下载历史数据：从 {datetime.fromtimestamp(before_time/1000)} "
                f"到 {datetime.fromtimestamp(target_time/1000) if target_time > 0 else '最早'}"
            )

            while before_time > target_time:
                candles = self.okx_client.get_history_candles(
                    inst_id=self.inst_id,
                    bar=self.interval,
                    before=str(before_time),
                    limit=300
                )

                if not candles:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"没有更多数据，停止下载"
                    )
                    break

                # 保存数据
                saved_count = self._save_candles(candles)
                downloaded_count += saved_count

                # OKX API 使用 before 参数时，返回的数据是从新到旧排序：
                # candles[0] 是最新的（时间戳最大，但 < before）
                # candles[-1] 是最早的（时间戳最小）
                newest_time = int(candles[0][0])
                oldest_time = int(candles[-1][0])

                logger.info(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"本批次下载 {len(candles)} 条，保存 {saved_count} 条，"
                    f"时间范围：{datetime.fromtimestamp(newest_time/1000)} ~ "
                    f"{datetime.fromtimestamp(oldest_time/1000)}, "
                    f"before={datetime.fromtimestamp(before_time/1000)}"
                )

                # 检查最新的数据是否 < before（这是API的正确行为）
                if newest_time >= before_time:
                    logger.warning(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"API返回了异常数据（newest={newest_time} >= before={before_time}），停止下载"
                    )
                    break

                # 检查最早的数据是否真的比before更早
                if oldest_time >= before_time:
                    logger.warning(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"最早数据时间戳未更新（oldest={oldest_time} >= before={before_time}），停止下载"
                    )
                    break

                # 使用最早的时间戳作为下一次的 before 参数
                before_time = oldest_time

                # 检查是否已经达到目标时间
                if before_time <= target_time:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"已达到目标时间，停止下载"
                    )
                    break

                # 如果返回数据少于300条，说明已经没有更多数据
                if len(candles) < 300:
                    logger.info(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"返回数据少于300条，可能已到达最早数据"
                    )
                    break

        return downloaded_count

    def _save_candles(self, candles: List) -> int:
        """
        保存K线数据到数据库（带去重）

        Args:
            candles: K线数据列表

        Returns:
            实际保存的数据条数
        """
        if not candles:
            return 0

        with get_db() as db:
            saved_count = 0

            for idx, candle in enumerate(candles):
                try:
                    parsed = self.okx_client.parse_candle_data(candle)
                    if not parsed:
                        logger.warning(
                            f"[{self.environment}][{self.inst_id}][{self.interval}] "
                            f"解析K线数据失败: candle={candle}"
                        )
                        continue

                    # 使用 PostgreSQL 的 INSERT ... ON CONFLICT DO UPDATE 实现更新或插入
                    # 如果时间戳已存在,更新 OHLCV 数据 (以最新下载的为准)
                    stmt = insert(KlineData).values(
                        inst_id=self.inst_id,
                        interval=self.interval,
                        timestamp=parsed['timestamp'],
                        open=parsed['open'],
                        high=parsed['high'],
                        low=parsed['low'],
                        close=parsed['close'],
                        volume=parsed['volume'],
                        amount=parsed['amount']
                    ).on_conflict_do_update(
                        index_elements=['inst_id', 'interval', 'timestamp'],
                        set_={
                            'open': parsed['open'],
                            'high': parsed['high'],
                            'low': parsed['low'],
                            'close': parsed['close'],
                            'volume': parsed['volume'],
                            'amount': parsed['amount']
                        }
                    )

                    result = db.execute(stmt)

                    # 检查是否实际插入了数据
                    if result.rowcount > 0:
                        saved_count += 1
                    elif idx == 0:
                        # 第一条数据如果没有插入，记录日志
                        logger.debug(
                            f"[{self.environment}][{self.inst_id}][{self.interval}] "
                            f"数据已存在或未插入: timestamp={parsed['timestamp']}"
                        )

                except Exception as e:
                    logger.error(
                        f"[{self.environment}][{self.inst_id}][{self.interval}] "
                        f"保存K线数据失败: {e}"
                    )
                    import traceback
                    traceback.print_exc()

            db.commit()

            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"批量保存: {saved_count}/{len(candles)} 条数据"
            )

            return saved_count

    def fill_missing_gaps(self, max_segments: int = 5) -> Dict:
        """
        填补缺失的数据段

        Args:
            max_segments: 最大填补段数

        Returns:
            填补结果
        """
        status = self.checker.check_data_status()

        if not status['has_data']:
            return {
                'success': False,
                'message': '无数据，请先执行完整下载'
            }

        missing_segments = status['missing_segments'][:max_segments]

        if not missing_segments:
            return {
                'success': True,
                'message': '无缺失数据段',
                'filled_count': 0
            }

        total_filled = 0

        for idx, segment in enumerate(missing_segments, 1):
            logger.info(
                f"[{self.environment}][{self.inst_id}][{self.interval}] "
                f"填补缺失段 {idx}/{len(missing_segments)}: "
                f"{segment['start_time']} ~ {segment['end_time']}"
            )

            try:
                params = {
                    'use_before': True,
                    'before_time': int(segment['end_time'].timestamp() * 1000),
                    'start_time': int(segment['start_time'].timestamp() * 1000)
                }

                count = self._download_data_segment(params)
                total_filled += count

            except Exception as e:
                logger.error(
                    f"[{self.environment}][{self.inst_id}][{self.interval}] "
                    f"填补缺失段失败: {e}"
                )

        return {
            'success': True,
            'message': f'填补完成，共填补 {total_filled} 条数据',
            'filled_count': total_filled
        }
