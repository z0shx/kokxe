"""
数据库K线数据集类
从PostgreSQL数据库读取K线数据用于Kronos模型训练
"""
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime
from typing import List, Optional
from database.db import get_db
from database.models import KlineData
from sqlalchemy import and_
from utils.logger import setup_logger

logger = setup_logger(__name__, "dataset.log")


class DatabaseKlineDataset(Dataset):
    """从数据库读取K线数据的Dataset"""

    def __init__(
        self,
        inst_id: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        data_type: str = 'train',
        lookback_window: int = 512,
        predict_window: int = 48,
        clip: float = 5.0,
        seed: int = 100,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0
    ):
        """
        初始化数据集

        Args:
            inst_id: 交易对
            interval: 时间颗粒度
            start_time: 开始时间
            end_time: 结束时间
            data_type: 'train', 'val', 或 'test'
            lookback_window: 历史窗口长度
            predict_window: 预测窗口长度
            clip: 标准化后的裁剪值
            seed: 随机种子
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        self.inst_id = inst_id
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']

        self.py_rng = random.Random(seed)

        # 加载并预处理数据
        self._load_and_preprocess_data()
        self._split_data_by_time()

        # n_samples 在 _split_data_by_time 中计算
        # self.n_samples = len(self.data) - self.window + 1

    def _load_and_preprocess_data(self):
        """从数据库加载并预处理数据"""
        logger.info(f"Loading data from database: {self.inst_id} {self.interval}")

        with get_db() as db:
            # 查询K线数据
            klines = db.query(KlineData).filter(
                and_(
                    KlineData.inst_id == self.inst_id,
                    KlineData.interval == self.interval,
                    KlineData.timestamp >= self.start_time,
                    KlineData.timestamp <= self.end_time
                )
            ).order_by(KlineData.timestamp).all()

            if not klines:
                raise ValueError(f"No data found for {self.inst_id} {self.interval}")

            # 转换为DataFrame
            data = []
            for kline in klines:
                data.append({
                    'timestamps': kline.timestamp,
                    'open': kline.open,
                    'high': kline.high,
                    'low': kline.low,
                    'close': kline.close,
                    'volume': kline.volume,
                    'amount': kline.amount
                })

            df = pd.DataFrame(data)
            df['timestamps'] = pd.to_datetime(df['timestamps'])

            logger.info(f"Loaded {len(df)} rows from database")

        # 提取时间特征
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month

        # 检查缺失值
        if df[self.feature_list].isnull().any().any():
            logger.warning("Found NaN values, filling with forward fill")
            df[self.feature_list] = df[self.feature_list].fillna(method='ffill').fillna(method='bfill')

        self.raw_df = df

    def _split_data_by_time(self):
        """按时间顺序分割数据集"""
        total_len = len(self.raw_df)

        # 检查数据是否足够（至少需要window条数据生成1个样本）
        min_required = self.window

        if total_len < min_required:
            raise ValueError(
                f"数据不足: 总共{total_len}条, 至少需要{min_required}条 "
                f"(lookback_window={self.lookback_window} + predict_window={self.predict_window} + 1)"
            )

        # 先计算初始分割点
        train_end_idx = int(total_len * self.train_ratio)
        val_end_idx = int(total_len * (self.train_ratio + self.val_ratio))

        # 检查训练集是否足够
        train_len = train_end_idx
        if train_len < min_required:
            logger.warning(
                f"训练集数据不足({train_len}条 < {min_required}条), "
                f"调整为使用 {min_required + 50} 条作为训练集（留50条作为buffer）"
            )
            train_end_idx = min(min_required + 50, int(total_len * 0.8))  # 至少保留20%给验证集
            val_end_idx = total_len

        # 计算各部分的实际长度
        train_len = train_end_idx
        val_len = val_end_idx - train_end_idx
        test_len = total_len - val_end_idx

        # 验证集数据不足时，统一调整分割点（所有数据集类型都用同样的调整）
        if val_len < min_required:
            logger.warning(
                f"验证集数据不足({val_len}条 < {min_required}条), "
                f"调整为使用所有数据的10%作为验证集"
            )
            # 使用最后10%作为验证集
            train_end_idx = int(total_len * 0.9)
            val_end_idx = total_len
            val_len = val_end_idx - train_end_idx

            # 如果还是不够，使用最后min_required条作为验证集
            if val_len < min_required:
                logger.warning(f"仍然不足，使用最后{min_required}条作为验证集")
                train_end_idx = total_len - min_required
                val_end_idx = total_len

        # 测试集数据不足时的处理
        if self.data_type == 'test' and test_len < min_required:
            logger.warning(
                f"测试集数据不足({test_len}条 < {min_required}条), "
                f"将跳过测试集"
            )
            # 测试集不足时，返回空数据
            self.data = self.raw_df.iloc[0:0].reset_index(drop=True)
            logger.info(
                f"[{self.data_type.upper()}] Split data: "
                f"total={total_len}, current={len(self.data)} (不足，已跳过)"
            )
            return

        # 根据统一调整后的分割点来分割数据
        if self.data_type == 'train':
            self.data = self.raw_df.iloc[:train_end_idx].reset_index(drop=True)
        elif self.data_type == 'val':
            self.data = self.raw_df.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
        elif self.data_type == 'test':
            self.data = self.raw_df.iloc[val_end_idx:].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")

        logger.info(
            f"[{self.data_type.upper()}] Split data: "
            f"total={total_len}, current={len(self.data)}"
        )

        # 重新计算样本数（每个数据集独立计算）
        self.n_samples = max(0, len(self.data) - self.window + 1)

        if self.n_samples <= 0:
            raise ValueError(
                f"[{self.data_type.upper()}] 数据不足以生成样本: "
                f"数据长度={len(self.data)}, 窗口大小={self.window}, "
                f"需要至少{self.window}条数据"
            )

        logger.info(
            f"[{self.data_type.upper()}] Dataset created: "
            f"inst_id={self.inst_id}, interval={self.interval}, "
            f"data_length={len(self.data)}, samples={self.n_samples}"
        )

    def set_epoch_seed(self, epoch_seed: int):
        """设置epoch级别的随机种子"""
        self.py_rng = random.Random(self.seed + epoch_seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            x: (window, 6) - 标准化后的特征
            y: (window, 6) - 标准化后的特征（同x）
        """
        # 训练集：随机起始索引
        if self.data_type == 'train':
            # 在整个有效范围内随机选择起始位置，确保不会超出边界
            max_start_idx = len(self.data) - self.window
            start_idx = self.py_rng.randint(0, max_start_idx)
        else:
            # 验证集/测试集：顺序索引
            start_idx = idx

        end_idx = start_idx + self.window

        # 提取数据
        sample = self.data.iloc[start_idx:end_idx][self.feature_list].values  # (window, 6)

        # 标准化
        mean = sample.mean(axis=0, keepdims=True)
        std = sample.std(axis=0, keepdims=True) + 1e-8
        sample_normalized = (sample - mean) / std

        # 裁剪
        sample_normalized = np.clip(sample_normalized, -self.clip, self.clip)

        # 转换为tensor，注意：不添加batch维度！
        x = torch.FloatTensor(sample_normalized)  # (window, 6)

        # 生成时间戳特征 (minute, hour, weekday, day, month) - 5个特征
        # 直接从已经预处理好的列中读取
        minute = torch.tensor(self.data.iloc[start_idx:end_idx]['minute'].values, dtype=torch.float32)
        hour = torch.tensor(self.data.iloc[start_idx:end_idx]['hour'].values, dtype=torch.float32)
        weekday = torch.tensor(self.data.iloc[start_idx:end_idx]['weekday'].values, dtype=torch.float32)
        day = torch.tensor(self.data.iloc[start_idx:end_idx]['day'].values, dtype=torch.float32)
        month = torch.tensor(self.data.iloc[start_idx:end_idx]['month'].values, dtype=torch.float32)

        # 组合成 (window, 5) - 顺序: minute, hour, weekday, day, month
        x_stamp = torch.stack([minute, hour, weekday, day, month], dim=1)  # (window, 5)

        return x, x_stamp


def get_kline_dataframe(
    inst_id: str,
    interval: str,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    从数据库获取K线数据的DataFrame（用于推理）

    Returns:
        DataFrame with columns: timestamps, open, high, low, close, volume, amount
    """
    with get_db() as db:
        klines = db.query(KlineData).filter(
            and_(
                KlineData.inst_id == inst_id,
                KlineData.interval == interval,
                KlineData.timestamp >= start_time,
                KlineData.timestamp <= end_time
            )
        ).order_by(KlineData.timestamp).all()

        if not klines:
            return pd.DataFrame()

        data = []
        for kline in klines:
            data.append({
                'timestamps': kline.timestamp,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume,
                'amount': kline.amount
            })

        df = pd.DataFrame(data)
        df['timestamps'] = pd.to_datetime(df['timestamps'])

        return df
