"""
Kronos模型训练封装
将数据库数据转换为Kronos可训练格式
"""
import os
import sys
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from database.db import get_db
from database.models import KlineData
from sqlalchemy import and_
from utils.logger import setup_logger

# 添加父目录到路径，以便导入 Kronos 模型
KRONOS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(KRONOS_ROOT))

from model import Kronos, KronosTokenizer

logger = setup_logger(__name__, "kronos_trainer.log")


class KronosTrainer:
    """Kronos训练器封装"""

    def __init__(self, training_config: Dict, progress_callback=None):
        """
        初始化训练器

        Args:
            training_config: 训练配置字典，包含：
                - plan_id: 计划ID
                - inst_id: 交易对
                - interval: 时间颗粒度
                - data_start_time: 训练数据开始时间
                - data_end_time: 训练数据结束时间
                - finetune_params: 微调参数
                - save_path: 模型保存路径
            progress_callback: 进度回调函数 callback(progress: float, stage: str, message: str)
        """
        self.config = training_config
        self.logger = logger
        self.progress_callback = progress_callback

    def _report_progress(self, progress: float, stage: str, message: str):
        """报告训练进度"""
        if self.progress_callback:
            try:
                self.progress_callback(progress, stage, message)
            except Exception as e:
                self.logger.error(f"进度回调失败: {e}")

    def _validate_and_fix_pretrained_paths(self, model_paths: Dict) -> tuple:
        """
        验证并修复预训练模型路径

        Args:
            model_paths: 模型路径配置字典

        Returns:
            (tokenizer_path, predictor_path) 元组
        """
        pretrained_tokenizer = model_paths.get('pretrained_tokenizer', '')
        pretrained_predictor = model_paths.get('pretrained_predictor', '')

        # 检查路径是否有效
        need_fix = False
        if not pretrained_tokenizer or not os.path.exists(pretrained_tokenizer):
            self.logger.warning(f"预训练Tokenizer路径无效: '{pretrained_tokenizer}'")
            need_fix = True

        if not pretrained_predictor or not os.path.exists(pretrained_predictor):
            self.logger.warning(f"预训练Predictor路径无效: '{pretrained_predictor}'")
            need_fix = True

        # 如果需要修复，自动下载默认模型
        if need_fix:
            self.logger.info("尝试自动下载 kronos-base 预训练模型...")

            try:
                from services.model_service import ModelService
                tokenizer_path, predictor_path = ModelService.download_model('kronos-base')

                # 更新 model_paths 字典（会影响原始配置）
                model_paths['pretrained_tokenizer'] = tokenizer_path
                model_paths['pretrained_predictor'] = predictor_path

                self.logger.info(f"✅ 预训练模型已下载:")
                self.logger.info(f"  - Tokenizer: {tokenizer_path}")
                self.logger.info(f"  - Predictor: {predictor_path}")

                return tokenizer_path, predictor_path

            except Exception as e:
                error_msg = f"自动下载预训练模型失败: {e}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        self.logger.info(f"使用预训练模型:")
        self.logger.info(f"  - Tokenizer: {pretrained_tokenizer}")
        self.logger.info(f"  - Predictor: {pretrained_predictor}")

        return pretrained_tokenizer, pretrained_predictor

    def train(self) -> Dict:
        """
        执行训练

        Returns:
            结果字典: {
                'success': bool,
                'tokenizer_path': str,
                'predictor_path': str,
                'metrics': dict,
                'error': str
            }
        """
        try:
            self._report_progress(0.0, 'init', '初始化训练环境...')

            # 1. 获取训练参数
            finetune_params = self.config.get('finetune_params', {})
            data_params = finetune_params.get('data', {})
            train_params = finetune_params.get('training', {})
            model_paths = finetune_params.get('model_paths', {})

            lookback_window = data_params.get('lookback_window', 512)
            predict_window = data_params.get('predict_window', 48)
            clip = data_params.get('clip', 5.0)
            train_ratio = data_params.get('train_ratio', 0.9)
            val_ratio = data_params.get('val_ratio', 0.1)

            tokenizer_epochs = train_params.get('tokenizer_epochs', 25)
            predictor_epochs = train_params.get('basemodel_epochs', 50)
            batch_size = train_params.get('batch_size', 16)
            tokenizer_lr = train_params.get('tokenizer_learning_rate', 0.0002)
            predictor_lr = train_params.get('predictor_learning_rate', 0.000001)
            seed = train_params.get('seed', 42)

            # 验证并修复预训练模型路径
            pretrained_tokenizer, pretrained_predictor = self._validate_and_fix_pretrained_paths(model_paths)

            # 2. 创建保存目录
            save_base = Path(self.config['save_path'])
            save_base.mkdir(parents=True, exist_ok=True)

            tokenizer_save_path = save_base / "tokenizer"
            predictor_save_path = save_base / "predictor"
            tokenizer_save_path.mkdir(exist_ok=True)
            predictor_save_path.mkdir(exist_ok=True)

            # 3. 训练Tokenizer
            self.logger.info("=" * 60)
            self.logger.info("开始训练Tokenizer...")
            self.logger.info("=" * 60)
            self._report_progress(0.05, 'tokenizer', '开始训练Tokenizer...')

            tokenizer_metrics = self._train_tokenizer(
                pretrained_path=pretrained_tokenizer,
                save_path=str(tokenizer_save_path),
                lookback_window=lookback_window,
                predict_window=predict_window,
                epochs=tokenizer_epochs,
                batch_size=batch_size,
                lr=tokenizer_lr,
                seed=seed,
                clip=clip,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            if not tokenizer_metrics['success']:
                return {'success': False, 'error': f"Tokenizer训练失败: {tokenizer_metrics.get('error')}"}

            # 4. 训练Predictor
            self.logger.info("=" * 60)
            self.logger.info("开始训练Predictor...")
            self.logger.info("=" * 60)
            self._report_progress(0.50, 'predictor', '开始训练Predictor...')

            predictor_metrics = self._train_predictor(
                tokenizer_path=str(tokenizer_save_path),
                pretrained_path=pretrained_predictor,
                save_path=str(predictor_save_path),
                lookback_window=lookback_window,
                predict_window=predict_window,
                epochs=predictor_epochs,
                batch_size=batch_size,
                lr=predictor_lr,
                seed=seed,
                clip=clip,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            if not predictor_metrics['success']:
                return {'success': False, 'error': f"Predictor训练失败: {predictor_metrics.get('error')}"}

            # 5. 返回结果
            self.logger.info("=" * 60)
            self.logger.info("训练完成！")
            self.logger.info(f"Tokenizer最佳验证损失: {tokenizer_metrics.get('best_val_loss', 0):.6f}")
            self.logger.info(f"Predictor最佳验证损失: {predictor_metrics.get('best_val_loss', 0):.6f}")
            self.logger.info("=" * 60)

            self._report_progress(1.0, 'completed', '训练完成！')

            return {
                'success': True,
                'tokenizer_path': str(tokenizer_save_path),
                'predictor_path': str(predictor_save_path),
                'metrics': {
                    'tokenizer_loss': tokenizer_metrics.get('best_val_loss', 0),
                    'predictor_loss': predictor_metrics.get('best_val_loss', 0)
                }
            }

        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            self._report_progress(0.0, 'failed', f'训练失败: {str(e)}')
            return {'success': False, 'error': str(e)}

    def _train_tokenizer(self, **kwargs) -> Dict:
        """
        训练Tokenizer

        参考 finetune_csv/finetune_tokenizer.py，使用数据库数据进行训练
        """
        try:
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader
            from services.database_dataset import DatabaseKlineDataset
            from tqdm import tqdm

            # 获取参数
            lookback_window = kwargs.get('lookback_window', 512)
            predict_window = kwargs.get('predict_window', 48)
            epochs = kwargs.get('epochs', 25)
            batch_size = kwargs.get('batch_size', 16)
            lr = kwargs.get('lr', 0.0002)
            seed = kwargs.get('seed', 42)
            clip = kwargs.get('clip', 5.0)
            train_ratio = kwargs.get('train_ratio', 0.9)
            val_ratio = kwargs.get('val_ratio', 0.1)
            pretrained_path = kwargs.get('pretrained_path', '')
            save_path = kwargs.get('save_path', '')
            weight_decay = kwargs.get('weight_decay', 0.01)

            # 设备
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"使用设备: {device}")

            # 1. 创建训练和验证数据集
            self.logger.info("创建训练数据集...")
            train_dataset = DatabaseKlineDataset(
                inst_id=self.config['inst_id'],
                interval=self.config['interval'],
                start_time=self.config['data_start_time'],
                end_time=self.config['data_end_time'],
                data_type='train',
                lookback_window=lookback_window,
                predict_window=predict_window,
                clip=clip,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            self.logger.info("创建验证数据集...")
            val_dataset = DatabaseKlineDataset(
                inst_id=self.config['inst_id'],
                interval=self.config['interval'],
                start_time=self.config['data_start_time'],
                end_time=self.config['data_end_time'],
                data_type='val',
                lookback_window=lookback_window,
                predict_window=predict_window,
                clip=clip,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            # 2. 创建DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Gradio环境建议使用0
                pin_memory=True if torch.cuda.is_available() else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )

            self.logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

            # 3. 加载预训练模型
            self.logger.info(f"加载预训练Tokenizer: {pretrained_path}")
            tokenizer = KronosTokenizer.from_pretrained(pretrained_path, local_files_only=True)
            tokenizer = tokenizer.to(device)

            # 4. 设置优化器
            optimizer = torch.optim.AdamW(
                tokenizer.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # 5. 训练循环
            best_val_loss = float('inf')
            self.logger.info(f"开始训练Tokenizer: {epochs}个epoch...")

            for epoch in range(epochs):
                # 计算并报告进度 (Tokenizer占总进度的5%-50%)
                tokenizer_progress = 0.05 + (epoch / epochs) * 0.45
                self._report_progress(
                    tokenizer_progress,
                    'tokenizer',
                    f'Tokenizer Epoch {epoch+1}/{epochs}'
                )

                # 训练阶段
                tokenizer.train()
                train_loss = 0.0
                train_recon_loss = 0.0
                train_entropy_loss = 0.0

                # 设置epoch种子（用于数据增强的随机性）
                train_dataset.set_epoch_seed(epoch)

                for batch_idx, (batch_x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=True)):
                    # 移动到设备，DataLoader已经返回正确的 (B, T, 6) 形状
                    batch_x = batch_x.to(device)  # (B, T, 6)

                    # 前向传播 - tokenizer返回4个值：(z_pre, z), bsq_loss, quantized, z_indices
                    zs, bsq_loss, _, _ = tokenizer(batch_x)
                    z_pre, z = zs  # z_pre: 使用s1_bits的重建, z: 使用完整codebook的重建

                    # 计算损失
                    recon_loss_pre = F.mse_loss(z_pre, batch_x)
                    recon_loss_all = F.mse_loss(z, batch_x)
                    recon_loss = recon_loss_pre + recon_loss_all
                    loss = (recon_loss + bsq_loss) / 2

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 累计损失
                    train_loss += loss.item()
                    train_recon_loss += recon_loss.item()
                    train_entropy_loss += bsq_loss.item()

                avg_train_loss = train_loss / len(train_loader)
                avg_recon_loss = train_recon_loss / len(train_loader)
                avg_entropy_loss = train_entropy_loss / len(train_loader)

                # 验证阶段
                tokenizer.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(device)

                        # 前向传播 - tokenizer返回4个值
                        zs, bsq_loss, _, _ = tokenizer(batch_x)
                        _, z = zs  # 使用完整codebook的重建

                        # 只计算重建损失用于验证
                        recon_loss = F.mse_loss(z, batch_x)
                        loss = (recon_loss + bsq_loss) / 2

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f} (Recon: {avg_recon_loss:.6f}, Entropy: {avg_entropy_loss:.6f}) - "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.logger.info(f"保存最佳模型 (val_loss={best_val_loss:.6f}) -> {save_path}")
                    tokenizer.save_pretrained(save_path)

            self.logger.info(f"Tokenizer训练完成! 最佳验证损失: {best_val_loss:.6f}")

            return {
                'success': True,
                'best_val_loss': float(best_val_loss)
            }

        except Exception as e:
            self.logger.error(f"Tokenizer训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _train_predictor(self, **kwargs) -> Dict:
        """
        训练Predictor

        参考 finetune_csv/finetune_base_model.py，使用数据库数据进行训练
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from services.database_dataset import DatabaseKlineDataset
            from tqdm import tqdm

            # 获取参数
            tokenizer_path = kwargs.get('tokenizer_path', '')
            lookback_window = kwargs.get('lookback_window', 512)
            predict_window = kwargs.get('predict_window', 48)
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 16)
            lr = kwargs.get('lr', 0.000001)
            seed = kwargs.get('seed', 42)
            clip = kwargs.get('clip', 5.0)
            train_ratio = kwargs.get('train_ratio', 0.9)
            val_ratio = kwargs.get('val_ratio', 0.1)
            pretrained_path = kwargs.get('pretrained_path', '')
            save_path = kwargs.get('save_path', '')
            weight_decay = kwargs.get('weight_decay', 0.01)

            # 设备
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"使用设备: {device}")

            # 1. 加载训练好的Tokenizer（冻结）
            self.logger.info(f"加载Tokenizer: {tokenizer_path}")
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            tokenizer = tokenizer.to(device)
            tokenizer.eval()  # 冻结tokenizer
            for param in tokenizer.parameters():
                param.requires_grad = False

            # 2. 加载预训练Predictor
            self.logger.info(f"加载预训练Predictor: {pretrained_path}")
            model = Kronos.from_pretrained(pretrained_path, local_files_only=True)
            model = model.to(device)

            # 3. 创建训练和验证数据集
            self.logger.info("创建训练数据集...")
            train_dataset = DatabaseKlineDataset(
                inst_id=self.config['inst_id'],
                interval=self.config['interval'],
                start_time=self.config['data_start_time'],
                end_time=self.config['data_end_time'],
                data_type='train',
                lookback_window=lookback_window,
                predict_window=predict_window,
                clip=clip,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            self.logger.info("创建验证数据集...")
            val_dataset = DatabaseKlineDataset(
                inst_id=self.config['inst_id'],
                interval=self.config['interval'],
                start_time=self.config['data_start_time'],
                end_time=self.config['data_end_time'],
                data_type='val',
                lookback_window=lookback_window,
                predict_window=predict_window,
                clip=clip,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            # 4. 创建DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )

            self.logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

            # 5. 设置优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # 6. 训练循环
            best_val_loss = float('inf')
            self.logger.info(f"开始训练Predictor: {epochs}个epoch...")

            for epoch in range(epochs):
                # 计算并报告进度 (Predictor占总进度的50%-100%)
                predictor_progress = 0.50 + (epoch / epochs) * 0.50
                self._report_progress(
                    predictor_progress,
                    'predictor',
                    f'Predictor Epoch {epoch+1}/{epochs}'
                )

                # 训练阶段
                model.train()
                train_loss = 0.0

                # 设置epoch种子
                train_dataset.set_epoch_seed(epoch * 10000)

                for batch_idx, (batch_x, batch_x_stamp) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=True)):
                    batch_x = batch_x.to(device)  # (B, window, 6)
                    batch_x_stamp = batch_x_stamp.to(device)  # (B, window, 5) - minute, hour, weekday, day, month

                    # 使用tokenizer编码为tokens（冻结，无梯度）
                    with torch.no_grad():
                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

                    # 构造输入和目标 (autoregressive)
                    # 输入: tokens[:, :-1], 输出: tokens[:, 1:]
                    token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                    token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                    # 调用Predictor forward
                    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])

                    # 计算损失
                    loss, s1_loss, s2_loss = model.head.compute_loss(
                        logits[0], logits[1], token_out[0], token_out[1]
                    )

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # 验证阶段
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_x_stamp in val_loader:
                        batch_x = batch_x.to(device)
                        batch_x_stamp = batch_x_stamp.to(device)

                        # 编码
                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

                        # 构造输入输出
                        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                        # 前向传播
                        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])

                        # 计算损失
                        loss, _, _ = model.head.compute_loss(
                            logits[0], logits[1], token_out[0], token_out[1]
                        )

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f} - "
                    f"Val Loss: {avg_val_loss:.6f}"
                )

                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.logger.info(f"保存最佳模型 (val_loss={best_val_loss:.6f}) -> {save_path}")
                    model.save_pretrained(save_path)

            self.logger.info(f"Predictor训练完成! 最佳验证损失: {best_val_loss:.6f}")

            return {
                'success': True,
                'best_val_loss': float(best_val_loss)
            }

        except Exception as e:
            self.logger.error(f"Predictor训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


class KronosInferencer:
    """Kronos推理器封装"""

    def __init__(self, tokenizer_path: str, predictor_path: str, interval: str, device: str = "cuda:0"):
        """
        初始化推理器

        Args:
            tokenizer_path: Tokenizer模型路径
            predictor_path: Predictor模型路径
            interval: 时间颗粒度（用于生成时间戳）
            device: 设备
        """
        self.tokenizer_path = tokenizer_path
        self.predictor_path = predictor_path
        self.interval = interval
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logger

        # 加载模型
        try:
            from model import KronosPredictor
            import os

            self.logger.info(f"加载模型: tokenizer={tokenizer_path}, predictor={predictor_path}")

            # 确保路径是绝对路径
            tokenizer_path = os.path.abspath(tokenizer_path)
            predictor_path = os.path.abspath(predictor_path)

            # 检查模型文件是否存在
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer路径不存在: {tokenizer_path}")
            if not os.path.exists(predictor_path):
                raise FileNotFoundError(f"Predictor路径不存在: {predictor_path}")

            # 加载本地模型，指定local_files_only=True
            tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            model = Kronos.from_pretrained(predictor_path, local_files_only=True)

            self.predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=512)
            self.logger.info("模型加载成功")

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    def _get_freq_from_interval(self, interval: str) -> str:
        """
        将OKX interval格式转换为pandas频率字符串

        Args:
            interval: OKX时间颗粒度，如 '1H', '30m', '4H'

        Returns:
            pandas频率字符串，如 '60min', '30min', '240min'
        """
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1H': '60min',
            '2H': '120min',
            '4H': '240min',
            '6H': '360min',
            '12H': '720min',
            '1D': '1D',
            '1W': '1W'
        }
        return interval_map.get(interval, '60min')

    def predict(
        self,
        historical_data: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        执行推理

        Args:
            historical_data: 历史数据DataFrame，需包含 timestamps, open, high, low, close, volume, amount
            pred_len: 预测长度
            T: 温度参数
            top_p: nucleus sampling参数
            sample_count: 采样次数

        Returns:
            预测结果DataFrame或None
        """
        try:
            # 准备输入数据
            x_df = historical_data[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = historical_data['timestamps']

            # 生成未来时间戳
            freq = self._get_freq_from_interval(self.interval)
            last_timestamp = x_timestamp.iloc[-1]

            self.logger.info(f"最后时间戳: {last_timestamp}, 预测频率: {freq}")

            # pd.date_range 返回 DatetimeIndex，需要转换为 Series
            # 生成北京时间戳（UTC+8），直接存储和显示
            from utils.timezone_helper import convert_to_beijing_time

            # 生成UTC时间戳
            y_timestamp_utc = pd.date_range(
                start=last_timestamp,
                periods=pred_len + 1,
                freq=freq,
                tz='UTC'  # 明确指定为UTC时区
            )[1:]  # 排除起始时间

            # 转换为北京时间戳（UTC+8）
            beijing_timestamps = [convert_to_beijing_time(ts) for ts in y_timestamp_utc]
            y_timestamp = pd.Series(beijing_timestamps)

            # 执行推理
            self.logger.info(f"开始推理: 历史数据{len(x_df)}条, 预测{pred_len}条")
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False
            )

            self.logger.info(f"推理完成: 生成{len(pred_df)}条预测数据")
            return pred_df

        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
