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

            # 优先从training节点获取epochs，如果没有则从顶层获取
            tokenizer_epochs = train_params.get('tokenizer_epochs', finetune_params.get('tokenizer_epochs', 25))
            predictor_epochs = train_params.get('basemodel_epochs', train_params.get('predictor_epochs', finetune_params.get('predictor_epochs', 50)))
            # 优先从training节点获取参数，如果没有则从顶层获取
            batch_size = train_params.get('batch_size', finetune_params.get('batch_size', 16))
            tokenizer_lr = train_params.get('tokenizer_learning_rate', finetune_params.get('learning_rate', 0.0002))
            predictor_lr = train_params.get('predictor_learning_rate', finetune_params.get('learning_rate', 0.000001))
            seed = train_params.get('seed', finetune_params.get('seed', 42))

            # 新增关键参数获取（使用kronos的默认值）
            weight_decay = train_params.get('weight_decay', finetune_params.get('weight_decay', 0.01))  # 使用kronos的默认值
            accumulation_steps = train_params.get('accumulation_steps', finetune_params.get('accumulation_steps', 1))
            log_interval = train_params.get('log_interval', finetune_params.get('log_interval', 50))
            adam_beta1 = train_params.get('adam_beta1', finetune_params.get('adam_beta1', 0.9))
            adam_beta2 = train_params.get('adam_beta2', finetune_params.get('adam_beta2', 0.95))
            gradient_clip_norm = train_params.get('gradient_clip_norm', finetune_params.get('gradient_clip_norm', 2.0))

            # 记录解析后的训练参数
            self.logger.info(f"解析的训练参数:")
            self.logger.info(f"  tokenizer_epochs: {tokenizer_epochs}")
            self.logger.info(f"  predictor_epochs: {predictor_epochs}")
            self.logger.info(f"  batch_size: {batch_size}")
            self.logger.info(f"  tokenizer_lr: {tokenizer_lr}")
            self.logger.info(f"  predictor_lr: {predictor_lr}")
            self.logger.info(f"  weight_decay: {weight_decay}")
            self.logger.info(f"  accumulation_steps: {accumulation_steps}")
            self.logger.info(f"  adam_beta1: {adam_beta1}")
            self.logger.info(f"  adam_beta2: {adam_beta2}")
            self.logger.info(f"  gradient_clip_norm: {gradient_clip_norm}")
            self.logger.info(f"  lookback_window: {lookback_window}")
            self.logger.info(f"  predict_window: {predict_window}")
            self.logger.info(f"  clip: {clip}")
            self.logger.info(f"  train_ratio: {train_ratio}")
            self.logger.info(f"  val_ratio: {val_ratio}")
            self.logger.info(f"  seed: {seed}")

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
                val_ratio=val_ratio,
                weight_decay=weight_decay,
                accumulation_steps=accumulation_steps,
                log_interval=log_interval,
                gradient_clip_norm=gradient_clip_norm
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
                val_ratio=val_ratio,
                weight_decay=weight_decay,
                log_interval=log_interval,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2
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

        精确移植kronos/finetune/train_tokenizer.py的训练逻辑
        包含完整的reconstruction loss和BSQ loss计算
        """
        try:
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader
            from torch.optim.lr_scheduler import OneCycleLR
            from services.database_dataset import DatabaseKlineDataset
            from tqdm import tqdm
            import time

            # 获取参数（全部从kwargs获取，确保参数传递完整）
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
            weight_decay = kwargs.get('weight_decay', 0.01)  # 使用kronos的默认值
            accumulation_steps = kwargs.get('accumulation_steps', 1)
            log_interval = kwargs.get('log_interval', 50)
            gradient_clip_norm = kwargs.get('gradient_clip_norm', 2.0)

            self.logger.info(f"Tokenizer训练参数:")
            self.logger.info(f"  epochs: {epochs}")
            self.logger.info(f"  batch_size: {batch_size}")
            self.logger.info(f"  lr: {lr}")
            self.logger.info(f"  weight_decay: {weight_decay}")
            self.logger.info(f"  accumulation_steps: {accumulation_steps}")
            self.logger.info(f"  gradient_clip_norm: {gradient_clip_norm}")
            self.logger.info(f"  log_interval: {log_interval}")

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

            # 2. 创建DataLoader（使用kronos相同的配置）
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,  # kronos使用batch_size=1，然后在内部手动处理
                shuffle=True,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,  # kronos使用batch_size=1，然后在内部手动处理
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False
            )

            self.logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

            # 3. 加载预训练模型
            self.logger.info(f"加载预训练Tokenizer: {pretrained_path}")
            tokenizer = KronosTokenizer.from_pretrained(pretrained_path, local_files_only=True)
            tokenizer = tokenizer.to(device)

            # 4. 设置优化器和调度器（完全复制kronos的逻辑）
            optimizer = torch.optim.AdamW(
                tokenizer.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # 添加OneCycleLR调度器（复制kronos逻辑）
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.03,
                div_factor=10
            )

            # 5. 完整的训练循环（精确复制kronos逻辑）
            best_val_loss = float('inf')
            batch_idx_global = 0
            start_time = time.time()
            self.logger.info(f"开始训练Tokenizer: {epochs}个epoch...")

            for epoch_idx in range(epochs):
                epoch_start_time = time.time()

                # 计算并报告进度 (Tokenizer占总进度的5%-50%)
                tokenizer_progress = 0.05 + (epoch_idx / epochs) * 0.45
                self._report_progress(
                    tokenizer_progress,
                    'tokenizer',
                    f'Tokenizer Epoch {epoch_idx+1}/{epochs}'
                )

                # 训练阶段
                tokenizer.train()

                # 设置epoch种子（用于数据增强的随机性）
                train_dataset.set_epoch_seed(epoch_idx * 10000)

                for i, (ori_batch_x, _) in enumerate(train_loader):
                    # 去掉第一个维度 (batch_size=1)，添加batch维度用于tokenizer
                    # ori_batch_x: (1, seq_len, 6) -> (seq_len, 6) -> (1, seq_len, 6)
                    ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                    # 为tokenizer添加batch维度: (seq_len, 6) -> (1, seq_len, 6)
                    ori_batch_x = ori_batch_x.unsqueeze(0)

                    # 确保形状正确: (1, seq_len, 6)
                    if len(ori_batch_x.shape) != 3 or ori_batch_x.shape[0] != 1:
                        self.logger.error(f"输入数据形状错误: {ori_batch_x.shape}, 期望 (1, seq_len, 6)")
                        continue

                    # --- 梯度累积循环（完全复制kronos逻辑）---
                    current_batch_total_loss = 0.0
                    # 如果梯度累积>1，需要拆分batch
                    if accumulation_steps > 1:
                        # 对于单个样本，拆分sequence维度
                        seq_len = ori_batch_x.shape[1]
                        chunk_size = seq_len // accumulation_steps
                        actual_batch_size = 1
                    else:
                        actual_batch_size = 1

                    for j in range(accumulation_steps):
                        if accumulation_steps > 1:
                            start_idx = j * chunk_size
                            end_idx = (j + 1) * chunk_size
                            # 拆分sequence维度： (1, seq_len, 6) -> (1, chunk_size, 6)
                            batch_x = ori_batch_x[:, start_idx:end_idx, :]
                        else:
                            # 不进行梯度累积，直接使用整个batch
                            batch_x = ori_batch_x

                        # 前向传播（精确复制kronos逻辑）
                        zs, bsq_loss, _, _ = tokenizer(batch_x)
                        z_pre, z = zs

                        # Loss计算（修复负值问题）
                        recon_loss_pre = F.mse_loss(z_pre, batch_x)
                        recon_loss_all = F.mse_loss(z, batch_x)
                        recon_loss = recon_loss_pre + recon_loss_all

                        # 确保bsq_loss为正值，如果为负则设为0
                        bsq_loss_positive = torch.clamp(bsq_loss, min=0.0)

                        # 总loss计算
                        loss = (recon_loss + bsq_loss_positive) / 2  # w_1=w_2=1

                        loss_scaled = loss / accumulation_steps
                        current_batch_total_loss += loss.item()
                        loss_scaled.backward()

                    # --- 梯度累积后的优化器步骤 ---
                    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # --- 日志记录（复制kronos逻辑）---
                    if (batch_idx_global + 1) % log_interval == 0:
                        avg_loss = current_batch_total_loss / accumulation_steps
                        self.logger.info(
                            f"[Epoch {epoch_idx + 1}/{epochs}, Step {i + 1}/{len(train_loader)}] "
                            f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                        )

                    batch_idx_global += 1

                # --- 验证循环（完全复制kronos逻辑）---
                tokenizer.eval()
                tot_val_loss_sum = 0.0
                val_sample_count = 0
                with torch.no_grad():
                    for ori_batch_x, _ in val_loader:
                        # 与训练循环相同的维度处理
                        ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                        ori_batch_x = ori_batch_x.unsqueeze(0)

                        # 确保形状正确
                        if len(ori_batch_x.shape) != 3 or ori_batch_x.shape[0] != 1:
                            self.logger.error(f"验证数据形状错误: {ori_batch_x.shape}, 期望 (1, seq_len, 6)")
                            continue

                        zs, _, _, _ = tokenizer(ori_batch_x)
                        _, z = zs
                        val_loss_item = F.mse_loss(z, ori_batch_x)

                        tot_val_loss_sum += val_loss_item.item() * ori_batch_x.size(0)
                        val_sample_count += ori_batch_x.size(0)

                avg_val_loss = tot_val_loss_sum / val_sample_count if val_sample_count > 0 else 0

                # --- Epoch总结和保存（复制kronos逻辑）---
                self.logger.info(
                    f"\n--- Epoch {epoch_idx + 1}/{epochs} Summary ---"
                    f"Validation Loss: {avg_val_loss:.4f}"
                    f"Time This Epoch: {time.time() - epoch_start_time:.2f}s"
                    f"Total Time Elapsed: {time.time() - start_time:.2f}s\n"
                )

                # 保存最佳模型（复制predictor逻辑）
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = kwargs.get('save_path', '')
                    if save_path:
                        tokenizer.save_pretrained(save_path)
                        self.logger.info(f"Best tokenizer saved to {save_path} (Val Loss: {best_val_loss:.4f})")

                self.logger.info(f"Tokenizer训练完成! 最佳验证损失: {best_val_loss:.6f}")

                # 确保最终模型被保存（如果还没有保存过）
                save_path = kwargs.get('save_path', '')
                if save_path and best_val_loss == float('inf'):
                    # 如果没有有效的验证损失，仍然保存最终模型
                    tokenizer.save_pretrained(save_path)
                    self.logger.info(f"Final tokenizer saved to {save_path} (no valid validation)")
                    best_val_loss = 0.0  # 设置为有效值

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

        精确移植kronos/finetune/train_predictor.py的训练逻辑
        包含完整的tokenization和语言模型训练
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from torch.optim.lr_scheduler import OneCycleLR
            from services.database_dataset import DatabaseKlineDataset
            from tqdm import tqdm
            import time

            # 获取参数（全部从kwargs获取，确保参数传递完整）
            tokenizer_path = kwargs.get('tokenizer_path', '')
            lookback_window = kwargs.get('lookback_window', 512)
            predict_window = kwargs.get('predict_window', 48)
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 16)
            lr = kwargs.get('lr', 0.00004)  # 使用kronos的默认值
            seed = kwargs.get('seed', 42)
            clip = kwargs.get('clip', 5.0)
            train_ratio = kwargs.get('train_ratio', 0.9)
            val_ratio = kwargs.get('val_ratio', 0.1)
            pretrained_path = kwargs.get('pretrained_path', '')
            save_path = kwargs.get('save_path', '')
            weight_decay = kwargs.get('weight_decay', 0.01)  # 使用kronos的默认值
            log_interval = kwargs.get('log_interval', 50)
            adam_beta1 = kwargs.get('adam_beta1', 0.9)
            adam_beta2 = kwargs.get('adam_beta2', 0.95)

            self.logger.info(f"Predictor训练参数:")
            self.logger.info(f"  epochs: {epochs}")
            self.logger.info(f"  batch_size: {batch_size}")
            self.logger.info(f"  lr: {lr}")
            self.logger.info(f"  weight_decay: {weight_decay}")
            self.logger.info(f"  adam_beta1: {adam_beta1}")
            self.logger.info(f"  adam_beta2: {adam_beta2}")
            self.logger.info(f"  log_interval: {log_interval}")

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

            # 5. 设置优化器和调度器（使用传递的参数）
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(adam_beta1, adam_beta2),  # 使用传递的参数
                weight_decay=weight_decay
            )

            # 添加OneCycleLR调度器（复制kronos逻辑）
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.03, div_factor=10
            )

            # 6. 训练循环（精确复制kronos逻辑）
            best_val_loss = float('inf')
            batch_idx_global = 0
            start_time = time.time()
            self.logger.info(f"开始训练Predictor: {epochs}个epoch...")

            for epoch_idx in range(epochs):
                epoch_start_time = time.time()

                # 计算并报告进度 (Predictor占总进度的50%-100%)
                predictor_progress = 0.50 + (epoch_idx / epochs) * 0.50
                self._report_progress(
                    predictor_progress,
                    'predictor',
                    f'Predictor Epoch {epoch_idx+1}/{epochs}'
                )

                # 训练阶段（完全复制kronos逻辑）
                model.train()
                train_dataset.set_epoch_seed(epoch_idx * 10000)
                val_dataset.set_epoch_seed(0)  # Keep validation sampling consistent

                for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
                    # 移动到设备，不进行squeeze操作
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                    # 确保形状正确: (batch_size, seq_len, features)
                    if len(batch_x.shape) != 3:
                        self.logger.error(f"Predictor输入数据形状错误: {batch_x.shape}, 期望 (batch_size, seq_len, 6)")
                        continue
                    if len(batch_x_stamp.shape) != 3:
                        self.logger.error(f"Predictor时间戳形状错误: {batch_x_stamp.shape}, 期望 (batch_size, seq_len, 5)")
                        continue

                    # --- Tokenize输入数据（完全复制kronos逻辑）---
                    with torch.no_grad():
                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

                    # --- 为语言模型准备输入和目标（完全复制kronos逻辑）---
                    token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                    token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                    # --- 前向传播和损失计算（完全复制kronos逻辑）---
                    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                    loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                    # --- 反向传播和优化（完全复制kronos逻辑）---
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
                    scheduler.step()

                    # --- 日志记录（复制kronos逻辑）---
                    if (batch_idx_global + 1) % log_interval == 0:
                        lr_current = optimizer.param_groups[0]['lr']
                        self.logger.info(
                            f"[Epoch {epoch_idx + 1}/{epochs}, Step {i + 1}/{len(train_loader)}] "
                            f"LR {lr_current:.6f}, Loss: {loss.item():.4f}"
                        )

                    batch_idx_global += 1

                # --- 验证循环（完全复制kronos逻辑）---
                model.eval()
                tot_val_loss_sum = 0.0
                val_batches_processed = 0
                with torch.no_grad():
                    for batch_x, batch_x_stamp in val_loader:
                        # 移动到设备，不进行squeeze操作
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                        # 确保形状正确: (batch_size, seq_len, features)
                        if len(batch_x.shape) != 3:
                            self.logger.error(f"Predictor验证输入形状错误: {batch_x.shape}, 期望 (batch_size, seq_len, 6)")
                            continue
                        if len(batch_x_stamp.shape) != 3:
                            self.logger.error(f"Predictor验证时间戳形状错误: {batch_x_stamp.shape}, 期望 (batch_size, seq_len, 5)")
                            continue

                        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                        val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                        tot_val_loss_sum += val_loss.item()
                        val_batches_processed += 1

                avg_val_loss = tot_val_loss_sum / val_batches_processed if val_batches_processed > 0 else 0

                # --- Epoch总结和保存（复制kronos逻辑）---
                self.logger.info(
                    f"\n--- Epoch {epoch_idx + 1}/{epochs} Summary ---"
                    f"Validation Loss: {avg_val_loss:.4f}"
                    f"Time This Epoch: {time.time() - epoch_start_time:.2f}s"
                    f"Total Time Elapsed: {time.time() - start_time:.2f}s\n"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = f"{save_path}/best_model"
                    model.save_pretrained(save_path)
                    self.logger.info(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

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

            # 检查predictor模型是否在best_model子目录中
            import os
            predictor_best_model_path = os.path.join(predictor_path, "best_model")
            if os.path.exists(predictor_best_model_path) and os.path.exists(os.path.join(predictor_best_model_path, "config.json")):
                model = Kronos.from_pretrained(predictor_best_model_path, local_files_only=True)
            else:
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

            # 处理时区感知的时间戳
            if hasattr(last_timestamp, 'tz') and last_timestamp.tz is not None:
                # 时间戳有时区信息，使用该时区
                start_timestamp = last_timestamp
                timezone = last_timestamp.tz
                self.logger.info(f"使用时区感知时间戳: {timezone}")
            else:
                # 时间戳无时区信息，假设为UTC
                start_timestamp = last_timestamp
                timezone = 'UTC'
                self.logger.info(f"使用naive时间戳，假设为UTC")

            # 生成未来时间戳
            y_timestamp = pd.date_range(
                start=start_timestamp,
                periods=pred_len + 1,
                freq=freq,
                tz=timezone
            )[1:]  # 排除起始时间

            # 移除时区信息以匹配数据库存储格式
            y_timestamp = y_timestamp.tz_localize(None)

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
