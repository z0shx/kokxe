"""
模型清理服务
定期清理旧的模型文件和推理记录，仅保留最近N个模型
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from sqlalchemy import and_, desc
import logging

from database.db import get_db
from database.models import TrainingRecord, PredictionData
from utils.logger import setup_logger

logger = setup_logger(__name__, "model_cleanup.log")

class ModelCleanupService:
    """模型清理服务"""

    def __init__(self, models_base_dir: str = "models"):
        self.models_base_dir = Path(models_base_dir)
        self.logger = logger

    def get_model_directories(self, plan_id: int) -> List[Path]:
        """获取指定计划的所有模型目录"""
        try:
            plan_model_dir = self.models_base_dir / f"plan_{plan_id}"
            if not plan_model_dir.exists():
                return []

            # 查找所有版本目录 (v1, v2, vv1, vv2, v3, ...)
            model_dirs = []
            for item in plan_model_dir.iterdir():
                if item.is_dir() and (item.name.startswith('v') or item.name.startswith('vv')):
                    try:
                        # 尝试解析版本号
                        if item.name.startswith('vv'):
                            # vv21 -> 21
                            version_num = int(item.name[2:])
                        else:
                            # v21 -> 21
                            version_num = int(item.name[1:])
                        model_dirs.append((version_num, item))
                    except ValueError:
                        continue

            # 按版本号排序
            model_dirs.sort(key=lambda x: x[0], reverse=True)
            return [dir_path for _, dir_path in model_dirs]

        except Exception as e:
            self.logger.error(f"获取模型目录失败: {e}")
            return []

    def get_training_records_by_version(self, plan_id: int) -> Dict[str, TrainingRecord]:
        """获取计划的训练记录，按版本分组"""
        try:
            with get_db() as db:
                records = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status.in_(['completed', 'failed'])
                    )
                ).order_by(desc(TrainingRecord.id)).all()

                version_records = {}
                for record in records:
                    version_records[record.version] = record

                return version_records

        except Exception as e:
            self.logger.error(f"获取训练记录失败: {e}")
            return {}

    def delete_model_directory(self, model_dir: Path) -> bool:
        """删除模型目录"""
        try:
            if model_dir.exists():
                self.logger.info(f"删除模型目录: {model_dir}")
                shutil.rmtree(model_dir)
                return True
            return False
        except Exception as e:
            self.logger.error(f"删除模型目录失败 {model_dir}: {e}")
            return False

    def delete_prediction_data(self, plan_id: int, version: str) -> int:
        """删除指定版本的预测数据"""
        try:
            with get_db() as db:
                # 删除该版本的预测数据
                deleted_count = db.query(PredictionData).filter(
                    and_(
                        PredictionData.plan_id == plan_id,
                        PredictionData.model_version == version
                    )
                ).delete()

                db.commit()
                self.logger.info(f"删除版本 {version} 的预测数据: {deleted_count} 条")
                return deleted_count

        except Exception as e:
            self.logger.error(f"删除预测数据失败: {e}")
            return 0

    def cleanup_old_models(self, plan_id: int, keep_count: int = 7) -> Dict[str, int]:
        """
        清理旧模型，仅保留最近keep_count个模型

        Args:
            plan_id: 计划ID
            keep_count: 保留的模型数量，默认7个

        Returns:
            Dict: 清理统计信息
        """
        try:
            self.logger.info(f"开始清理计划 {plan_id} 的旧模型，保留最近 {keep_count} 个")

            stats = {
                'models_deleted': 0,
                'predictions_deleted': 0,
                'errors': 0,
                'kept_models': 0
            }

            # 获取所有模型目录
            model_dirs = self.get_model_directories(plan_id)

            # 获取训练记录
            version_records = self.get_training_records_by_version(plan_id)

            # 判断是否需要删除
            if len(model_dirs) <= keep_count:
                self.logger.info(f"模型数量 ({len(model_dirs)}) 不超过保留数量 ({keep_count})，无需清理")
                stats['kept_models'] = len(model_dirs)
                return stats

            # 找出需要删除的旧模型
            models_to_delete = model_dirs[keep_count:]
            stats['kept_models'] = keep_count

            for model_dir in models_to_delete:
                # 转换目录名到版本号格式
                if model_dir.name.startswith('vv'):
                    version = f"v{model_dir.name[2:]}"  # vv21 -> v21
                else:
                    version = model_dir.name  # v21 -> v21

                try:
                    # 删除模型目录
                    if self.delete_model_directory(model_dir):
                        stats['models_deleted'] += 1

                    # 删除相关的预测数据
                    predictions_deleted = self.delete_prediction_data(plan_id, version)
                    stats['predictions_deleted'] += predictions_deleted

                    self.logger.info(f"已清理模型版本: {version}")

                except Exception as e:
                    self.logger.error(f"清理模型版本 {version} 失败: {e}")
                    stats['errors'] += 1

            self.logger.info(f"计划 {plan_id} 模型清理完成: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"清理旧模型失败: {e}")
            return {
                'models_deleted': 0,
                'predictions_deleted': 0,
                'errors': 1,
                'kept_models': 0
            }

    def cleanup_all_plans(self, keep_count: int = 7) -> Dict[int, Dict[str, int]]:
        """
        清理所有计划的旧模型

        Args:
            keep_count: 每个计划保留的模型数量，默认7个

        Returns:
            Dict: 每个计划的清理统计信息
        """
        try:
            self.logger.info(f"开始清理所有计划的旧模型，每个计划保留最近 {keep_count} 个")

            all_stats = {}

            # 获取所有有模型的计划
            if not self.models_base_dir.exists():
                self.logger.warning("模型目录不存在")
                return all_stats

            for plan_dir in self.models_base_dir.iterdir():
                if plan_dir.is_dir() and plan_dir.name.startswith('plan_'):
                    try:
                        plan_id = int(plan_dir.name.split('_')[1])
                        plan_stats = self.cleanup_old_models(plan_id, keep_count)
                        all_stats[plan_id] = plan_stats

                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"无效的计划目录名: {plan_dir.name}")
                        continue

            # 汇总统计
            total_models_deleted = sum(stats['models_deleted'] for stats in all_stats.values())
            total_predictions_deleted = sum(stats['predictions_deleted'] for stats in all_stats.values())
            total_errors = sum(stats['errors'] for stats in all_stats.values())

            self.logger.info(f"所有计划模型清理完成: 删除 {total_models_deleted} 个模型目录, "
                           f"{total_predictions_deleted} 条预测数据, {total_errors} 个错误")

            return all_stats

        except Exception as e:
            self.logger.error(f"清理所有计划模型失败: {e}")
            return {}

# 创建全局实例
model_cleanup_service = ModelCleanupService()

def cleanup_old_models(plan_id: int, keep_count: int = 7) -> Dict[str, int]:
    """清理指定计划的旧模型（便捷函数）"""
    return model_cleanup_service.cleanup_old_models(plan_id, keep_count)

def cleanup_all_plans_models(keep_count: int = 7) -> Dict[int, Dict[str, int]]:
    """清理所有计划的旧模型（便捷函数）"""
    return model_cleanup_service.cleanup_all_plans(keep_count)

def manual_cleanup_plan_models(plan_id: int, keep_count: int = 7) -> Dict[str, int]:
    """手动清理指定计划的旧模型（用于UI调用）"""
    try:
        logger.info(f"手动清理计划 {plan_id} 的旧模型，保留 {keep_count} 个最新模型")
        return model_cleanup_service.cleanup_old_models(plan_id, keep_count)
    except Exception as e:
        logger.error(f"手动清理计划 {plan_id} 模型失败: {e}")
        return {
            'models_deleted': 0,
            'predictions_deleted': 0,
            'errors': 1,
            'kept_models': 0
        }