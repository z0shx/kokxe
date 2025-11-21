"""
Kronos 模型管理服务
负责从 Hugging Face 下载和管理预训练模型
"""
import os
from pathlib import Path
from typing import Optional, Tuple
from utils.logger import setup_logger
from model import Kronos, KronosTokenizer

logger = setup_logger(__name__, "model_service.log")


class ModelService:
    """Kronos 模型管理服务"""

    # 模型存储基础路径
    BASE_MODEL_DIR = Path(__file__).parent.parent / "models"
    PRETRAINED_DIR = BASE_MODEL_DIR / "pretrained"
    FINETUNED_DIR = BASE_MODEL_DIR / "train"

    # Hugging Face 上的默认模型
    DEFAULT_MODELS = {
        "kronos-mini": {
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-mini",
            "predictor": "NeoQuasar/Kronos-mini"
        },
        "kronos-small": {
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-small",
            "predictor": "NeoQuasar/Kronos-small"
        },
        "kronos-base": {
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
            "predictor": "NeoQuasar/Kronos-base"
        },
        "kronos-large": {
            "tokenizer": "NeoQuasar/Kronos-Tokenizer-large",
            "predictor": "NeoQuasar/Kronos-large"
        }
    }

    @classmethod
    def get_model_choices(cls) -> list:
        """获取可用的模型选项列表"""
        return list(cls.DEFAULT_MODELS.keys())

    @classmethod
    def ensure_dirs(cls):
        """确保模型目录存在"""
        cls.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        cls.FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_local_model_path(cls, model_size: str, model_type: str) -> Path:
        """
        获取模型的本地存储路径

        Args:
            model_size: 模型大小 (kronos-mini, kronos-small, kronos-base, kronos-large)
            model_type: 模型类型 (tokenizer, predictor)

        Returns:
            本地模型路径
        """
        if model_type == "tokenizer":
            return cls.PRETRAINED_DIR / f"Kronos-Tokenizer-{model_size.split('-')[-1]}"
        else:
            return cls.PRETRAINED_DIR / f"Kronos-{model_size.split('-')[-1]}"

    @classmethod
    def download_model(
        cls,
        model_size: str,
        force_download: bool = False
    ) -> Tuple[str, str]:
        """
        下载预训练模型到本地

        Args:
            model_size: 模型大小 (kronos-mini, kronos-small, kronos-base, kronos-large)
            force_download: 是否强制重新下载

        Returns:
            (tokenizer_path, predictor_path) 元组
        """
        cls.ensure_dirs()

        if model_size not in cls.DEFAULT_MODELS:
            raise ValueError(f"不支持的模型: {model_size}, 可选: {list(cls.DEFAULT_MODELS.keys())}")

        model_config = cls.DEFAULT_MODELS[model_size]

        # 获取本地路径
        tokenizer_path = cls.get_local_model_path(model_size, "tokenizer")
        predictor_path = cls.get_local_model_path(model_size, "predictor")

        try:
            # 下载 Tokenizer
            if force_download or not tokenizer_path.exists():
                logger.info(f"正在从 Hugging Face 下载 Tokenizer: {model_config['tokenizer']}")
                tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer'])
                tokenizer.save_pretrained(str(tokenizer_path))
                logger.info(f"Tokenizer 已保存到: {tokenizer_path}")
            else:
                logger.info(f"Tokenizer 已存在: {tokenizer_path}")

            # 下载 Predictor
            if force_download or not predictor_path.exists():
                logger.info(f"正在从 Hugging Face 下载 Predictor: {model_config['predictor']}")
                predictor = Kronos.from_pretrained(model_config['predictor'])
                predictor.save_pretrained(str(predictor_path))
                logger.info(f"Predictor 已保存到: {predictor_path}")
            else:
                logger.info(f"Predictor 已存在: {predictor_path}")

            return str(tokenizer_path), str(predictor_path)

        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            raise

    @classmethod
    def get_finetuned_save_path(cls, inst_id: str, interval: str) -> str:
        """
        获取微调模型的保存路径

        Args:
            inst_id: 交易对
            interval: 时间颗粒度

        Returns:
            微调模型保存基础路径
        """
        from datetime import datetime
        exp_name = f"{inst_id.replace('-', '_')}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path = cls.FINETUNED_DIR / exp_name
        save_path.mkdir(parents=True, exist_ok=True)
        return str(save_path)

    @classmethod
    def list_finetuned_models(cls) -> list:
        """列出所有微调后的模型"""
        cls.ensure_dirs()
        finetuned_models = []

        if cls.FINETUNED_DIR.exists():
            for model_dir in cls.FINETUNED_DIR.iterdir():
                if model_dir.is_dir():
                    finetuned_models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "tokenizer_path": str(model_dir / "tokenizer" / "best_model"),
                        "predictor_path": str(model_dir / "basemodel" / "best_model")
                    })

        return finetuned_models
