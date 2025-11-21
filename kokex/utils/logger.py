"""
日志工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import config


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件名（可选）
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = config.LOG_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_ws_logger(inst_id: str, interval: str):
    """
    获取 WebSocket 专用日志记录器

    Args:
        inst_id: 交易对
        interval: 时间颗粒度
    """
    log_file = f"ws_{inst_id.replace('-', '_')}_{interval}_{datetime.now().strftime('%Y%m%d')}.log"
    return setup_logger(f"ws.{inst_id}.{interval}", log_file)
