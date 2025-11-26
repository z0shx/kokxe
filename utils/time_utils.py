"""
时间工具函数 - 统一使用UTC+8北京时间
"""
from datetime import datetime
import pytz

# 统一时区：北京时区 UTC+8
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

def now_beijing():
    """获取当前北京时间"""
    return datetime.now(BEIJING_TZ)

def now_beijing_str():
    """获取当前北京时间字符串"""
    return datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')

def now_beijing_iso():
    """获取当前北京时间ISO格式"""
    return datetime.now(BEIJING_TZ).isoformat()

def to_beijing(dt):
    """将datetime转换为北京时间"""
    if dt.tzinfo is None:
        # 如果没有时区信息，假设是UTC
        dt = pytz.utc.localize(dt)
    return dt.astimezone(BEIJING_TZ)

def format_beijing(dt, format_str='%Y-%m-%d %H:%M:%S'):
    """格式化datetime为北京时间字符串"""
    beijing_dt = to_beijing(dt)
    return beijing_dt.strftime(format_str)