"""
时区处理工具模块

提供统一的时区转换功能，确保所有数据在Gradio界面中正确显示为UTC+8
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import pytz

# 定义时区常量
UTC_TZ = timezone.utc
BEIJING_TZ = timezone(timedelta(hours=8))  # UTC+8
SHANGHAI_TZ = pytz.timezone('Asia/Shanghai')  # 备用，更准确的时区


def convert_to_beijing_time(dt: Optional[Union[datetime, str]]) -> Optional[datetime]:
    """
    将datetime对象或字符串转换为UTC+8北京时间

    Args:
        dt: 输入时间(datetime对象或字符串)

    Returns:
        UTC+8时区的datetime对象
    """
    if dt is None:
        return None

    # 如果是字符串，先解析为datetime
    if isinstance(dt, str):
        try:
            # 尝试多种时间格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%m-%d %H:%M:%S',
                '%m-%d %H:%M'
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(dt, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None  # 无法解析
        except Exception:
            return None

    # 确保输入是datetime对象
    if not isinstance(dt, datetime):
        return None

    # 处理时区转换
    if dt.tzinfo is None:
        # naive datetime，假设它已经是北京时间（因为数据库使用UTC+8存储）
        # 直接返回naive datetime，不进行时区转换
        return dt
    else:
        # timezone-aware datetime，先转换为UTC
        dt_utc = dt.astimezone(UTC_TZ)
        # 转换为UTC+8
        dt_beijing = dt_utc.astimezone(BEIJING_TZ)
        return dt_beijing


def format_datetime_beijing(dt: Optional[Union[datetime, str]],
                           format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    格式化时间为UTC+8北京时间字符串

    Args:
        dt: 输入时间(datetime对象或字符串)
        format_str: 格式化字符串

    Returns:
        格式化后的时间字符串，如果输入为None返回空字符串
    """
    if dt is None:
        return ""

    dt_beijing = convert_to_beijing_time(dt)
    if dt_beijing is None:
        return ""

    return dt_beijing.strftime(format_str)


def format_datetime_short_beijing(dt: Optional[Union[datetime, str]]) -> str:
    """
    格式化时间为简短的UTC+8北京时间字符串 (月-日 时:分)

    Args:
        dt: 输入时间

    Returns:
        格式化后的简短时间字符串
    """
    return format_datetime_beijing(dt, '%m-%d %H:%M')


def format_datetime_full_beijing(dt: Optional[Union[datetime, str]]) -> str:
    """
    格式化时间为完整的UTC+8北京时间字符串 (年-月-日 时:分:秒)

    Args:
        dt: 输入时间

    Returns:
        格式化后的完整时间字符串
    """
    return format_datetime_beijing(dt, '%Y-%m-%d %H:%M:%S')


def format_datetime_detail_beijing(dt: Optional[Union[datetime, str]]) -> str:
    """
    格式化时间为详细时间字符串 (包含时区信息)

    Args:
        dt: 输入时间

    Returns:
        格式化后的详细时间字符串
    """
    if dt is None:
        return ""

    dt_beijing = convert_to_beijing_time(dt)
    if dt_beijing is None:
        return ""

    return dt_beijing.strftime('%Y-%m-%d %H:%M:%S (UTC+8)')


def get_current_beijing_time() -> datetime:
    """
    获取当前UTC+8北京时间

    Returns:
        当前UTC+8时间的datetime对象
    """
    return datetime.now(UTC_TZ).astimezone(BEIJING_TZ)


def get_current_beijing_str(format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    获取当前UTC+8时间字符串

    Args:
        format_str: 格式化字符串

    Returns:
        格式化后的当前时间字符串
    """
    return get_current_beijing_time().strftime(format_str)




# 批量转换函数
def convert_list_to_beijing(dt_list, format_str: str = '%Y-%m-%d %H:%M:%S') -> list:
    """
    批量转换时间列表为UTC+8字符串

    Args:
        dt_list: 时间列表
        format_str: 格式化字符串

    Returns:
        格式化后的时间字符串列表
    """
    return [format_datetime_beijing(dt, format_str) for dt in dt_list]


# 时间范围格式化
def format_time_range_utc8(start_dt: Optional[Union[datetime, str]],
                          end_dt: Optional[Union[datetime, str]],
                          format_str: str = '%m-%d %H:%M') -> str:
    """
    格式化时间范围为UTC+8字符串

    Args:
        start_dt: 开始时间
        end_dt: 结束时间
        format_str: 格式化字符串

    Returns:
        格式化后的时间范围字符串
    """
    start_str = format_datetime_beijing(start_dt, format_str)
    end_str = format_datetime_beijing(end_dt, format_str)

    if start_str and end_str:
        return f"{start_str} ~ {end_str}"
    elif start_str:
        return f"{start_str} ~"
    elif end_str:
        return f"~ {end_str}"
    else:
        return "N/A"


# 验证函数
def is_valid_datetime(dt) -> bool:
    """
    验证是否是有效的datetime对象或时间字符串

    Args:
        dt: 待验证的对象

    Returns:
        是否有效
    """
    if dt is None:
        return False

    if isinstance(dt, datetime):
        return True

    if isinstance(dt, str):
        try:
            # 尝试解析
            convert_to_beijing_time(dt)
            return True
        except:
            return False

    return False