"""
通用工具函数
"""
from typing import Optional, Union, Any, Tuple
from database.models import now_beijing


def safe_plan_id(pid: Union[str, int, None]) -> Optional[int]:
    """安全转换为 plan_id"""
    if pid is None or pid == "":
        return None
    try:
        return int(pid)
    except (ValueError, TypeError):
        return None


def validate_plan_id(pid: Union[str, int, None]) -> Tuple[bool, Optional[int], str]:
    """验证 plan_id"""
    if not pid:
        return False, None, "计划ID不能为空"

    try:
        plan_id = int(pid)
        if plan_id <= 0:
            return False, None, "计划ID必须是正整数"
        return True, plan_id, ""
    except (ValueError, TypeError):
        return False, None, "计划ID必须是有效的数字"


def validate_plan_exists(pid: Union[str, int, None]) -> Tuple[bool, Optional[int], str]:
    """验证 plan_id 并检查计划是否存在"""
    is_valid, plan_id, error_msg = validate_plan_id(pid)
    if not is_valid:
        return False, None, error_msg

    # 验证计划是否存在
    try:
        from database.db import get_db
        from database.models import TradingPlan

        with get_db() as db:
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
            if not plan:
                return False, plan_id, f"计划ID {plan_id} 不存在"

        return True, plan_id, ""
    except Exception as e:
        return False, plan_id, f"验证计划ID失败: {str(e)}"


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数"""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            if value.strip() == "":
                return default
            return int(float(value))
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            if value.strip() == "":
                return default
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default


def format_timestamp(dt) -> str:
    """格式化时间戳"""
    if dt is None:
        return "未知时间"
    try:
        if hasattr(dt, 'strftime'):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return str(dt)
    except:
        return str(dt)


def format_duration(seconds: Optional[Union[int, float]]) -> str:
    """格式化持续时间"""
    if seconds is None or seconds <= 0:
        return "未知"

    try:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes}分{remaining_seconds}秒"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours}小时{remaining_minutes}分钟"
    except:
        return "未知"


def format_number(num: Optional[Union[int, float]], decimals: int = 2) -> str:
    """格式化数字显示"""
    if num is None:
        return "0"

    try:
        if isinstance(num, (int, float)):
            if decimals == 0:
                return f"{int(num)}"
            return f"{num:.{decimals}f}"
        return str(num)
    except:
        return "0"


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """格式化百分比"""
    if value is None:
        return "0%"

    try:
        return f"{value:.{decimals}f}%"
    except:
        return "0%"


def get_current_beijing_time_str() -> str:
    """获取当前北京时间字符串"""
    return now_beijing().strftime('%Y-%m-%d %H:%M:%S')


def truncate_string(text: Optional[str], max_length: int = 50) -> str:
    """截断字符串"""
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length-3] + "..."


def safe_json_loads(json_str: Optional[str], default: Any = None) -> Any:
    """安全解析JSON字符串"""
    if not json_str:
        return default

    import json
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """安全序列化为JSON字符串"""
    if obj is None:
        return default

    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return default