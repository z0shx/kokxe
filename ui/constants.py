"""
UI组件常量定义
"""
import pandas as pd

class DataFrameHeaders:
    """DataFrame 列头常量"""
    LLM_CONFIG = ["ID", "名称", "提供商", "模型", "状态", "默认"]
    PROMPT_TEMPLATE = ["ID", "名称", "分类", "描述", "状态", "默认"]
    TASK_HISTORY = ['ID', '任务类型', '任务名称', '状态', '计划时间', '开始时间',
                   '完成时间', '执行时长(秒)', '触发方式', '进度(%)']
    PLANS_TABLE = ["ID", "计划名称", "交易对", "时间颗粒度", "状态", "WebSocket", "环境", "创建时间"]

class StatusEmoji:
    """状态映射常量"""
    BASIC = {
        'created': '⚪',
        'running': '🟢',
        'paused': '🟡',
        'stopped': '🔴'
    }

    DETAILED = {
        'created': '⚪ 已创建',
        'running': '🟢 运行中',
        'paused': '🟡 已暂停',
        'stopped': '🔴 已停止',
        'created_unnamed': '⚪ 未命名',
        'running_ws': '🟢 已连接',
        'stopped_ws': '🔴 未连接',
        'unknown': '❓ 未知'
    }

class DataTypes:
    """DataFrame 数据类型常量"""
    TASK_HISTORY = {
        'ID': 'int',
        '执行时长(秒)': 'int',
        '进度(%)': 'int'
    }

    PLANS_TABLE = ["number", "str", "str", "str", "str", "str", "str", "str"]

def create_empty_dataframe(columns: list, dtypes: dict = None):
    """创建指定结构的空DataFrame"""
    df = pd.DataFrame(columns=columns)
    if dtypes:
        df = df.astype(dtypes)
    return df