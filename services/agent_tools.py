"""
AI Agent 交易工具定义
定义 Agent 可以调用的所有交易工具及其参数
"""
from typing import Dict, List, Any
from enum import Enum


class ToolCategory(str, Enum):
    """工具分类"""
    QUERY = "query"  # 查询类工具
    TRADE = "trade"  # 交易类工具
    MONITOR = "monitor"  # 监控类工具


class AgentTool:
    """Agent 工具定义"""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Dict[str, Any],
        required_params: List[str],
        risk_level: str = "low"
    ):
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters
        self.required_params = required_params
        self.risk_level = risk_level

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "required_params": self.required_params,
            "risk_level": self.risk_level
        }


# ============================================
# 定义所有可用工具
# ============================================

AGENT_TOOLS = {
    # 查询类工具
    "get_account_balance": AgentTool(
        name="get_account_balance",
        description="查询账户余额。返回指定币种或所有币种的可用余额、冻结余额等信息。下单前必须调用此工具确认有足够的资金。根据交易限制配置，如果固定USDT资金不足，将使用账户总资金的百分比。",
        category=ToolCategory.QUERY,
        parameters={
            "ccy": {
                "type": "string",
                "description": "币种，如 BTC、USDT。不填则返回所有币种余额",
                "required": False
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "get_account_positions": AgentTool(
        name="get_account_positions",
        description="查询账户持仓信息。返回当前所有持仓的详细信息，包括持仓数量、均价、未实现盈亏等。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT。不填则返回所有持仓",
                "required": False
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "get_order_info": AgentTool(
        name="get_order_info",
        description="查询订单详细信息。可以查询订单的状态、成交情况、剩余数量等。用于确认订单是否成功下单或成交。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            },
            "order_id": {
                "type": "string",
                "description": "订单ID (order_id 和 client_order_id 至少提供一个)",
                "required": False
            },
            "client_order_id": {
                "type": "string",
                "description": "客户端订单ID",
                "required": False
            }
        },
        required_params=["inst_id"],
        risk_level="low"
    ),

    "get_pending_orders": AgentTool(
        name="get_pending_orders",
        description="查询当前所有未成交订单（挂单）。返回所有等待成交的限价单信息。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT。不填则返回所有交易对的挂单",
                "required": False
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "get_order_history": AgentTool(
        name="get_order_history",
        description="查询历史订单。返回最近7天内的订单历史记录，包括已成交、已撤销的订单。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT。不填则返回所有交易对",
                "required": False
            },
            "begin": {
                "type": "string",
                "description": "开始时间（毫秒时间戳）",
                "required": False
            },
            "end": {
                "type": "string",
                "description": "结束时间（毫秒时间戳）",
                "required": False
            },
            "limit": {
                "type": "string",
                "description": "返回数量，默认100，最大100",
                "required": False,
                "default": "100"
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "get_fills": AgentTool(
        name="get_fills",
        description="查询成交明细。返回最近3个月的成交记录，包括成交价格、数量、手续费等。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": False
            },
            "order_id": {
                "type": "string",
                "description": "订单ID",
                "required": False
            },
            "begin": {
                "type": "string",
                "description": "开始时间（毫秒时间戳）",
                "required": False
            },
            "end": {
                "type": "string",
                "description": "结束时间（毫秒时间戳）",
                "required": False
            },
            "limit": {
                "type": "string",
                "description": "返回数量，默认100，最大100",
                "required": False,
                "default": "100"
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "get_current_price": AgentTool(
        name="get_current_price",
        description="获取交易对当前市场价格。返回最新成交价、买一价、卖一价等实时行情信息。用于判断合适的下单价格。",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            }
        },
        required_params=["inst_id"],
        risk_level="low"
    ),

    # 交易类工具
    "place_limit_order": AgentTool(
        name="place_limit_order",
        description="""下限价单。限价单以指定价格买入或卖出，只有当市场价格达到或优于限价时才会成交。

重要提示：
1. 下单前必须先调用 get_account_balance 确认有足够的资金
2. 将根据交易限制配置中的可用USDT数量或百分比计算最大可用资金
3. 支持平摊操作：如果配置了平摊单量，会将交易金额分成多笔小额订单
4. 买入时需要足够的计价货币（如USDT），卖出时需要足够的基础货币（如BTC）
5. 价格和数量必须符合交易对的最小交易单位和精度要求
6. 建议下单后调用 get_order_info 确认订单状态""",
        category=ToolCategory.TRADE,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            },
            "side": {
                "type": "string",
                "description": "订单方向：buy(买入) 或 sell(卖出)",
                "required": True,
                "enum": ["buy", "sell"]
            },
            "price": {
                "type": "string",
                "description": "限价价格。买入时不能高于卖一价的105%，卖出时不能低于买一价的95%",
                "required": True
            },
            "size": {
                "type": "string",
                "description": "委托数量（基础货币数量，如BTC的数量）。如果不指定，系统将根据交易限制自动计算",
                "required": False
            },
            "total_amount": {
                "type": "string",
                "description": "总交易金额（USDT）。如果不指定size，系统将根据此金额和价格计算数量",
                "required": False
            },
            "client_order_id": {
                "type": "string",
                "description": "客户端订单ID，用于追踪订单。建议使用有意义的标识",
                "required": False
            }
        },
        required_params=["inst_id", "side", "price"],
        risk_level="high"
    ),

    "cancel_order": AgentTool(
        name="cancel_order",
        description="""撤销未成交的订单。撤单后，冻结的资金会立即释放。

注意事项：
1. 只能撤销状态为未成交或部分成交的订单
2. 已完全成交的订单无法撤销
3. 撤单成功后，资金会解冻并返回账户可用余额
4. order_id 和 client_order_id 至少提供一个""",
        category=ToolCategory.TRADE,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            },
            "order_id": {
                "type": "string",
                "description": "订单ID",
                "required": False
            },
            "client_order_id": {
                "type": "string",
                "description": "客户端订单ID",
                "required": False
            }
        },
        required_params=["inst_id"],
        risk_level="medium"
    ),

    "amend_order": AgentTool(
        name="amend_order",
        description="""修改未成交订单的价格或数量。改单可以在不撤销订单的情况下调整交易参数。

注意事项：
1. 只能修改未成交或部分成交的订单
2. 每次改单必须至少修改价格或数量之一
3. 改单后订单会丢失队列优先级（重新排队）
4. 如果改单失败，原订单仍然有效
5. 改后的价格和数量仍需符合交易规则""",
        category=ToolCategory.TRADE,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            },
            "order_id": {
                "type": "string",
                "description": "订单ID",
                "required": False
            },
            "client_order_id": {
                "type": "string",
                "description": "客户端订单ID",
                "required": False
            },
            "new_price": {
                "type": "string",
                "description": "新的委托价格",
                "required": False
            },
            "new_size": {
                "type": "string",
                "description": "新的委托数量",
                "required": False
            }
        },
        required_params=["inst_id"],
        risk_level="medium"
    ),

    "cancel_all_orders": AgentTool(
        name="cancel_all_orders",
        description="""批量撤销所有未成交订单。紧急情况下可以一次性撤销所有挂单。

使用场景：
1. 市场异常波动时需要快速平仓
2. 策略失效需要停止所有交易
3. 重新调整交易策略前清空所有订单

注意：此操作不可逆，请谨慎使用""",
        category=ToolCategory.TRADE,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT。不填则撤销所有交易对的订单",
                "required": False
            }
        },
        required_params=[],
        risk_level="high"
    ),

    "place_stop_loss_order": AgentTool(
        name="place_stop_loss_order",
        description="""设置止损订单。当价格达到指定止损比例时自动卖出，用于控制风险。

重要提示：
1. 止损基于配置的止损百分比，计算当前持仓的平均成本
2. 止损价格 = 平均成本 * (1 - 止损百分比)
3. 下单前会检查当前持仓情况和止损必要性
4. 止损订单使用市价单，确保快速成交
5. 建议在持仓后及时设置止损保护""",
        category=ToolCategory.TRADE,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT",
                "required": True
            },
            "stop_loss_percentage": {
                "type": "string",
                "description": "止损百分比，如 0.2 表示20%的止损。如果不指定，使用计划配置的默认值",
                "required": False
            },
            "client_order_id": {
                "type": "string",
                "description": "客户端订单ID，用于追踪止损订单",
                "required": False
            }
        },
        required_params=["inst_id"],
        risk_level="high"
    ),

"get_prediction_history": AgentTool(
        name="get_prediction_history",
        description="""查询Kronos模型的历史预测数据。返回指定训练版本的所有历史推理批次及其预测结果。

使用场景：
1. 查看之前的预测结果，评估模型准确性
2. 对比不同时间点的预测差异
3. 分析预测趋势的变化
4. 回顾历史预测与实际走势的对比

返回信息包括：
- 推理批次ID
- 推理时间
- 预测数据数量
- 预测时间范围
- 具体的OHLC预测值
- 上涨概率、波动性放大概率等指标""",
        category=ToolCategory.QUERY,
        parameters={
            "training_id": {
                "type": "integer",
                "description": "训练记录ID。如果不指定，则返回当前计划最新训练版本的预测历史",
                "required": False
            },
            "inference_batch_id": {
                "type": "string",
                "description": "推理批次ID。如果指定，则只返回该批次的详细预测数据；不指定则返回所有批次列表",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "返回批次数量，默认10，最大50",
                "required": False,
                "default": 10
            }
        },
        required_params=[],
        risk_level="low"
    ),

"query_prediction_data": AgentTool(
        name="query_prediction_data",
        description="""查询数据库中存储的预测数据。可以按时间范围、批次ID等条件查询，支持详细的OHLC数据查询。

使用场景：
1. 查询特定时间段的预测数据
2. 分析预测数据的统计特征
3. 对比不同批次的预测结果
4. 获取预测数据的详细信息用于决策分析

返回信息包括：
- 符合条件的预测数据列表
- 每条数据的OHLC值、时间戳
- 批次信息和推理时间
- 统计信息（如数据量、时间范围等）
- 概率指标（如果有）""",
        category=ToolCategory.QUERY,
        parameters={
            "training_id": {
                "type": "integer",
                "description": "训练记录ID，用于筛选特定训练版本的预测数据",
                "required": False
            },
            "inference_batch_id": {
                "type": "string",
                "description": "推理批次ID，用于筛选特定批次的预测数据",
                "required": False
            },
            "start_time": {
                "type": "string",
                "description": "开始时间，格式：YYYY-MM-DD HH:MM:SS",
                "required": False
            },
            "end_time": {
                "type": "string",
                "description": "结束时间，格式：YYYY-MM-DD HH:MM:SS",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "返回数据条数，默认50，最大200",
                "required": False,
                "default": 50
            },
            "order_by": {
                "type": "string",
                "description": "排序方式：'time_asc'(时间升序), 'time_desc'(时间降序), 'created_asc', 'created_desc'",
                "required": False,
                "default": "time_asc"
            }
        },
        required_params=[],
        risk_level="low"
    ),

    "query_historical_kline_data": AgentTool(
        name="query_historical_kline_data",
        description="""查询真实历史交易数据。通过查询数据库中的K线数据表，使用UTC+0时间戳作为查询条件，返回指定时间范围内的历史OHLCV数据。

使用场景：
1. 查看特定时间段的历史价格走势
2. 分析历史价格波动和成交量
3. 获取技术分析所需的历史数据
4. 对比模型预测与实际历史数据
5. 回测交易策略的历史表现

返回信息包括：
- 符合条件的历史K线数据列表
- 每条数据的时间戳（UTC+0）、开盘价、最高价、最低价、收盘价、成交量、成交额
- 数据统计信息（数据量、时间范围、价格波动等）
- 数据完整性说明""",
        category=ToolCategory.QUERY,
        parameters={
            "inst_id": {
                "type": "string",
                "description": "交易对，如 BTC-USDT、ETH-USDT",
                "required": True
            },
            "interval": {
                "type": "string",
                "description": "K线时间间隔，支持：30m、1H、2H、4H",
                "required": False,
                "default": "1H",
                "enum": ["30m", "1H", "2H", "4H"]
            },
            "start_time": {
                "type": "string",
                "description": "开始时间，格式：YYYY-MM-DD HH:MM:SS（UTC+0时间）",
                "required": False
            },
            "end_time": {
                "type": "string",
                "description": "结束时间，格式：YYYY-MM-DD HH:MM:SS（UTC+0时间）",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "返回数据条数，默认100，最大500",
                "required": False,
                "default": 100
            },
            "order_by": {
                "type": "string",
                "description": "排序方式：'time_asc'(时间升序)、'time_desc'(时间降序)",
                "required": False,
                "default": "time_asc",
                "enum": ["time_asc", "time_desc"]
            }
        },
        required_params=["inst_id"],
        risk_level="low"
    ),

    "get_current_utc_time": AgentTool(
        name="get_current_utc_time",
        description="""获取当前的UTC+0日期与时间。返回精确的时间戳和格式化的时间字符串，用于时间相关的操作和查询。

使用场景：
1. 确定当前时间用于数据查询范围
2. 计算时间差和时间间隔
3. 生成时间相关的交易决策依据
4. 同步不同系统的时间标准
5. 记录精确的操作时间点

返回信息包括：
- 当前UTC+0时间戳（毫秒）
- 格式化的时间字符串（YYYY-MM-DD HH:MM:SS）
- ISO格式的时间字符串
- 时区信息说明""",
        category=ToolCategory.QUERY,
        parameters={},
        required_params=[],
        risk_level="low"
    ),
}


def get_tool(tool_name: str) -> AgentTool:
    """获取工具定义"""
    return AGENT_TOOLS.get(tool_name)


def get_tools_by_category(category: ToolCategory) -> List[AgentTool]:
    """按分类获取工具列表"""
    return [tool for tool in AGENT_TOOLS.values() if tool.category == category]


def get_all_tools() -> Dict[str, AgentTool]:
    """获取所有工具"""
    return AGENT_TOOLS


def get_tools_config() -> List[Dict]:
    """获取所有工具的配置（字典格式）"""
    return [tool.to_dict() for tool in AGENT_TOOLS.values()]


def validate_tool_params(tool_name: str, params: Dict) -> tuple[bool, str]:
    """
    验证工具参数是否有效

    Args:
        tool_name: 工具名称
        params: 参数字典

    Returns:
        (是否有效, 错误信息)
    """
    tool = get_tool(tool_name)
    if not tool:
        return False, f"工具 {tool_name} 不存在"

    # 检查必填参数
    for required_param in tool.required_params:
        if required_param not in params or not params[required_param]:
            return False, f"缺少必填参数: {required_param}"

    # 检查参数类型和枚举值
    for param_name, param_value in params.items():
        if param_name not in tool.parameters:
            return False, f"未知参数: {param_name}"

        param_def = tool.parameters[param_name]

        # 检查枚举值
        if "enum" in param_def and param_value not in param_def["enum"]:
            return False, f"参数 {param_name} 的值必须是: {param_def['enum']}"

    return True, ""
