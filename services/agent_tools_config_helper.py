"""
Agent 工具配置示例和帮助函数
"""
from services.agent_tools import get_all_tools, ToolCategory


def get_default_tools_config(risk_level: str = "conservative") -> dict:
    """
    获取默认的工具配置

    Args:
        risk_level: 风险级别
            - conservative: 保守（只允许查询类工具）
            - moderate: 适中（允许查询 + 撤单/改单）
            - aggressive: 激进（允许所有工具）

    Returns:
        工具配置字典
    """
    all_tools = get_all_tools()

    if risk_level == "conservative":
        # 保守模式：只允许查询类工具
        enabled_tools = [
            "get_account_balance",
            "get_account_positions",
            "get_order_info",
            "get_pending_orders",
            "get_order_history",
            "get_fills",
            "get_current_price",
            "get_latest_prediction_analysis"
        ]
    elif risk_level == "moderate":
        # 适中模式：允许查询 + 撤单/改单
        enabled_tools = [
            "get_account_balance",
            "get_account_positions",
            "get_order_info",
            "get_pending_orders",
            "get_order_history",
            "get_fills",
            "get_current_price",
            "get_latest_prediction_analysis",
            "cancel_order",
            "amend_order"
        ]
    else:  # aggressive
        # 激进模式：允许所有工具
        enabled_tools = [tool_name for tool_name in all_tools.keys()]

    return {
        "enabled_tools": enabled_tools,
        "tool_settings": {
            "require_confirmation": risk_level != "aggressive",
            "auto_cancel_on_error": True,
            "max_retries": 3
        }
    }


def get_default_trading_limits(risk_level: str = "conservative") -> dict:
    """
    获取默认的交易限制配置

    Args:
        risk_level: 风险级别

    Returns:
        交易限制字典
    """
    if risk_level == "conservative":
        return {
            "max_order_amount": 100.0,  # 单笔最大交易金额（USDT）
            "min_order_amount": 10.0,   # 单笔最小交易金额（USDT）
            "max_daily_trades": 10,     # 每日最大交易次数
            "max_position_size": 500.0, # 最大持仓金额（USDT）
            "allowed_inst_ids": [],     # 允许的交易对（空表示不限制）
            "stop_loss_percentage": 0.03,  # 止损百分比（3%）
            "take_profit_percentage": 0.05 # 止盈百分比（5%）
        }
    elif risk_level == "moderate":
        return {
            "max_order_amount": 500.0,
            "min_order_amount": 10.0,
            "max_daily_trades": 30,
            "max_position_size": 2000.0,
            "allowed_inst_ids": [],
            "stop_loss_percentage": 0.05,
            "take_profit_percentage": 0.10
        }
    else:  # aggressive
        return {
            "max_order_amount": 2000.0,
            "min_order_amount": 10.0,
            "max_daily_trades": 100,
            "max_position_size": 10000.0,
            "allowed_inst_ids": [],
            "stop_loss_percentage": 0.08,
            "take_profit_percentage": 0.15
        }


def print_all_tools():
    """打印所有可用的工具"""
    all_tools = get_all_tools()

    print("=" * 80)
    print("所有可用的 Agent 交易工具")
    print("=" * 80)

    # 按分类打印
    for category in ToolCategory:
        tools = [t for t in all_tools.values() if t.category == category]
        if tools:
            print(f"\n【{category.value.upper()}】")
            print("-" * 80)

            for tool in tools:
                print(f"\n名称: {tool.name}")
                print(f"风险等级: {tool.risk_level}")
                print(f"描述: {tool.description[:100]}...")
                print(f"必填参数: {', '.join(tool.required_params) if tool.required_params else '无'}")

    print("\n" + "=" * 80)


def print_tool_details(tool_name: str):
    """打印工具的详细信息"""
    all_tools = get_all_tools()
    tool = all_tools.get(tool_name)

    if not tool:
        print(f"工具 '{tool_name}' 不存在")
        return

    print("=" * 80)
    print(f"工具详情: {tool.name}")
    print("=" * 80)
    print(f"\n分类: {tool.category.value}")
    print(f"风险等级: {tool.risk_level}")
    print(f"\n描述:\n{tool.description}")
    print(f"\n参数:")

    for param_name, param_info in tool.parameters.items():
        required = "【必填】" if param_name in tool.required_params else "【可选】"
        print(f"\n  {required} {param_name}")
        print(f"    类型: {param_info.get('type')}")
        print(f"    说明: {param_info.get('description')}")

        if "enum" in param_info:
            print(f"    可选值: {param_info['enum']}")
        if "default" in param_info:
            print(f"    默认值: {param_info['default']}")

    print("\n" + "=" * 80)


# 使用示例
if __name__ == "__main__":
    # 打印所有工具
    print_all_tools()

    print("\n\n")

    # 打印具体工具详情
    print_tool_details("place_limit_order")

    print("\n\n")

    # 获取默认配置
    print("=" * 80)
    print("默认工具配置示例")
    print("=" * 80)

    for risk_level in ["conservative", "moderate", "aggressive"]:
        print(f"\n{risk_level.upper()} 模式:")
        print("-" * 40)

        tools_config = get_default_tools_config(risk_level)
        print(f"启用的工具数量: {len(tools_config['enabled_tools'])}")
        print(f"工具列表: {', '.join(tools_config['enabled_tools'][:5])}...")

        limits = get_default_trading_limits(risk_level)
        print(f"最大单笔交易: ${limits['max_order_amount']}")
        print(f"每日交易次数: {limits['max_daily_trades']}")
        print(f"止损比例: {limits['stop_loss_percentage'] * 100}%")
