"""
测试 Agent 工具和配置
"""
import sys
sys.path.insert(0, '.')

from services.agent_tools import get_all_tools, validate_tool_params, ToolCategory
from services.agent_tools_config_helper import (
    get_default_tools_config,
    get_default_trading_limits
)


def test_tool_validation():
    """测试工具参数验证"""
    print("=" * 80)
    print("测试工具参数验证")
    print("=" * 80)

    # 测试 place_limit_order
    print("\n1. 测试 place_limit_order - 有效参数")
    valid_params = {
        "inst_id": "BTC-USDT",
        "side": "buy",
        "price": "50000",
        "size": "0.01"
    }
    is_valid, error = validate_tool_params("place_limit_order", valid_params)
    print(f"   结果: {'✓ 通过' if is_valid else f'✗ 失败: {error}'}")

    print("\n2. 测试 place_limit_order - 缺少必填参数")
    invalid_params = {
        "inst_id": "BTC-USDT",
        "side": "buy"
    }
    is_valid, error = validate_tool_params("place_limit_order", invalid_params)
    print(f"   结果: {'✓ 通过' if is_valid else f'✗ 失败: {error}'}")
    print(f"   预期: 缺少必填参数 - {error}")

    print("\n3. 测试 place_limit_order - 无效的 side 值")
    invalid_side = {
        "inst_id": "BTC-USDT",
        "side": "invalid",
        "price": "50000",
        "size": "0.01"
    }
    is_valid, error = validate_tool_params("place_limit_order", invalid_side)
    print(f"   结果: {'✓ 通过' if is_valid else f'✗ 失败: {error}'}")
    print(f"   预期: side 值必须是 ['buy', 'sell'] - {error}")

    print("\n4. 测试 get_account_balance - 无必填参数")
    params = {"ccy": "USDT"}
    is_valid, error = validate_tool_params("get_account_balance", params)
    print(f"   结果: {'✓ 通过' if is_valid else f'✗ 失败: {error}'}")


def test_default_configs():
    """测试默认配置"""
    print("\n" + "=" * 80)
    print("测试默认配置")
    print("=" * 80)

    risk_levels = ["conservative", "moderate", "aggressive"]

    for level in risk_levels:
        print(f"\n{level.upper()} 模式:")
        print("-" * 40)

        tools_config = get_default_tools_config(level)
        print(f"启用的工具: {len(tools_config['enabled_tools'])} 个")
        print(f"需要确认: {tools_config['tool_settings']['require_confirmation']}")

        limits = get_default_trading_limits(level)
        print(f"最大单笔: ${limits['max_order_amount']}")
        print(f"最大持仓: ${limits['max_position_size']}")
        print(f"每日交易: {limits['max_daily_trades']} 次")


def test_tools_by_category():
    """测试按分类获取工具"""
    print("\n" + "=" * 80)
    print("按分类统计工具")
    print("=" * 80)

    all_tools = get_all_tools()

    for category in ToolCategory:
        tools = [t for t in all_tools.values() if t.category == category]
        print(f"\n{category.value.upper()}: {len(tools)} 个工具")

        for tool in tools:
            params_count = len(tool.parameters)
            required_count = len(tool.required_params)
            print(f"  - {tool.name:25s} (参数: {params_count}, 必填: {required_count}, 风险: {tool.risk_level})")


def print_tool_example(tool_name: str):
    """打印工具使用示例"""
    all_tools = get_all_tools()
    tool = all_tools.get(tool_name)

    if not tool:
        print(f"工具 {tool_name} 不存在")
        return

    print("\n" + "=" * 80)
    print(f"工具使用示例: {tool.name}")
    print("=" * 80)

    print(f"\n分类: {tool.category.value}")
    print(f"风险: {tool.risk_level}")
    print(f"\n描述:")
    print(f"{tool.description}")

    print(f"\n必填参数:")
    for param in tool.required_params:
        param_info = tool.parameters[param]
        print(f"  - {param}: {param_info['description']}")

    print(f"\n可选参数:")
    optional_params = [p for p in tool.parameters.keys() if p not in tool.required_params]
    for param in optional_params:
        param_info = tool.parameters[param]
        print(f"  - {param}: {param_info['description']}")

    print(f"\n调用示例:")
    example_params = {}
    for param in tool.required_params:
        param_info = tool.parameters[param]
        if "enum" in param_info:
            example_params[param] = param_info["enum"][0]
        elif param_info["type"] == "string":
            if "BTC" in param_info["description"]:
                example_params[param] = "BTC-USDT"
            else:
                example_params[param] = "example_value"

    print(f'  execute_tool("{tool_name}", {example_params})')


if __name__ == "__main__":
    # 运行所有测试
    test_tool_validation()
    test_default_configs()
    test_tools_by_category()

    # 打印具体工具示例
    print_tool_example("place_limit_order")
    print_tool_example("get_account_balance")
    print_tool_example("cancel_order")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
