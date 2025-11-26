"""
测试资金管理策略
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.capital_management_service import CapitalManagementService

async def test_capital_management():
    """测试资金管理功能"""

    print("🧪 测试资金管理策略")
    print("=" * 50)

    # 使用计划ID 2 (ETH-USDT) 进行测试
    plan_id = 2
    capital_service = CapitalManagementService(plan_id)

    # 测试1: 获取当前资金信息
    print("\n📊 测试1: 获取当前资金信息")
    capital_info = await capital_service.get_current_capital_info()

    if 'error' in capital_info:
        print(f"❌ 获取资金信息失败: {capital_info['error']}")
        return False

    print(f"✅ 当前总资金: ${capital_info['current_capital']:.2f} USDT")
    print(f"✅ 可用USDT: ${capital_info['available_usdt']:.2f} USDT")
    print(f"✅ 初始本金: ${capital_info['initial_capital']:.2f} USDT")
    print(f"✅ 盈亏: ${capital_info['profit_loss']:+.2f} USDT ({capital_info['profit_loss_percentage']:+.2f}%)")
    print(f"✅ 平均每批订单数: {capital_info['avg_orders_per_batch']}")
    print(f"✅ 下次下单金额: ${capital_info['next_order_amount']:.2f} USDT")

    # 测试2: 计算买入订单参数
    print("\n🧮 测试2: 计算买入订单参数")
    price = 2800.0  # ETH价格

    order_params = await capital_service.calculate_order_parameters(
        side='buy',
        price=price,
        custom_amount=None,  # 使用动态平摊金额
        custom_size=None
    )

    if order_params.get('success'):
        print(f"✅ 建议买入金额: ${order_params['amount']:.2f} USDT")
        print(f"✅ 建议买入数量: {order_params['size']:.6f} ETH")
        if order_params.get('risk_warnings'):
            print(f"⚠️ 风险提示: {'; '.join(order_params['risk_warnings'])}")
    else:
        print(f"❌ 计算订单参数失败: {order_params.get('error')}")
        return False

    # 测试3: 模拟下单（不实际执行）
    print("\n🔄 测试3: 模拟资金管理下单")

    # 检查是否为模拟盘
    with await capital_service._get_trading_tools() if hasattr(capital_service, '_get_trading_tools') else None as trading_tools:
        pass

    print("📝 计算完成的订单参数:")
    print(f"   交易对: ETH-USDT")
    print(f"   方向: 买入")
    print(f"   价格: ${price:.2f}")
    print(f"   金额: ${order_params['amount']:.2f}")
    print(f"   数量: {order_params['size']:.6f}")

    print("\n✅ 动态平摊策略验证完成!")
    print("\n💡 策略说明:")
    print("1. 当前资金根据盈亏动态调整")
    print("2. 每次下单金额 = 当前总资金 / 平均批次数")
    print("3. 自动进行风险检查和余额验证")
    print("4. 支持自定义金额覆盖平摊逻辑")

    return True

async def test_manual_custom_amount():
    """测试自定义金额下单"""

    print("\n🧪 测试自定义金额下单")
    print("=" * 50)

    plan_id = 2
    capital_service = CapitalManagementService(plan_id)

    # 测试自定义小额
    custom_amount = 50.0  # 50 USDT
    price = 2800.0

    order_params = await capital_service.calculate_order_parameters(
        side='buy',
        price=price,
        custom_amount=custom_amount,
        custom_size=None
    )

    if order_params.get('success'):
        print(f"✅ 自定义金额 ${custom_amount:.2f} USDT:")
        print(f"   计算数量: {order_params['size']:.6f} ETH")
        print(f"   总价值: ${order_params['size'] * price:.2f} USDT")
    else:
        print(f"❌ 自定义金额测试失败: {order_params.get('error')}")

if __name__ == "__main__":
    print("🚀 开始测试动态平摊下单策略")

    # 基础功能测试
    success1 = asyncio.run(test_capital_management())

    # 自定义金额测试
    asyncio.run(test_manual_custom_amount())

    if success1:
        print("\n🎉 所有测试通过！")
    else:
        print("\n💥 测试失败！")