#!/usr/bin/env python3
"""
测试数据验证服务修复
"""
import asyncio
import sys
from datetime import datetime
from api.okx_client import OKXClient
from services.data_validation_service import data_validation_service

async def test_okx_data_parsing():
    """测试OKX数据解析和数据库保存"""
    print("🧪 测试数据验证服务修复...")

    try:
        # 初始化OKX客户端
        okx_client = OKXClient()
        print("✅ OKX客户端初始化成功")

        # 测试获取历史数据
        print("📡 获取测试数据...")
        kline_data = okx_client.get_history_candles(
            inst_id="ETH-USDT",
            bar="1H",
            after="1763758800000",  # 最近一个时间点
            limit=1
        )

        if not kline_data:
            print("❌ 未能获取测试数据")
            return False

        print(f"✅ 获取到 {len(kline_data)} 条原始K线数据")

        # 测试数据解析
        formatted_data = []
        for candle in kline_data:
            parsed = okx_client.parse_candle_data(candle)
            if parsed:
                from utils.timezone_helper import convert_to_beijing_time
                beijing_time = convert_to_beijing_time(parsed['timestamp']).replace(tzinfo=None)

                formatted_data.append({
                    'instId': "ETH-USDT",
                    'ts': str(int(parsed['timestamp'].timestamp() * 1000)),
                    'o': str(parsed['open']),
                    'h': str(parsed['high']),
                    'l': str(parsed['low']),
                    'c': str(parsed['close']),
                    'vol': str(parsed['volume']),
                    'ccy': str(parsed['amount'])
                })

        print(f"✅ 成功解析 {len(formatted_data)} 条数据")

        # 测试数据库保存
        print("💾 测试数据库保存...")
        await data_validation_service._save_kline_data(2, formatted_data, "1H")
        print("✅ 数据库保存测试成功")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("=" * 50)
    print("数据验证服务修复测试")
    print("=" * 50)

    # 初始化数据验证服务
    await data_validation_service.initialize()

    # 运行测试
    success = await test_okx_data_parsing()

    if success:
        print("\n🎉 所有测试通过！数据验证服务修复成功")
        sys.exit(0)
    else:
        print("\n💥 测试失败，需要进一步修复")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())