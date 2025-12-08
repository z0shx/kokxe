#!/usr/bin/env python3
"""
测试自动训练状态更新修复
"""
import asyncio
import sys
from services.schedule_service import ScheduleService

async def test_auto_training():
    """测试自动训练是否能正确完成状态更新"""
    print("开始测试自动训练状态更新修复...")

    # 使用计划ID 2（ETH-USDT）
    plan_id = 2

    try:
        print(f"触发自动训练: plan_id={plan_id}")
        result = await ScheduleService._trigger_finetune(plan_id)

        print(f"训练结果: {result}")

        if result.get('success'):
            print(f"✅ 自动训练成功完成!")
            print(f"   - training_id: {result.get('training_id')}")
            print(f"   - final_status: {result.get('final_status')}")
            print(f"   - duration: {result.get('duration')}s")
        else:
            print(f"❌ 自动训练失败: {result.get('error')}")

        return result

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # 运行测试
    result = asyncio.run(test_auto_training())
    sys.exit(0 if result.get('success') else 1)