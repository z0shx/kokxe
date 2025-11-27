#!/usr/bin/env python3
"""
测试调度器状态和任务配置
"""

import asyncio
from datetime import datetime
from database.db import get_db
from database.models import TradingPlan
from services.schedule_service import ScheduleService

def main():
    print("=== 调度器状态检查 ===")

    # 检查调度器是否初始化
    scheduler = ScheduleService.get_scheduler()
    print(f"调度器状态: {'运行中' if scheduler.running else '未运行'}")
    print(f"调度器任务数量: {len(scheduler.get_jobs())}")

    # 检查计划配置
    print("\n=== 计划配置检查 ===")
    with get_db() as db:
        plans = db.query(TradingPlan).all()
        for plan in plans:
            print(f"\n计划ID: {plan.id}")
            # print(f"  名称: {plan.name}")  # TradingPlan可能没有name字段
            print(f"  状态: {plan.status}")
            print(f"  自动预测启用: {plan.auto_inference_enabled}")
            print(f"  预测间隔时间: {plan.auto_inference_interval_hours}小时")
            print(f"  自动微调启用: {plan.auto_finetune_enabled}")
            print(f"  微调时间表: {plan.auto_finetune_schedule}")

            # 检查该计划的任务
            jobs = ScheduleService.get_plan_jobs(plan.id)
            print(f"  关联任务数量: {len(jobs)}")
            for job in jobs:
                print(f"    - {job.name} (ID: {job.id})")
                if job.next_run_time:
                    next_run_beijing = job.next_run_time.astimezone(ScheduleService.BEIJING_TZ)
                    print(f"      下次执行(UTC+8): {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"      下次执行: 未设置")

    print("\n=== 所有调度器任务 ===")
    for job in scheduler.get_jobs():
        print(f"- {job.name} (ID: {job.id})")
        print(f"  触发器: {job.trigger}")
        if job.next_run_time:
            next_run_beijing = job.next_run_time.astimezone(ScheduleService.BEIJING_TZ)
            print(f"  下次执行(UTC+8): {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

async def test_start_schedule():
    print("\n=== 手动测试启动计划调度 ===")
    try:
        # 手动为计划ID 2 启动调度
        result = await ScheduleService.start_schedule(2)
        print(f"启动计划调度结果: {result}")

        # 再次检查调度器状态
        print("\n=== 启动后的调度器状态 ===")
        scheduler = ScheduleService.get_scheduler()
        print(f"调度器任务数量: {len(scheduler.get_jobs())}")

        for job in scheduler.get_jobs():
            print(f"- {job.name} (ID: {job.id})")
            if job.next_run_time:
                next_run_beijing = job.next_run_time.astimezone(ScheduleService.BEIJING_TZ)
                print(f"  下次执行(UTC+8): {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"启动计划调度失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    # 手动测试启动调度
    asyncio.run(test_start_schedule())