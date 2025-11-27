#!/usr/bin/env python3
"""
测试重新加载调度器功能
"""

from services.schedule_service import ScheduleService
from database.db import get_db
from database.models import TradingPlan

def main():
    print("=== 测试重新加载调度器功能 ===")

    # 先清空当前调度器
    scheduler = ScheduleService.get_scheduler()
    print(f"当前调度器任务数: {len(scheduler.get_jobs())}")

    # 手动调用重新加载所有调度器
    print("\n调用 reload_all_schedules()...")
    ScheduleService.reload_all_schedules()

    # 检查结果
    print(f"\n重新加载后的调度器任务数: {len(scheduler.get_jobs())}")

    for job in scheduler.get_jobs():
        print(f"- {job.name} (ID: {job.id})")
        if job.next_run_time:
            next_run_beijing = job.next_run_time.astimezone(ScheduleService.BEIJING_TZ)
            print(f"  下次执行(UTC+8): {next_run_beijing.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()