#!/usr/bin/env python3
"""
重新加载调度器任务
用于在运行中的应用中重新加载定时任务
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from services.schedule_service import ScheduleService
from utils.logger import setup_logger

logger = setup_logger(__name__, "reload_scheduler.log")

def main():
    """主函数"""
    try:
        print("=== 重新加载调度器任务 ===")

        # 重新加载所有调度任务
        ScheduleService.reload_all_schedules()

        # 测试调度器状态
        ScheduleService.test_scheduler()

        print("✅ 调度器任务重新加载完成")

    except Exception as e:
        logger.error(f"重新加载调度器失败: {e}")
        print(f"❌ 重新加载失败: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())