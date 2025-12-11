"""
调度器配置
用于控制使用哪个调度器服务，并支持平滑迁移
"""

import os
from typing import Optional

# 环境变量控制
USE_UNIFIED_SCHEDULER = os.getenv('USE_UNIFIED_SCHEDULER', 'true').lower() == 'true'
DISABLE_LEGACY_SCHEDULERS = os.getenv('DISABLE_LEGACY_SCHEDULERS', 'false').lower() == 'true'

class SchedulerConfig:
    """调度器配置管理"""

    @staticmethod
    def should_use_unified_scheduler() -> bool:
        """是否使用统一调度器"""
        return USE_UNIFIED_SCHEDULER

    @staticmethod
    def should_disable_legacy() -> bool:
        """是否禁用旧的调度器"""
        return DISABLE_LEGACY_SCHEDULERS

    @staticmethod
    def get_scheduler_info() -> dict:
        """获取当前调度器配置信息"""
        return {
            'use_unified': USE_UNIFIED_SCHEDULER,
            'disable_legacy': DISABLE_LEGACY_SCHEDULERS,
            'unified_class': 'services.unified_scheduler.UnifiedScheduler' if USE_UNIFIED_SCHEDULER else None,
            'legacy_classes': [
                'services.scheduler_service.SchedulerService',
                'services.schedule_service.ScheduleService'
            ] if not DISABLE_LEGACY_SCHEDULERS else []
        }

# 导出配置
scheduler_config = SchedulerConfig()