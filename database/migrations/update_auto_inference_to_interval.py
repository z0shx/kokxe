"""
数据库迁移：将自动预测时间表改为固定时间间隔
将 auto_inference_schedule (JSON数组) 改为 auto_inference_interval_hours (整数，小时)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from sqlalchemy import text
from database.db import get_db

logger = logging.getLogger(__name__)


def migrate():
    """执行迁移"""
    try:
        with get_db() as db:
            # 检查列是否存在
            result = db.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'trading_plans'
                AND column_name = 'auto_inference_schedule'
            """)).fetchone()

            if result:
                logger.info("开始迁移自动预测配置...")

                # 1. 添加新列
                try:
                    db.execute(text("""
                        ALTER TABLE trading_plans
                        ADD COLUMN auto_inference_interval_hours INTEGER DEFAULT 4
                    """))
                    logger.info("添加了 auto_inference_interval_hours 列")
                except Exception as e:
                    if "already exists" not in str(e):
                        raise e
                    logger.info("auto_inference_interval_hours 列已存在")

                # 2. 为有旧数据的记录设置默认值（如果有时间表，设置为4小时）
                db.execute(text("""
                    UPDATE trading_plans
                    SET auto_inference_interval_hours = 4
                    WHERE auto_inference_schedule IS NOT NULL
                    AND jsonb_array_length(auto_inference_schedule) > 0
                    AND auto_inference_interval_hours IS NULL
                """))

                # 3. 删除旧列
                try:
                    db.execute(text("ALTER TABLE trading_plans DROP COLUMN auto_inference_schedule"))
                    logger.info("删除了 auto_inference_schedule 列")
                except Exception as e:
                    logger.warning(f"删除旧列失败: {e}")

                # 4. 添加列注释
                db.execute(text("""
                    COMMENT ON COLUMN trading_plans.auto_inference_interval_hours IS
                    '自动预测间隔时间（小时），如 3, 6, 12, 24'
                """))

                db.commit()
                logger.info("✅ 自动预测配置迁移完成")

            else:
                logger.info("auto_inference_schedule 列不存在，无需迁移")

    except Exception as e:
        logger.error(f"迁移失败: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()