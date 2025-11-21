"""
数据库迁移工具
用于更新已存在的表结构
"""
from sqlalchemy import text
from database.db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__, "db_migration.log")


def migrate_database():
    """执行数据库迁移"""

    migrations = [
        # 2025-11-17: 添加自动化配置字段到 trading_plans
        {
            'version': '2025-11-17-001',
            'description': '添加自动化配置字段',
            'sql': [
                'ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_finetune_enabled BOOLEAN DEFAULT FALSE',
                'ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_inference_enabled BOOLEAN DEFAULT FALSE',
                'ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_agent_enabled BOOLEAN DEFAULT FALSE',
                'ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS latest_training_record_id INTEGER'
            ]
        },
        # 2025-11-18: 添加自动工具执行开关
        {
            'version': '2025-11-18-001',
            'description': '添加自动工具执行开关',
            'sql': [
                'ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_tool_execution_enabled BOOLEAN DEFAULT FALSE'
            ]
        },
        # 2025-11-19: 添加不确定性范围和概率指标到 prediction_data
        {
            'version': '2025-11-19-001',
            'description': '添加不确定性范围和概率指标',
            'sql': [
                # 不确定性范围字段
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_min REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_max REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_std REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS open_min REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS open_max REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS high_min REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS high_max REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS low_min REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS low_max REAL',
                # 概率指标字段
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS upward_probability REAL',
                'ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS volatility_amplification_probability REAL',
                # 更新 close 字段注释
                'COMMENT ON COLUMN prediction_data.close IS \'预测收盘价（平均值）\''
            ]
        }
    ]

    logger.info("开始数据库迁移...")

    with get_db() as db:
        for migration in migrations:
            logger.info(f"执行迁移: {migration['version']} - {migration['description']}")

            for sql in migration['sql']:
                try:
                    db.execute(text(sql))
                    logger.info(f"  ✓ {sql[:100]}...")
                except Exception as e:
                    logger.warning(f"  ⚠ 跳过: {str(e)[:100]}")

            db.commit()

    logger.info("数据库迁移完成")
    print("✅ 数据库迁移完成")


if __name__ == "__main__":
    migrate_database()
