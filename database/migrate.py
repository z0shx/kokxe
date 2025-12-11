"""
数据库迁移工具（清理版）
用于更新已存在的表结构，修复跳过警告问题
"""
from sqlalchemy import text
from database.db import get_db
from utils.logger import setup_logger

logger = setup_logger(__name__, "db_migration.log")


def execute_migration_safe(db, sql, description):
    """安全执行单个SQL语句"""
    try:
        db.execute(text(sql))
        logger.info(f"  ✓ {description}")
        return True
    except Exception as e:
        # 只记录真正需要关注的错误
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in [
            'duplicate column', 'already exists', 'does not exist'
        ]):
            logger.info(f"  ○ 跳过（已存在）: {description}")
        else:
            logger.warning(f"  ⚠ 跳过: {description} - {str(e)[:100]}")
        return False


def migrate_database():
    """执行数据库迁移"""

    migrations = [
        # 2025-11-17: 添加自动化配置字段
        {
            'version': '2025-11-17-001',
            'description': '添加自动化配置字段',
            'sql': [
                ('ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_finetune_enabled BOOLEAN DEFAULT FALSE', 'auto_finetune_enabled'),
                ('ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_inference_enabled BOOLEAN DEFAULT FALSE', 'auto_inference_enabled'),
                ('ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_agent_enabled BOOLEAN DEFAULT FALSE', 'auto_agent_enabled'),
                ('ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS latest_training_record_id INTEGER', 'latest_training_record_id')
            ]
        },
        # 2025-11-18: 添加自动工具执行开关
        {
            'version': '2025-11-18-001',
            'description': '添加自动工具执行开关',
            'sql': [
                ('ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS auto_tool_execution_enabled BOOLEAN DEFAULT FALSE', 'auto_tool_execution_enabled')
            ]
        },
        # 2025-11-19: 添加不确定性范围和概率指标到 prediction_data
        {
            'version': '2025-11-19-001',
            'description': '添加不确定性范围和概率指标',
            'sql': [
                # 不确定性范围字段
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_min REAL', 'close_min'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_max REAL', 'close_max'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS close_std REAL', 'close_std'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS open_min REAL', 'open_min'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS open_max REAL', 'open_max'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS high_min REAL', 'high_min'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS high_max REAL', 'high_max'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS low_min REAL', 'low_min'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS low_max REAL', 'low_max'),
                # 概率指标字段
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS upward_probability REAL', 'upward_probability'),
                ('ALTER TABLE prediction_data ADD COLUMN IF NOT EXISTS volatility_amplification_probability REAL', 'volatility_amplification_probability')
            ]
        },
        # 2025-12-04: 增强Agent对话记录功能
        {
            'version': '2025-12-04-001',
            'description': '增强Agent对话记录功能',
            'sql': [
                ('ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS agent_message_id INTEGER', 'trade_orders.agent_message_id'),
                ('ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS conversation_id INTEGER', 'trade_orders.conversation_id'),
                ('ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS tool_call_id VARCHAR(100)', 'trade_orders.tool_call_id'),
                ('ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS tool_call_id VARCHAR(100)', 'agent_messages.tool_call_id'),
                ('ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS tool_execution_time FLOAT', 'agent_messages.tool_execution_time'),
                ('ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS related_order_id VARCHAR(100)', 'agent_messages.related_order_id'),
                # 添加索引
                ('CREATE INDEX IF NOT EXISTS idx_trade_order_agent_message_id ON trade_orders(agent_message_id)', 'idx_trade_order_agent_message_id'),
                ('CREATE INDEX IF NOT EXISTS idx_trade_order_conversation_id ON trade_orders(conversation_id)', 'idx_trade_order_conversation_id'),
                ('CREATE INDEX IF NOT EXISTS idx_trade_order_tool_call_id ON trade_orders(tool_call_id)', 'idx_trade_order_tool_call_id'),
                ('CREATE INDEX IF NOT EXISTS idx_agent_message_tool_call_id ON agent_messages(tool_call_id)', 'idx_agent_message_tool_call_id'),
                ('CREATE INDEX IF NOT EXISTS idx_agent_message_related_order_id ON agent_messages(related_order_id)', 'idx_agent_message_related_order_id')
            ]
        },
        # 2025-12-10: 订单频道订阅支持（已通过单独脚本执行，跳过）
        {
            'version': '2025-12-10-001',
            'description': '订单频道订阅支持（已完成）',
            'sql': []  # 已通过 sql/fix_order_subscription_simple.sql 执行
        }
    ]

    logger.info("开始数据库迁移（清理版）...")

    with get_db() as db:
        for migration in migrations:
            logger.info(f"执行迁移: {migration['version']} - {migration['description']}")

            # 检查是否已执行过此迁移（简单的版本记录）
            try:
                # 尝试创建迁移记录表（如果不存在）
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS migration_history (
                        version VARCHAR(50) PRIMARY KEY,
                        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        description TEXT
                    )
                """))
                db.commit()
            except Exception:
                pass  # 忽略创建表错误

            # 检查是否已执行过
            try:
                result = db.execute(text(
                    "SELECT COUNT(*) FROM migration_history WHERE version = :version"
                ), {"version": migration['version']}).scalar()

                if result > 0:
                    logger.info(f"  ✓ 迁移已执行过，跳过: {migration['version']}")
                    continue
            except Exception:
                pass  # 如果没有迁移记录表，继续执行

            # 执行SQL语句
            success_count = 0
            for sql, description in migration['sql']:
                if execute_migration_safe(db, sql, description):
                    success_count += 1

            # 记录迁移执行历史
            if success_count > 0:
                try:
                    db.execute(text("""
                        INSERT INTO migration_history (version, description)
                        VALUES (:version, :description)
                        ON CONFLICT (version) DO NOTHING
                    """), {
                        "version": migration['version'],
                        "description": migration['description']
                    })
                except Exception:
                    pass  # 忽略记录错误

            db.commit()

    logger.info("数据库迁移完成")
    print("✅ 数据库迁移完成（清理版）")


if __name__ == "__main__":
    migrate_database()