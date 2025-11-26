"""
添加资金管理字段的数据库迁移
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import SessionLocal
from sqlalchemy import text

def add_capital_management_fields():
    """添加资金管理相关字段到trading_plans表"""

    migrations = [
        # 添加资金管理字段
        "ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS initial_capital FLOAT DEFAULT 1000.0",
        "ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS avg_orders_per_batch INTEGER DEFAULT 10",
        "ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS max_single_order_ratio FLOAT DEFAULT 0.2",
        "ALTER TABLE trading_plans ADD COLUMN IF NOT EXISTS capital_management_enabled BOOLEAN DEFAULT TRUE",

        # 添加字段注释
        "COMMENT ON COLUMN trading_plans.initial_capital IS '初始本金（USDT）'",
        "COMMENT ON COLUMN trading_plans.avg_orders_per_batch IS '平均每批订单数（用于平摊策略）'",
        "COMMENT ON COLUMN trading_plans.max_single_order_ratio IS '单次订单最大占总资金比例'",
        "COMMENT ON COLUMN trading_plans.capital_management_enabled IS '是否启用资金管理策略'",
    ]

    db = SessionLocal()
    try:
        for migration in migrations:
            print(f"执行: {migration}")
            db.execute(text(migration))

        db.commit()
        print("✅ 资金管理字段添加完成")

        # 验证字段是否添加成功
        result = db.execute(text("""
            SELECT column_name, data_type, column_default
            FROM information_schema.columns
            WHERE table_name = 'trading_plans'
            AND column_name IN ('initial_capital', 'avg_orders_per_batch', 'max_single_order_ratio', 'capital_management_enabled')
        """))

        columns = result.fetchall()
        print("\n📋 新增字段验证:")
        for col in columns:
            print(f"  ✓ {col[0]} ({col[1]}) - 默认值: {col[2]}")

    except Exception as e:
        db.rollback()
        print(f"❌ 迁移失败: {e}")
        return False
    finally:
        db.close()

    return True

if __name__ == "__main__":
    print("开始添加资金管理字段...")
    success = add_capital_management_fields()
    if success:
        print("🎉 迁移成功完成！")
    else:
        print("💥 迁移失败！")