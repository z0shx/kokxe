#!/usr/bin/env python
"""
添加 llm_config_id 字段到 trading_plans 表
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import engine
from sqlalchemy import text

print("=" * 60)
print("添加 llm_config_id 字段到 trading_plans 表")
print("=" * 60)

try:
    with engine.connect() as conn:
        # 检查字段是否已存在
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'trading_plans'
            AND column_name = 'llm_config_id'
        """))

        if result.fetchone():
            print("\n✅ llm_config_id 字段已存在")
        else:
            print("\n正在添加 llm_config_id 字段...")
            conn.execute(text("""
                ALTER TABLE trading_plans
                ADD COLUMN llm_config_id INTEGER
            """))
            conn.commit()
            print("✅ llm_config_id 字段添加成功")

        # 验证字段
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'trading_plans'
            AND column_name = 'llm_config_id'
        """))

        row = result.fetchone()
        if row:
            print(f"\n验证结果:")
            print(f"  字段名: {row[0]}")
            print(f"  数据类型: {row[1]}")

        print("\n" + "=" * 60)
        print("✅ 操作完成！")
        print("=" * 60)

except Exception as e:
    print(f"\n❌ 操作失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
