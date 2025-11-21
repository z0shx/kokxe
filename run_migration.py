#!/usr/bin/env python3
"""
执行数据库迁移脚本
"""
import sys
from pathlib import Path
from database.db import engine
from sqlalchemy import text
from utils.logger import setup_logger

logger = setup_logger(__name__, "migration.log")


def run_migration(sql_file: Path):
    """
    执行迁移脚本

    Args:
        sql_file: SQL 文件路径
    """
    if not sql_file.exists():
        logger.error(f"迁移文件不存在: {sql_file}")
        return False

    logger.info(f"开始执行迁移: {sql_file.name}")

    try:
        # 读取 SQL 文件
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # 执行 SQL
        with engine.connect() as conn:
            # 分割多个语句并执行
            statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]

            for idx, statement in enumerate(statements, 1):
                # 跳过注释
                if statement.startswith('--'):
                    continue

                logger.info(f"执行语句 {idx}/{len(statements)}: {statement[:100]}...")

                try:
                    conn.execute(text(statement))
                    conn.commit()
                    logger.info(f"✓ 语句 {idx} 执行成功")
                except Exception as e:
                    logger.error(f"✗ 语句 {idx} 执行失败: {e}")
                    # 某些语句失败是可以接受的（如 DROP CONSTRAINT IF EXISTS）
                    if "does not exist" not in str(e):
                        raise

        logger.info(f"✓ 迁移完成: {sql_file.name}")
        return True

    except Exception as e:
        logger.error(f"✗ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    migration_file = Path(__file__).parent / "database" / "migrations" / "2025-11-20-001-add-inference-batch-id.sql"

    print(f"准备执行迁移: {migration_file}")
    print("=" * 60)

    success = run_migration(migration_file)

    print("=" * 60)
    if success:
        print("✓ 迁移成功完成！")
        return 0
    else:
        print("✗ 迁移失败，请查看日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())
