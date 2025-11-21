"""
数据库重置脚本
警告：这将删除所有数据！
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from database.db import engine
from database.models import Base
from utils.logger import setup_logger

logger = setup_logger(__name__, "reset_db.log")


def reset_database():
    """重置数据库（删除所有表并重新创建）"""
    print("=" * 60)
    print("⚠️  警告：这将删除所有数据库表和数据！")
    print("=" * 60)

    response = input("确认要继续吗？(yes/no): ")
    if response.lower() != 'yes':
        print("已取消")
        return

    try:
        logger.info("开始重置数据库...")

        # 删除所有表
        logger.info("删除所有表...")
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ 表删除成功")

        # 重新创建所有表
        logger.info("重新创建所有表...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ 表创建成功")

        # 导出 schema
        logger.info("导出数据库 schema...")
        from database.db import export_schema
        export_schema()
        logger.info("✅ Schema 导出成功")

        print("\n" + "=" * 60)
        print("✅ 数据库重置完成！")
        print("=" * 60)

    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        print(f"\n❌ 错误: {e}")
        raise


if __name__ == "__main__":
    reset_database()
