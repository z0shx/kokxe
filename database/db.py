"""
数据库连接和操作
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from config import config
from utils.logger import setup_logger
from database.models import Base

logger = setup_logger(__name__, "database.log")

# 创建数据库引擎，设置时区为UTC+8
engine = create_engine(
    config.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
    connect_args={"options": "-c timezone=Asia/Shanghai"}
)

# 创建会话工厂
# 设置 expire_on_commit=False，防止对象在 commit 后过期
# 这样在 session 关闭后仍然可以访问对象属性
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


def init_db():
    """初始化数据库，创建所有表"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化成功")
    except Exception as e:
        logger.error(f"数据库表初始化失败: {e}")
        raise


def export_schema():
    """导出数据库 schema 到 SQL 文件"""
    from sqlalchemy.schema import CreateTable
    import os

    sql_file = config.SQL_DIR / "schema.sql"
    with open(sql_file, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("-- KOKEX Database Schema\n")
        f.write(f"-- Generated at: {__import__('datetime').datetime.now()}\n\n")

        # 导出每个表的 CREATE TABLE 语句
        for table in Base.metadata.sorted_tables:
            f.write(f"\n-- Table: {table.name}\n")
            create_stmt = str(CreateTable(table).compile(engine))
            f.write(create_stmt)
            f.write(";\n")

    logger.info(f"数据库 schema 已导出到: {sql_file}")


from contextlib import contextmanager

@contextmanager
def get_db():
    """
    获取数据库会话（上下文管理器）

    用法:
        with get_db() as db:
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"数据库操作错误: {e}")
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    获取数据库会话（需手动关闭）

    用法:
        db = get_db_session()
        try:
            db.query(...)
            db.commit()
        finally:
            db.close()
    """
    return SessionLocal()
