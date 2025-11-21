#!/usr/bin/env python
"""
数据库初始化脚本
创建或更新所有数据库表
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import engine, get_db
from database.models import Base, LLMConfig, AgentPromptTemplate, TradingPlan
from sqlalchemy import inspect, text
from utils.logger import setup_logger

logger = setup_logger(__name__, "db_init.log")


def check_table_exists(table_name: str) -> bool:
    """检查表是否存在"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def init_database():
    """初始化数据库"""
    print("=" * 60)
    print("KOKEX 数据库初始化")
    print("=" * 60)

    # 检查现有表
    print("\n1. 检查现有表...")
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    print(f"   现有表: {', '.join(existing_tables) if existing_tables else '无'}")

    # 创建所有表
    print("\n2. 创建/更新数据库表...")
    try:
        Base.metadata.create_all(engine)
        print("   ✅ 表创建/更新成功")
    except Exception as e:
        print(f"   ❌ 表创建失败: {e}")
        return False

    # 验证新表
    print("\n3. 验证新表...")
    new_tables = ['llm_configs', 'agent_prompt_templates']
    for table in new_tables:
        if check_table_exists(table):
            print(f"   ✅ {table} - 存在")
        else:
            print(f"   ❌ {table} - 不存在")

    # 检查 trading_plans 表的 llm_config_id 字段
    print("\n4. 检查 trading_plans 表字段...")
    try:
        columns = inspector.get_columns('trading_plans')
        column_names = [col['name'] for col in columns]
        if 'llm_config_id' in column_names:
            print("   ✅ llm_config_id 字段存在")
        else:
            print("   ⚠️  llm_config_id 字段不存在，需要手动添加")
            print("   执行: ALTER TABLE trading_plans ADD COLUMN llm_config_id INTEGER;")
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")

    # 插入默认数据
    print("\n5. 插入默认数据...")
    try:
        with get_db() as db:
            # 检查是否已有默认提示词模版
            existing_template = db.query(AgentPromptTemplate).filter(
                AgentPromptTemplate.name == "默认交易策略"
            ).first()

            if not existing_template:
                default_template = AgentPromptTemplate(
                    name="默认交易策略",
                    description="基础的加密货币交易策略提示词",
                    content="""你是一个专业的加密货币交易员。根据预测的K线数据，分析市场趋势并做出交易决策。

你的职责：
1. 分析预测的价格走势和成交量
2. 识别潜在的买入和卖出信号
3. 考虑风险管理和仓位控制
4. 提供明确的交易建议

请基于数据分析，给出你的交易决策。""",
                    category="balanced",
                    is_default=True
                )
                db.add(default_template)
                db.commit()
                print("   ✅ 已插入默认 Agent 提示词模版")
            else:
                print("   ℹ️  默认提示词模版已存在，跳过插入")

    except Exception as e:
        print(f"   ⚠️  插入默认数据失败: {e}")

    print("\n6. 数据库统计...")
    try:
        with get_db() as db:
            llm_count = db.query(LLMConfig).count()
            template_count = db.query(AgentPromptTemplate).count()
            plan_count = db.query(TradingPlan).count()

            print(f"   - LLM 配置: {llm_count} 条")
            print(f"   - Agent 提示词模版: {template_count} 条")
            print(f"   - 交易计划: {plan_count} 条")
    except Exception as e:
        print(f"   ⚠️  统计失败: {e}")

    print("\n" + "=" * 60)
    print("✅ 数据库初始化完成！")
    print("=" * 60)
    print("\n你现在可以：")
    print("1. 在配置中心创建 LLM 配置")
    print("2. 在配置中心创建 Agent 提示词模版")
    print("3. 在新建计划中使用这些配置")
    print()

    return True


if __name__ == "__main__":
    try:
        success = init_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
