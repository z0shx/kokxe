"""
测试Agent工具定义
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from database.models import TradingPlan
from database.db import get_db

def check_plan_tools():
    """检查计划启用的工具"""
    with get_db() as db:
        plan = db.query(TradingPlan).filter(
            TradingPlan.llm_config_id.isnot(None),
            TradingPlan.status == 'running'
        ).first()

        if not plan:
            print("❌ 没有找到可用的计划")
            return

        print(f"计划: {plan.inst_id} (ID: {plan.id})")

        tools_config = plan.agent_tools_config or {}
        enabled_tools = [name for name, enabled in tools_config.items() if enabled]

        print(f"\n启用的工具数量: {len(enabled_tools)}")
        print("启用的工具列表:")
        for i, tool in enumerate(enabled_tools, 1):
            print(f"  {i}. {tool}")

if __name__ == "__main__":
    check_plan_tools()