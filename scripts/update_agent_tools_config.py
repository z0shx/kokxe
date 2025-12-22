#!/usr/bin/env python3
"""
更新现有计划的Agent工具配置，添加新的预测分析工具
"""

import sys
import os
sys.path.insert(0, '.')

from database.db import SessionLocal
from database.models import TradingPlan
import json

def update_plan_tools_config(plan_id: int = None):
    """更新计划的工具配置"""
    db = SessionLocal()

    try:
        # 获取计划
        query = db.query(TradingPlan)
        if plan_id:
            query = query.filter(TradingPlan.id == plan_id)

        plans = query.all()

        if not plans:
            print("没有找到符合条件的计划")
            return

        updated_count = 0

        for plan in plans:
            print(f"\\n处理计划: {plan.plan_name} (ID: {plan.id})")

            # 获取现有工具配置
            tools_config = plan.agent_tools_config
            if isinstance(tools_config, str):
                if tools_config.strip():
                    try:
                        tools_config = json.loads(tools_config)
                    except json.JSONDecodeError:
                        tools_config = {}
                else:
                    tools_config = {}
            elif tools_config is None:
                tools_config = {}

            # 检查是否已包含新工具
            new_tool = 'get_latest_prediction_analysis'
            if new_tool not in tools_config:
                # 添加新工具并启用
                tools_config[new_tool] = True
                updated_count += 1
                print(f"  ✅ 添加新工具: {new_tool} -> 启用")
            else:
                print(f"  ⚠️  工具已存在: {new_tool} -> {tools_config[new_tool]}")
                continue

            # 更新配置
            plan.agent_tools_config = tools_config

            print(f"  📋 更新后的工具数量: {len(tools_config)}")

            # 显示启用的工具
            enabled_tools = [name for name, enabled in tools_config.items() if enabled]
            print(f"  🔧 启用的工具: {enabled_tools}")

        # 提交更改
        if updated_count > 0:
            db.commit()
            print(f"\\n✅ 成功更新 {updated_count} 个计划的工具配置")
        else:
            print("\\n⚠️  没有计划需要更新")

    except Exception as e:
        db.rollback()
        print(f"\\n❌ 更新失败: {e}")
        raise
    finally:
        db.close()

def main():
    """主函数"""
    print("🔄 更新Agent工具配置")
    print("=" * 50)

    if len(sys.argv) > 1:
        # 指定计划ID
        plan_id = int(sys.argv[1])
        print(f"🎯 更新指定计划ID: {plan_id}")
        update_plan_tools_config(plan_id)
    else:
        # 更新所有计划
        print("🎯 更新所有计划")
        update_plan_tools_config()

if __name__ == "__main__":
    main()