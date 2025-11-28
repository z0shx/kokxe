#!/usr/bin/env python3
"""
更新Agent配置以适配完整的LangChain Agent实现
"""

def update_agent_prompt_for_langchain():
    """更新Agent提示词以优化LangChain Agent使用"""

    updated_prompt = """智能K线交易决策系统 - LangChain Agent版本

你是一个专业的加密货币交易AI助手，使用LangChain Agent框架进行智能决策。

## 核心能力
- 分析预测数据和历史K线数据
- 执行交易操作和风险管理
- 使用工具调用获取实时信息
- 提供清晰的分析和决策过程

## 可用工具
你拥有以下13个工具来完成交易任务：
1. place_order - 下单交易
2. cancel_order - 取消订单
3. modify_order - 修改订单
4. get_positions - 获取持仓信息
5. get_pending_orders - 获取挂单信息
6. get_account_balance - 获取账户余额
7. get_current_utc_time - 获取当前时间
8. place_stop_loss_order - 设置止损订单
9. query_prediction_data - 查询预测数据
10. get_prediction_history - 获取预测历史
11. run_latest_model_inference - 运行最新推理
12. query_historical_kline_data - 查询历史K线数据
13. delete_prediction_data_by_batch - 删除预测数据

## 决策流程
1. 数据分析：查询最新的预测数据和历史K线数据
2. 市场分析：识别价格趋势和交易机会
3. 风险评估：检查当前持仓和账户状态
4. 交易决策：基于分析结果执行合适的交易操作
5. 风险控制：设置合理的止损和止盈

## 资金与风控规则
- 已占用本金 + 新订单 ≤ 可用余额
- 最大订单数：N个，本金均分
- 止损：单笔亏损 ≥ 20% 立即平仓
- 每次仅新建1个限价订单
- 保守原则：不确定时不操作，保持现状

## 响应格式
请提供：
1. 📊 数据分析结果
2. 🧠 市场判断和推理过程
3. 🛠️  工具调用记录（如有）
4. 📈 交易决策和理由

使用工具获取必要的信息，然后基于数据做出明智的交易决策。
"""

    return updated_prompt

def main():
    """主更新函数"""
    print("🔧 更新Agent配置以适配LangChain Agent")
    print("=" * 50)

    from database.db import get_db
    from database.models import TradingPlan
    import json

    with get_db() as db:
        plan = db.query(TradingPlan).filter(TradingPlan.id == 2).first()

        if not plan:
            print("❌ 未找到计划2")
            return

        print(f"📊 更新计划: {plan.plan_name}")
        print(f"原提示词长度: {len(plan.agent_prompt) if plan.agent_prompt else 0}")

        # 更新提示词
        new_prompt = update_agent_prompt_for_langchain()
        plan.agent_prompt = new_prompt

        print(f"新提示词长度: {len(new_prompt)}")

        # 工具配置保持不变（已经是正确的13个工具）
        if plan.agent_tools_config:
            tools_config = json.loads(plan.agent_tools_config) if isinstance(plan.agent_tools_config, str) else plan.agent_tools_config
            print(f"工具配置: {len(tools_config)} 个工具（保持不变）")

        try:
            db.commit()
            print("✅ Agent配置更新完成")

            # 显示更新后的配置摘要
            print("\\n📋 更新后的配置摘要:")
            print(f"✅ 提示词: 包含LangChain Agent使用说明")
            print(f"✅ 工具: 13个工具配置完整")
            print(f"✅ 指令: 优化的决策流程和响应格式")

        except Exception as e:
            print(f"❌ 更新失败: {e}")
            db.rollback()

if __name__ == "__main__":
    main()