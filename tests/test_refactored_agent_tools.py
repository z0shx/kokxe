#!/usr/bin/env python3
"""
测试重构后的Agent工具
基于用户需求的10个核心工具进行全面测试
"""

import asyncio
import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service

class RefactoredToolTester:
    def __init__(self, plan_id=2):
        self.plan_id = plan_id
        self.results = {}
        self.plan_info = None
        self.llm_config = None

    async def setup(self):
        """设置测试环境"""
        from database.db import get_db
        from database.models import TradingPlan, LLMConfig

        with get_db() as db:
            self.plan_info = db.query(TradingPlan).filter(TradingPlan.id == self.plan_id).first()
            if not self.plan_info:
                raise Exception(f"未找到计划 {self.plan_id}")

            self.llm_config = db.query(LLMConfig).filter(LLMConfig.id == self.plan_info.llm_config_id).first()
            if not self.llm_config:
                raise Exception(f"未找到LLM配置")

    async def test_tool(self, tool_name, test_message, expected_params=None):
        """测试单个工具"""
        print(f"\n🧪 测试工具: {tool_name}")
        print(f"📝 测试消息: {test_message}")

        if expected_params:
            print(f"🔧 期望参数: {expected_params}")

        print("-" * 50)

        messages = []
        tool_calls = 0
        tool_executed = False
        tool_success = False
        error_message = ""

        try:
            async for message_batch in langchain_agent_v2_service.stream_conversation(
                plan_id=self.plan_id,
                user_message=test_message
            ):
                for msg in message_batch:
                    content = msg.get("content", "")
                    role = msg.get("role", "assistant")
                    messages.append(content)

                    # 检测工具调用
                    if "🛠️ 调用工具" in content and tool_name in content:
                        tool_calls += 1
                        tool_executed = True
                        print(f"✅ 检测到工具调用")

                    # 检测执行结果
                    if f"✅ 工具执行结果" in content:
                        if "工具执行失败" not in content:
                            tool_success = True
                            print(f"✅ 工具执行成功")
                        else:
                            print(f"❌ 工具执行失败")
                            error_message += "执行失败; "

                    # 只显示前几条消息避免刷屏
                    if len(messages) <= 3:
                        print(f"📨 [{role}] {content[:150]}...")

                    # 限制测试长度
                    if len(messages) > 10:
                        break

        except Exception as e:
            error_message += f"测试异常: {str(e)}; "
            print(f"❌ 测试异常: {e}")

        # 记录结果
        self.results[tool_name] = {
            "tool_calls": tool_calls,
            "tool_executed": tool_executed,
            "tool_success": tool_success,
            "error_message": error_message.strip("; "),
            "messages_count": len(messages),
            "status": "success" if tool_success else "failed" if tool_executed else "no_call"
        }

        print(f"📊 结果: 调用={tool_calls}, 执行={tool_executed}, 成功={tool_success}")
        if error_message:
            print(f"❌ 错误: {error_message}")

async def main():
    """主测试函数"""
    print("🚀 重构后的Agent工具全面测试")
    print("=" * 60)

    tester = RefactoredToolTester()

    try:
        await tester.setup()

        print(f"📊 测试计划: {tester.plan_info.plan_name}")
        print(f"🔧 交易对: {tester.plan_info.inst_id}")
        print(f"🤖 LLM: {tester.llm_config.provider} - {tester.llm_config.model_name}")

        # 重构后的10个核心工具测试用例
        refactored_test_cases = [
            {
                "name": "query_prediction_data",
                "message": f"请查询计划 {tester.plan_id} 的最新预测数据，包含上涨概率和波动性概率",
                "params": {"plan_id": tester.plan_id, "limit": 10}
            },
            {
                "name": "get_prediction_history",
                "message": f"请查询计划 {tester.plan_id} 的历史预测批次，最多显示30个批次",
                "params": {"plan_id": tester.plan_id, "limit": 30}
            },
            {
                "name": "query_historical_kline_data",
                "message": f"请查询 {tester.plan_info.inst_id} 最近24小时的历史K线数据，使用UTC+8时间",
                "params": {"inst_id": tester.plan_info.inst_id, "limit": 50}
            },
            {
                "name": "get_current_utc_time",
                "message": "请查询当前UTC+8时间",
                "params": {}
            },
            {
                "name": "run_latest_model_inference",
                "message": f"请为计划 {tester.plan_id} 运行最新的模型推理",
                "params": {"plan_id": tester.plan_id}
            },
            {
                "name": "get_account_balance",
                "message": "请查询当前账户的USDT余额信息",
                "params": {"ccy": "USDT"}
            },
            {
                "name": "get_pending_orders",
                "message": f"请查询 {tester.plan_info.inst_id} 的未成交订单，状态为等待成交",
                "params": {"inst_id": tester.plan_info.inst_id, "state": "live", "limit": 300}
            },
            {
                "name": "place_order",
                "message": f"为 {tester.plan_info.inst_id} 下一个测试限价买单，数量0.001，价格1000（测试模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "side": "buy", "sz": "0.001", "px": "1000"}
            },
            {
                "name": "cancel_order",
                "message": "请取消一个测试订单（测试模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "cl_ord_id": "test_order_123"}
            },
            {
                "name": "amend_order",
                "message": "请修改一个测试订单，数量改为0.002，价格改为1100（测试模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "cl_ord_id": "test_order_456", "new_sz": "0.002", "new_px": "1100"}
            }
        ]

        # 执行测试
        for test_case in refactored_test_cases:
            await tester.test_tool(
                test_case["name"],
                test_case["message"],
                test_case["params"]
            )

            # 添加延迟避免API限制
            await asyncio.sleep(1)

        # 生成报告
        await generate_refactored_report(tester.results, tester.plan_info, tester.llm_config)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def generate_refactored_report(results, plan_info, llm_config):
    """生成重构后的测试报告"""
    print("\n" + "="*60)
    print("📊 生成重构后测试报告...")

    # 统计数据
    total_tools = len(results)
    successful_tools = sum(1 for r in results.values() if r["status"] == "success")
    failed_tools = sum(1 for r in results.values() if r["status"] == "failed")
    no_call_tools = sum(1 for r in results.values() if r["status"] == "no_call")

    # 生成Markdown报告
    report = f"""# LangChain Agent重构后工具测试报告

## 📊 测试概览

- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **计划名称**: {plan_info.plan_name}
- **交易对**: {plan_info.inst_id}
- **LLM配置**: {llm_config.provider} - {llm_config.model_name}
- **Agent实现**: 重构后的LangChain Agent，专注于10个核心工具

## 📈 测试统计

- **总工具数**: {total_tools}
- **✅ 成功工具**: {successful_tools}
- **❌ 失败工具**: {failed_tools}
- **⚠️ 未调用**: {no_call_tools}
- **成功率**: {successful_tools/total_tools*100:.1f}%

## 🛠️ 工具详细状态

"""

    # 工具详细状态
    for tool_name, result in results.items():
        status_emoji = {"success": "✅", "failed": "❌", "no_call": "⚠️"}
        report += f"### {status_emoji[result['status']]} {tool_name}\n\n"
        report += f"- **调用次数**: {result['tool_calls']}\n"
        report += f"- **是否执行**: {'是' if result['tool_executed'] else '否'}\n"
        report += f"- **执行状态**: {'成功' if result['tool_success'] else '失败'}\n"
        report += f"- **消息数量**: {result['messages_count']}\n"

        if result['error_message']:
            report += f"- **错误信息**: `{result['error_message']}`\n"

        report += "\n"

    # 问题总结
    report += "## 🔧 重构成果与问题分析\n\n"

    if failed_tools > 0:
        report += f"### ❌ 失败工具 ({failed_tools}个)\n\n"
        for tool_name, result in results.items():
            if result['status'] == 'failed':
                report += f"**{tool_name}**: {result['error_message'] or '执行失败'}\n\n"

    if no_call_tools > 0:
        report += f"### ⚠️ 未调用工具 ({no_call_tools}个)\n\n"
        for tool_name, result in results.items():
            if result['status'] == 'no_call':
                report += f"**{tool_name}**: Agent未调用此工具，可能需要改进提示词\n\n"

    # 重构成果总结
    report += "## 🎯 重构成果总结\n\n"
    report += "### ✅ 成功改进\n"
    report += "1. **工具精简**: 从原来的13个工具精简为10个核心工具\n"
    report += "2. **参数规范**: 统一参数名称，符合OKX API规范\n"
    report += "3. **功能增强**: 增加上涨概率和波动性概率查询\n"
    report += "4. **时间统一**: 所有时间查询统一使用UTC+8时区\n"
    report += "5. **API优化**: 改进OKX API参数传递和错误处理\n\n"

    report += "### 🔧 技术改进\n"
    report += "1. **数据库查询**: 直接查询prediction_data表，使用inference_batch_id字段\n"
    report += "2. **工具绑定**: 使用真正的LangChain bind_tools方法\n"
    report += "3. **消息流格式**: 支持标准的role:system/user/assistant/tool序列\n"
    report += "4. **参数验证**: 增强工具参数验证和错误处理\n\n"

    # 下一步优化建议
    report += "## 📋 下一步优化建议\n\n"
    report += "1. **API配置**: 检查并配置正确的OKX API密钥\n"
    report += "2. **参数提示**: 改进Agent提示词，明确参数要求\n"
    report += "3. **错误处理**: 增强工具执行的错误恢复机制\n"
    report += "4. **性能优化**: 优化数据库查询性能\n"

    # 保存报告
    os.makedirs("docs", exist_ok=True)

    with open("docs/langchain_refactored_tools_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 报告已保存到: docs/langchain_refactored_tools_test_report.md")

    # 控制台输出总结
    print(f"\n📊 重构测试总结:")
    print(f"   成功工具: {successful_tools}/{total_tools}")
    print(f"   失败工具: {failed_tools}/{total_tools}")
    print(f"   未调用: {no_call_tools}/{total_tools}")
    print(f"   成功率: {successful_tools/total_tools*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())