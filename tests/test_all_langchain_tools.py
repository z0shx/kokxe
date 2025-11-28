#!/usr/bin/env python3
"""
测试所有LangChain工具的真实调用情况
基于真正的LangChain Agent + bind_tools实现
"""

import asyncio
import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_agent_v2 import langchain_agent_v2_service

class ToolTester:
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
                    messages.append(content)

                    # 检测工具调用
                    if "结构化工具调用" in content and tool_name in content:
                        tool_calls += 1
                        tool_executed = True
                        print(f"✅ 检测到工具调用")

                        # 提取参数
                        if "参数:" in content:
                            import re
                            param_match = re.search(r'参数：`([^`]+)`', content)
                            if param_match:
                                params_str = param_match.group(1)
                                try:
                                    params = json.loads(params_str)
                                    print(f"🔧 实际参数: {params}")

                                    # 验证参数
                                    if expected_params:
                                        for key in expected_params:
                                            if key not in params:
                                                error_message += f"缺少参数: {key}; "
                                except:
                                    print(f"⚠️ 参数解析失败: {params_str}")

                    # 检测执行结果
                    if f"{tool_name} 执行完成" in content:
                        if "工具执行失败" not in content:
                            tool_success = True
                            print(f"✅ 工具执行成功")
                        else:
                            print(f"❌ 工具执行失败")
                            if "工具执行失败:" in content:
                                import re
                                error_match = re.search(r'工具执行失败：([^`]+)', content)
                                if error_match:
                                    error_message += f"执行错误: {error_match.group(1)}; "

                    # 只显示前几条消息避免刷屏
                    if len(messages) <= 3:
                        print(f"📨 [{len(messages)}] {content[:150]}...")

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
    print("🚀 LangChain工具全面测试")
    print("=" * 60)

    tester = ToolTester()

    try:
        await tester.setup()

        print(f"📊 测试计划: {tester.plan_info.plan_name}")
        print(f"🔧 交易对: {tester.plan_info.inst_id}")
        print(f"🤖 LLM: {tester.llm_config.provider} - {tester.llm_config.model_name}")

        # 定义工具测试用例
        test_cases = [
            {
                "name": "get_account_balance",
                "message": "请查询当前账户余额信息",
                "params": {}
            },
            {
                "name": "get_positions",
                "message": f"请查询 {tester.plan_info.inst_id} 的持仓信息",
                "params": {"inst_id": tester.plan_info.inst_id}
            },
            {
                "name": "get_pending_orders",
                "message": f"请查询 {tester.plan_info.inst_id} 的未成交订单",
                "params": {"inst_id": tester.plan_info.inst_id}
            },
            {
                "name": "query_prediction_data",
                "message": f"请查询计划 {tester.plan_id} 的最新预测数据",
                "params": {"plan_id": tester.plan_id, "limit": 10}
            },
            {
                "name": "get_prediction_history",
                "message": f"请查询计划 {tester.plan_id} 的预测历史",
                "params": {"plan_id": tester.plan_id}
            },
            {
                "name": "get_current_utc_time",
                "message": "请查询当前UTC时间",
                "params": {}
            },
            {
                "name": "query_historical_kline_data",
                "message": f"请查询 {tester.plan_info.inst_id} 最近24小时的历史K线数据",
                "params": {"inst_id": tester.plan_info.inst_id, "limit": 50}
            },
            {
                "name": "run_latest_model_inference",
                "message": f"请为计划 {tester.plan_id} 运行最新的模型推理",
                "params": {"plan_id": tester.plan_id}
            },
            # 交易相关工具（可能需要有效的API密钥）
            {
                "name": "place_order",
                "message": f"为 {tester.plan_info.inst_id} 下一个测试限价买单（演示模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "side": "buy", "order_type": "limit", "size": 0.001, "price": 1000}
            },
            {
                "name": "cancel_order",
                "message": "请取消一个测试订单（演示模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "order_id": "test_order_id"}
            },
            {
                "name": "modify_order",
                "message": "请修改一个测试订单（演示模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "order_id": "test_order_id", "size": 0.002, "price": 1100}
            },
            {
                "name": "place_stop_loss_order",
                "message": f"为 {tester.plan_info.inst_id} 设置止损订单（演示模式）",
                "params": {"inst_id": tester.plan_info.inst_id, "size": 0.001, "stop_price": 900}
            },
            {
                "name": "delete_prediction_data_by_batch",
                "message": "删除一个测试预测数据批次（演示模式，请勿实际执行）",
                "params": {"batch_id": "test_batch_id"}
            }
        ]

        # 执行测试
        for test_case in test_cases:
            await tester.test_tool(
                test_case["name"],
                test_case["message"],
                test_case["params"]
            )

            # 添加延迟避免API限制
            await asyncio.sleep(1)

        # 生成报告
        await generate_report(tester.results, tester.plan_info, tester.llm_config)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def generate_report(results, plan_info, llm_config):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📊 生成测试报告...")

    # 统计数据
    total_tools = len(results)
    successful_tools = sum(1 for r in results.values() if r["status"] == "success")
    failed_tools = sum(1 for r in results.values() if r["status"] == "failed")
    no_call_tools = sum(1 for r in results.values() if r["status"] == "no_call")

    # 生成Markdown报告
    report = f"""# LangChain Agent工具测试报告

## 📊 测试概览

- **测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **计划名称**: {plan_info.plan_name}
- **交易对**: {plan_info.inst_id}
- **LLM配置**: {llm_config.provider} - {llm_config.model_name}
- **Agent实现**: 改进的bind_tools版本（真正的LangChain工具调用）

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
    report += "## 🔧 问题与建议\n\n"

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

    # 修复建议
    report += "## 🛠️ 修复建议\n\n"
    report += "1. **API密钥配置**: 检查OKX API密钥配置是否正确\n"
    report += "2. **工具方法实现**: 确保所有工具方法在OKXTradingTools中正确实现\n"
    report += "3. **参数验证**: 改进工具参数验证和错误处理\n"
    report += "4. **提示词优化**: 优化Agent提示词以提高工具调用准确性\n"
    report += "5. **权限管理**: 确保API账户具有执行相关操作的权限\n"

    # 保存报告
    os.makedirs("docs", exist_ok=True)

    with open("docs/langchain_agent_tools_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 报告已保存到: docs/langchain_agent_tools_test_report.md")

    # 控制台输出总结
    print(f"\n📊 测试总结:")
    print(f"   成功工具: {successful_tools}/{total_tools}")
    print(f"   失败工具: {failed_tools}/{total_tools}")
    print(f"   未调用: {no_call_tools}/{total_tools}")
    print(f"   成功率: {successful_tools/total_tools*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())