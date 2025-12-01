#!/usr/bin/env python3
"""
交易流程完整测试脚本
测试完整的订单生命周期：下单 -> 查询 -> 修改 -> 取消
"""
import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agent_tool_executor import AgentToolExecutor
from database.db import get_db
from database.models import TradingPlan, KlineData
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingWorkflowTester:
    """交易流程测试器"""

    def __init__(self):
        self.test_results: List[Dict] = []
        self.placed_orders: List[str] = []

        # Initialize tool executor with demo settings
        config = Config()
        self.tool_executor = AgentToolExecutor(
            api_key=config.OKX_API_KEY,
            secret_key=config.OKX_SECRET_KEY,
            passphrase=config.OKX_PASSPHRASE,
            is_demo=True,  # Always use demo for testing
            plan_id=1,     # Provide a plan_id for database operations
            conversation_id=1
        )

    def add_result(self, step: str, success: bool, result: Any = None, error: str = None):
        """添加测试结果"""
        test_result = {
            'step': step,
            'success': success,
            'result': result,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(test_result)

        if success:
            logger.info(f"✅ {step}: SUCCESS")
        else:
            logger.error(f"❌ {step}: FAILED - {error}")

    async def step_1_place_orders(self):
        """步骤1: 下单测试"""
        logger.info("🔄 步骤1: 测试下单功能...")

        test_instrument = "ETH-USDT"
        # 使用更合理的测试价格（接近市场但不会立即成交）
        buy_price = "2000.0"    # 低于当前市价的限价买单
        sell_price = "4000.0"   # 高于当前市价的限价卖单
        test_size = "0.001"     # 小数量测试

        # 测试买单
        try:
            logger.info(f"测试买单: {test_instrument} @ ${buy_price}")
            result = await self.tool_executor.execute_tool("place_order", {
                "inst_id": test_instrument,
                "side": "buy",
                "order_type": "limit",
                "size": test_size,
                "price": buy_price,
                "tag": "workflow_test_buy"
            })

            # 检查是否真的成功下单
            if isinstance(result, dict):
                if result.get('success') and 'order_id' in result:
                    self.placed_orders.append(result['order_id'])
                    self.add_result("place_order_buy", True, result)
                    logger.info(f"✅ 买单成功，订单ID: {result['order_id']}")
                elif result.get('success') and result.get('data'):
                    # 有些API返回格式不同
                    order_data = result['data']
                    if isinstance(order_data, list) and len(order_data) > 0:
                        order_id = order_data[0].get('ordId')
                        if order_id:
                            self.placed_orders.append(order_id)
                            self.add_result("place_order_buy", True, result)
                            logger.info(f"✅ 买单成功，订单ID: {order_id}")
                    else:
                        self.add_result("place_order_buy", False, result, "无法解析订单ID")
                else:
                    error_msg = result.get('error', '未知错误')
                    self.add_result("place_order_buy", False, result, f"下单失败: {error_msg}")
            else:
                self.add_result("place_order_buy", False, result, "返回格式异常")

        except Exception as e:
            self.add_result("place_order_buy", False, None, str(e))

        # 测试卖单
        try:
            logger.info(f"测试卖单: {test_instrument} @ ${sell_price}")
            result = await self.tool_executor.execute_tool("place_order", {
                "inst_id": test_instrument,
                "side": "sell",
                "order_type": "limit",
                "size": test_size,
                "price": sell_price,
                "tag": "workflow_test_sell"
            })

            # 检查是否真的成功下单
            if isinstance(result, dict):
                if result.get('success') and 'order_id' in result:
                    self.placed_orders.append(result['order_id'])
                    self.add_result("place_order_sell", True, result)
                    logger.info(f"✅ 卖单成功，订单ID: {result['order_id']}")
                elif result.get('success') and result.get('data'):
                    # 有些API返回格式不同
                    order_data = result['data']
                    if isinstance(order_data, list) and len(order_data) > 0:
                        order_id = order_data[0].get('ordId')
                        if order_id:
                            self.placed_orders.append(order_id)
                            self.add_result("place_order_sell", True, result)
                            logger.info(f"✅ 卖单成功，订单ID: {order_id}")
                    else:
                        self.add_result("place_order_sell", False, result, "无法解析订单ID")
                else:
                    error_msg = result.get('error', '未知错误')
                    self.add_result("place_order_sell", False, result, f"下单失败: {error_msg}")
            else:
                self.add_result("place_order_sell", False, result, "返回格式异常")

        except Exception as e:
            self.add_result("place_order_sell", False, None, str(e))

        logger.info(f"步骤1完成，共成功下单 {len(self.placed_orders)} 个订单")

    async def step_2_get_pending_orders(self):
        """步骤2: 查询未成交订单"""
        logger.info("🔄 步骤2: 测试查询未成交订单...")

        try:
            result = await self.tool_executor.execute_tool("get_pending_orders", {})

            if isinstance(result, dict):
                if result.get('success') and 'orders' in result:
                    orders = result['orders']
                    found_orders = []

                    # 检查是否能找到我们刚下的订单
                    for order in orders:
                        order_id = order.get('order_id') or order.get('ordId')
                        if order_id in self.placed_orders:
                            found_orders.append(order)

                    self.add_result("get_pending_orders", True, {
                        'total_orders': len(orders),
                        'found_orders': len(found_orders),
                        'our_orders': found_orders
                    })

                    logger.info(f"✅ 查询成功，共 {len(orders)} 个挂单，找到 {len(found_orders)} 个我们的订单")

                    # 如果找到了我们的订单，保存详细信息用于后续测试
                    if found_orders:
                        self.our_order_details = found_orders

                elif result.get('success'):
                    # 可能API返回成功但没有订单
                    self.add_result("get_pending_orders", True, {
                        'message': '查询成功但没有找到订单',
                        'result': result
                    })
                else:
                    error_msg = result.get('error', '未知错误')
                    self.add_result("get_pending_orders", False, result, f"查询失败: {error_msg}")
            else:
                self.add_result("get_pending_orders", False, result, "返回格式异常")

        except Exception as e:
            self.add_result("get_pending_orders", False, None, str(e))

    async def step_3_amend_order(self):
        """步骤3: 修改订单"""
        logger.info("🔄 步骤3: 测试修改订单...")

        if not self.placed_orders:
            self.add_result("amend_order", False, None, "没有可修改的订单")
            return

        order_to_amend = self.placed_orders[0]  # 修改第一个订单
        new_price = "2100.0"  # 修改后的价格

        try:
            logger.info(f"修改订单 {order_to_amend} 价格到 ${new_price}")
            result = await self.tool_executor.execute_tool("amend_order", {
                "inst_id": "ETH-USDT",
                "order_id": order_to_amend,
                "new_price": new_price
            })

            if isinstance(result, dict):
                if result.get('success'):
                    self.add_result("amend_order", True, result)
                    logger.info(f"✅ 订单修改成功")
                else:
                    error_msg = result.get('error', '未知错误')
                    self.add_result("amend_order", False, result, f"修改失败: {error_msg}")
            else:
                self.add_result("amend_order", False, result, "返回格式异常")

        except Exception as e:
            self.add_result("amend_order", False, None, str(e))

    async def step_4_cancel_orders(self):
        """步骤4: 取消所有订单"""
        logger.info("🔄 步骤4: 测试取消订单...")

        if not self.placed_orders:
            self.add_result("cancel_orders", False, None, "没有可取消的订单")
            return

        cancelled_count = 0

        for order_id in self.placed_orders:
            try:
                logger.info(f"取消订单 {order_id}")
                result = await self.tool_executor.execute_tool("cancel_order", {
                    "inst_id": "ETH-USDT",
                    "order_id": order_id
                })

                if isinstance(result, dict):
                    if result.get('success'):
                        cancelled_count += 1
                        logger.info(f"✅ 订单 {order_id} 取消成功")
                    else:
                        error_msg = result.get('error', '未知错误')
                        logger.warning(f"⚠️ 订单 {order_id} 取消失败: {error_msg}")
                else:
                    logger.warning(f"⚠️ 订单 {order_id} 取消响应格式异常")

            except Exception as e:
                logger.error(f"❌ 取消订单 {order_id} 异常: {e}")

        if cancelled_count > 0:
            self.add_result("cancel_orders", True, {
                'total_orders': len(self.placed_orders),
                'cancelled_orders': cancelled_count
            })
            logger.info(f"✅ 步骤4完成，成功取消 {cancelled_count} 个订单")
        else:
            self.add_result("cancel_orders", False, None, "没有成功取消任何订单")

    async def verify_final_state(self):
        """验证最终状态：确保没有未成交订单"""
        logger.info("🔄 验证最终状态...")

        try:
            result = await self.tool_executor.execute_tool("get_pending_orders", {})

            if isinstance(result, dict) and result.get('success'):
                orders = result.get('orders', [])
                our_orders_remaining = []

                for order in orders:
                    order_id = order.get('order_id') or order.get('ordId')
                    if order_id in self.placed_orders:
                        our_orders_remaining.append(order_id)

                if our_orders_remaining:
                    self.add_result("final_verification", False, {
                        'remaining_orders': our_orders_remaining
                    }, f"仍有 {len(our_orders_remaining)} 个订单未取消")
                else:
                    self.add_result("final_verification", True, {
                        'message': '所有订单已正确处理'
                    })
                    logger.info("✅ 验证通过：所有订单已正确处理")

        except Exception as e:
            self.add_result("final_verification", False, None, f"验证异常: {e}")

    async def run_complete_workflow(self):
        """运行完整的交易流程测试"""
        logger.info("🚀 开始完整交易流程测试...")
        logger.info("=" * 60)

        # 按顺序执行交易流程
        await self.step_1_place_orders()
        await asyncio.sleep(1)  # 等待订单处理

        await self.step_2_get_pending_orders()
        await asyncio.sleep(1)  # 等待查询结果

        await self.step_3_amend_order()
        await asyncio.sleep(1)  # 等待修改处理

        await self.step_4_cancel_orders()
        await asyncio.sleep(2)  # 等待取消处理完成

        await self.verify_final_state()

        logger.info("=" * 60)
        logger.info("🏁 交易流程测试完成！")

        # 统计结果
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        failed = total_tests - passed

        logger.info(f"总测试: {total_tests}, 通过: {passed}, 失败: {failed}")

        return {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'success_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
            'detailed_results': self.test_results
        }

    def save_workflow_report(self, results: Dict):
        """保存工作流测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_data = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Trading Workflow Testing',
                'description': 'Complete order lifecycle testing: place -> query -> amend -> cancel'
            },
            'workflow_summary': {
                'total_tests': results['total_tests'],
                'passed': results['passed'],
                'failed': results['failed'],
                'success_rate': results['success_rate']
            },
            'placed_orders': self.placed_orders,
            'detailed_results': results['detailed_results']
        }

        # 保存JSON报告
        json_file = f"docs/trading_workflow_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 保存Markdown报告
        md_file = f"docs/trading_workflow_report_{timestamp}.md"
        self._generate_workflow_markdown(report_data, md_file)

        logger.info(f"📄 工作流测试报告已保存:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Markdown: {md_file}")

        return json_file, md_file

    def _generate_workflow_markdown(self, report_data: Dict, filename: str):
        """生成Markdown格式的工作流报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# KOKEX 交易流程测试报告\n\n")
            f.write(f"**生成时间:** {report_data['test_metadata']['timestamp']}\n\n")
            f.write(f"**测试类型:** {report_data['test_metadata']['test_type']}\n\n")
            f.write(f"**描述:** {report_data['test_metadata']['description']}\n\n")

            # 工作流摘要
            summary = report_data['workflow_summary']
            f.write("## 工作流测试摘要\n\n")
            f.write(f"- **总测试步骤:** {summary['total_tests']}\n")
            f.write(f"- **成功步骤:** {summary['passed']} ✅\n")
            f.write(f"- **失败步骤:** {summary['failed']} ❌\n")
            f.write(f"- **成功率:** {summary['success_rate']:.1f}%\n\n")

            # 下单信息
            f.write("## 下单信息\n\n")
            f.write(f"- **成功下单数量:** {len(report_data['placed_orders'])}\n")
            if report_data['placed_orders']:
                f.write("- **订单ID列表:**\n")
                for order_id in report_data['placed_orders']:
                    f.write(f"  - `{order_id}`\n")
            f.write("\n")

            # 详细测试步骤
            f.write("## 详细测试步骤\n\n")
            step_num = 1
            for result in report_data['detailed_results']:
                status_icon = "✅" if result['success'] else "❌"
                f.write(f"### {status_icon} 步骤 {step_num}: {result['step']}\n\n")
                f.write(f"**时间:** {result['timestamp']}\n\n")
                f.write(f"**结果:** {result['success']}\n\n")

                if result['success'] and result['result']:
                    f.write("**详细信息:**\n```json\n")
                    f.write(json.dumps(result['result'], indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                elif not result['success']:
                    f.write(f"**错误信息:** {result['error']}\n\n")

                step_num += 1

            f.write("---\n")
            f.write("*报告由 KOKEX 交易流程测试套件生成*\n")

    async def cleanup(self):
        """清理资源"""
        try:
            await self.tool_executor.close()
        except Exception as e:
            logger.warning(f"清理资源时出现异常: {e}")


async def main():
    """主测试执行"""
    tester = TradingWorkflowTester()

    try:
        # 运行完整工作流测试
        results = await tester.run_complete_workflow()

        # 保存测试报告
        json_file, md_file = tester.save_workflow_report(results)

        # 清理资源
        await tester.cleanup()

        # 返回结果
        return {
            'success': results['failed'] == 0,
            'summary': results,
            'reports': {
                'json': json_file,
                'markdown': md_file
            }
        }

    except Exception as e:
        logger.error(f"工作流测试执行失败: {e}")
        try:
            await tester.cleanup()
        except:
            pass
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 运行交易流程测试
    result = asyncio.run(main())

    if result['success']:
        print("\n🎉 交易流程测试完成！")
        print(f"📊 报告已保存到: {result['reports']['json']} 和 {result['reports']['markdown']}")
    else:
        print(f"\n❌ 测试失败: {result.get('error', '未知错误')}")
        sys.exit(1)