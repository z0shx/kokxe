#!/usr/bin/env python3
"""
模拟工具测试脚本
测试工具的基本功能，不依赖实际的API调用
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

from services.agent_tools import AGENT_TOOLS, get_tool, validate_tool_params
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockToolTester:
    """模拟工具测试器"""

    def __init__(self):
        self.test_results: List[Dict] = []
        self.test_summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': []
        }

    def add_result(self, test_name: str, success: bool, result: Any = None, error: str = None):
        """添加测试结果"""
        test_result = {
            'test_name': test_name,
            'success': success,
            'result': result,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(test_result)
        self.test_summary['total_tests'] += 1

        if success:
            self.test_summary['passed'] += 1
            logger.info(f"✅ {test_name}: SUCCESS")
        else:
            self.test_summary['failed'] += 1
            self.test_summary['errors'].append(f"{test_name}: {error}")
            logger.error(f"❌ {test_name}: FAILED - {error}")

    def test_tool_availability(self):
        """测试工具可用性"""
        logger.info("🔧 测试工具可用性...")

        required_tools = [
            "get_prediction_history",
            "query_historical_kline_data",
            "run_latest_model_inference",
            "place_order",
            "get_pending_orders",
            "amend_order",
            "cancel_order"
        ]

        for tool_name in required_tools:
            tool = get_tool(tool_name)
            if tool:
                self.add_result(f"tool_availability_{tool_name}", True, {
                    'name': tool.name,
                    'description': tool.description[:100] + "...",
                    'category': tool.category,
                    'risk_level': tool.risk_level
                })
            else:
                self.add_result(f"tool_availability_{tool_name}", False, None, f"工具 {tool_name} 不存在")

    def test_parameter_validation(self):
        """测试参数验证"""
        logger.info("🔍 测试参数验证...")

        # 测试 place_order 参数验证
        test_cases = [
            {
                'tool': 'place_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'side': 'buy',
                    'size': '0.001',
                    'order_type': 'limit',
                    'price': '2000.0'
                },
                'should_pass': True,
                'description': '有效的买单参数'
            },
            {
                'tool': 'place_order',
                'params': {
                    'side': 'buy'
                },
                'should_pass': False,
                'description': '缺少必需参数'
            },
            {
                'tool': 'place_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'side': 'invalid_side',
                    'size': '0.001',
                    'order_type': 'limit',
                    'price': '2000.0'
                },
                'should_pass': False,
                'description': '无效的枚举值'
            },
            {
                'tool': 'get_pending_orders',
                'params': {},
                'should_pass': True,
                'description': '有效的查询参数'
            },
            {
                'tool': 'cancel_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'order_id': 'test_order_id'
                },
                'should_pass': True,
                'description': '有效的取消订单参数'
            },
            {
                'tool': 'amend_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'order_id': 'test_order_id',
                    'new_price': '2100.0'
                },
                'should_pass': True,
                'description': '有效的修改订单参数'
            }
        ]

        for i, case in enumerate(test_cases):
            tool_name = case['tool']
            params = case['params']
            should_pass = case['should_pass']
            description = case['description']

            is_valid, error_msg = validate_tool_params(tool_name, params)

            if should_pass:
                if is_valid:
                    self.add_result(f"param_validation_{i+1}", True, {
                        'tool': tool_name,
                        'params': params,
                        'description': description
                    })
                else:
                    self.add_result(f"param_validation_{i+1}", False, None,
                                  f"应该通过但失败: {description}, 错误: {error_msg}")
            else:
                if not is_valid:
                    self.add_result(f"param_validation_{i+1}", True, {
                        'tool': tool_name,
                        'params': params,
                        'description': description,
                        'expected_error': error_msg
                    })
                else:
                    self.add_result(f"param_validation_{i+1}", False, None,
                                  f"应该失败但通过: {description}")

    def test_tool_definitions(self):
        """测试工具定义完整性"""
        logger.info("📋 测试工具定义完整性...")

        for tool_name, tool in AGENT_TOOLS.items():
            issues = []

            # 检查基本属性
            if not tool.name:
                issues.append("缺少名称")
            if not tool.description:
                issues.append("缺少描述")
            if not tool.category:
                issues.append("缺少分类")
            # 注意：某些工具（如 get_current_utc_time）确实不需要参数，所以空参数定义是有效的
            if tool.parameters is None:
                issues.append("缺少参数定义")

            # 检查参数定义
            if tool.parameters:
                for param_name, param_def in tool.parameters.items():
                    if 'type' not in param_def:
                        issues.append(f"参数 {param_name} 缺少类型定义")
                    if 'description' not in param_def:
                        issues.append(f"参数 {param_name} 缺少描述")

            # 检查必需参数
            required_in_params = tool.required_params
            all_params = set(tool.parameters.keys())
            missing_in_params = set(required_in_params) - all_params
            if missing_in_params:
                issues.append(f"必需参数 {missing_in_params} 不在参数定义中")

            if issues:
                self.add_result(f"tool_definition_{tool_name}", False, None,
                              f"定义问题: {', '.join(issues)}")
            else:
                self.add_result(f"tool_definition_{tool_name}", True, {
                    'name': tool.name,
                    'category': tool.category,
                    'param_count': len(tool.parameters),
                    'required_params': len(tool.required_params)
                })

    def test_trading_workflow_logic(self):
        """测试交易工作流逻辑"""
        logger.info("🔄 测试交易工作流逻辑...")

        # 模拟完整的交易流程参数
        workflow_steps = [
            {
                'step': '1_place_order_buy',
                'tool': 'place_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'side': 'buy',
                    'order_type': 'limit',
                    'size': '0.001',
                    'price': '2000.0',
                    'tag': 'test_buy'
                },
                'description': '下买单'
            },
            {
                'step': '2_place_order_sell',
                'tool': 'place_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'side': 'sell',
                    'order_type': 'limit',
                    'size': '0.001',
                    'price': '4000.0',
                    'tag': 'test_sell'
                },
                'description': '下卖单'
            },
            {
                'step': '3_get_pending_orders',
                'tool': 'get_pending_orders',
                'params': {},
                'description': '查询挂单'
            },
            {
                'step': '4_amend_order',
                'tool': 'amend_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'order_id': 'mock_order_123',
                    'new_price': '2100.0'
                },
                'description': '修改订单'
            },
            {
                'step': '5_cancel_order',
                'tool': 'cancel_order',
                'params': {
                    'inst_id': 'ETH-USDT',
                    'order_id': 'mock_order_123'
                },
                'description': '取消订单'
            }
        ]

        for workflow_step in workflow_steps:
            tool_name = workflow_step['tool']
            params = workflow_step['params']
            step_name = workflow_step['step']
            description = workflow_step['description']

            # 验证工具存在
            tool = get_tool(tool_name)
            if not tool:
                self.add_result(f"workflow_{step_name}", False, None, f"工具 {tool_name} 不存在")
                continue

            # 验证参数
            is_valid, error_msg = validate_tool_params(tool_name, params)
            if is_valid:
                self.add_result(f"workflow_{step_name}", True, {
                    'tool': tool_name,
                    'description': description,
                    'params': params,
                    'risk_level': tool.risk_level
                })
            else:
                self.add_result(f"workflow_{step_name}", False, None,
                              f"参数验证失败: {description}, 错误: {error_msg}")

    def test_tool_categories(self):
        """测试工具分类"""
        logger.info("🏷️ 测试工具分类...")

        from services.agent_tools import ToolCategory, get_tools_by_category

        categories = [ToolCategory.QUERY, ToolCategory.TRADE, ToolCategory.MONITOR]

        for category in categories:
            tools = get_tools_by_category(category)
            if tools:
                self.add_result(f"category_{category.value}", True, {
                    'category': category.value,
                    'tool_count': len(tools),
                    'tools': [tool.name for tool in tools]
                })
            else:
                self.add_result(f"category_{category.value}", False, None,
                              f"分类 {category.value} 没有工具")

    def run_all_tests(self):
        """运行所有模拟测试"""
        logger.info("🚀 开始模拟工具测试...")
        logger.info("=" * 60)

        # 运行所有测试
        self.test_tool_availability()
        self.test_parameter_validation()
        self.test_tool_definitions()
        self.test_trading_workflow_logic()
        self.test_tool_categories()

        logger.info("=" * 60)
        logger.info("🏁 模拟测试完成！")

        # 统计结果
        logger.info(f"总测试: {self.test_summary['total_tests']}, "
                   f"通过: {self.test_summary['passed']}, "
                   f"失败: {self.test_summary['failed']}")

        if self.test_summary['failed'] > 0:
            logger.error("失败的测试:")
            for error in self.test_summary['errors']:
                logger.error(f"  - {error}")

        return self.test_summary

    def save_mock_test_report(self, results: Dict):
        """保存模拟测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_data = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Mock Tool Testing',
                'description': 'Tool availability, validation, and workflow logic testing without API calls'
            },
            'test_summary': results,
            'detailed_results': self.test_results
        }

        # 保存JSON报告
        json_file = f"docs/mock_tool_test_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 保存Markdown报告
        md_file = f"docs/mock_tool_test_report_{timestamp}.md"
        self._generate_mock_markdown_report(report_data, md_file)

        logger.info(f"📄 模拟测试报告已保存:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Markdown: {md_file}")

        return json_file, md_file

    def _generate_mock_markdown_report(self, report_data: Dict, filename: str):
        """生成Markdown格式的模拟测试报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# KOKEX 模拟工具测试报告\n\n")
            f.write(f"**生成时间:** {report_data['test_metadata']['timestamp']}\n\n")
            f.write(f"**测试类型:** {report_data['test_metadata']['test_type']}\n\n")
            f.write(f"**描述:** {report_data['test_metadata']['description']}\n\n")

            # 测试摘要
            summary = report_data['test_summary']
            f.write("## 测试摘要\n\n")
            f.write(f"- **总测试数:** {summary['total_tests']}\n")
            f.write(f"- **通过测试:** {summary['passed']} ✅\n")
            f.write(f"- **失败测试:** {summary['failed']} ❌\n")
            f.write(f"- **成功率:** {(summary['passed'] / max(summary['total_tests'], 1)) * 100:.1f}%\n\n")

            # 详细结果
            f.write("## 详细测试结果\n\n")
            for result in report_data['detailed_results']:
                status_icon = "✅" if result['success'] else "❌"
                f.write(f"### {status_icon} {result['test_name']}\n\n")
                f.write(f"**时间:** {result['timestamp']}\n\n")
                f.write(f"**结果:** {result['success']}\n\n")

                if result['success'] and result['result']:
                    f.write("**详细信息:**\n```json\n")
                    f.write(json.dumps(result['result'], indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                elif not result['success']:
                    f.write(f"**错误信息:** {result['error']}\n\n")

            # 错误总结
            if summary['errors']:
                f.write("## 错误总结\n\n")
                for error in summary['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")

            f.write("---\n")
            f.write("*报告由 KOKEX 模拟工具测试套件生成*\n")


async def main():
    """主测试执行"""
    tester = MockToolTester()

    try:
        # 运行所有模拟测试
        results = tester.run_all_tests()

        # 保存测试报告
        json_file, md_file = tester.save_mock_test_report(results)

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
        logger.error(f"模拟测试执行失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # 运行模拟工具测试
    result = asyncio.run(main())

    if result['success']:
        print("\n🎉 模拟工具测试完成！")
        print(f"📊 报告已保存到: {result['reports']['json']} 和 {result['reports']['markdown']}")
    else:
        print(f"\n❌ 测试失败: {result.get('error', '未知错误')}")
        sys.exit(1)