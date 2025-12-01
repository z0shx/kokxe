#!/usr/bin/env python3
"""
Comprehensive test script for all available tools in KOKEX.
Tests database, inference, and API tools by directly invoking them.
Results are saved to docs/ directory.
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

class ToolTestResult:
    """Container for tool test results"""
    def __init__(self, tool_name: str, success: bool, result: Any = None, error: str = None):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.timestamp = datetime.now()

class ToolTester:
    """Comprehensive tool testing suite"""

    def __init__(self):
        self.test_results: List[ToolTestResult] = []
        self.test_summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': []
        }

        # Initialize tool executor with demo settings
        config = Config()
        self.tool_executor = AgentToolExecutor(
            api_key=config.OKX_API_KEY,
            secret_key=config.OKX_SECRET_KEY,
            passphrase=config.OKX_PASSPHRASE,
            is_demo=True,  # Always use demo for testing
            plan_id=None,
            conversation_id=None
        )

    def add_result(self, tool_name: str, success: bool, result: Any = None, error: str = None):
        """Add a test result"""
        test_result = ToolTestResult(tool_name, success, result, error)
        self.test_results.append(test_result)
        self.test_summary['total_tests'] += 1

        if success:
            self.test_summary['passed'] += 1
            logger.info(f"✅ {tool_name}: SUCCESS")
        else:
            self.test_summary['failed'] += 1
            self.test_summary['errors'].append(f"{tool_name}: {error}")
            logger.error(f"❌ {tool_name}: FAILED - {error}")

    async def test_database_tools(self):
        """Test all database tools"""
        logger.info("🗄️ Testing Database Tools...")

        # Test 1: get_prediction_history
        try:
            logger.info("Testing get_prediction_history...")
            result = await self.tool_executor.execute_tool("get_prediction_history", {"limit": 10})
            self.add_result("get_prediction_history", True, result)

            # Print prediction history summary
            if result and 'batch_count' in result:
                logger.info(f"Found {result['batch_count']} inference batches")
                if result.get('batches'):
                    latest_batch = result['batches'][0]
                    logger.info(f"Latest batch: {latest_batch.get('batch_id')} - {latest_batch.get('status')}")

        except Exception as e:
            self.add_result("get_prediction_history", False, error=str(e))

        # Test 2: query_historical_kline_data with time range
        try:
            logger.info("Testing query_historical_kline_data with time range...")
            result = await self.tool_executor.execute_tool("query_historical_kline_data", {
                "inst_id": "ETH-USDT",
                "interval": "1H",
                "start_time": "2024-01-01 00:00:00",
                "end_time": "2024-01-02 00:00:00"
            })
            self.add_result("query_historical_kline_data (time range)", True, result)

            # Print data summary
            if result and 'data' in result:
                data_count = len(result['data'])
                logger.info(f"Retrieved {data_count} K-line records")

        except Exception as e:
            self.add_result("query_historical_kline_data (time range)", False, error=str(e))

        # Test 3: query_historical_kline_data with batch ID
        try:
            # First get a batch ID from prediction history
            batch_id = None
            for test_result in self.test_results:
                if test_result.tool_name == "get_prediction_history" and test_result.success:
                    if test_result.result and test_result.result.get('batches'):
                        batch_id = test_result.result['batches'][0].get('batch_id')
                        break

            if batch_id:
                logger.info(f"Testing query_prediction_data with batch_id: {batch_id}")
                result = await self.tool_executor.execute_tool("query_prediction_data", {"inference_batch_id": batch_id})
                self.add_result("query_prediction_data (batch_id)", True, result)
            else:
                logger.warning("No batch ID available for testing query_prediction_data with batch_id")
                self.add_result("query_prediction_data (batch_id)", False,
                              error="No batch ID available for testing")

        except Exception as e:
            self.add_result("query_prediction_data (batch_id)", False, error=str(e))

    async def test_inference_tools(self):
        """Test inference tools"""
        logger.info("🤖 Testing Inference Tools...")

        try:
            logger.info("Testing run_latest_model_inference...")
            result = await self.tool_executor.execute_tool("run_latest_model_inference", {
                "lookback_window": 512,
                "predict_window": 48,
                "force_rerun": False
            })
            self.add_result("run_latest_model_inference", True, result)

            # Print inference summary
            if result:
                logger.info(f"Inference triggered successfully")
                if isinstance(result, dict):
                    if 'batch_id' in result:
                        logger.info(f"Generated batch ID: {result['batch_id']}")
                    if 'status' in result:
                        logger.info(f"Status: {result['status']}")

        except Exception as e:
            self.add_result("run_latest_model_inference", False, error=str(e))

    async def test_api_tools(self):
        """Test API trading tools with test parameters"""
        logger.info("📡 Testing API Trading Tools...")

        # Test parameters - using far-from-market prices to avoid actual trades
        test_instrument = "ETH-USDT"
        buy_price = "100.0"      # Far below market for testing
        sell_price = "100000.0"  # Far above market for testing
        test_size = "0.001"      # Small size for testing

        placed_order_ids = []

        # Test 1: place_order (buy)
        try:
            logger.info(f"Testing place_order (buy) for {test_instrument} at ${buy_price}")
            result = await self.tool_executor.execute_tool("place_order", {
                "inst_id": test_instrument,
                "side": "buy",
                "order_type": "limit",
                "size": test_size,
                "price": buy_price,
                "tag": "test_buy_order"
            })
            self.add_result("place_order (buy)", True, result)

            if result and 'order_id' in result:
                placed_order_ids.append(result['order_id'])
                logger.info(f"Placed buy order: {result['order_id']}")

        except Exception as e:
            self.add_result("place_order (buy)", False, error=str(e))

        # Test 2: place_order (sell)
        try:
            logger.info(f"Testing place_order (sell) for {test_instrument} at ${sell_price}")
            result = await self.tool_executor.execute_tool("place_order", {
                "inst_id": test_instrument,
                "side": "sell",
                "order_type": "limit",
                "size": test_size,
                "price": sell_price,
                "tag": "test_sell_order"
            })
            self.add_result("place_order (sell)", True, result)

            if result and 'order_id' in result:
                placed_order_ids.append(result['order_id'])
                logger.info(f"Placed sell order: {result['order_id']}")

        except Exception as e:
            self.add_result("place_order (sell)", False, error=str(e))

        # Test 3: get_pending_orders
        try:
            logger.info("Testing get_pending_orders...")
            result = await self.tool_executor.execute_tool("get_pending_orders", {})
            self.add_result("get_pending_orders", True, result)

            # Print pending orders summary
            if result and 'orders' in result:
                logger.info(f"Found {len(result['orders'])} pending orders")
                for order in result['orders'][:5]:  # Show first 5 orders
                    logger.info(f"  Order {order.get('order_id')}: {order.get('side')} {order.get('size')} @ {order.get('price')}")

        except Exception as e:
            self.add_result("get_pending_orders", False, error=str(e))

        # Test 4: amend_order (modify one of our placed orders)
        if placed_order_ids:
            try:
                order_to_amend = placed_order_ids[0]
                new_price = str(float(buy_price) * 1.02)  # Increase price by 2%
                logger.info(f"Testing amend_order for order {order_to_amend} to price ${new_price}")
                result = await self.tool_executor.execute_tool("amend_order", {
                    "inst_id": test_instrument,
                    "order_id": order_to_amend,
                    "new_price": new_price
                })
                self.add_result("amend_order", True, result)

                if result:
                    logger.info(f"Amended order {order_to_amend} successfully")

            except Exception as e:
                self.add_result("amend_order", False, error=str(e))
        else:
            logger.warning("No orders available to test amend_order")
            self.add_result("amend_order", False, error="No orders available to test")

        # Test 5: cancel_order (cancel all placed orders)
        for order_id in placed_order_ids:
            try:
                logger.info(f"Testing cancel_order for order {order_id}")
                result = await self.tool_executor.execute_tool("cancel_order", {
                    "inst_id": test_instrument,
                    "order_id": order_id
                })
                self.add_result(f"cancel_order ({order_id[:8]}...)", True, result)

                if result:
                    logger.info(f"Cancelled order {order_id} successfully")

            except Exception as e:
                self.add_result(f"cancel_order ({order_id[:8]}...)", False, error=str(e))

    def format_result_for_report(self, result: ToolTestResult) -> Dict[str, Any]:
        """Format a test result for the report"""
        formatted = {
            'tool_name': result.tool_name,
            'success': result.success,
            'timestamp': result.timestamp.isoformat(),
        }

        if result.success:
            # Limit result size for readability
            if isinstance(result.result, dict):
                formatted['result'] = {
                    k: v for k, v in result.result.items()
                    if not isinstance(v, (list, dict)) or len(str(v)) < 500
                }
                if isinstance(result.result, dict):
                    for k, v in result.result.items():
                        if isinstance(v, list) and len(v) > 0:
                            formatted['result'][k] = f"List with {len(v)} items"
                        elif isinstance(v, dict) and len(v) > 0:
                            formatted['result'][k] = f"Dict with {len(v)} keys"
            else:
                formatted['result'] = str(result.result)[:500]
        else:
            formatted['error'] = result.error

        return formatted

    def save_results_to_docs(self):
        """Save comprehensive test results to docs directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results data structure
        report_data = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Comprehensive Tool Testing',
                'environment': 'Direct Tool Invocation',
                'description': 'Testing all available database, inference, and API tools'
            },
            'test_summary': self.test_summary,
            'detailed_results': [self.format_result_for_report(r) for r in self.test_results],
            'tool_categories': {
                'database_tools': {
                    'description': 'Database query and data retrieval tools',
                    'tools_tested': [
                        r.tool_name for r in self.test_results
                        if 'query_historical' in r.tool_name or 'prediction_history' in r.tool_name
                    ]
                },
                'inference_tools': {
                    'description': 'Model inference and prediction tools',
                    'tools_tested': [
                        r.tool_name for r in self.test_results
                        if 'inference' in r.tool_name
                    ]
                },
                'api_tools': {
                    'description': 'OKX API trading and order management tools',
                    'tools_tested': [
                        r.tool_name for r in self.test_results
                        if any(word in r.tool_name for word in ['place_order', 'get_pending', 'amend_order', 'cancel_order'])
                    ]
                }
            }
        }

        # Save JSON report
        json_file = f"docs/tool_test_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Save markdown report
        md_file = f"docs/tool_test_report_{timestamp}.md"
        self._generate_markdown_report(report_data, md_file)

        logger.info(f"📄 Test reports saved:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Markdown: {md_file}")

        return json_file, md_file

    def _generate_markdown_report(self, report_data: Dict, filename: str):
        """Generate a markdown report"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# KOKEX Tool Test Report\n\n")
            f.write(f"**Generated:** {report_data['test_metadata']['timestamp']}\n\n")
            f.write(f"**Test Type:** {report_data['test_metadata']['test_type']}\n\n")
            f.write(f"**Environment:** {report_data['test_metadata']['environment']}\n\n")

            # Summary
            summary = report_data['test_summary']
            f.write("## Test Summary\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed']} ✅\n")
            f.write(f"- **Failed:** {summary['failed']} ❌\n")
            f.write(f"- **Success Rate:** {(summary['passed'] / max(summary['total_tests'], 1)) * 100:.1f}%\n\n")

            # Tool Categories
            f.write("## Tool Categories Tested\n\n")
            for category, info in report_data['tool_categories'].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write(f"{info['description']}\n\n")
                f.write("**Tools Tested:**\n")
                for tool in info['tools_tested']:
                    f.write(f"- {tool}\n")
                f.write("\n")

            # Detailed Results
            f.write("## Detailed Test Results\n\n")
            for result in report_data['detailed_results']:
                status_icon = "✅" if result['success'] else "❌"
                f.write(f"### {status_icon} {result['tool_name']}\n\n")
                f.write(f"**Timestamp:** {result['timestamp']}\n\n")
                f.write(f"**Success:** {result['success']}\n\n")

                if result['success']:
                    f.write("**Result:**\n```json\n")
                    f.write(json.dumps(result.get('result', {}), indent=2))
                    f.write("\n```\n\n")
                else:
                    f.write(f"**Error:** {result['error']}\n\n")

            # Errors Summary
            if summary['errors']:
                f.write("## Errors Summary\n\n")
                for error in summary['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")

            f.write("---\n")
            f.write("*Report generated by KOKEX Tool Testing Suite*\n")

    async def run_all_tests(self):
        """Run comprehensive tests for all tools"""
        logger.info("🚀 Starting Comprehensive Tool Testing...")
        logger.info("=" * 60)

        # Test all tool categories
        await self.test_database_tools()
        await self.test_inference_tools()
        await self.test_api_tools()

        # Print summary
        logger.info("=" * 60)
        logger.info("🏁 Testing Complete!")
        logger.info(f"Total: {self.test_summary['total_tests']}, "
                   f"Passed: {self.test_summary['passed']}, "
                   f"Failed: {self.test_summary['failed']}")

        if self.test_summary['failed'] > 0:
            logger.error("Failed tests:")
            for error in self.test_summary['errors']:
                logger.error(f"  - {error}")

        # Save results
        json_file, md_file = self.save_results_to_docs()

        return self.test_summary, json_file, md_file

async def main():
    """Main test execution"""
    tester = ToolTester()

    try:
        summary, json_file, md_file = await tester.run_all_tests()

        # Clean up connections
        await tester.tool_executor.close()

        # Return summary for potential use in scripts
        return {
            'success': summary['failed'] == 0,
            'summary': summary,
            'reports': {
                'json': json_file,
                'markdown': md_file
            }
        }

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        try:
            await tester.tool_executor.close()
        except:
            pass
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Run the comprehensive test suite
    result = asyncio.run(main())

    if result.get('success'):
        print("\n🎉 All tests completed successfully!")
        print(f"📊 Reports saved to: {result['reports']['json']} and {result['reports']['markdown']}")
    else:
        print(f"\n❌ Some tests failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)