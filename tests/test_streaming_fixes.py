"""
流式输出和工具调用修复功能的综合测试
测试所有修复的功能是否正常工作
"""
import asyncio
import json
import sys
import os
from typing import Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import AgentMessage
from database.db import get_db
from services.langchain_agent import LangChainAgentService
from ui.streaming_handler import StreamingHandler
from ui.custom_chatbot import process_streaming_messages


async def test_realtime_streaming():
    """测试实时流式输出功能"""
    print("\n🧪 测试实时流式输出功能...")

    handler = StreamingHandler()

    async def mock_agent_stream():
        """模拟Agent流式输出"""
        # 模拟工具调用
        yield [{"role": "tool_call", "content": json.dumps({
            "tool_name": "get_current_price",
            "arguments": {"inst_id": "BTC-USDT"},
            "status": "calling"
        })}]

        await asyncio.sleep(0.1)  # 模拟执行时间

        # 模拟工具结果
        yield [{"role": "tool_result", "content": json.dumps({
            "tool_name": "get_current_price",
            "result": {"price": "45000.5", "success": True},
            "status": "success"
        })}]

        await asyncio.sleep(0.1)

        # 模拟AI回复
        yield [{"role": "assistant", "content": "当前BTC价格为 $45,000.50"}]

    message_count = 0
    message_types = []

    try:
        async for message_batch in handler.process_agent_stream_realtime(
            mock_agent_stream(), "test_session"
        ):
            message_count += 1
            print(f"  📨 收到消息批次 {message_count}: {len(message_batch)} 条消息")

            for message in message_batch:
                message_type = message.get('role', 'unknown')
                message_types.append(message_type)
                print(f"    • 类型: {message_type}, 内容长度: {len(str(message.get('content', '')))}")

        # 验证消息数量和类型
        expected_count = 3
        expected_types = ["tool_call", "tool_result", "assistant"]

        if message_count == expected_count:
            print(f"  ✅ 消息批次数量正确: {message_count}/{expected_count}")
        else:
            print(f"  ❌ 消息批次数量错误: {message_count}/{expected_count}")
            return False

        if all(t in message_types for t in expected_types):
            print(f"  ✅ 消息类型完整: {message_types}")
        else:
            print(f"  ❌ 消息类型缺失: {message_types}, 期望: {expected_types}")
            return False

        print("✅ 实时流式输出测试通过")
        return True

    except Exception as e:
        print(f"❌ 实时流式输出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_field_fix():
    """测试数据库字段名修复（简化版，避免事件循环冲突）"""
    print("\n🧪 测试数据库字段名修复...")

    try:
        agent = LangChainAgentService()

        # 验证方法存在且参数正确
        if hasattr(agent, '_save_message'):
            print("  ✅ _save_message 方法存在")

            # 检查方法签名
            import inspect
            sig = inspect.signature(agent._save_message)
            params = list(sig.parameters.keys())

            if 'tool_args' in params:
                print("  ✅ 字段名已修复为 tool_args")
            else:
                print("  ❌ 字段名仍为旧格式")
                return False

            if 'tool_arguments' in params:
                print("  ✅ 新字段名 tool_arguments 也存在")
            else:
                print("  ⚠️  新字段名 tool_arguments 不存在")

        else:
            print("  ❌ _save_message 方法不存在")
            return False

        # 验证订单ID提取方法
        if hasattr(agent, 'extract_order_ids_from_tool_results'):
            print("  ✅ 订单ID提取方法已添加")
        else:
            print("  ❌ 订单ID提取方法缺失")
            return False

        print("✅ 数据库字段修复测试通过（方法级验证）")
        return True

    except Exception as e:
        print(f"❌ 数据库字段修复测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_order_id_extraction():
    """测试订单ID提取逻辑"""
    print("\n🧪 测试订单ID提取逻辑...")

    try:
        agent = LangChainAgentService()

        # 测试不同格式的工具结果
        test_cases = [
            {
                "name": "标准place_order结果",
                "tool_results": [
                    {
                        "success": True,
                        "result": {
                            "order_id": "12345",
                            "state": "live"
                        }
                    }
                ],
                "expected_ids": ["12345"]
            },
            {
                "name": "OKX API响应格式",
                "tool_results": [
                    {
                        "success": True,
                        "result": {
                            "data": [
                                {"ordId": "67890"},
                                {"ordId": "67891"}
                            ]
                        }
                    }
                ],
                "expected_ids": ["67890", "67891"]
            },
            {
                "name": "混合格式",
                "tool_results": [
                    {
                        "success": True,
                        "result": {
                            "order_id": "11111"
                        }
                    },
                    {
                        "success": True,
                        "result": {
                            "data": [{"ordId": "22222"}]
                        }
                    }
                ],
                "expected_ids": ["11111", "22222"]
            }
        ]

        all_passed = True

        for test_case in test_cases:
            extracted_ids = agent.extract_order_ids_from_tool_results(test_case["tool_results"])
            expected_ids = set(test_case["expected_ids"])
            extracted_set = set(extracted_ids)

            if extracted_set == expected_ids:
                print(f"  ✅ {test_case['name']}: {extracted_ids}")
            else:
                print(f"  ❌ {test_case['name']}: 期望 {expected_ids}, 实际 {extracted_set}")
                all_passed = False

        if all_passed:
            print("✅ 订单ID提取逻辑测试通过")
            return True
        else:
            print("❌ 订单ID提取逻辑测试失败")
            return False

    except Exception as e:
        print(f"❌ 订单ID提取逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_message_formatting():
    """测试消息格式化功能"""
    print("\n🧪 测试消息格式化功能...")

    try:
        from ui.custom_chatbot import _format_tool_call_message, _format_tool_result_message

        # 测试工具调用消息格式化
        tool_call_content = json.dumps({
            "tool_name": "place_order",
            "arguments": {
                "inst_id": "BTC-USDT",
                "side": "buy",
                "sz": "0.1",
                "px": "45000"
            },
            "status": "calling"
        })

        formatted_call = _format_tool_call_message(tool_call_content)

        if "🔄" in formatted_call and "place_order" in formatted_call and "BTC-USDT" in formatted_call:
            print("  ✅ 工具调用消息格式化正确")
        else:
            print(f"  ❌ 工具调用消息格式化错误: {formatted_call}")
            return False

        # 测试工具结果消息格式化
        tool_result_content = json.dumps({
            "tool_name": "place_order",
            "result": {
                "order_id": "12345",
                "state": "live",
                "success": True
            },
            "status": "success"
        })

        formatted_result = _format_tool_result_message(tool_result_content)

        if "✅" in formatted_result and "place_order" in formatted_result and "12345" in formatted_result:
            print("  ✅ 工具结果消息格式化正确")
        else:
            print(f"  ❌ 工具结果消息格式化错误: {formatted_result}")
            return False

        print("✅ 消息格式化功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 消息格式化功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_message_processing():
    """测试流式消息处理"""
    print("\n🧪 测试流式消息处理...")

    try:
        # 模拟流式消息批次
        test_batches = [
            [{"role": "tool_call", "content": json.dumps({
                "tool_name": "get_current_price",
                "arguments": {"inst_id": "BTC-USDT"},
                "status": "calling"
            })}],
            [{"role": "tool_result", "content": json.dumps({
                "tool_name": "get_current_price",
                "result": {"price": "45000.5", "success": True},
                "status": "success"
            })}],
            [{"role": "assistant", "content": "基于当前价格，我建议..."}]
        ]

        processed_messages = process_streaming_messages(test_batches)

        if len(processed_messages) == 3:
            print(f"  ✅ 处理了正确的消息数量: {len(processed_messages)}")

            # 验证消息类型
            roles = [msg.get('role') for msg in processed_messages]
            expected_roles = ['tool_call', 'tool_result', 'assistant']

            if roles == expected_roles:
                print(f"  ✅ 消息角色正确: {roles}")
                print("✅ 流式消息处理测试通过")
                return True
            else:
                print(f"  ❌ 消息角色错误: {roles}, 期望: {expected_roles}")
                return False
        else:
            print(f"  ❌ 消息数量错误: {len(processed_messages)}, 期望: 3")
            return False

    except Exception as e:
        print(f"❌ 流式消息处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行流式输出修复功能综合测试")
    print("=" * 50)

    test_functions = [
        ("实时流式输出", test_realtime_streaming),
        ("数据库字段修复", test_database_field_fix),
        ("订单ID提取逻辑", test_order_id_extraction),
        ("消息格式化功能", test_message_formatting),
        ("流式消息处理", test_streaming_message_processing)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_functions:
        print(f"\n📋 执行测试: {test_name}")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} - 通过")
            else:
                failed += 1
                print(f"❌ {test_name} - 失败")

        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - 异常: {e}")

    print("\n" + "=" * 50)
    print(f"📊 测试结果汇总:")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📈 成功率: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 所有测试通过！流式输出修复功能正常工作")
        return True
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，需要检查相关功能")
        return False


if __name__ == "__main__":
    # 运行所有测试
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)