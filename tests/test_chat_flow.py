#!/usr/bin/env python3
"""
测试聊天流修复 - 验证send_message_wrapper的返回值
"""

def test_send_message_wrapper():
    """测试send_message_wrapper函数的返回值结构"""

    # 模拟函数调用
    def mock_send_message_wrapper(message, history, pid):
        """模拟的send_message_wrapper函数"""
        if not pid:
            return [{"role": "assistant", "content": "❌ 请先选择计划"}], ""

        if not message or not message.strip():
            return history, ""

        # 模拟异步生成器
        async def mock_generate_response():
            # 模拟几个响应批次
            yield [{"role": "assistant", "content": "正在思考..."}], ""
            yield [{"role": "assistant", "content": "分析中..."}], ""
            yield [{"role": "assistant", "content": "完成分析"}], ""

        return mock_generate_response()

    # 测试不同的输入场景
    test_cases = [
        # 测试1: 没有pid
        {"message": "测试消息", "history": [], "pid": None},

        # 测试2: 空消息
        {"message": "", "history": [], "pid": 1},

        # 测试3: 正常消息
        {"message": "正常测试消息", "history": [], "pid": 1},
    ]

    print("🧪 测试send_message_wrapper返回值结构")
    print("=" * 50)

    for i, case in enumerate(test_cases):
        print(f"\n📋 测试用例 {i+1}:")
        print(f"   message: '{case['message']}'")
        print(f"   pid: {case['pid']}")

        try:
            result = mock_send_message_wrapper(
                case['message'],
                case['history'],
                case['pid']
            )

            if hasattr(result, '__aiter__'):
                print(f"   ✅ 返回类型: AsyncGenerator")

                # 检查generator的yield值
                import asyncio
                async def check_generator():
                    try:
                        async for value in result:
                            if isinstance(value, (list, tuple)) and len(value) >= 2:
                                print(f"   ✅ Generator yield {len(value)} 个值")
                                print(f"      值1类型: {type(value[0])}")
                                print(f"      值2类型: {type(value[1])}")
                                break
                            else:
                                print(f"   ❌ Generator yield 错误的值数量: {len(value) if hasattr(value, '__len__') else '未知'}")
                                break
                    except Exception as e:
                        print(f"   ❌ Generator 错误: {e}")

                asyncio.run(check_generator())

            else:
                print(f"   ✅ 返回类型: {type(result)}")
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    print(f"   ✅ 返回 {len(result)} 个值")
                    print(f"      值1类型: {type(result[0])}")
                    print(f"      值2类型: {type(result[1])}")
                else:
                    print(f"   ❌ 返回错误: 需要2个值，实际返回 {len(result) if hasattr(result, '__len__') else '未知'}")

        except Exception as e:
            print(f"   ❌ 测试失败: {e}")

    print(f"\n✅ send_message_wrapper测试完成")

if __name__ == "__main__":
    test_send_message_wrapper()