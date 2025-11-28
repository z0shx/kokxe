#!/usr/bin/env python3
"""
测试Qwen的thinking模式
"""

import os
from openai import OpenAI

def test_qwen_thinking():
    """测试Qwen的thinking模式"""
    print("🧪 测试Qwen的thinking模式")
    print("=" * 50)

    try:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key="sk-dummy-key",  # 需要替换为真实的API key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        messages = [{"role": "user", "content": "你是谁？请详细说明你的思考过程。"}]

        completion = client.chat.completions.create(
            model="qwen-plus",  # 或者你实际使用的模型
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True,
            stream_options={
                "include_usage": True
            },
        )

        reasoning_content = ""  # 完整思考过程
        answer_content = ""  # 完整回复
        is_answering = False  # 是否进入回复阶段

        print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # 只收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                reasoning_content += delta.reasoning_content

            # 收到content，开始进行回复
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

        print(f"\n\n📊 统计:")
        print(f"思考过程长度: {len(reasoning_content)} 字符")
        print(f"回答长度: {len(answer_content)} 字符")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_qwen_thinking()