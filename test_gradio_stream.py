#!/usr/bin/env python3
"""
测试Gradio流式输出
"""

import asyncio
import gradio as gr
from services.langchain_agent_v2 import langchain_agent_v2_service


async def test_gradio_stream():
    """测试Gradio流式输出"""
    plan_id = 2
    print(f"开始测试计划 {plan_id} 的Gradio流式输出...")

    # 收集所有消息
    all_messages = []

    try:
        async for message_batch in langchain_agent_v2_service.stream_manual_inference(plan_id):
            print(f"收到消息批次: {len(message_batch)} 条消息")
            for msg in message_batch:
                print(f"  [{msg['role']}]: {msg['content'][:50]}...")
                all_messages.append(msg)

            # 模拟Gradio的yield行为
            yield all_messages.copy()

    except Exception as e:
        print(f"流式输出错误: {e}")
        import traceback
        traceback.print_exc()
        yield [{"role": "assistant", "content": f"❌ 错误: {str(e)}"}]

    print(f"总共收到 {len(all_messages)} 条消息")


def create_test_interface():
    """创建测试界面"""
    with gr.Blocks(title="Gradio流式测试") as demo:
        gr.Markdown("# 🧪 Gradio流式输出测试")

        with gr.Row():
            test_btn = gr.Button("🚀 开始测试", variant="primary")

        chatbot = gr.Chatbot(height=500, show_copy_button=True)

        async def wrapper():
            async for messages in test_gradio_stream():
                yield messages

        test_btn.click(
            fn=wrapper,
            outputs=[chatbot],
            show_progress="full"
        )

    return demo


if __name__ == "__main__":
    demo = create_test_interface()
    demo.launch(server_port=7882, share=False)