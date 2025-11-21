"""
测试 WebSocket header 是否正确发送
"""
import asyncio
import websockets

async def test_header():
    """测试连接"""
    print("测试 WebSocket 连接with header...")

    # 模拟盘需要添加特殊 header
    headers = {
        'x-simulated-trading': '1',
        'User-Agent': 'Test'
    }

    url = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

    try:
        async with websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10
        ) as websocket:
            print("✅ WebSocket 连接成功")
            print(f"连接信息: {websocket}")

            # 等待第一条消息
            message = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"收到消息: {message}")

    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_header())
