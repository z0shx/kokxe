"""
测试 OKX 账户 WebSocket 连接
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.account_ws_service import OKXAccountWebSocket
from utils.logger import setup_logger

logger = setup_logger(__name__, "test_account_ws.log")


async def test_account_ws():
    """测试账户 WebSocket"""

    # 替换为你的 API 凭证
    API_KEY = "your-api-key"
    SECRET_KEY = "your-secret-key"
    PASSPHRASE = "your-passphrase"
    IS_DEMO = True  # True: 模拟盘, False: 真实盘

    def callback(data):
        """回调函数"""
        print(f"\n收到数据推送:")
        print(f"  频道: {data['channel']}")
        print(f"  时间: {data['timestamp']}")
        print(f"  数据: {data['data']}")

    # 创建 WebSocket 服务
    ws_service = OKXAccountWebSocket(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        passphrase=PASSPHRASE,
        is_demo=IS_DEMO,
        callback=callback
    )

    # 启动连接
    task = asyncio.create_task(ws_service.start())

    # 等待一段时间观察数据
    print("WebSocket 已启动，等待数据推送...")
    print("按 Ctrl+C 停止")

    try:
        while True:
            await asyncio.sleep(5)

            # 获取当前状态
            status = ws_service.get_status()
            account_info = ws_service.get_account_info()

            print(f"\n连接状态: {'🟢 已连接' if status['connected'] else '⚪ 未连接'}")
            print(f"总接收消息: {status['total_received']}")
            print(f"最后更新: {status['last_update']}")

            if account_info['balances']:
                print(f"\n账户余额:")
                for ccy, data in account_info['balances'].items():
                    print(f"  {ccy}: 可用 {data['available']:.4f}, 余额 {data['balance']:.4f}")

            if account_info['positions']:
                print(f"\n持仓:")
                for pos in account_info['positions']:
                    print(f"  {pos['inst_id']}: {pos['pos']} @ {pos['avg_price']:.4f}")

    except KeyboardInterrupt:
        print("\n\n正在停止...")
        await ws_service.stop()
        print("已停止")


if __name__ == "__main__":
    asyncio.run(test_account_ws())
