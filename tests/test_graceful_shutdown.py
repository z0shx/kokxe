#!/usr/bin/env python3
"""
测试优雅关闭功能
"""

import asyncio
import signal
import time
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.graceful_shutdown_service import graceful_shutdown_service

def test_graceful_shutdown():
    """测试优雅关闭功能"""
    print("🧪 测试优雅关闭功能")
    print("=" * 60)

    # 模拟注册一些关闭处理器
    def cleanup_handler_1():
        print("🧹 清理处理器1: 清理临时文件")
        time.sleep(0.5)

    def cleanup_handler_2():
        print("🧹 清理处理器2: 关闭数据库连接")
        time.sleep(0.3)

    async def async_cleanup_handler():
        print("🧹 异步清理处理器: 发送剩余数据")
        await asyncio.sleep(0.2)

    # 注册关闭处理器
    graceful_shutdown_service.register_shutdown_handler(
        cleanup_handler_1,
        "临时文件清理",
        priority=1
    )

    graceful_shutdown_service.register_shutdown_handler(
        cleanup_handler_2,
        "数据库连接清理",
        priority=2
    )

    graceful_shutdown_service.register_shutdown_handler(
        async_cleanup_handler,
        "异步数据发送",
        priority=3
    )

    # 模拟后台线程
    def background_task():
        print("🔄 后台任务开始运行...")
        while not graceful_shutdown_service.is_shutting_down:
            try:
                time.sleep(1)
                print("💼 后台任务工作中...")
            except KeyboardInterrupt:
                print("⚠️ 后台任务收到中断信号")
                break
        print("✅ 后台任务已停止")

    background_thread = threading.Thread(target=background_task, daemon=True)
    background_thread.start()

    # 注册后台线程
    graceful_shutdown_service.register_background_thread(
        background_thread,
        "测试后台任务"
    )

    print("✅ 优雅关闭服务配置完成")
    print("💡 使用 Ctrl+C 测试优雅关闭")
    print("⏱️  程序将运行30秒后自动关闭...")

    # 设置信号处理器
    graceful_shutdown_service.setup_signal_handlers()

    # 运行30秒后自动关闭
    try:
        time.sleep(30)
        print("⏰ 时间到，触发优雅关闭...")

        # 手动触发优雅关闭
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(graceful_shutdown_service.stop_all_services())
        loop.close()

        if result:
            print("🎉 优雅关闭测试成功完成")
        else:
            print("❌ 优雅关闭测试失败")

    except KeyboardInterrupt:
        print("\n⚠️ 收到中断信号，触发优雅关闭...")

        # 手动触发优雅关闭
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(graceful_shutdown_service.stop_all_services())
        loop.close()

        if result:
            print("🎉 优雅关闭测试成功完成")
        else:
            print("❌ 优雅关闭测试失败")

if __name__ == "__main__":
    test_graceful_shutdown()