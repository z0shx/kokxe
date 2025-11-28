# 优雅关闭功能使用指南

## 📋 概述

KOKEX应用现在支持优雅关闭功能，可以在程序关闭时按顺序停止所有服务，确保数据完整性和资源正确释放。

## 🎯 功能特性

### 支持的服务关闭顺序

1. **训练服务** (优先级: 0)
   - 停止正在运行的训练任务
   - 将训练记录标记为取消状态
   - 记录训练结束时间

2. **定时任务调度器** (优先级: 1)
   - 停止APScheduler调度器
   - 取消所有定时任务

3. **WebSocket连接** (优先级: 2)
   - 停止K线数据WebSocket连接
   - 停止账户WebSocket连接
   - 清理连接资源

4. **Agent服务** (优先级: 3)
   - 停止LangChain Agent服务
   - 清理对话上下文

5. **数据验证服务** (优先级: 4)
   - 停止数据完整性验证服务
   - 取消验证调度任务

6. **后台线程** (优先级: 5)
   - 停止健康检查线程
   - 停止其他后台任务

## 🚀 使用方法

### 1. 自动初始化

应用启动时会自动初始化优雅关闭服务，无需额外配置：

```bash
python app.py
```

启动日志会显示：
```
2025-11-28 17:59:25 - services.graceful_shutdown_service - INFO - 初始化优雅关闭服务...
2025-11-28 17:59:25 - services.graceful_shutdown_service - INFO - ✅ 信号处理器已设置
2025-11-28 17:59:25 - services.graceful_shutdown_service - INFO - ✅ 优雅关闭服务初始化完成
```

### 2. 触发优雅关闭

#### 方式1: Ctrl+C (推荐)
```bash
# 在终端中运行的应用，使用 Ctrl+C
python app.py
# 按下 Ctrl+C
```

#### 方式2: SIGTERM信号
```bash
# 发送SIGTERM信号
kill -TERM <pid>

# 或使用killall
killall -TERM python
```

### 3. 关闭过程示例

当触发优雅关闭时，会看到如下日志：

```
============================================================
🚨 开始优雅关闭所有服务...
============================================================
🛑 停止训练服务...
找到 2 个运行中的训练记录
✅ 训练服务已停止
🛑 停止定时任务调度器...
✅ 定时任务调度器已停止
🛑 停止WebSocket连接...
✅ K线数据WebSocket连接已停止
✅ 账户WebSocket连接已停止
🛑 停止Agent服务...
✅ Agent服务已停止
🛑 停止数据验证服务...
✅ 数据验证服务已停止
🛑 停止后台线程...
✅ 后台线程已停止
🛑 执行注册的关闭处理器...
✅ 关闭处理器 清理处理器 已执行
✅ 所有服务已优雅关闭
🎉 优雅关闭完成，程序可以安全退出
```

## 🛠️ 扩展功能

### 注册自定义关闭处理器

如果需要添加自定义的清理逻辑：

```python
from services.graceful_shutdown_service import graceful_shutdown_service

# 同步处理器
def my_cleanup_handler():
    print("执行自定义清理...")
    # 清理逻辑

# 异步处理器
async def my_async_cleanup_handler():
    print("执行异步清理...")
    await asyncio.sleep(0.1)
    # 异步清理逻辑

# 注册处理器
graceful_shutdown_service.register_shutdown_handler(
    my_cleanup_handler,
    "自定义清理处理器",
    priority=6  # 优先级，数字越小越先执行
)

graceful_shutdown_service.register_shutdown_handler(
    my_async_cleanup_handler,
    "异步清理处理器",
    priority=7
)
```

### 注册后台线程

```python
import threading
from services.graceful_shutdown_service import graceful_shutdown_service

# 创建可停止的线程
class StoppableThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            # 工作逻辑
            time.sleep(1)

    def stop(self):
        self._stop_event.set()

# 注册线程
my_thread = StoppableThread()
my_thread.start()

graceful_shutdown_service.register_background_thread(
    my_thread,
    "我的后台线程"
)
```

## 🔧 故障排除

### 常见问题

1. **程序仍然卡住**
   - 检查是否有无限循环的线程
   - 确保所有线程都支持停止信号
   - 查看日志中的错误信息

2. **某些服务未停止**
   - 检查服务是否实现了正确的停止方法
   - 确认服务是否正确注册到优雅关闭服务

3. **数据丢失**
   - 优雅关闭会确保训练记录正确标记
   - WebSocket连接会正常关闭，不会丢失数据
   - 数据库连接会正确释放

### 调试模式

启用详细日志：

```python
import logging
logging.getLogger('services.graceful_shutdown_service').setLevel(logging.DEBUG)
```

## 📝 注意事项

1. **超时处理**: 如果某个服务关闭时间过长，其他服务仍会继续关闭
2. **强制关闭**: 如果优雅关闭失败，系统会记录错误但不会阻止程序退出
3. **兼容性**: 优雅关闭功能与现有的所有服务兼容
4. **性能影响**: 优雅关闭服务的性能开销极小

## 🧪 测试

运行测试脚本验证功能：

```bash
python tests/test_graceful_shutdown.py
```

这个测试脚本会：
- 模拟后台线程
- 注册关闭处理器
- 测试信号处理
- 验证关闭顺序

## 📊 日志文件

优雅关闭的日志会记录在：
- `logs/graceful_shutdown.log` - 专用日志
- `logs/app.log` - 主应用日志
- 终端输出 - 实时关闭过程

---

**更新日期**: 2025-11-28
**版本**: v1.0.0