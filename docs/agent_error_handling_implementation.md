# Agent工具调用错误处理实现文档

## 概述

本文档描述了KOKEX系统中Agent工具调用的完整错误处理和记录机制，确保即使在工具调用失败的情况下，Agent也能够继续对话而不是停止。

## 实现的功能

### 1. 新增AgentErrorHandler服务

**文件位置**: `/home/zshx/code/kokex/services/agent_error_handler.py`

**核心功能**:
- **错误记录**: 将所有工具调用错误记录到数据库`AgentDecision`表
- **智能判断**: 根据错误类型和连续错误次数判断是否应该继续对话
- **格式化错误信息**: 为Agent提供易于理解的错误信息和建议
- **容错响应**: 创建标准化的失败响应，确保对话能够继续

#### 主要方法

```python
# 记录工具错误到数据库
@classmethod
def record_tool_error(cls, plan_id, tool_name, error_message, ...)

# 判断是否应该继续对话
@classmethod
def should_continue_conversation(cls, error_message, tool_name, consecutive_errors)

# 格式化错误信息供Agent理解
@classmethod
def format_error_for_agent(cls, tool_name, error_message, suggestion)

# 创建失败响应，确保对话能够继续
@classmethod
def create_fallback_response(cls, tool_name, error_message, plan_context)

# 获取最近的错误次数
@classmethod
def get_recent_error_count(cls, plan_id, time_window_minutes)
```

### 2. 增强AgentToolExecutor

**文件位置**: `/home/zshx/code/kokex/services/agent_tool_executor.py`

**主要改进**:

#### 构造函数增强
```python
def __init__(self, api_key, secret_key, passphrase, is_demo=True,
             trading_limits=None, plan_id=None, conversation_id=None):
    # 新增plan_id和conversation_id参数用于错误记录
```

#### 错误处理集成
- **工具不存在**: 记录错误并返回容错响应
- **参数验证失败**: 记录错误并返回容错响应
- **交易限制**: 记录错误并返回容错响应
- **执行异常**: 记录错误并返回容错响应

### 3. 错误分类和处理策略

#### 致命错误（停止对话）
- API密钥无效
- 认证失败
- 账户被冻结
- 权限不足
- 配置错误
- 连续错误≥3次

#### 网络错误（继续对话）
- 连接超时
- 网络错误
- 502/503/504错误

#### 数据错误（继续对话）
- 数据不存在
- 参数验证失败
- 查询结果为空
- 格式错误

#### 默认策略
- 未知错误类型默认继续对话
- 异常情况默认继续对话

### 4. 数据库记录

**记录表**: `AgentDecision`

**记录字段**:
```sql
- plan_id: 计划ID
- conversation_id: 对话ID（可选）
- tool_name: 工具名称
- tool_params: 工具参数（JSON）
- decision: "ERROR"（表示错误记录）
- reason: 错误描述
- error_message: 详细错误信息
- status: "failed"
- created_at: 错误发生时间
```

### 5. 响应格式

#### 成功响应
```json
{
    "success": true,
    "data": "...",
    "continue_conversation": true
}
```

#### 失败响应
```json
{
    "success": false,
    "error": "错误描述",
    "tool_name": "工具名称",
    "fallback_message": "工具调用失败，但我们可以继续分析其他方面",
    "continue_conversation": true/false,
    "timestamp": "2025-11-27T...",
    "context": {"plan_id": 123}
}
```

## 错误日志位置

**主要日志文件**:
- `/home/zshx/code/kokex/logs/agent_tool_executor.log` - 工具执行错误
- `/home/zshx/code/kokex/logs/agent_error_handler.log` - 错误处理器日志

## 使用示例

### 1. 创建AgentToolExecutor时传入上下文

```python
executor = AgentToolExecutor(
    api_key="your_api_key",
    secret_key="your_secret",
    passphrase="your_passphrase",
    is_demo=True,
    plan_id=123,  # 用于错误记录
    conversation_id=456  # 用于错误记录
)
```

### 2. 执行工具

```python
result = await executor.execute_tool("get_current_price", {"inst_id": "BTC-USDT"})

# 检查是否应该继续对话
if result.get("continue_conversation", True):
    # 继续对话
    pass
else:
    # 停止对话
    pass
```

## 优势

1. **完整记录**: 所有工具调用错误都被记录到数据库
2. **智能容错**: 根据错误类型智能判断是否继续对话
3. **详细上下文**: 包含plan_id和conversation_id便于追踪
4. **标准化响应**: 统一的响应格式便于处理
5. **可扩展性**: 易于添加新的错误类型和处理策略

## 解决的问题

1. ✅ **错误记录缺失**: 现在所有错误都被记录到AgentDecision表
2. ✅ **对话中断**: 工具调用失败不再导致Agent停止对话
3. ✅ **错误追踪**: 通过plan_id和conversation_id可以追踪完整的错误历史
4. ✅ **智能判断**: 根据错误类型和频率智能决定是否继续
5. ✅ **调试友好**: 详细的日志和错误信息便于问题排查

## 后续改进建议

1. **监控仪表板**: 创建错误统计和监控界面
2. **告警机制**: 对高频错误设置告警
3. **错误分析**: 定期分析错误模式和优化建议
4. **重试机制**: 对特定类型的错误实现自动重试
5. **错误恢复**: 对某些错误类型实现自动恢复策略