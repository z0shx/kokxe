-- 移除工具确认相关表和字段
-- 迁移ID: 2025-11-24-001-remove-tool-confirmation
-- 描述: 移除废弃的工具确认机制相关的数据表和字段

-- 删除待确认工具调用表
DROP TABLE IF EXISTS pending_tool_calls;

-- 注意：保留 auto_tool_execution_enabled 字段但标记为废弃
-- AI Agent现在直接使用启用的工具，无需此开关