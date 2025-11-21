-- 添加待确认工具调用表
-- 迁移ID: 2025-11-18-002
-- 描述: 添加待确认工具调用表，支持工具确认机制

-- 创建待确认工具调用表
CREATE TABLE IF NOT EXISTS pending_tool_calls (
    id SERIAL PRIMARY KEY,
    plan_id INTEGER NOT NULL,
    agent_decision_id INTEGER,

    -- 工具信息
    tool_name VARCHAR(100) NOT NULL,
    tool_arguments JSONB NOT NULL,

    -- 预期效果和风险
    expected_effect TEXT,
    risk_warning TEXT,

    -- 状态
    status VARCHAR(20) DEFAULT 'pending',

    -- 执行结果
    execution_result JSONB,
    error_message TEXT,

    -- 操作信息
    confirmed_at TIMESTAMP,
    confirmed_by VARCHAR(100),

    -- 超时设置
    expires_at TIMESTAMP,

    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_pending_tool_plan_id ON pending_tool_calls(plan_id);
CREATE INDEX IF NOT EXISTS idx_pending_tool_status ON pending_tool_calls(status);
CREATE INDEX IF NOT EXISTS idx_pending_tool_created_at ON pending_tool_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_pending_tool_expires_at ON pending_tool_calls(expires_at);

-- 添加注释
COMMENT ON TABLE pending_tool_calls IS '待确认工具调用表';
COMMENT ON COLUMN pending_tool_calls.plan_id IS '关联的交易计划ID';
COMMENT ON COLUMN pending_tool_calls.agent_decision_id IS '关联的Agent决策ID';
COMMENT ON COLUMN pending_tool_calls.tool_name IS '工具名称';
COMMENT ON COLUMN pending_tool_calls.tool_arguments IS '工具参数（JSON）';
COMMENT ON COLUMN pending_tool_calls.expected_effect IS '预期效果说明';
COMMENT ON COLUMN pending_tool_calls.risk_warning IS '风险提示';
COMMENT ON COLUMN pending_tool_calls.status IS '状态：pending/confirmed/rejected/expired';
COMMENT ON COLUMN pending_tool_calls.execution_result IS '执行结果（JSON）';
COMMENT ON COLUMN pending_tool_calls.error_message IS '错误信息';
COMMENT ON COLUMN pending_tool_calls.confirmed_at IS '确认时间';
COMMENT ON COLUMN pending_tool_calls.confirmed_by IS '确认人（系统/手动）';
COMMENT ON COLUMN pending_tool_calls.expires_at IS '过期时间';
COMMENT ON COLUMN pending_tool_calls.created_at IS '创建时间';
COMMENT ON COLUMN pending_tool_calls.updated_at IS '更新时间';
