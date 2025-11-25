-- 创建对话系统表
-- 迁移ID: 2025-11-25-001-add-agent-conversation-tables

-- 创建 AI Agent 对话会话表
CREATE TABLE IF NOT EXISTS agent_conversations (
    id SERIAL PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES trading_plans(id) ON DELETE CASCADE,
    training_record_id INTEGER,
    session_name VARCHAR(200),

    -- 会话类型和状态
    conversation_type VARCHAR(50) DEFAULT 'auto_inference',
    status VARCHAR(20) DEFAULT 'active',

    -- 统计信息
    total_messages INTEGER DEFAULT 0,
    total_tool_calls INTEGER DEFAULT 0,

    -- 时间戳
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建 AI Agent 对话消息表
CREATE TABLE IF NOT EXISTS agent_messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES agent_conversations(id) ON DELETE CASCADE,

    -- 消息基本信息
    role VARCHAR(20) NOT NULL,
    content TEXT,
    message_type VARCHAR(50) DEFAULT 'text',

    -- React 循环信息
    react_iteration INTEGER,
    react_stage VARCHAR(50),

    -- 工具调用信息
    tool_name VARCHAR(100),
    tool_arguments JSONB,
    tool_result JSONB,
    tool_status VARCHAR(20) DEFAULT 'pending',

    -- 模型信息
    llm_model VARCHAR(100),

    -- 时间戳
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_agent_conversation_plan_id ON agent_conversations(plan_id);
CREATE INDEX IF NOT EXISTS idx_agent_conversation_training_record_id ON agent_conversations(training_record_id);
CREATE INDEX IF NOT EXISTS idx_agent_conversation_type ON agent_conversations(conversation_type);
CREATE INDEX IF NOT EXISTS idx_agent_conversation_status ON agent_conversations(status);
CREATE INDEX IF NOT EXISTS idx_agent_conversation_started_at ON agent_conversations(started_at);

CREATE INDEX IF NOT EXISTS idx_agent_message_conversation_id ON agent_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_message_role ON agent_messages(role);
CREATE INDEX IF NOT EXISTS idx_agent_message_type ON agent_messages(message_type);
CREATE INDEX IF NOT EXISTS idx_agent_message_timestamp ON agent_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_agent_message_react_iteration ON agent_messages(react_iteration);

-- 添加表注释
COMMENT ON TABLE agent_conversations IS 'AI Agent 对话会话表';
COMMENT ON TABLE agent_messages IS 'AI Agent 对话消息表';

-- 添加字段注释
COMMENT ON COLUMN agent_conversations.plan_id IS '关联的交易计划ID';
COMMENT ON COLUMN agent_conversations.training_record_id IS '关联的训练记录ID（触发时的模型版本）';
COMMENT ON COLUMN agent_conversations.session_name IS '会话名称';
COMMENT ON COLUMN agent_conversations.conversation_type IS '对话类型：auto_inference, manual_chat, analysis';
COMMENT ON COLUMN agent_conversations.status IS '状态：active, completed, archived';
COMMENT ON COLUMN agent_conversations.total_messages IS '总消息数';
COMMENT ON COLUMN agent_conversations.total_tool_calls IS '总工具调用数';
COMMENT ON COLUMN agent_conversations.started_at IS '开始时间';
COMMENT ON COLUMN agent_conversations.last_message_at IS '最后消息时间';
COMMENT ON COLUMN agent_conversations.completed_at IS '完成时间';

COMMENT ON COLUMN agent_messages.conversation_id IS '关联的对话会话ID';
COMMENT ON COLUMN agent_messages.role IS '角色：user, assistant, system, tool';
COMMENT ON COLUMN agent_messages.content IS '消息内容';
COMMENT ON COLUMN agent_messages.message_type IS '消息类型：text, thinking, tool_call, tool_result, system';
COMMENT ON COLUMN agent_messages.react_iteration IS 'ReAct循环迭代次数';
COMMENT ON COLUMN agent_messages.react_stage IS 'ReAct阶段：thought, action, observation';
COMMENT ON COLUMN agent_messages.tool_name IS '工具名称';
COMMENT ON COLUMN agent_messages.tool_arguments IS '工具参数';
COMMENT ON COLUMN agent_messages.tool_result IS '工具执行结果';
COMMENT ON COLUMN agent_messages.tool_status IS '工具状态：pending, success, failed';
COMMENT ON COLUMN agent_messages.llm_model IS '使用的LLM模型';
COMMENT ON COLUMN agent_messages.timestamp IS '消息时间';