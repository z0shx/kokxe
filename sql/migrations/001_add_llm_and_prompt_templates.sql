-- 数据库迁移 SQL
-- 版本: v1.1.0
-- 日期: 2025-11-14
-- 描述: 添加 LLM 配置和 Agent 提示词模版管理功能

-- ========================================
-- 1. 创建 LLM 配置表
-- ========================================

CREATE TABLE IF NOT EXISTS llm_configs (
    id SERIAL NOT NULL,
    name VARCHAR(100) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL,

    -- API 配置
    api_key VARCHAR(200),
    api_base_url VARCHAR(200),
    model_name VARCHAR(100),

    -- 高级配置
    max_tokens INTEGER DEFAULT 4096,
    temperature FLOAT DEFAULT 0.7,
    top_p FLOAT DEFAULT 1.0,
    extra_params JSONB,

    -- 状态
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,

    -- 时间戳
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),

    PRIMARY KEY (id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_llm_config_provider ON llm_configs(provider);
CREATE INDEX IF NOT EXISTS idx_llm_config_is_active ON llm_configs(is_active);

-- 添加注释
COMMENT ON TABLE llm_configs IS 'LLM 配置表';
COMMENT ON COLUMN llm_configs.name IS '配置名称';
COMMENT ON COLUMN llm_configs.provider IS 'LLM 提供商：claude/qwen/ollama/openai';
COMMENT ON COLUMN llm_configs.api_key IS 'API Key（加密存储）';
COMMENT ON COLUMN llm_configs.api_base_url IS 'API 基础 URL';
COMMENT ON COLUMN llm_configs.model_name IS '模型名称，如 claude-3-sonnet-20240229';
COMMENT ON COLUMN llm_configs.max_tokens IS '最大 token 数';
COMMENT ON COLUMN llm_configs.temperature IS '温度参数';
COMMENT ON COLUMN llm_configs.top_p IS 'Top P 参数';
COMMENT ON COLUMN llm_configs.extra_params IS '其他参数（JSON）';
COMMENT ON COLUMN llm_configs.is_active IS '是否启用';
COMMENT ON COLUMN llm_configs.is_default IS '是否默认配置';
COMMENT ON COLUMN llm_configs.created_at IS '创建时间';
COMMENT ON COLUMN llm_configs.updated_at IS '更新时间';

-- ========================================
-- 2. 创建 Agent 提示词模版表
-- ========================================

CREATE TABLE IF NOT EXISTS agent_prompt_templates (
    id SERIAL NOT NULL,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    content TEXT NOT NULL,

    -- 分类和标签
    category VARCHAR(50),
    tags JSONB,

    -- 状态
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,

    -- 时间戳
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),

    PRIMARY KEY (id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_agent_prompt_template_category ON agent_prompt_templates(category);
CREATE INDEX IF NOT EXISTS idx_agent_prompt_template_is_active ON agent_prompt_templates(is_active);

-- 添加注释
COMMENT ON TABLE agent_prompt_templates IS 'Agent 提示词模版表';
COMMENT ON COLUMN agent_prompt_templates.name IS '模版名称';
COMMENT ON COLUMN agent_prompt_templates.description IS '模版描述';
COMMENT ON COLUMN agent_prompt_templates.content IS '提示词内容';
COMMENT ON COLUMN agent_prompt_templates.category IS '分类：conservative/aggressive/balanced/custom';
COMMENT ON COLUMN agent_prompt_templates.tags IS '标签（JSON数组）';
COMMENT ON COLUMN agent_prompt_templates.is_active IS '是否启用';
COMMENT ON COLUMN agent_prompt_templates.is_default IS '是否默认模版';
COMMENT ON COLUMN agent_prompt_templates.created_at IS '创建时间';
COMMENT ON COLUMN agent_prompt_templates.updated_at IS '更新时间';

-- ========================================
-- 3. 在 trading_plans 表中添加 llm_config_id 字段
-- ========================================

-- 检查字段是否存在，如果不存在则添加
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'trading_plans'
        AND column_name = 'llm_config_id'
    ) THEN
        ALTER TABLE trading_plans ADD COLUMN llm_config_id INTEGER;
        COMMENT ON COLUMN trading_plans.llm_config_id IS 'LLM 配置 ID';
    END IF;
END $$;

-- ========================================
-- 4. 插入默认数据（可选）
-- ========================================

-- 插入默认的 Agent 提示词模版
INSERT INTO agent_prompt_templates (name, description, content, category, is_default)
VALUES (
    '默认交易策略',
    '基础的加密货币交易策略提示词',
    '你是一个专业的加密货币交易员。根据预测的K线数据，分析市场趋势并做出交易决策。

你的职责：
1. 分析预测的价格走势和成交量
2. 识别潜在的买入和卖出信号
3. 考虑风险管理和仓位控制
4. 提供明确的交易建议

请基于数据分析，给出你的交易决策。',
    'balanced',
    true
)
ON CONFLICT (name) DO NOTHING;

-- ========================================
-- 5. 创建更新时间触发器函数
-- ========================================

-- 创建或替换触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW() AT TIME ZONE 'utc';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为 llm_configs 表添加触发器
DROP TRIGGER IF EXISTS update_llm_configs_updated_at ON llm_configs;
CREATE TRIGGER update_llm_configs_updated_at
    BEFORE UPDATE ON llm_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 为 agent_prompt_templates 表添加触发器
DROP TRIGGER IF EXISTS update_agent_prompt_templates_updated_at ON agent_prompt_templates;
CREATE TRIGGER update_agent_prompt_templates_updated_at
    BEFORE UPDATE ON agent_prompt_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- 迁移完成
-- ========================================

-- 验证表是否创建成功
SELECT
    'llm_configs' AS table_name,
    COUNT(*) AS row_count
FROM llm_configs
UNION ALL
SELECT
    'agent_prompt_templates' AS table_name,
    COUNT(*) AS row_count
FROM agent_prompt_templates;

-- 显示迁移完成消息
DO $$
BEGIN
    RAISE NOTICE '✅ 数据库迁移完成！';
    RAISE NOTICE '已创建表：llm_configs, agent_prompt_templates';
    RAISE NOTICE '已更新表：trading_plans（添加 llm_config_id 字段）';
END $$;
