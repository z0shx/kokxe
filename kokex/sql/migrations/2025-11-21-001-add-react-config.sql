-- 添加ReAct配置字段到TradingPlan表
-- 迁移时间: 2025-11-21-001

-- 1. 添加ReAct配置字段
ALTER TABLE trading_plans
ADD COLUMN react_config JSONB COMMENT 'ReAct推理配置（JSON），包含推理轮数等参数';

-- 2. 为现有记录设置默认ReAct配置
UPDATE trading_plans
SET react_config = '{"max_iterations": 3, "enable_thinking": true, "tool_approval": false, "thinking_style": "详细"}'
WHERE react_config IS NULL;

-- 3. 添加注释
COMMENT ON COLUMN trading_plans.react_config IS 'ReAct推理配置，支持max_iterations(最大推理轮数)、enable_thinking(启用思考过程)、tool_approval(工具审批)等参数';