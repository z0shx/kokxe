-- 增强Agent对话记录功能
-- 为TradeOrder表添加与Agent对话的关联字段
-- 为AgentMessage表添加更详细的工具调用信息

-- 为trade_orders表添加与Agent对话的关联字段
ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS agent_message_id INTEGER;
ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS conversation_id INTEGER;
ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS tool_call_id VARCHAR(100);

-- 为agent_messages表添加更详细的工具调用信息
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS tool_call_id VARCHAR(100);
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS tool_execution_time FLOAT;
ALTER TABLE agent_messages ADD COLUMN IF NOT EXISTS related_order_id VARCHAR(100);

-- 添加索引以提升查询性能
CREATE INDEX IF NOT EXISTS idx_trade_order_agent_message_id ON trade_orders(agent_message_id);
CREATE INDEX IF NOT EXISTS idx_trade_order_conversation_id ON trade_orders(conversation_id);
CREATE INDEX IF NOT EXISTS idx_trade_order_tool_call_id ON trade_orders(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_agent_message_tool_call_id ON agent_messages(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_agent_message_related_order_id ON agent_messages(related_order_id);

-- 添加注释
COMMENT ON COLUMN trade_orders.agent_message_id IS '关联到触发此订单的Agent消息，用于追踪决策上下文';
COMMENT ON COLUMN trade_orders.conversation_id IS '关联到Agent对话会话ID，用于查询完整对话历史';
COMMENT ON COLUMN trade_orders.tool_call_id IS '工具调用ID，关联到具体的工具调用记录';
COMMENT ON COLUMN agent_messages.tool_call_id IS '工具调用的唯一标识符，用于关联订单';
COMMENT ON COLUMN agent_messages.tool_execution_time IS '工具执行所需的时间，用于性能分析';
COMMENT ON COLUMN agent_messages.related_order_id IS '工具调用产生的订单ID，用于关联交易结果';

-- 更新现有记录（如果需要）
-- UPDATE trade_orders SET is_from_agent = TRUE WHERE plan_id IN (
--     SELECT DISTINCT plan_id FROM agent_conversations
-- ) AND is_from_agent = FALSE;