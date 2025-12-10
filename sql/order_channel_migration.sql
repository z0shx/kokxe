-- 添加订单频道订阅支持的数据库迁移脚本
-- 执行命令: psql -d kokex -f sql/order_channel_migration.sql

-- 扩展 WebSocketSubscription 表支持订单频道
ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS subscribed_channels JSONB;
ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS last_order_update TIMESTAMP;
ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS order_count INTEGER DEFAULT 0;

-- 新增订单事件日志表
CREATE TABLE IF NOT EXISTS order_event_logs (
    id SERIAL PRIMARY KEY,
    plan_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    inst_id VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    event_data JSONB NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_conversation_id INTEGER,
    FOREIGN KEY (plan_id) REFERENCES trading_plans(id),
    FOREIGN KEY (agent_conversation_id) REFERENCES agent_conversations(id),
    UNIQUE(plan_id, order_id, event_type)
);

-- 添加索引
CREATE INDEX IF NOT EXISTS idx_order_event_logs_plan_id ON order_event_logs(plan_id);
CREATE INDEX IF NOT EXISTS idx_order_event_logs_order_id ON order_event_logs(order_id);
CREATE INDEX IF NOT EXISTS idx_order_event_logs_event_type ON order_event_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_order_event_logs_processed_at ON order_event_logs(processed_at);
CREATE INDEX IF NOT EXISTS idx_order_event_logs_conversation_id ON order_event_logs(agent_conversation_id);

-- 添加注释
COMMENT ON COLUMN web_socket_subscriptions.subscribed_channels IS '订阅的频道列表（JSON）';
COMMENT ON COLUMN web_socket_subscriptions.last_order_update IS '最后订单更新时间';
COMMENT ON COLUMN web_socket_subscriptions.order_count IS '接收订单数量';
COMMENT ON TABLE order_event_logs IS '订单事件日志表，记录所有订单相关的WebSocket事件';
COMMENT ON COLUMN order_event_logs.event_type IS '事件类型：buy_order_done, sell_order_done等';
COMMENT ON COLUMN order_event_logs.event_data IS '订单事件的完整数据（JSON格式）';
COMMENT ON COLUMN order_event_logs.agent_conversation_id IS '关联的Agent对话会话ID';