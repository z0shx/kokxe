-- 修复订单频道订阅数据库迁移脚本
-- 为 KOKEX 系统添加订单事件日志表和相关字段

-- 检查并扩展 WebSocketSubscription 表
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'web_socket_subscriptions') THEN
        ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS subscribed_channels JSONB;
        ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS last_order_update TIMESTAMP;
        ALTER TABLE web_socket_subscriptions ADD COLUMN IF NOT EXISTS order_count INTEGER DEFAULT 0;
        RAISE NOTICE 'WebSocketSubscription table has been extended with order subscription columns';
    ELSE
        RAISE NOTICE 'WebSocketSubscription table does not exist, skipping extension';
    END IF;
END $$;

-- 创建订单事件日志表
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
CREATE INDEX IF NOT EXISTS idx_order_event_logs_plan_order_event ON order_event_logs(plan_id, order_id, event_type);

-- 创建复合索引用于查询优化
CREATE INDEX IF NOT EXISTS idx_order_event_logs_plan_event_type ON order_event_logs(plan_id, event_type);
CREATE INDEX IF NOT EXISTS idx_order_event_logs_plan_processed ON order_event_logs(plan_id, processed_at);

RAISE NOTICE 'Order event logs table and indexes have been created successfully';