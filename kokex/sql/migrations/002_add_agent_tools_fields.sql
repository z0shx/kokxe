-- 为 TradingPlan 表添加 Agent 工具相关字段的迁移
-- 日期: 2025-11-14

-- 注意: 这些字段已经在 models.py 中定义，如果表已存在但缺少这些字段，需要手动添加

-- 检查并添加 agent_tools_config 字段（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trading_plans' AND column_name = 'agent_tools_config'
    ) THEN
        ALTER TABLE trading_plans ADD COLUMN agent_tools_config JSONB;
        COMMENT ON COLUMN trading_plans.agent_tools_config IS 'Agent 工具配置（JSON）';
    END IF;
END $$;

-- 检查并添加 trading_limits 字段（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trading_plans' AND column_name = 'trading_limits'
    ) THEN
        ALTER TABLE trading_plans ADD COLUMN trading_limits JSONB;
        COMMENT ON COLUMN trading_plans.trading_limits IS '交易限制配置（JSON）';
    END IF;
END $$;

-- 示例：agent_tools_config 字段的 JSON 结构
-- {
--   "enabled_tools": [
--     "get_account_balance",
--     "get_current_price",
--     "get_order_info",
--     "get_pending_orders",
--     "place_limit_order",
--     "cancel_order"
--   ],
--   "tool_settings": {
--     "require_confirmation": true,
--     "auto_cancel_on_error": false
--   }
-- }

-- 示例：trading_limits 字段的 JSON 结构
-- {
--   "max_order_amount": 1000.0,
--   "min_order_amount": 10.0,
--   "max_daily_trades": 50,
--   "max_position_size": 5000.0,
--   "allowed_inst_ids": ["BTC-USDT", "ETH-USDT"],
--   "stop_loss_percentage": 0.05,
--   "take_profit_percentage": 0.10
-- }
