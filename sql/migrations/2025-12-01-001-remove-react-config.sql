-- 清理 ReAct 相关配置
-- 迁移时间: 2025-12-01-001

-- 1. 删除 TradingPlan 表中的 react_config 字段
ALTER TABLE trading_plans DROP COLUMN IF EXISTS react_config;

-- 2. 清理相关注释和索引（如果存在）
-- (由于是直接删除字段，相关索引和注释会自动删除)

-- 3. 记录清理操作
DO $$
BEGIN
    RAISE NOTICE '已清理 ReAct 相关配置字段 react_config';
END $$;