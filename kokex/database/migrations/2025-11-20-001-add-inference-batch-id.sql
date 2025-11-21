-- Migration: Add inference_batch_id to prediction_data table
-- Date: 2025-11-20
-- Purpose: 支持同一训练记录的多次推理数据保存，不覆盖历史推理记录

-- 1. 添加 inference_batch_id 字段
ALTER TABLE prediction_data
ADD COLUMN IF NOT EXISTS inference_batch_id VARCHAR(50);

-- 2. 为现有数据生成默认的 batch_id（基于 created_at 时间戳分组）
-- 这样可以将已有数据按创建时间归类到不同的批次
UPDATE prediction_data
SET inference_batch_id =
    CONCAT(
        TO_CHAR(created_at, 'YYYYMMDDHH24MISS'),
        '_legacy_',
        SUBSTR(MD5(CONCAT(training_record_id::text, created_at::text)), 1, 8)
    )
WHERE inference_batch_id IS NULL;

-- 3. 设置字段为 NOT NULL
ALTER TABLE prediction_data
ALTER COLUMN inference_batch_id SET NOT NULL;

-- 4. 删除旧的唯一约束（training_record_id + timestamp）
ALTER TABLE prediction_data
DROP CONSTRAINT IF EXISTS uq_prediction_data_record_timestamp;

-- 5. 添加新的唯一约束（training_record_id + inference_batch_id + timestamp）
ALTER TABLE prediction_data
ADD CONSTRAINT uq_prediction_data_batch_timestamp
UNIQUE (training_record_id, inference_batch_id, timestamp);

-- 6. 添加索引以加速批次查询
CREATE INDEX IF NOT EXISTS idx_prediction_data_inference_batch_id
ON prediction_data(inference_batch_id);

-- 7. 添加注释
COMMENT ON COLUMN prediction_data.inference_batch_id IS '推理批次ID，用于区分同一训练记录的不同推理';

-- 验证迁移结果
-- SELECT
--     training_record_id,
--     inference_batch_id,
--     COUNT(*) as predictions_count,
--     MIN(timestamp) as time_start,
--     MAX(timestamp) as time_end,
--     MIN(created_at) as inference_time
-- FROM prediction_data
-- GROUP BY training_record_id, inference_batch_id
-- ORDER BY inference_time DESC;
