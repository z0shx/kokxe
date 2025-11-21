-- Update PendingToolCall status field to support new confirmation workflow
-- This migration updates the status field to include 'executed' status

-- First, update any existing records with old status values
UPDATE pending_tool_calls
SET status = 'approved'
WHERE status = 'confirmed';

-- Update the status field comment to reflect new possible values
COMMENT ON COLUMN pending_tool_calls.status IS '状态：pending/approved/rejected/expired/executed';