# 智能Data Offset预测间隔管理指南

## 📋 概述

KOKEX系统现已实现智能Data Offset计算功能，确保预测间隔的准确性和数据连续性。该功能自动调整推理数据的时间窗口，确保预测间隔满足配置要求。

## 🎯 核心功能

### 1. 智能间隔计算
- **自动检测**: 计算距离上次预测的时间差
- **智能偏移**: 根据目标间隔自动计算Data Offset
- **精确控制**: 确保回看窗口的最后一个数据点正确

### 2. 双模式支持
- **自动触发模式**: 定时任务触发，严格遵循间隔时间
- **手动触发模式**: 用户手动触发，智能计算最佳偏移

### 3. 间隔时间管理
- **标准间隔**: 支持2h, 4h, 6h, 8h, 12h, 24h等
- **灵活配置**: 可在计划配置中自定义间隔时间
- **时区统一**: 所有时间计算基于UTC+8

## 🔧 技术实现

### 核心组件

#### 1. InferenceDataOffsetService
**位置**: `services/inference_data_offset_service.py`

**主要功能**:
- 计算最优Data Offset
- 更新推理参数
- 提供预测状态摘要

#### 2. 调度服务集成
**位置**: `services/schedule_service.py`

**集成点**:
- 自动预测触发前计算偏移
- 手动预测触发前计算偏移
- 更新推理参数到计划配置

### 计算逻辑

#### 偏移量计算公式
```python
# 1. 计算时间差
time_diff_hours = current_time - latest_prediction_time

# 2. 计算需要回退的时间
if time_diff_hours < target_interval:
    additional_backoff = target_interval - time_diff_hours
else:
    additional_backoff = time_diff_hours % target_interval

# 3. 转换为K线条数
data_offset = int(additional_backoff / kline_interval_hours)

# 4. 实际间隔
actual_interval = time_diff_hours + (data_offset * kline_interval_hours)
```

#### 应用场景
1. **间隔不足**: 需要进一步回退以满足最小间隔
2. **间隔过长**: 尽量接近目标间隔的倍数
3. **首次预测**: 使用默认偏移（offset=0）

## 🚀 使用方法

### 自动触发场景

当定时任务触发自动预测时：

1. **检查最新预测时间**
2. **计算时间差**
3. **应用Data Offset**
4. **执行推理**
5. **记录结果**

```python
# 日志示例
2025-11-28 18:07:22 - 计划 2: K线间隔=1小时, 目标预测间隔=4小时
2025-11-28 18:07:22 - 计划 2: 最新预测时间=2025-11-28 14:00:00, 距今=4.12小时
2025-11-28 18:07:22 - 偏移计算: 时间差=4.12h, 目标=4h, 需要回退=0.12h, 偏移K线数=0
2025-11-28 18:07:22 - 计划 2: Data Offset计算完成 - offset=0
```

### 手动触发场景

当用户点击"执行推理"按钮时：

1. **分析当前状态**
2. **计算最佳偏移**
3. **更新推理参数**
4. **执行推理**

```python
# 手动触发日志
2025-11-28 18:07:22 - 手动推理数据偏移计算完成: plan_id=2, offset=1
2025-11-28 18:07:22 - 手动推理偏移说明: 当前时间差21.88小时，为满足4小时目标间隔，回退1条K线数据（1.00小时）
2025-11-28 18:07:22 - 手动推理参数已更新: training_id=34, data_offset=1
```

## 📊 配置说明

### 计划配置参数

```sql
-- 交易计划表中的相关字段
auto_inference_enabled = TRUE              -- 启用自动推理
auto_inference_interval_hours = 4           -- 推理间隔（小时）
finetune_params = {
  "inference": {
    "data_offset": 0,                      -- 数据偏移量（自动计算）
    "temperature": 1.0,
    "top_p": 0.9,
    "sample_count": 30
  }
}
```

### K线间隔支持

| 间隔代码 | 小时数 | 说明 |
|---------|--------|------|
| 1m/1M | 0.0167 | 1分钟 |
| 5m/5M | 0.0833 | 5分钟 |
| 15m/15M | 0.25 | 15分钟 |
| 30m/30M | 0.5 | 30分钟 |
| 1H/1h | 1 | 1小时 |
| 4H/4h | 4 | 4小时 |
| 1D/1d | 24 | 1天 |

## 📈 测试结果

### 测试场景覆盖

#### 1. 不同间隔测试
```
目标间隔: 2h  -> 偏移: 1条K线, 实际间隔: 22.9h
目标间隔: 4h  -> 偏移: 1条K线, 实际间隔: 22.9h
目标间隔: 6h  -> 偏移: 3条K线, 实际间隔: 24.9h
目标间隔: 8h  -> 偏移: 5条K线, 实际间隔: 26.9h
目标间隔: 12h -> 偏移: 9条K线, 实际间隔: 30.9h
目标间隔: 24h -> 偏移: 2条K线, 实际间隔: 23.9h
```

#### 2. 边界情况测试
- ✅ 不存在计划ID：正确返回错误
- ✅ 无预测数据：使用默认偏移
- ✅ 无训练记录：正确返回错误
- ✅ K线间隔转换：支持所有常用格式

## 🔍 监控和日志

### 关键日志信息

#### 偏移计算日志
```
INFO - 偏移计算: 时间差=4.12h, 目标=4h, K线间隔=1h, 需要回退=0.12h, 偏移K线数=0
INFO - 计算完成: 偏移说明: 自动触发：距离上次预测4.12小时，满足4小时目标间隔
```

#### 参数更新日志
```
INFO - 更新推理参数: plan_id=2, training_id=34, old_offset=0, new_offset=1
```

#### 任务执行日志
```
INFO - 定时预测已启动: plan_id=2, inference_id=auto_2_20251128_180722, data_offset=1
```

### 监控指标

#### 1. 预测间隔准确性
- **目标**: 实际间隔与目标间隔误差 < 10%
- **监控**: 定期检查间隔分布

#### 2. 数据连续性
- **目标**: 无数据缺口和重复
- **监控**: 检查时间戳连续性

#### 3. 系统稳定性
- **目标**: 计算成功率 > 99%
- **监控**: 记录计算失败和异常

## 🛠️ 故障排除

### 常见问题

#### 1. 时间差为负值
**原因**: 时区转换问题
**解决**: 系统自动使用绝对值，记录警告日志

#### 2. 偏移量过大
**原因**: 预测数据异常或间隔配置错误
**解决**: 检查预测数据表和计划配置

#### 3. K线间隔无法识别
**原因**: 不支持的间隔格式
**解决**: 检查K线数据表中的interval字段

### 调试方法

#### 1. 查看预测状态摘要
```python
from services.inference_data_offset_service import inference_data_offset_service
summary = inference_data_offset_service.get_prediction_status_summary(plan_id=2)
print(summary)
```

#### 2. 手动测试偏移计算
```python
result = inference_data_offset_service.calculate_optimal_data_offset(
    plan_id=2,
    target_interval_hours=4,
    manual_trigger=True
)
print(result)
```

#### 3. 检查推理参数
```sql
SELECT id, plan_name, finetune_params
FROM trading_plans
WHERE id = 2;
```

## 📋 最佳实践

### 1. 间隔时间选择
- **短周期交易**: 2h, 4h（高频预测）
- **中长周期**: 6h, 8h（平衡频率和性能）
- **长期投资**: 12h, 24h（稳定预测）

### 2. 数据质量管理
- 确保K线数据完整性和准确性
- 定期检查数据缺口
- 维护历史数据备份

### 3. 参数调优
- 根据交易策略调整采样数量
- 优化温度参数以平衡创造性和稳定性
- 调整预测窗口以匹配交易周期

---

**更新日期**: 2025-11-28
**版本**: v1.0.0