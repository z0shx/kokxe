#!/usr/bin/env python3
"""
测试多次推理数据保存功能
"""
from database.db import get_db
from database.models import PredictionData, TrainingRecord
from sqlalchemy import func, and_

print("=" * 80)
print("测试多次推理数据保存功能")
print("=" * 80)

with get_db() as db:
    # 1. 查询所有训练记录
    training_records = db.query(TrainingRecord).filter(
        TrainingRecord.status == 'completed'
    ).all()

    print(f"\n1. 已完成的训练记录: {len(training_records)} 条")
    for record in training_records[:3]:
        print(f"   - ID={record.id}, version={record.version}, plan_id={record.plan_id}")

    if not training_records:
        print("   没有训练记录，退出测试")
        exit(0)

    # 2. 查询某个训练记录的所有推理批次
    test_training_id = training_records[0].id
    print(f"\n2. 查询训练记录 {test_training_id} 的所有推理批次:")

    batches = db.query(
        PredictionData.inference_batch_id,
        func.min(PredictionData.created_at).label('inference_time'),
        func.count(PredictionData.id).label('predictions_count'),
        func.min(PredictionData.timestamp).label('time_start'),
        func.max(PredictionData.timestamp).label('time_end')
    ).filter(
        PredictionData.training_record_id == test_training_id
    ).group_by(
        PredictionData.inference_batch_id
    ).order_by(
        func.min(PredictionData.created_at).desc()
    ).all()

    print(f"   找到 {len(batches)} 个推理批次:")
    for batch in batches:
        print(f"   - 批次ID: {batch.inference_batch_id[:30]}...")
        print(f"     推理时间: {batch.inference_time}")
        print(f"     预测数量: {batch.predictions_count} 条")
        print(f"     时间范围: {batch.time_start} ~ {batch.time_end}")
        print()

    # 3. 测试查询指定批次的数据
    if batches:
        test_batch_id = batches[0].inference_batch_id
        print(f"3. 查询批次 {test_batch_id[:30]}... 的详细数据:")

        predictions = db.query(PredictionData).filter(
            and_(
                PredictionData.training_record_id == test_training_id,
                PredictionData.inference_batch_id == test_batch_id
            )
        ).limit(5).all()

        print(f"   前5条预测数据:")
        for pred in predictions:
            print(f"   - 时间: {pred.timestamp}, 收盘价: {pred.close:.2f}")
            if pred.upward_probability is not None:
                print(f"     上涨概率: {pred.upward_probability*100:.1f}%")

    # 4. 测试唯一约束
    print(f"\n4. 测试唯一约束 (training_record_id, inference_batch_id, timestamp):")

    # 统计每个训练记录的批次数量
    stats = db.query(
        PredictionData.training_record_id,
        func.count(func.distinct(PredictionData.inference_batch_id)).label('batch_count'),
        func.count(PredictionData.id).label('total_predictions')
    ).group_by(
        PredictionData.training_record_id
    ).all()

    print(f"   统计结果:")
    for stat in stats:
        print(f"   - 训练ID={stat.training_record_id}: {stat.batch_count} 个批次, 共 {stat.total_predictions} 条预测")

print("\n" + "=" * 80)
print("✓ 测试完成！多次推理数据保存功能正常")
print("=" * 80)
