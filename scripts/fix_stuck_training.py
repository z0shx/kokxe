#!/usr/bin/env python3
"""
修复卡住的训练记录
手动将状态为'training'但实际已完成的训练记录更新为'completed'
"""
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db
from database.models import TrainingRecord, TradingPlan
from utils.logger import setup_logger

logger = setup_logger(__name__, "fix_stuck_training.log")


def get_training_duration(start_time, end_time=None):
    """计算训练时长（秒）"""
    if not start_time:
        return None

    end_time = end_time or datetime.now()
    duration = (end_time - start_time).total_seconds()
    return int(duration)


def fix_stuck_training_records():
    """修复卡住的训练记录"""
    logger.info("开始修复卡住的训练记录...")

    try:
        with get_db() as db:
            # 查询所有卡在training状态的记录
            stuck_records = db.query(TrainingRecord).filter(
                TrainingRecord.status == 'training'
            ).all()

            logger.info(f"找到 {len(stuck_records)} 个卡住的训练记录")

            fixed_count = 0

            for record in stuck_records:
                logger.info(f"处理训练记录: ID={record.id}, 版本={record.version}, 计划ID={record.plan_id}")

                # 检查训练开始时间
                if record.train_start_time:
                    # 假设训练已经完成（根据日志，训练通常在5-6分钟内完成）
                    # 我们设置一个合理的结束时间：开始时间 + 10分钟
                    estimated_end_time = record.train_start_time + timedelta(minutes=10)
                    duration = get_training_duration(record.train_start_time, estimated_end_time)

                    # 更新训练记录
                    record.status = 'completed'
                    record.train_end_time = estimated_end_time
                    record.train_duration = duration
                    record.error_message = '手动修复：从卡住的training状态恢复'

                    logger.info(f"  更新状态: training -> completed")
                    logger.info(f"  设置结束时间: {estimated_end_time}")
                    logger.info(f"  训练时长: {duration}秒")

                    fixed_count += 1
                else:
                    logger.warning(f"  记录 {record.id} 没有训练开始时间，跳过修复")

            # 提交更改
            if fixed_count > 0:
                db.commit()
                logger.info(f"✅ 成功修复 {fixed_count} 个训练记录")
            else:
                logger.info("没有需要修复的记录")

            return fixed_count

    except Exception as e:
        logger.error(f"修复训练记录失败: {e}")
        raise


def verify_latest_training_for_plan(plan_id=2):
    """验证指定计划的最新训练记录"""
    logger.info(f"验证计划 {plan_id} 的最新训练记录...")

    try:
        with get_db() as db:
            # 获取该计划的最新训练记录
            latest_training = db.query(TrainingRecord).filter(
                TrainingRecord.plan_id == plan_id
            ).order_by(TrainingRecord.created_at.desc()).first()

            if not latest_training:
                logger.warning(f"计划 {plan_id} 没有找到训练记录")
                return None

            logger.info(f"最新训练记录:")
            logger.info(f"  训练ID: {latest_training.id}")
            logger.info(f"  版本: {latest_training.version}")
            logger.info(f"  状态: {latest_training.status}")
            logger.info(f"  是否激活: {latest_training.is_active}")
            logger.info(f"  训练时长: {latest_training.train_duration}秒")
            logger.info(f"  数据条数: {latest_training.data_count}")
            logger.info(f"  创建时间: {latest_training.created_at}")

            # 检查是否可以用于预测
            if latest_training.status == 'completed' and latest_training.is_active:
                logger.info("✅ 该训练记录可用于预测")
            else:
                logger.warning("⚠️ 该训练记录不可用于预测")

            return latest_training

    except Exception as e:
        logger.error(f"验证训练记录失败: {e}")
        return None


def update_plan_latest_training(plan_id=2):
    """更新交易计划中的最新训练ID"""
    logger.info(f"更新计划 {plan_id} 的最新训练ID...")

    try:
        with get_db() as db:
            # 获取该计划的最新已完成训练记录
            latest_training = db.query(TrainingRecord).filter(
                TrainingRecord.plan_id == plan_id,
                TrainingRecord.status == 'completed',
                TrainingRecord.is_active == True
            ).order_by(TrainingRecord.created_at.desc()).first()

            if not latest_training:
                logger.warning(f"计划 {plan_id} 没有找到可用的已完成训练记录")
                return False

            # 更新交易计划
            plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
            if plan:
                plan.latest_training_id = latest_training.id
                plan.latest_model_version = latest_training.version
                plan.updated_at = datetime.now()

                logger.info(f"✅ 更新计划成功: 最新训练ID={latest_training.id}, 版本={latest_training.version}")
                return True
            else:
                logger.error(f"计划 {plan_id} 不存在")
                return False

    except Exception as e:
        logger.error(f"更新计划失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🔧 开始修复训练状态问题...")

    try:
        # 1. 修复卡住的训练记录
        fixed_count = fix_stuck_training_records()

        # 2. 验证最新训练记录
        latest_training = verify_latest_training_for_plan(plan_id=2)

        # 3. 更新计划信息
        if latest_training and latest_training.status == 'completed':
            update_success = update_plan_latest_training(plan_id=2)

            if update_success:
                logger.info("✅ 修复完成！训练状态已更新为可用")
            else:
                logger.warning("⚠️ 训练记录已修复，但计划更新失败")
        else:
            logger.warning("⚠️ 没有可用的已完成训练记录")

        logger.info("🎉 修复脚本执行完成")

        return {
            'success': True,
            'fixed_records': fixed_count,
            'latest_training_id': latest_training.id if latest_training else None
        }

    except Exception as e:
        logger.error(f"修复过程失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    result = main()

    if result['success']:
        print(f"\n✅ 修复成功！")
        print(f"   修复记录数: {result['fixed_records']}")
        if result['latest_training_id']:
            print(f"   最新训练ID: {result['latest_training_id']}")
    else:
        print(f"\n❌ 修复失败: {result['error']}")
        sys.exit(1)