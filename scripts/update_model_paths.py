#!/usr/bin/env python3
"""
更新训练记录中的模型路径
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db
from database.models import TrainingRecord
from utils.logger import setup_logger

logger = setup_logger(__name__, "update_model_paths.log")


def update_model_paths():
    """更新训练记录中的模型路径"""
    logger.info("开始更新训练记录中的模型路径...")

    try:
        with get_db() as db:
            # 查询所有已完成但缺少模型路径的训练记录
            incomplete_records = db.query(TrainingRecord).filter(
                TrainingRecord.status == 'completed',
                TrainingRecord.is_active == True,
                TrainingRecord.tokenizer_path.is_(None)
            ).all()

            logger.info(f"找到 {len(incomplete_records)} 个需要更新的训练记录")

            updated_count = 0

            for record in incomplete_records:
                logger.info(f"处理训练记录: ID={record.id}, 版本={record.version}, 计划ID={record.plan_id}")

                # 构建预期的模型路径
                plan_id = record.plan_id
                version = record.version

                # 模型保存路径模式: models/plan_{plan_id}/v{version}
                base_path = f"models/plan_{plan_id}/v{version}"
                tokenizer_path = os.path.join(base_path, "tokenizer")
                predictor_path = os.path.join(base_path, "predictor")

                # 检查模型文件是否存在
                tokenizer_exists = os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "model.safetensors"))
                predictor_exists = os.path.exists(predictor_path) and os.path.exists(os.path.join(predictor_path, "model.safetensors"))

                if tokenizer_exists and predictor_exists:
                    # 更新训练记录
                    record.tokenizer_path = tokenizer_path
                    record.predictor_path = predictor_path

                    # 添加一些基本的训练指标
                    record.train_metrics = {
                        "tokenizer_loss": "completed",
                        "predictor_loss": "completed",
                        "updated_at": datetime.now().isoformat()
                    }

                    logger.info(f"  ✅ 更新模型路径:")
                    logger.info(f"     Tokenizer: {tokenizer_path}")
                    logger.info(f"     Predictor: {predictor_path}")

                    updated_count += 1
                else:
                    logger.warning(f"  ⚠️ 模型文件不完整:")
                    logger.warning(f"     Tokenizer存在: {tokenizer_exists}")
                    logger.warning(f"     Predictor存在: {predictor_exists}")

            # 提交更改
            if updated_count > 0:
                db.commit()
                logger.info(f"✅ 成功更新 {updated_count} 个训练记录的模型路径")
            else:
                logger.info("没有需要更新的记录")

            return updated_count

    except Exception as e:
        logger.error(f"更新模型路径失败: {e}")
        raise


def verify_model_paths():
    """验证模型路径更新结果"""
    logger.info("验证模型路径更新结果...")

    try:
        with get_db() as db:
            # 查询最新的训练记录
            latest_training = db.query(TrainingRecord).filter(
                TrainingRecord.plan_id == 2,
                TrainingRecord.status == 'completed',
                TrainingRecord.is_active == True
            ).order_by(TrainingRecord.created_at.desc()).first()

            if latest_training:
                logger.info(f"最新训练记录:")
                logger.info(f"  训练ID: {latest_training.id}")
                logger.info(f"  版本: {latest_training.version}")
                logger.info(f"  Tokenizer路径: {latest_training.tokenizer_path}")
                logger.info(f"  Predictor路径: {latest_training.predictor_path}")

                # 验证文件存在性
                if latest_training.tokenizer_path and latest_training.predictor_path:
                    tokenizer_file = os.path.join(latest_training.tokenizer_path, "model.safetensors")
                    predictor_file = os.path.join(latest_training.predictor_path, "model.safetensors")

                    tokenizer_exists = os.path.exists(tokenizer_file)
                    predictor_exists = os.path.exists(predictor_file)

                    logger.info(f"  Tokenizer文件存在: {tokenizer_exists}")
                    logger.info(f"  Predictor文件存在: {predictor_exists}")

                    if tokenizer_exists and predictor_exists:
                        logger.info("✅ 模型路径验证通过")
                        return True
                    else:
                        logger.error("❌ 模型文件不存在")
                        return False
                else:
                    logger.error("❌ 模型路径为空")
                    return False
            else:
                logger.error("❌ 没有找到有效的训练记录")
                return False

    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🔧 开始更新模型路径...")

    try:
        # 1. 更新模型路径
        updated_count = update_model_paths()

        # 2. 验证更新结果
        verification_passed = verify_model_paths()

        if verification_passed:
            logger.info("🎉 模型路径更新完成并验证通过")
        else:
            logger.warning("⚠️ 模型路径更新完成但验证失败")

        return {
            'success': verification_passed,
            'updated_records': updated_count
        }

    except Exception as e:
        logger.error(f"更新过程失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    result = main()

    if result['success']:
        print(f"\n✅ 模型路径更新成功！")
        print(f"   更新记录数: {result['updated_records']}")
        print("   验证状态: 通过")
    else:
        print(f"\n❌ 模型路径更新失败: {result.get('error', '验证失败')}")
        sys.exit(1)