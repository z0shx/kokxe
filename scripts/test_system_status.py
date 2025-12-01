#!/usr/bin/env python3
"""
系统状态综合测试脚本
测试所有核心功能是否正常工作
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db
from database.models import TrainingRecord, TradingPlan
from services.agent_tool_executor import AgentToolExecutor
from ui.plan_detail import PlanDetailUI
from config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__, "system_status_test.log")


def test_database_connection():
    """测试数据库连接"""
    logger.info("🔗 测试数据库连接...")

    try:
        with get_db() as db:
            # 简单查询测试连接
            count = db.query(TradingPlan).count()
            logger.info(f"✅ 数据库连接正常，交易计划数量: {count}")
            return True

    except Exception as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        return False


def test_training_status():
    """测试训练状态"""
    logger.info("🧠 测试训练状态...")

    try:
        with get_db() as db:
            # 查询最新训练记录
            latest_training = db.query(TrainingRecord).filter(
                TrainingRecord.plan_id == 2,
                TrainingRecord.status == 'completed',
                TrainingRecord.is_active == True
            ).order_by(TrainingRecord.created_at.desc()).first()

            if latest_training:
                logger.info(f"✅ 找到最新训练记录:")
                logger.info(f"   训练ID: {latest_training.id}")
                logger.info(f"   版本: {latest_training.version}")
                logger.info(f"   状态: {latest_training.status}")
                logger.info(f"   训练时长: {latest_training.train_duration}秒")
                logger.info(f"   Tokenizer路径: {latest_training.tokenizer_path}")
                logger.info(f"   Predictor路径: {latest_training.predictor_path}")

                # 验证模型文件存在
                import os
                tokenizer_exists = os.path.exists(latest_training.tokenizer_path) if latest_training.tokenizer_path else False
                predictor_exists = os.path.exists(latest_training.predictor_path) if latest_training.predictor_path else False

                if tokenizer_exists and predictor_exists:
                    logger.info("✅ 模型文件验证通过")
                    return True
                else:
                    logger.warning("⚠️ 模型文件不存在")
                    return False
            else:
                logger.error("❌ 没有找到可用的训练记录")
                return False

    except Exception as e:
        logger.error(f"❌ 训练状态检查失败: {e}")
        return False


def test_ui_components():
    """测试UI组件"""
    logger.info("🖥️ 测试UI组件...")

    try:
        # 测试PlanDetailUI
        ui = PlanDetailUI()
        messages = ui.get_latest_conversation_messages(2)
        logger.info(f"✅ PlanDetailUI正常，对话消息数量: {len(messages)}")

        return True

    except Exception as e:
        logger.error(f"❌ UI组件测试失败: {e}")
        return False


def test_tool_executor():
    """测试工具执行器"""
    logger.info("🛠️ 测试工具执行器...")

    try:
        import asyncio

        async def test_tool():
            config = Config()
            executor = AgentToolExecutor(
                api_key=config.OKX_API_KEY,
                secret_key=config.OKX_SECRET_KEY,
                passphrase=config.OKX_PASSPHRASE,
                is_demo=True,
                plan_id=2,
                conversation_id=None
            )

            try:
                result = await executor.execute_tool("get_prediction_history", {"limit": 1})
                success = result.get('success', False)
                logger.info(f"✅ 工具执行器测试: {'成功' if success else '失败'}")
                if 'error' in result:
                    logger.info(f"   错误信息: {result['error']}")
                return success
            finally:
                await executor.close()

        return asyncio.run(test_tool())

    except Exception as e:
        logger.error(f"❌ 工具执行器测试失败: {e}")
        return False


def test_conversation_types():
    """测试对话类型枚举"""
    logger.info("💬 测试对话类型枚举...")

    try:
        from services.langchain_agent_v2 import ConversationType

        # 测试枚举值
        auto_value = ConversationType.AUTO_INFERENCE.value
        manual_value = ConversationType.MANUAL_CHAT.value

        logger.info(f"✅ 对话类型枚举正常:")
        logger.info(f"   AUTO_INFERENCE: {auto_value}")
        logger.info(f"   MANUAL_CHAT: {manual_value}")

        return auto_value == "auto_inference" and manual_value == "manual_chat"

    except Exception as e:
        logger.error(f"❌ 对话类型枚举测试失败: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始系统状态综合测试...")
    logger.info("=" * 60)

    tests = [
        ("数据库连接", test_database_connection),
        ("训练状态", test_training_status),
        ("UI组件", test_ui_components),
        ("工具执行器", test_tool_executor),
        ("对话类型枚举", test_conversation_types)
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} 测试出现异常: {e}")
            results[test_name] = False

    logger.info("=" * 60)
    logger.info("🏁 系统状态测试完成！")
    logger.info(f"总测试: {total}, 通过: {passed}, 失败: {total - passed}")
    logger.info(f"成功率: {(passed / total * 100):.1f}%")

    # 详细结果
    logger.info("\n详细结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")

    return {
        'success': passed == total,
        'passed': passed,
        'total': total,
        'success_rate': passed / total * 100,
        'detailed_results': results
    }


if __name__ == "__main__":
    result = main()

    if result['success']:
        print(f"\n🎉 系统状态测试完全通过！")
        print(f"   成功率: {result['success_rate']:.1f}%")
        print("   所有核心功能都正常工作！")
    else:
        print(f"\n⚠️ 系统测试发现问题:")
        print(f"   成功率: {result['success_rate']:.1f}%")
        failed_tests = [name for name, result in result['detailed_results'].items() if not result]
        print(f"   失败的测试: {', '.join(failed_tests)}")
        sys.exit(1)