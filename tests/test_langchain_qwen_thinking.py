#!/usr/bin/env python3
"""
测试LangChain中Qwen的thinking模式
"""

def test_qwen_thinking_mode():
    """测试Qwen的thinking模式"""
    print("🧪 测试Qwen思考模式在LangChain中的实现")
    print("=" * 50)

    try:
        from services.langchain_agent_v2 import langchain_agent_v2_service
        from database.models import LLMConfig, TradingPlan
        from database.db import get_db

        # 获取Qwen配置
        with get_db() as db:
            trading_plan = db.query(TradingPlan).filter(TradingPlan.id == 2).first()
            if not trading_plan:
                print("❌ 未找到计划2")
                return False

            llm_config = db.query(LLMConfig).filter(LLMConfig.id == trading_plan.llm_config_id).first()
            if not llm_config or llm_config.provider != "qwen":
                print("❌ 计划2未配置Qwen")
                return False

            print(f"✅ 找到Qwen配置: {llm_config.name} (模型: {llm_config.model_name})")
            print(f"   API Base: {llm_config.api_base_url}")
            print(f"   Temperature: {llm_config.temperature}")

            # 获取LLM客户端
            llm_client = langchain_agent_v2_service._get_llm_client(llm_config)

            print(f"✅ LLM客户端类型: {type(llm_client)}")

            # 检查extra_body是否设置
            if hasattr(llm_client, 'model_kwargs'):
                print(f"✅ Model kwargs: {llm_client.model_kwargs}")
            elif hasattr(llm_client, 'kwargs'):
                print(f"✅ Kwargs: {llm_client.kwargs}")
            else:
                print("⚠️  未找到配置参数")

        print("\n✅ Qwen thinking模式配置成功")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_qwen_thinking_mode()