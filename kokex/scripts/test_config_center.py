#!/usr/bin/env python
"""
测试配置中心功能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.config_service import ConfigService

print("=" * 60)
print("测试配置中心功能")
print("=" * 60)

# 测试 1: 获取所有 LLM 配置
print("\n1. 获取所有 LLM 配置...")
try:
    configs = ConfigService.get_all_llm_configs(active_only=False)
    print(f"   ✅ 成功获取 {len(configs)} 条 LLM 配置")
    for config in configs:
        print(f"      - ID: {config.id}, 名称: {config.name}, 提供商: {config.provider}")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 2: 获取所有 Agent 提示词模版
print("\n2. 获取所有 Agent 提示词模版...")
try:
    templates = ConfigService.get_all_prompt_templates(active_only=False)
    print(f"   ✅ 成功获取 {len(templates)} 条提示词模版")
    for template in templates:
        print(f"      - ID: {template.id}, 名称: {template.name}, 分类: {template.category}")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 创建测试 LLM 配置
print("\n3. 创建测试 LLM 配置...")
try:
    config_id = ConfigService.create_llm_config(
        name="测试 Claude 配置",
        provider="claude",
        model_name="claude-3-5-sonnet-20241022",
        api_key="test-api-key",
        is_default=True
    )
    if config_id:
        print(f"   ✅ 创建成功，配置 ID: {config_id}")
    else:
        print("   ⚠️  创建失败（可能已存在）")
except Exception as e:
    print(f"   ⚠️  创建失败: {e}")

# 测试 4: 获取默认配置
print("\n4. 获取默认 LLM 配置...")
try:
    default_config = ConfigService.get_default_llm_config()
    if default_config:
        print(f"   ✅ 找到默认配置: {default_config.name} ({default_config.provider})")
    else:
        print("   ⚠️  未设置默认配置")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试 5: 获取默认模版
print("\n5. 获取默认 Agent 提示词模版...")
try:
    default_template = ConfigService.get_default_prompt_template()
    if default_template:
        print(f"   ✅ 找到默认模版: {default_template.name}")
        print(f"      内容长度: {len(default_template.content)} 字符")
    else:
        print("   ⚠️  未设置默认模版")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print("\n" + "=" * 60)
print("✅ 测试完成！")
print("=" * 60)
