# 配置中心功能更新

## 概述

本次更新将配置管理从 `.env` 文件迁移到数据库，实现了：

1. **LLM 配置管理**：支持 Claude、Qwen、Ollama、兼容 OpenAI 的自定义接口
2. **Agent 提示词模版管理**：可复用的交易策略提示词模版
3. **新建计划集成**：在创建交易计划时可选择 LLM 配置和提示词模版

## 数据库迁移

### 运行迁移脚本

执行以下命令应用数据库更改：

```bash
psql -U <username> -d <database_name> -f kokex/sql/migrations/001_add_llm_and_prompt_templates.sql
```

或者使用环境变量：

```bash
psql $DATABASE_URL -f kokex/sql/migrations/001_add_llm_and_prompt_templates.sql
```

### 迁移内容

该迁移脚本会：

1. 创建 `llm_configs` 表 - 存储 LLM 配置
2. 创建 `agent_prompt_templates` 表 - 存储 Agent 提示词模版
3. 在 `trading_plans` 表中添加 `llm_config_id` 字段
4. 插入默认的 Agent 提示词模版
5. 创建自动更新 `updated_at` 字段的触发器

## 使用指南

### 1. 配置中心

访问 Web UI 的 **配置中心** 选项卡，分为两个子选项卡：

#### LLM 配置

- **创建配置**：
  - 配置名称：例如 "Claude Sonnet 3.5"
  - LLM 提供商：claude, qwen, ollama, openai
  - API Key：你的 API 密钥
  - API Base URL：可选，自定义接口地址
  - 模型名称：例如 "claude-3-5-sonnet-20241022"
  - 温度/Top P：生成参数
  - 设为默认配置：勾选后作为默认选项

- **管理配置**：
  - 查看所有配置列表
  - 删除不需要的配置

#### Agent 提示词模版

- **创建模版**：
  - 模版名称：例如 "保守型策略"
  - 分类：conservative（保守）、aggressive（激进）、balanced（平衡）、custom（自定义）
  - 模版描述：简要说明
  - 提示词内容：完整的 Agent 提示词
  - 设为默认模版：勾选后作为默认选项

- **管理模版**：
  - 查看所有模版列表
  - 删除不需要的模版

### 2. 新建计划

在创建交易计划时，可以：

1. **选择 LLM 配置**：
   - 点击 "🔄 刷新 LLM 配置列表" 加载最新配置
   - 从下拉框选择 LLM 配置
   - 如果设置了默认配置，会自动选中

2. **使用提示词模版**：
   - 点击 "🔄 刷新模版列表" 加载最新模版
   - 从下拉框选择提示词模版
   - 选择后会自动填充到 "Agent 提示词内容" 文本框
   - 可以在自动填充后继续编辑

## 代码结构

### 新增文件

- `kokex/services/config_service.py` - 配置管理服务
- `kokex/sql/migrations/001_add_llm_and_prompt_templates.sql` - 数据库迁移脚本

### 修改文件

- `kokex/database/models.py` - 添加 `LLMConfig` 和 `AgentPromptTemplate` 模型
- `kokex/services/plan_service.py` - 添加 `llm_config_id` 参数
- `kokex/ui/config_center.py` - 完全重写，实现配置管理界面
- `kokex/ui/plan_create.py` - 添加 LLM 和模版选择功能
- `kokex/sql/schema.sql` - 更新完整数据库架构

## API 参考

### ConfigService

#### LLM 配置管理

```python
from services.config_service import ConfigService

# 创建 LLM 配置
config_id = ConfigService.create_llm_config(
    name="Claude Sonnet 3.5",
    provider="claude",
    api_key="sk-...",
    model_name="claude-3-5-sonnet-20241022",
    is_default=True
)

# 获取所有配置
configs = ConfigService.get_all_llm_configs(active_only=True)

# 获取默认配置
default_config = ConfigService.get_default_llm_config()

# 更新配置
ConfigService.update_llm_config(config_id, temperature=0.8)

# 删除配置
ConfigService.delete_llm_config(config_id)
```

#### Agent 提示词模版管理

```python
# 创建提示词模版
template_id = ConfigService.create_prompt_template(
    name="保守型策略",
    content="你是一个保守的交易员...",
    category="conservative",
    is_default=True
)

# 获取所有模版
templates = ConfigService.get_all_prompt_templates(active_only=True)

# 获取默认模版
default_template = ConfigService.get_default_prompt_template()

# 更新模版
ConfigService.update_prompt_template(template_id, content="新的提示词内容")

# 删除模版
ConfigService.delete_prompt_template(template_id)
```

## 注意事项

1. **API Key 安全**：
   - API Key 存储在数据库中，建议使用数据库加密功能
   - 在生产环境中考虑使用密钥管理服务（KMS）

2. **默认配置**：
   - 每个提供商只能有一个默认 LLM 配置
   - 整个系统只能有一个默认 Agent 提示词模版
   - 设置新的默认配置会自动取消之前的默认配置

3. **兼容性**：
   - 旧的交易计划（没有 `llm_config_id` 的）仍然可以正常运行
   - 建议逐步为所有计划配置 LLM

## 故障排除

### 迁移失败

如果迁移脚本执行失败，可以手动检查：

```bash
# 检查表是否存在
psql $DATABASE_URL -c "\dt llm_configs"
psql $DATABASE_URL -c "\dt agent_prompt_templates"

# 检查字段是否添加
psql $DATABASE_URL -c "\d trading_plans"
```

### UI 无法加载配置

1. 确认数据库迁移已成功执行
2. 检查日志文件 `kokex/logs/config_service.log`
3. 确认 ConfigService 正确导入

### 创建计划时找不到 LLM 配置

1. 先在配置中心创建至少一个 LLM 配置
2. 点击 "🔄 刷新 LLM 配置列表" 按钮
3. 确认配置状态为 "启用"

## 未来改进

- [ ] API Key 加密存储
- [ ] 配置版本控制
- [ ] 提示词模版的导入/导出功能
- [ ] 批量管理功能
- [ ] 配置审计日志

## 更新日志

### v1.1.0 (2025-11-14)

- ✨ 新增 LLM 配置管理功能
- ✨ 新增 Agent 提示词模版管理功能
- ♻️ 配置中心从 .env 迁移到数据库
- 🔧 新建计划支持选择 LLM 和提示词模版
