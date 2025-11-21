# KOKEX - AI Crypto Trading Platform

基于 Kronos 时序预测模型的全自动加密货币交易平台

## ✨ 核心功能

- 🤖 **Kronos 模型训练**: 真实的时序预测模型训练和推理
- ⏰ **定时任务调度**: 自动化训练、推理、决策流程
- 🧠 **AI Agent 决策**: 支持 Claude、OpenAI、Qwen 等多种 LLM
- 💹 **OKX 交易集成**: 完整的交易工具（下单、修改、取消）
- 📊 **Web UI**: 直观的管理和监控界面
- 🔄 **完全自动化**: 从数据同步到交易执行的全流程自动化

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/zshx/code/kronos/kokex
pip install -r requirements.txt
```

### 2. 配置环境

创建 `.env` 文件:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/kokex
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

### 3. 启动应用

```bash
python app.py
```

### 4. 访问界面

打开浏览器访问: `http://localhost:7860`

## 📖 文档

- **[USER_GUIDE.md](USER_GUIDE.md)**: 详细使用指南和配置说明
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: 项目完整实现总结
- **[FULL_IMPLEMENTATION_SUMMARY.md](FULL_IMPLEMENTATION_SUMMARY.md)**: 技术实现详解

## 🎯 核心流程

```
定时器触发 → 训练模型 → 生成预测 → AI决策 → 执行交易
```

## 🛠️ 技术栈

- **Python 3.8+**
- **PyTorch**: 深度学习
- **PostgreSQL**: 数据库
- **Gradio**: Web UI
- **APScheduler**: 定时任务
- **Anthropic/OpenAI**: LLM API
- **OKX API**: 交易所集成

## 📊 系统架构

```
┌─────────────────────────────────────────────────┐
│                   Web UI (Gradio)               │
├─────────────────────────────────────────────────┤
│  Schedule Service  │  Training Service          │
│  Agent Service     │  Inference Service         │
│  Trading Tools     │  WebSocket Service         │
├─────────────────────────────────────────────────┤
│  Kronos Trainer    │  Database Dataset          │
│  KronosInferencer  │  OKX API Client            │
├─────────────────────────────────────────────────┤
│         PostgreSQL Database + LLM API           │
└─────────────────────────────────────────────────┘
```

## 🔐 安全性

- ✅ API Key 加密存储
- ✅ 模拟盘/实盘隔离
- ✅ 交易限制检查
- ✅ 完整的日志追踪

## ⚙️ 主要配置

### 训练参数
- Lookback Window: 512
- Predict Window: 48
- Batch Size: 16
- Tokenizer Epochs: 25
- Predictor Epochs: 50

### 定时任务
支持配置多个时间点，如:
- 00:00 (每天凌晨)
- 12:00 (每天中午)
- 18:00 (每天下午)

### LLM 支持
- Claude (Anthropic)
- GPT (OpenAI)
- Qwen (阿里云)

## 📝 使用示例

### 1. 创建交易计划

在 "新增计划" 页面配置:
- 交易对: ETH-USDT
- 时间粒度: 1H
- 自动微调时间: 00:00, 12:00
- LLM: Claude
- 环境: 模拟盘

### 2. 启动计划

在 "计划详情" 页面:
1. 点击 "启动WebSocket" 开始数据同步
2. 点击 "启动计划" 开始自动化流程

### 3. 监控运行

查看:
- 训练记录和指标
- K线图和预测数据
- Agent 决策历史
- 交易执行结果

## 🧪 测试

运行完整流程测试:

```bash
python test_full_workflow.py <plan_id>
```

## 📊 性能

- **训练时间**: 30-90分钟 (取决于硬件)
- **推理时间**: 1-5秒
- **决策时间**: 2-10秒
- **资源需求**: 16GB RAM + GPU (推荐)

## 🔮 特性

### 已实现 ✅
- [x] Kronos 模型训练和推理
- [x] 定时任务调度
- [x] LLM 集成 (Claude/OpenAI/Qwen)
- [x] OKX 交易工具
- [x] Web UI 管理界面
- [x] WebSocket 实时数据
- [x] 数据库持久化
- [x] 完整日志系统

### 计划中 🚧
- [ ] 多GPU训练
- [ ] 回测系统
- [ ] 更多交易所支持
- [ ] 移动端适配

## 📜 许可证

基于 Kronos 项目，遵循相同的许可证条款。

## 🙏 致谢

- Kronos - 优秀的时序预测模型
- OKX - 交易所 API
- Anthropic/OpenAI - LLM API

## 📮 支持

如有问题，请查看:
1. [USER_GUIDE.md](USER_GUIDE.md) - 使用指南
2. `logs/` 目录 - 日志文件
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 项目总结

---

**版本**: v3.0
**状态**: 生产就绪 ✅
**更新**: 2025-11-17
