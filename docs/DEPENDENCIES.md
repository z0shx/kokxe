# Kokex 依赖说明

本文档说明 Kokex 项目的依赖包及其用途。

## 依赖包分类

### 1. Kronos 核心依赖（推理和微调必需）

这些依赖来自 Kronos 主项目，是运行 Kronos 模型推理和微调所必需的：

| 包名 | 版本 | 用途 |
|------|------|------|
| `numpy` | latest | 数值计算基础库 |
| `pandas` | latest | 数据处理（K线数据加载、处理） |
| `torch` | latest | PyTorch 深度学习框架 |
| `einops` | 0.8.1 | 张量操作（用于 model/module.py） |
| `huggingface_hub` | 0.33.1 | 模型下载和管理（从 Hugging Face Hub） |
| `matplotlib` | 3.9.3 | 数据可视化 |
| `tqdm` | 4.67.1 | 进度条显示 |
| `safetensors` | 0.6.2 | 模型序列化和加载 |

**代码位置**：
- `model/kronos.py`: KronosTokenizer, Kronos, KronosPredictor
- `model/module.py`: BinarySphericalQuantizer, TransformerBlock
- `kokex/services/model_service.py`: 模型下载和管理

### 2. 微调配置相关依赖

| 包名 | 用途 |
|------|------|
| `pyyaml` | 读取 YAML 配置文件（finetune_csv/config_loader.py） |

### 3. Kokex 应用依赖

#### Web UI
| 包名 | 版本 | 用途 |
|------|------|------|
| `gradio` | ≥5.4.0 | Web UI 框架 |

#### 数据库
| 包名 | 版本 | 用途 |
|------|------|------|
| `psycopg2-binary` | latest | PostgreSQL 数据库驱动 |
| `sqlalchemy` | ≥2.0.0 | ORM 框架 |
| `alembic` | latest | 数据库迁移工具 |

#### 配置和环境
| 包名 | 用途 |
|------|------|
| `python-dotenv` | 环境变量加载（.env 文件） |

#### HTTP/WebSocket 客户端
| 包名 | 用途 |
|------|------|
| `requests` | HTTP 请求（OKX API） |
| `websocket-client` | WebSocket 客户端 |
| `websockets` | WebSocket 服务器/客户端 |
| `aiohttp` | 异步 HTTP 客户端 |

#### 数据验证
| 包名 | 版本 | 用途 |
|------|------|------|
| `pydantic` | ≥2.0.0 | 数据验证和设置管理 |

#### 可视化
| 包名 | 版本 | 用途 |
|------|------|------|
| `plotly` | ≥5.0.0 | 交互式图表 |

### 4. 可选依赖

这些依赖只在特定场景下需要：

| 包名 | 用途 | 使用场景 |
|------|------|----------|
| `pytest` | 单元测试 | 运行 tests/ 目录下的测试 |
| `pyqlib` | Qlib 量化平台 | 使用 finetune/ 目录中的 Qlib 微调功能 |
| `akshare` | 中国市场数据下载 | 运行 examples/prediction_cn_markets_day.py |

## 安装说明

### 基础安装
```bash
cd /home/zshx/code/kronos/kokex
pip install -r requirements.txt
```

### 可选依赖安装

#### 测试环境
```bash
pip install pytest
```

#### Qlib 微调（中国市场）
```bash
pip install pyqlib
```

#### 中国市场数据下载
```bash
pip install akshare
```

## 依赖来源分析

### 从 Kronos 主项目继承
- `requirements.txt` 中的核心依赖直接来自 `/home/zshx/code/kronos/requirements.txt`
- 这些依赖是运行 Kronos 模型所必需的

### 微调相关依赖
从以下文件的分析得出：
- `finetune_csv/finetune_tokenizer.py`: PyTorch 分布式训练
- `finetune_csv/finetune_base_model.py`: 模型微调
- `finetune_csv/config_loader.py`: YAML 配置加载（需要 pyyaml）
- `finetune/qlib_test.py`: Qlib 回测（需要 pyqlib）

### Kokex 应用依赖
从以下文件的分析得出：
- `kokex/services/model_service.py`: 模型管理
- `kokex/api/okx_client.py`: OKX API 客户端
- `kokex/database/`: 数据库操作
- `kokex/ui/`: Gradio UI 组件

## 注意事项

1. **PyTorch 安装**：建议根据您的 CUDA 版本安装对应的 PyTorch：
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **模型导入问题**：
   - 错误: `cannot import name 'Kronos' from 'model'`
   - 原因: `kokex/model/__init__.py` 为空
   - 解决: `kokex/services/model_service.py` 通过 `sys.path.append` 导入父目录的 model 包

3. **分布式训练**：
   - 如需使用多 GPU 微调，确保正确配置 `torch.distributed`
   - 使用 `torchrun` 启动分布式训练

4. **数据库**：
   - PostgreSQL 需要单独安装并配置
   - 使用 Alembic 管理数据库迁移

## 版本兼容性

- Python: 建议 ≥ 3.8
- PyTorch: 建议 ≥ 2.0
- CUDA: 根据您的 GPU 选择（如果使用 GPU）
