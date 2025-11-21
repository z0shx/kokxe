# Kronos 项目依赖整理报告

## 概述

本报告分析了 `/home/zshx/code/kronos` 项目（除 kokex 子目录外）的依赖包，并整理到 `/home/zshx/code/kronos/kokex/requirements.txt`。

## 分析范围

### 分析的目录和文件
- `model/`: Kronos 核心模型代码
- `examples/`: 推理示例
- `finetune/`: Qlib 微调代码
- `finetune_csv/`: CSV 微调代码
- `tests/`: 测试代码

### 排除的目录
- `kokex/`: Kokex 应用代码（已有自己的依赖）

## 依赖对比

### 原 Kronos 项目 requirements.txt
```
numpy
pandas
torch

einops==0.8.1
huggingface_hub==0.33.1
matplotlib==3.9.3
pandas==2.2.2
tqdm==4.67.1
safetensors==0.6.2
```

**问题**:
1. numpy 和 pandas 重复声明（一次无版本，一次有版本）
2. 缺少微调配置相关的依赖（如 pyyaml）
3. 缺少可选依赖的说明

### 更新后的 kokex/requirements.txt

#### 结构改进
1. **分类清晰**: 将依赖分为 4 大类
   - Kronos 核心依赖
   - 微调配置相关依赖
   - Kokex 应用依赖
   - 可选依赖

2. **添加注释**: 每个依赖都有用途说明

3. **版本管理**: 保持与主项目一致的版本号

#### 新增依赖

**微调配置相关**:
- `pyyaml`: 用于读取 finetune_csv 的 YAML 配置文件

**Kokex 应用相关**:
- `gradio`: Web UI
- `psycopg2-binary`: PostgreSQL 驱动
- `sqlalchemy`: ORM
- `alembic`: 数据库迁移
- `python-dotenv`: 环境变量
- `requests`: HTTP 客户端
- `websocket-client`: WebSocket 客户端
- `websockets`: WebSocket 服务器
- `aiohttp`: 异步 HTTP
- `pydantic`: 数据验证
- `plotly`: 交互式图表

**可选依赖** (注释):
- `pytest`: 测试框架
- `pyqlib`: Qlib 量化平台
- `akshare`: 中国市场数据

## 代码依赖分析

### 核心模型依赖
来自 `model/kronos.py` 和 `model/module.py`:
```python
import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
from tqdm import trange
from einops import rearrange, reduce
from safetensors import safe_open
```

### 推理示例依赖
来自 `examples/prediction_example.py`:
```python
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor
```

### 微调依赖
来自 `finetune_csv/config_loader.py`:
```python
import yaml  # 需要 pyyaml
from typing import Dict, Any
```

来自 `finetune_csv/finetune_tokenizer.py`:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from logging.handlers import RotatingFileHandler
```

### 测试依赖
来自 `tests/test_kronos_regression.py`:
```python
import pytest
from pathlib import Path
```

### 可选依赖
来自 `finetune/qlib_test.py`:
```python
import qlib  # 需要 pyqlib
from qlib.config import REG_CN
from qlib.backtest import backtest, executor
```

来自 `examples/prediction_cn_markets_day.py`:
```python
import akshare as ak  # 需要 akshare
```

## 关键发现

### 1. 导入路径问题
**问题**: `kokex/services/model_service.py` 出现错误：
```
cannot import name 'Kronos' from 'model'
```

**原因**:
- `kokex/model/__init__.py` 为空
- `kokex/services/model_service.py` 尝试通过 `sys.path.append` 导入父目录的 model

**解决方案**:
```python
# kokex/services/model_service.py:12
sys.path.append(str(Path(__file__).parent.parent.parent))
from model import Kronos, KronosTokenizer  # 导入顶层的 model 包
```

**建议**:
1. 保持 `kokex/model/__init__.py` 为空（避免命名冲突）
2. 或者在 `kokex/model/__init__.py` 中添加：
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent))
   from model.kronos import Kronos, KronosTokenizer, KronosPredictor
   __all__ = ['Kronos', 'KronosTokenizer', 'KronosPredictor']
   ```

### 2. 缺失的依赖
原 Kronos 项目 `requirements.txt` 缺少：
- `pyyaml`: 微调配置加载需要
- `pytest`: 测试需要（可选）
- `pyqlib`: Qlib 微调需要（可选）
- `akshare`: 中国市场数据需要（可选）

### 3. 标准库 vs 第三方库
以下导入是 Python 标准库，无需在 requirements.txt 中声明：
- `os`, `sys`, `time`, `json`, `pickle`
- `datetime`, `logging`, `math`
- `typing`, `pathlib`
- `collections`, `random`
- `base64`, `hmac`, `hashlib`

### 4. PyTorch 相关
PyTorch 相关的子模块无需单独声明：
- `torch.nn`, `torch.nn.functional`
- `torch.utils.data`
- `torch.distributed`
- `torch.autograd`

## 使用建议

### 完整安装（推理 + 微调 + Kokex 应用）
```bash
cd /home/zshx/code/kronos/kokex
pip install -r requirements.txt
```

### 仅 Kronos 核心（推理）
```bash
pip install numpy pandas torch einops==0.8.1 huggingface_hub==0.33.1 \
    matplotlib==3.9.3 tqdm==4.67.1 safetensors==0.6.2
```

### 添加微调支持
```bash
pip install pyyaml
```

### 添加 Qlib 微调支持
```bash
pip install pyqlib
```

### 添加测试支持
```bash
pip install pytest
```

## 总结

### 已完成
✅ 分析了 Kronos 项目（除 kokex）的所有依赖
✅ 整理到 `kokex/requirements.txt`，结构清晰
✅ 添加了完整的注释和分类
✅ 识别了核心依赖和可选依赖
✅ 创建了依赖说明文档 `kokex/docs/DEPENDENCIES.md`

### 依赖统计
- **Kronos 核心**: 8 个包
- **微调配置**: 1 个包 (pyyaml)
- **Kokex 应用**: 11 个包
- **可选依赖**: 3 个包 (pytest, pyqlib, akshare)
- **总计**: 23 个包（20 个必需，3 个可选）

### 建议
1. 在 CI/CD 中使用 `pip install -r requirements.txt` 安装所有必需依赖
2. 为开发环境创建 `requirements-dev.txt` 包含可选依赖
3. 定期更新依赖版本，注意兼容性
4. 考虑使用 `pip-tools` 或 `poetry` 进行依赖管理

## 相关文件

- `/home/zshx/code/kronos/requirements.txt` - Kronos 原始依赖
- `/home/zshx/code/kronos/kokex/requirements.txt` - 更新后的完整依赖
- `/home/zshx/code/kronos/kokex/docs/DEPENDENCIES.md` - 依赖说明文档
