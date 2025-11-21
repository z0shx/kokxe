#!/bin/bash

# KOKEX 启动脚本

echo "========================================="
echo "  KOKEX - AI 智投平台 version 0.1.0"
echo "========================================="

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 检查依赖
echo ""
echo "检查依赖..."
pip show gradio > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 依赖未安装，请先运行: pip install -r requirements.txt"
    exit 1
fi
echo "✅ 依赖检查通过"

# 检查数据库连接
echo ""
echo "检查数据库连接..."
# 这里可以添加数据库连接检查逻辑

# 启动应用
echo ""
echo "启动应用..."
python app.py

