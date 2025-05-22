#!/bin/bash
export PYTHONPATH="/data/home/sczc725/run/DeepCodeRAG:$PYTHONPATH"         
# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

# 加载CUDA和cuDNN模块
module load cuda/12.1
module load cudnn/8.9.6.50_cuda12

# 定义Python文件的路径
PYTHON_FILE="./rag/vector_store_manager.py"

# 运行Python文件
python3 "$PYTHON_FILE"