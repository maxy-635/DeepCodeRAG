#!/bin/bash
export PYTHONPATH="/data/home/sczc725/run/DeepCodeRAG:$PYTHONPATH"      
# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

# 加载CUDA和cuDNN模块
module load cuda/12.1
module load cudnn/8.9.6.50_cuda12


# 定义Python文件的路径
PYTHON_FILE="./rag/main_deepeval_v3.py"

python3 "$PYTHON_FILE" \
     --model_local_path "/data/home/sczc725/run/huggingface/hub" \
     --inference_model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
     --api_docs_path "./api_parser/tensorflow/apis_parsed_results" \
     --matching_index_path "./database/whoosh_tf_apis_index_0522" \
     --benchmark "./benchmark/DeepEval/" \
     --repeats 1 \
     --experiment_id ""
