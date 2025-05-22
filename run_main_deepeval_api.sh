#!/bin/bash
export PYTHONPATH="/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG:$PYTHONPATH"       
# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DeepCodeRAG

# 定义Python文件的路径
PYTHON_FILE="./rag/main_deepeval_api_v3.py"

python3 "$PYTHON_FILE" \
     --inference_model_id "gpt-4o-mini" \
     --api_docs_path "./api_parser/tensorflow/apis_parsed_results" \
     --matching_index_path "./database/whoosh_tf_apis_index_0522" \
     --benchmark "./benchmark/DeepEval/" \
     --repeats 1 \
     --experiment_id "" 
