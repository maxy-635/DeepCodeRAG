#!/bin/bash
export PYTHONPATH="/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG:$PYTHONPATH"       
# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DeepEval

# 加载CUDA和cuDNN模块
module load cuda/12.1
module load cudnn/8.9.6.50_cuda12


# 定义Python文件的路径
PYTHON_FILE="./rag/main_deepeval_v3.py"

python3 "$PYTHON_FILE" \
     --model_local_path "/data/home/sczc725/run/huggingface/hub" \
     --inference_model_id "gpt-4o-mini" \
     --embedding_model_id "BAAI/bge-m3" \
     --api_docs_path "./api_parser/tensorflow/apis_parsed_results" \
     --embedding_vector_path "./database/api_name_description_detail.faiss" \
     --matching_index_path "./database/whoosh_tf_apis_index_0521" \
     --benchmark "./benchmark/DeepEval/" \
     --repeats 1 \
     --experiment_id "DeepEvalRAG"
