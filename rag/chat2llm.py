# ========== 大模型推理模块 ==========
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class Chat2LLM:
    """
    大模型处理器，用于加载和输入提示词给LLM
    """
    def __init__(self, generative_model, cache_dir):
        '''
        初始化LLM
        :param generative_model: 生成式LLM模型id
        :param cache_dir: 模型缓存目录
        '''
        logger.info(f"step4. 加载 LLM 和 Tokenizer: {generative_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
                                        pretrained_model_name_or_path=generative_model, cache_dir=cache_dir, 
                                        local_files_only=True,
                                        use_fast=False
                                    )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
                                        pretrained_model_name_or_path=generative_model,
                                        cache_dir=cache_dir,
                                        local_files_only=True,
                                        device_map='cuda',
                                        torch_dtype=torch.bfloat16
                                    ).eval()

    def ask(self, prompt, temperature, top_p, max_new_tokens):
        '''
        输入提示词给LLM并获取回答
        :param prompt: 提示词
        :param max_tokens: 最大生成token数量
        :return: LLM的回答
        '''
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        ]
        input_ids = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')

        # 生成模型输出
        output_ids = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 解码输出
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):]

        return response


if __name__ == "__main__":
    logger.add('./test_chat2llm.log')
    LLM_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    CACHE_DIR = '/data/home/sczc725/run/huggingface/hub' # 缓存目录
    # 创建LLM处理器
    llm = Chat2LLM(generative_model=LLM_MODEL, cache_dir=CACHE_DIR)
    # 测试LLM
    prompt = """
As a developer specializing in deep learning, your task is as follows.

## Step 1: Analyze the constraints of following API usage

**API Signature**:
`tf.keras.layers.Add(**kwargs)`
**API Usage Details**:
It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).


## Step 2: Generate Code for the Task. The generated code should be satisfied the API usage constraints above, if need, you could adjust the parameters of the other APIs.
### Task:
"Please assist me in creating a deep learning model for image classification using the CIFAR-10 dataset. The model will include three sequential blocks, each comprising a convolutional layer, a batch normalization layer, and a ReLU activation function to extract image features. These blocks will produce three separate output paths, each corresponding to one block's output. Additionally, a parallel branch of convolutional layers will process the input directly. The outputs from all paths will be added, and the aggregated result will pass through two fully connected layers for classification."

Note: use TensorFlow Functional API. Only use layers that conform to the above constraints.
Complete only this function:
```python
def dl_model():
    
    return model
    """
    
    answer = llm.ask(prompt=prompt, temperature=0.8, top_p=0.95, max_new_tokens=1500)
    logger.info(f"step6. LLM回答: {answer}")
