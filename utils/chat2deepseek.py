import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Chat2DeepSeekLLM:
    """
    分为两部分：
    1.load_model: 加载模型和分词器
    2.chat: 与模型交互，生成代码
    参考示例：
    1. https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
    2. https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
    """

    def __init__(self):
        pass

    def load_model(self, model_local_path, cache_dir, data_type):

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            device_map="auto",
            torch_dtype=data_type,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        model.generation_config = GenerationConfig.from_pretrained(model_local_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return model, tokenizer

    def chat(self, model, tokenizer, prompting, temperature, top_p, max_new_tokens):
        
        chat = [{'role': 'user', 'content': prompting}]
        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True, 
            return_tensors="pt").to("cuda")

        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][len(inputs[0]-1):], skip_special_tokens=True)

        # 释放显存
        del inputs
        del outputs
        torch.cuda.empty_cache()

        return response


if __name__ == "__main__":
    # 设置缓存路径 (并行计算服务器)
    cache_dir_paras_a100 = "/home/bingxing2/home/scx8amp/huggingface/hub"

    model_id = [
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        ]

    prompting = "Write me a Python function to add two number."

    model, tokenizer = Chat2DeepSeekLLM().load_model(
                        model_local_path=model_id[0], 
                        cache_dir=cache_dir_paras_a100,
                        data_type=torch.bfloat16
                    )

    response = Chat2DeepSeekLLM().chat(
        model=model,
        tokenizer=tokenizer,
        prompting=prompting,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=100,
    )
    print(response)
