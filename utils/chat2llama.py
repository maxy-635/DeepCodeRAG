import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_access_token = "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"


class Chat2LlamaLLM:
    """
    分为两部分：
    1.load_model: 加载模型和分词器
    2.chat: 与模型交互，生成代码
    参考示例：
    1.https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    备选：
    1. https://huggingface.co/meta-llama 中的其他模型，需要提前申请access token
    """

    def __init__(self):
        pass

    def load_model(self, model_local_path, cache_dir, data_dtype):

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path, 
            token=hf_access_token,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_local_path,
            device_map="auto",
            torch_dtype=data_dtype,
            token=hf_access_token,
            cache_dir=cache_dir,
        )

        return model, tokenizer

    def chat(self, model, tokenizer, prompting, temperature, top_p, max_new_tokens):

        chat = [{"role": "user", "content": prompting}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        # 获取并解码生成的 token IDs，移除输入提示词,只保留模型输出
        generated_text = tokenizer.decode(outputs[0])
        response = generated_text[len(prompt)-1:].strip()

        # 释放显存
        del inputs
        del outputs
        torch.cuda.empty_cache()

        return response


if __name__ == "__main__":
    # 设置缓存路径 (并行计算服务器)
    cache_dir_paras_v100 = "/data/home/scv7f1z/run/huggingface/hub"

    model_id = ["meta-llama/CodeLlama-7b-Instruct-hf",
                "meta-llama/Meta-Llama-3.1-8B-Instruct"]

    prompting = "Write me a Python function to add two number."

    model, tokenizer = Chat2LlamaLLM().load_model(
        model_local_path=model_id[1], 
        cache_dir=cache_dir_paras_v100,
        data_dtype=torch.bfloat16
    )

    response = Chat2LlamaLLM().chat(
        model=model,
        tokenizer=tokenizer,
        prompting=prompting,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=1500,
    )
    print(response)
