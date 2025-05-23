import os
import time
import openai


class Chat2GPTLLM:
    """
    Chat with a GPT model using OpenAI's API.
    参考：
    模型列表： https://platform.openai.com/docs/models
    API参数：
    1. https://platform.openai.com/docs/api-reference/chat
    2. https://platform.openai.com/playground/chat?models=gpt-3.5-turbo-0125
        API 默认参数配置：
        1) max_tokens,如果用户没有指定，会默认为对应模型的最大token数量, default=4096
        2) temperature = 0.8
        3) top_p = 0.95
        备注：没有top_k参数
    计算token数量：
    https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    可调用模型：
    print(openai.Engine.list())
    """

    def __init__(self):
        pass

    def chat(self, prompting, model_id, temperature, top_p, max_tokens):

        api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = api_key  # openai==0.27.2版本API的调用方法

        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[{"role": "user", "content": prompting}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        # completion中还包含其他很多信息，比如调用的模型ID,role,content等
        completion = response.get("choices")[0]["message"]["content"]

        # 获取输入输出的tokens数量
        # token_count = response['usage']
        # print("Token 数量：", token_count)

        return completion


if __name__ == "__main__":

    LLM_ID = {"GPT": ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o"]}
    chater = Chat2GPTLLM()
    prompting = "Write a function to add two numbers in Python."

    time1 = time.time()
    reponse = chater.chat(
        prompting=prompting,
        model_id=LLM_ID["GPT"][2],
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    print(reponse)
    time2 = time.time()
    print("time:", time2 - time1)
