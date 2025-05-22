""" 
单个实验的代码生成任务
"""

import os
from utils.chat_log import ChatLog
from utils.chatlog2code import Chatlog2Code


class DLcodeGeneration:
    """
    针对某个benchmark任务,设计并使用某种prompt技术,生成深度学习代码,并将代码和聊天日志按照时间和顺序进行保存.
    """

    def __init__(self, count_num, save_code_path, save_log_path):
        """
        Paras:
        1.count_num: int, 标记是第几次实验
        2.save_code_path: str, 保存LLM生成的代码的路径
        3.save_log_path: str, 保存与LLM交互的聊天日志的路径
        """
        self.count_num = count_num
        self.save_code_path = save_code_path
        self.save_log_path = save_log_path

    def prompt2llm(
        self,
        model,
        tokenizer,
        prompt,
        chat2llm,
        temperature,
        top_p,
        max_new_tokens,
    ):
        """
        根据具体的任务要求，设计prompt提示词，与LLM模型进行交互，生成代码
        Paras:
        1.model: str, 要调用的LLM模型,注意部署方式的模型，model对应的是一个实体
        2.prompt_designer: class, 要采用的prompt提示词
        3.tasks_requirement: 具体的任务要求
        4.chat2llm: class, 与LLM模型交互的类
        5.temperature: float, 生成代码的温度
        6.top_p: float, 生成代码的top_p
        7.max_new_tokens: int, 生成代码的最大长度
        """

        # get conversation from LLM
        response = chat2llm.chat(
            model=model,
            tokenizer=tokenizer,
            prompting=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        return response

    def save(self, prompting, response):
        """
        保存生成的代码和聊天日志
        """
        # save chat logs
        if not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)
        chatloger = ChatLog(self.save_log_path, prompting, response)
        _, log_file = chatloger.save_chat_log(self.count_num)

        # 根据 chatlog 提取代码 20240906更新
        if not os.path.exists(self.save_code_path):
            os.makedirs(self.save_code_path) 
            
        chatlog2coder = Chatlog2Code(
            log_file_path=log_file, save_code_path=self.save_code_path
        )
        matched_code = chatlog2coder.read_yamldata_to_pycode()

        return matched_code
