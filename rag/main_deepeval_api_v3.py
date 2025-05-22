"""主程序，用于生成深度学习代码
此版本对应 方案2，GPT系列模型，没有使用向量存储，只采用了LLM推理+关键词检索
没有 model.load_model()函数
"""
import os
import re
import openai
import time
import argparse
import torch
import traceback
from loguru import logger
from utils.chat2deepseek import Chat2DeepSeekLLM
from utils.chat2gemma import Chat2GemmaLLM
from utils.chat2llama import Chat2LlamaLLM
from utils.chat2gpt import Chat2GPTLLM
from utils.chat_log import ChatLog
from utils.chatlog2code import Chatlog2Code
from rag.whoosh_search import WhooshSearch,JiebaAnalyzer
from utils.utils import get_all_files,get_llm_name, read_yaml_data
from prompts.first_prompt import FirstPromptDesigner
from prompts.second_prompt import SecondPromptDesigner


class DLcodeGeneration:
    """
    针对某个benchmark任务，设计并使用某种prompt技术，生成深度学习代码，并将代码和聊天日志按照时间和顺序进行保存
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
        model_id,
        prompting,
        tasks_requirement,
        chat2llm,
        temperature,
        top_p,
        max_tokens,
    ):
        """
        根据具体的任务要求，设计prompt提示词，与LLM模型进行交互，生成代码
        Paras:
        1.model_id: str, 要调用的LLM模型
        2.prompt_designer: class, 要采用的prompt提示词
        3.tasks_requirement: 具体的任务要求
        """
        time.sleep(0.1)  # 控制访问频率,防止频繁调用API引发报错
        try:
            response = chat2llm().chat(
                prompting=prompting,
                model_id=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            # print("Prompt text:\n", prompting)
            # print("Response text:\n", response)

            return response

        # 此处的APIError是openai.error.APIError，需要对Replication的APIError进行修改
        except (openai.error.APIError) as e:  
            logger.error(f"Encountered error: {e}")
            time.sleep(5)
            return self.prompt2llm(
                model_id=model_id,
                prompting=prompting,
                tasks_requirement=tasks_requirement,
                chat2llm=chat2llm,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

    def save(self, prompting, response):
        """
        保存生成的代码和聊天日志
        """
        # save chat logs
        if not os.path.exists(self.save_log_path):
            os.makedirs(self.save_log_path)
        chatloger = ChatLog(self.save_log_path, prompting, response)
        _, log_file = chatloger.save_chat_log(self.count_num)

        # save generated code
        if not os.path.exists(self.save_code_path):
            os.makedirs(self.save_code_path)  # 如果路径不存在，创建文件夹

        # 根据 chatlog 提取代码 20240906更新
        chatlog2coder = Chatlog2Code(
            log_file_path=log_file, save_code_path=self.save_code_path
        )
        matched_code = chatlog2coder.read_yamldata_to_pycode()

        return matched_code



class MultiDLcodeGeneration:
    """
    针对每个benchmark任务，设计并使用多种prompt技术，重复多次生成深度学习代码，并将代码和聊天日志按照时间和顺序进行保存
    """
    def __init__(
        self,
        inference_model_id,
        api_docs_path,
        matching_index_path,
        sava_code_last_path,
        sava_log_last_path,
    ):
        """
        Paras:
        1.inference_model_id: str, 要调用的LLM推理模型
        2.api_docs_path: str, API文档路径
        3.matching_index_path: str, 匹配索引路径
        4.sava_code_last_path: str, 需要用户指定的代码保存文件夹命名
        5.sava_log_last_path: str, 需要用户指定的日志保存文件夹命名
        """
        # 1. 初始化 LLM推理模型
        if "gpt" in inference_model_id:
            self.chat2llm = Chat2GPTLLM
        else:
            raise ValueError("The inference_model_id is invalid, please check!")
        
        self.model = inference_model_id
        logger.info(f"LLM推理模型采用{self.model}")

        # 2. 初始化 关键词检索模型
        self.api_docs_path = api_docs_path
        self.matching_index_path = matching_index_path
        self.bm25searcher = WhooshSearch(docs_path=self.api_docs_path, index_dir=self.matching_index_path)
        logger.info(f"关键词检索模型{self.matching_index_path}初始化成功！")

        self.sava_code_last_path = sava_code_last_path
        self.sava_log_last_path = sava_log_last_path
        self.llm_name = get_llm_name(inference_model_id)

    def generate_code(self, task_requirement, task_id, count):
        """
        Paras:
        2.task_requirement: 具体的任务要求
        3.task_id: str, 任务名称
        4.count: int, 标记是第几次实验
        会两次调用这个函数
        分别是 first_prompt和second_prompt
        """

        # 设置代码和log保存的路径，按照llm名称，prompting类型，实验id，task id进行保存
        save_code_full_path = f"./response/{self.llm_name}/{self.sava_code_last_path}{task_id}"
        save_log_full_path = f"./chat_logs/{self.llm_name}/{self.sava_log_last_path}{task_id}"


        # 1. 第一次向LLM提问，直接让LLM判断得到的备选APIs
        dlcode_generator = DLcodeGeneration(
            count_num=count,
            save_code_path=save_code_full_path,
            save_log_path=save_log_full_path,
        )
        # 注意此处为FirstPromptDesigner 中的 prompt_v2
        first_prompt = FirstPromptDesigner().prompt_v2(task_requirement)
        response = dlcode_generator.prompt2llm(
            model_id=self.model,
            prompting=first_prompt,
            tasks_requirement=task_requirement,
            chat2llm=self.chat2llm,
            temperature=0.8,
            top_p=0.95,
            max_tokens=1500,
        )
        logger.info(f"第一次LLM推理的提示词: {first_prompt}")
        logger.info(f"第一次LLM推理的结果: {response}")
        
        # 对第一次的LLM response进行文本处理
        match = re.search(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
        if match:
            code_inside = match.group(1).strip()
            # 检查是否包含 candidate_apis = [...] 或 candidate_apis.append(...)格式的代码块
            pattern1 = r"candidate_apis\s*=\s*\[(.*?)\]"
            pattern2 = r"candidate_apis\.append\((\{.*?\})\)"
            if re.search(pattern1, code_inside, re.DOTALL) or re.search(pattern2, code_inside, re.DOTALL):
                try:
                    exec(code_inside, globals()) # 执行代码，生成candidate_apis 列表变量
                except Exception as e:
                    logger.error(f"执行代码时发生错误: {e}")
                logger.info(f"第一次LLM回答得到的candidate_apis: {candidate_apis}")
        
        
        # 3. 第二次检索（关键词检索），使用原来的简化的 API文档信息 结果，然后进行与原文档的 match，得到 对于每个API的完备信息；
        # 并对第二次检索的结果进行处理；
        # 并对第二次检索的结果进行处理；
        # 注意：如果上一个阶段的检索结果为空，则不存在candidate_apis变量
        api_docs = []
        try:
            if not candidate_apis:
                logger.info("candidate_apis 为空，跳过 API 检索。")
            else:
                api_name_set = set()
                for candidate_api in candidate_apis:
                    candidate_api_name = candidate_api['api_name'].replace('tensorflow', 'tf')
                    if candidate_api_name not in api_name_set: # 去重

                        # 只进行层相关layer API的检索，避免提示词过长
                        if 'layer' in candidate_api_name:
                            logger.info(f"开始检索: {candidate_api_name}")
                            # 进行关键词检索, 注意每次只取top-1的结果
                            results = self.bm25searcher.main(query_str=candidate_api_name, limit=1)
                            if not results: # 如果 results 为空
                                logger.info(f"未找到与 {candidate_api_name} 匹配的 API 文档")
                                continue

                            for result in results:
                                api_doc = (
                                    f"{result['api_name']}\n"
                                    # f"{result['api_description']}\n"
                                    # f"{result['api_signature']}\n"
                                    # f"{result['api_details']}\n"
                                    f"{result['api_usage_description']}\n"
                                    f"{result['api_parameters']}\n"
                                    # f"{result['api_usage_example']}\n\n"
                                )

                                api_docs.append("\n")
                                api_docs.append(api_doc)

                            api_name_set.add(candidate_api_name)
        except:
            logger.info(f"执行代码时发生错误: {traceback.format_exc()}")
        
        # ----------------------------------------------------------------------------------------
        # 4. 第二次向LLM提问，使用完备的 API文档信息 作为上下文，得到LLM判断得到的精选API
        second_prompt = SecondPromptDesigner().prompt_v2(task_requirement, api_docs, dll='Tensorflow')
        response = dlcode_generator.prompt2llm(
            model_id=self.model,
            prompting=second_prompt,
            tasks_requirement=task_requirement,
            chat2llm=self.chat2llm,
            temperature=0.8,
            top_p=0.95,
            max_tokens=6000, # api docs 可能会比较长
        )
        logger.info(f"第二次LLM推理的提示词: {second_prompt}")
        logger.info(f"第二次LLM推理的结果: {response}")
        dlcode_generator.save(second_prompt, response)

    def multi_dlcode_generation(self, benchmark_path, repeats_count):
        """
        Paras:
        1.prompt_designers: list, prompt技术
        2.benchmark_path: str, benchmark路径
        3.repeats_count: int, 重复实验次数
        """
        
        benchmark_files = get_all_files(directory=benchmark_path, file_type=".yaml")

        logger.info(f"开始进行深度学习代码生成任务")
        for yaml_file in benchmark_files:  
            task_id = "/" + os.path.basename(yaml_file)[:-5]
            task= read_yaml_data(yaml_file)
            task_requirement = task['Requirement']

            logger.info(f"此次任务{task_id}需求: {task_requirement}")
            for count in range(1, repeats_count + 1):
                self.generate_code(task_requirement, task_id, count)
        
        logger.info(f"全部深度学习代码生成任务完成")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_local_path", type = str, help="The LLM local path")
    parser.add_argument("--inference_model_id", type = str, help="The Inference LLM id")
    parser.add_argument("--embedding_model_id", type = str, help="The Embedding model id")
    parser.add_argument("--api_docs_path", type = str, help="The API docs path for embedding and best matching")
    parser.add_argument("--embedding_vector_path", type = str, help="The vector store path")
    parser.add_argument("--matching_index_path", type = str, help="The index path for best matching")
    parser.add_argument("--benchmark", type = str, help="The benchmark dataset path")
    parser.add_argument("--repeats", type = int, help="The repeat times")
    parser.add_argument("--experiment_id", type = str, help="The experiment id")

    args = parser.parse_args()

    logger.add("logs/" + args.experiment_id + ".log")

    # 初始化 推理模型和BM25F模型
    multi_DLcode_Generator = MultiDLcodeGeneration(
        inference_model_id=args.inference_model_id,
        api_docs_path=args.api_docs_path,
        matching_index_path=args.matching_index_path,
        sava_code_last_path="models_" + args.experiment_id,
        sava_log_last_path="logs_" + args.experiment_id,
    )
    multi_DLcode_Generator.multi_dlcode_generation(
        benchmark_path=args.benchmark,
        repeats_count=args.repeats,
    )