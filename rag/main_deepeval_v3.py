"""主程序，用于生成深度学习代码
此版本对应 运行从huggingface中加载的LLM模型
此版本对应方案2，没有使用向量存储，只采用了LLM推理+关键词检索
"""
import os
import re
import argparse
import torch
from loguru import logger
from deepcode_generation import DLcodeGeneration
from rag.whoosh_search import WhooshSearch,JiebaAnalyzer
from utils.utils import get_all_files,get_llm_name, read_yaml_data
from utils.chat2deepseek import Chat2DeepSeekLLM
from utils.chat2gemma import Chat2GemmaLLM
from utils.chat2llama import Chat2LlamaLLM
from prompts.first_prompt import FirstPromptDesigner
from prompts.second_prompt import SecondPromptDesigner

class MultiDLcodeGeneration:
    """
    针对每个benchmark任务，设计并使用多种prompt技术，重复多次生成深度学习代码，并将代码和聊天日志按照时间和顺序进行保存
    """
    def __init__(
        self,
        inference_model_id,
        data_dtype,
        api_docs_path,
        matching_index_path,
        model_cache_path,
        sava_code_last_path,
        sava_log_last_path,
    ):
        """
        Paras:
        1.inference_model_id: str, 要调用的LLM推理模型
        2.embedding_model_id: str, 要使用的Embedding模型
        3.api_docs_path: str, API文档路径
        4.embedding_vector_path: str, 向量存储路径
        5.matching_index_path: str, 匹配索引路径
        6.model_cache_path: str, 模型缓存路径
        7.sava_code_last_path: str, 需要用户指定的代码保存文件夹命名
        8.sava_log_last_path: str, 需要用户指定的日志保存文件夹命名
        """
        # 1. 初始化 LLM推理模型,只在这里加载一次
        if "deepseek" in inference_model_id:
            self.chat2llm = Chat2DeepSeekLLM
        elif "gemma" in inference_model_id:
            self.chat2llm = Chat2GemmaLLM
        elif "llama" in inference_model_id:
            self.chat2llm = Chat2LlamaLLM
        else:
            raise ValueError("The inference_model_id is invalid, please check!")
 
        self.model, self.tokenizer = self.chat2llm().load_model(
            inference_model_id, model_cache_path, data_dtype
        )
        logger.info(f"推理模型 {inference_model_id} 初始化加载成功！")

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
        first_prompt = FirstPromptDesigner().prompt_v2(task_requirement)
        # 注意此处为FirstPromptDesigner
        response = dlcode_generator.prompt2llm(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=first_prompt,
            chat2llm=self.chat2llm(),
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=500,
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
                    candidate_api_name = candidate_api['api_name']
                    if candidate_api_name not in api_name_set: # 去重

                        # 只进行层相关layer API的检索，避免提示词过长
                        if 'layer' in candidate_api_name:
                            # 下面这行 关于tf.keras.layers.MaxPooling2D的修改是 为了 GPT-4o-mini
                            if candidate_api_name == 'tf.keras.layers.MaxPooling2D':
                                candidate_api_name = 'tf.keras.layers.MaxPool2D'

                            logger.info(f"第二次关键词检索的candidate_api_names: {candidate_api_name}")
                            # 进行关键词检索, 注意每次只取top-1的结果
                            results = self.bm25searcher.main(query_str=candidate_api_name, limit=1)
                            if not results:
                                logger.info(f"未找到与 {candidate_api_name} 匹配的 API 文档")
                                continue

                            for result in results:
                                # 0521修改，只取用top-5的api_parameters，此处输出的api_parameters 长字符串
                                # 0521修改，取消 api_signature 的默认参数值
                                # 0521修改，整合 api_description 和 api_details
                                api_doc = (
                                    f"{result['api_name']}\n\n"
                                    # f"{result['api_description']}\n"
                                    f"{result['api_signature']}\n\n"
                                    # f"{result['api_details']}\n"
                                    f"{result['api_usage_description']}\n\n"
                                    f"{result['api_parameters']}\n\n"
                                    f"{result['api_usage_example']}\n\n"
                                )
                                api_docs.append(api_doc)

                        api_name_set.add(candidate_api_name)
        except Exception as e:
            import traceback
            # 打印完整的异常信息
            logger.info(f"执行代码时发生错误: {traceback.format_exc()}")
        
        # ----------------------------------------------------------------------------------------
        # 4. 第二次向LLM提问，使用完备的 API文档信息 作为上下文，得到LLM判断得到的精选API
        second_prompt = SecondPromptDesigner().prompt_v2(task_requirement, api_docs, dll='Tensorflow')
        response = dlcode_generator.prompt2llm(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=second_prompt,
            chat2llm=self.chat2llm(),
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=1500,
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

            logger.info(f"此次任务需求: {task_requirement}")
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
    # 初始化 推理模型和Embedding模型及实验设置参数
    multi_DLcode_Generator = MultiDLcodeGeneration(
        model_cache_path=args.model_local_path,
        inference_model_id=args.inference_model_id,
        embedding_model_id= args.embedding_model_id,
        api_docs_path=args.api_docs_path,
        data_dtype=torch.bfloat16,
        embedding_vector_path=args.embedding_vector_path,
        matching_index_path=args.matching_index_path,
        sava_code_last_path="models_" + args.experiment_id,
        sava_log_last_path="logs_" + args.experiment_id,
    ) 

    # 生成代码
    multi_DLcode_Generator.multi_dlcode_generation(
        benchmark_path=args.benchmark,
        repeats_count=args.repeats,
    )