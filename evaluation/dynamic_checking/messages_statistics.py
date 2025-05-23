"""
统计每个LLMs下每个提示词中json文件中记录下来的compile_error信息
"""
import re
import json
from utils.utils import dict2json

class MessagesStatistics:
    
    def __init__(self, report_jsons, llms, promptings, experiment_id, save_path):
        self.report_jsons = report_jsons
        self.llms = llms
        self.promptings = promptings
        self.experiment_id = experiment_id
        self.save_path = save_path

    def preprocess_error_message(self, error_message):
        """
        智能处理error_message，将相同的error_message进行合并，统计数量,打标签
        """
        # 1.对于 Add 和 Concatenate 的错误信息, 删除Received后面的内容
        error = error_message.split("Received")[0].strip()
        # 2.对于\n\nname 'tf' is not defined
        if "\n\nname 'tf' is not defined" in error:
            error = "name 'tf' is not defined"
        # 3.对于 Reshape 相关错误信息，total size of new array must be unchanged 
        if "\n\ntotal size of new array must be unchanged" in error:
            error = "Reshape:total size of new array must be unchanged"
        if "\n\nTried to convert 'shape' to a tensor and failed" in error:
            error = "Reshape:Tried to convert 'shape' to a tensor and failed"
        # 4.对于 Lambda中使用split的错误信息，Dimension size must be evenly divisible by 3 but
        if "\n\nDimension size must be evenly divisible by" in error:
            error = "Lambda:Dimension size must be evenly divisible by"
        # 5.对于"name 'Add' is not defined"类型的错误信息
        if "is not defined" in error:
            # 使用正则表达式匹配模式，捕获 API 或变量名称
            pattern = r"name '([A-Za-z][A-Za-z0-9_]*)' is not defined"
            match = re.search(pattern, error)
            if match:
                name = match.group(1)  # 获取捕获的名称
                if name[0].isupper():  # 判断是否以大写字母开头
                    # 大写开头，替换为 "API"
                    error = "Unimported API: name API is not defined"
                else:
                    # 小写开头，替换为 "variable"
                    if name == "tf":
                        error = "Unimported API: name API is not defined"
                    else:
                        error = "Undefined-Variable: name variable is not defined"

        # 6. 对于Add层输入维度不匹配的错误信息
        if "Inputs have incompatible shapes." in error:
            error = "Add:Inputs have incompatible shapes."
        # 7. 对于Concatenate层输入维度不匹配的错误信息
        if "A `Concatenate` layer requires inputs" in error:
            error = "Concatenate:A `Concatenate` layer requires inputs"
        # 8. 对于 “Keyword argument not understood”
        if "Keyword argument not understood" in error:
            error = "Conv2D:Keyword argument not understood"
        # 9. 对于 “Invalid target shape for Reshape layer: dynamic tensor used instead of static shape.”
        if "(type Reshape).\n\nKeras symbolic inputs/outputs do not implement `__len__`." in error:
            error = "Reshape:Invalid target shape for Reshape layer: dynamic tensor used instead of static shape."
        if "Keras symbolic inputs/outputs do not implement `__len__`." in error:
            error = "Reshape:Invalid target shape for Reshape layer: dynamic tensor used instead of static shape."
        # 10.对于\n\n\nThe following Variables were created within a Lambda layer
        if "\n\n\nThe following Variables were created within a Lambda layer" in error:
            error = "Lambda:The following Variables were created within a Lambda layer"
        # 11.对于One of the dimensions in the output is <= 0 due to downsampling in
        if "One of the dimensions in the output is <= 0 due to downsampling in" in error:
            error = "Conv2D:One of the dimensions in the output is <= 0 due to downsampling in"
        # 12.(type AveragePooling2D).\n\nNegative dimension size caused by subtracting
        if "(type AveragePooling2D).\n\nNegative dimension size caused by subtracting" in error:
            error = "AveragePooling2D:Negative dimension size caused by subtracting"
        # 13.(type MaxPooling2D).\n\nNegative dimension size caused by subtracting
        if "(type MaxPooling2D).\n\nNegative dimension size caused by subtracting" in error:
            error = "MaxPooling2D:Negative dimension size caused by subtracting"
        # (type DepthwiseConv2D).\n\nNegative dimension size caused
        if "(type DepthwiseConv2D).\n\nNegative dimension size caused" in error:
            error = "DepthwiseConv2D:Negative dimension size caused by subtracting"
        # (type SeparableConv2D).\n\nNegative dimension size caused by subtracting
        if "(type SeparableConv2D).\n\nNegative dimension size caused by subtracting" in error:
            error = "SeparableConv2D:Negative dimension size caused by subtracting"
        # 14.incompatible with the layer: expected min_ndim=4, found ndim=2
        if "incompatible with the layer" in error:
            pattern =  r"incompatible with the layer: expected min_ndim=([0-9]+), found ndim=([0-9]+)"
            match = re.search(pattern, error)
            if match:
                error = "incompatible with the layer"
        # 15.(type Conv2D).\n\nNegative dimension size caused by subtracting
        if "(type Conv2D).\n\nNegative dimension size caused by subtracting" in error:
            error = "Conv2D:Negative dimension size caused by subtracting"
        # 16. missing ** required positional argument
        if " required positional argument" in error:
            pattern =  r"missing ([0-9]+) required positional argument"
            error = "missing required positional argument"
        if "Missing required positional argument" in error:
            error = "missing required positional argument"
        # 17.got an unexpected keyword argument
        if "got an unexpected keyword argument" in error or "\n\nGot an unexpected keyword argument" in error:
            error = "got an unexpected keyword argument"
        # 18. cannot import name 'TransposedConv2D' from 'keras.layers'
        if "cannot import name" in error:
            pattern =  r"cannot import name '([^']+)' from '([^']+)'"
            match = re.search(pattern, error)
            if match:
                error = f"cannot import name API from DLL"
        # 19.module 'keras.api._v2.keras.backend' has no attribute 'split'
        if "has no attribute" in error and "dl_model" not in error:
            pattern =  r"module '([^']+)' has no attribute '([^']+)'"
            match = re.search(pattern, error)
            if match:
                error = f"module API has no attribute API"
        if "has no attribute 'dl_model'" in error:
            error = "has no attribute dl_model"
        # 21 (type TFOpLambda).\n\nDimensions
        if "(type TFOpLambda).\n\nDimensions" in error or "(type TFOpLambda).\n\nDimension" in error or "(type TFOpLambda).\n\nShape" in error:
            error = "Dimensions or shape mismatch"
        # expects 1 input(s), but it received 3 input tensors.
        if "but it received" in error:
            pattern =  r"expects ([0-9]+) input\(s\), but it received ([0-9]+) input tensors."
            match = re.search(pattern, error)
            if match:
                error = "received input tensors numbers mismatch"
        # 22. No module named
        if "No module named" in error:
            error = "No module named"
        # 23. Graph disconnected: cannot obtain value for tensor KerasTensor
        if "Graph disconnected: cannot obtain value for tensor" in error:
            error = "Graph disconnected: cannot obtain value for tensor"
        # 24. got multiple values for argument
        if "got multiple values for argument" in error:
            error = "got multiple values for argument"
        # 25. (type Conv2D).\n\nDepth of filter must not be 0 
        if "(type Conv2D).\n\nDepth of filter must not be 0" in error:
            error = "Conv2D:Depth of filter must not be 0"
        # (type TFOpLambda).\n\nFailed to convert elements of 
        if "(type TFOpLambda).\n\nFailed to convert elements of" in error:
            error = "TFOpLambda:Failed to convert elements of"
        # 26. (type Lambda).\n\nDimensions must be equal
        if "(type Lambda).\n\nDimensions must be equal" in error:
            error = "Lambda:Dimensions must be equal"
        # 27. (type Conv2D).\n\nUsing a symbolic `tf.Tensor` as a Python `bool` is not allowed in Graph execution.
        if "(type Conv2D).\n\nUsing a symbolic `tf.Tensor` as a Python `bool` is not allowed in Graph execution." in error:
            error = "Conv2D:Using a symbolic `tf.Tensor` as a Python `bool` is not allowed in Graph execution."
        if "'return' outside function" in error:
            error = "'return' outside function"
        if "referenced before assignment" in error:
            error = "referenced before assignment"
        if "invalid decimal literal" in error:
            error = "invalid decimal literal"
        if "Unknown activation function" in error:
            error = "unknown activation function"
        if "Could not build a TypeSpec for KerasTensor" in error:
            error = "Could not build a TypeSpec for KerasTensor"
        if "invalid syntax. Perhaps you forgot a comma" in error:
            error = "invalid syntax. missing comma"

        return error
    
    def main(self):

        for llm in self.llms:
            message_counts = {} # 初始化为空的字典
            message_counts[llm] = {}
            message_summary = {}
            message_summary['summary_error'] = {}
            message_summary['summary_type'] = {}
            message_summary['summary_type']['total'] = {}

            for prompting in self.promptings:
                # 读取json文件，统计messages信息
                with open(self.report_jsons+f'/{llm}/{self.experiment_id}/{prompting}.json', "r") as file:
                    data_list = json.load(file)

                message_counts[llm][prompting] = []
                if prompting not in message_summary['summary_error']:
                    message_summary['summary_type'][prompting] = {}

                for data in data_list:
                    if data["compile_error"] != str(None):
                        if data["compile_error"] not in message_counts[llm][prompting]:
                            message_counts[llm][prompting].append([
                                {"compile_error":data["compile_error"]},
                                {"error_type":data["error_type"]}
                            ])

                # 汇总不同 llm 下的相同 compile_error数量，保存为summary dict
                for message in message_counts[llm][prompting]:
                    # 对于message["compile_error"]进行预处理
                    error = self.preprocess_error_message(message[0]["compile_error"])

                    # 统计每个错误信息出现的次数
                    if error not in message_summary['summary_error']:
                        message_summary['summary_error'][error] = 0
                    message_summary['summary_error'][error] += 1

                    # 统计每个错误类型出现的次数
                    error_type = message[1]["error_type"]

                    # 总计 total
                    if error_type not in message_summary['summary_type']['total']:
                        message_summary['summary_type']['total'][error_type] = 0
                    message_summary['summary_type']['total'][error_type] += 1
                    # 按照错误类型数量从多到少排序
                    sorted_types = sorted(message_summary['summary_type']['total'].items(), key=lambda x: x[1], reverse=True)
                    message_summary['summary_type']['total'] = dict(sorted_types)

                    # 统计每个错误类型出现的次数,细化到每个提示词下的数量统计
                    if error_type not in message_summary['summary_type'][prompting]:
                        message_summary['summary_type'][prompting][error_type] = 0
                    message_summary['summary_type'][prompting][error_type] += 1
                    # 按照错误类型数量从多到少排序
                    sorted_types = sorted(message_summary['summary_type'][prompting].items(), key=lambda x: x[1], reverse=True)
                    message_summary['summary_type'][prompting] = dict(sorted_types)

                # 按照错误数量从多到少排序
                sorted_errors = sorted(message_summary['summary_error'].items(), key=lambda x: x[1], reverse=True)
                message_summary['summary_error'] = dict(sorted_errors)
                
                
            # Dict2Json(save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l1.json').save(message_counts)
            # Dict2Json(save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l2.json').save(message_summary)

            dict2json(source_dict=message_counts, save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l1.json')
            dict2json(source_dict=message_summary, save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l2.json')
        
        return message_counts

    def postprocess(self):
        """
        对preprocess_error_message处理后的错误信息进行进一步处理
        1. 合并同类项
        2. 打标签 value 扩充为(Python Error, DL Error, Root Cause)
        """
        for llm in self.llms:
            with open(self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l2.json', "r") as file:
                error = json.load(file)

            summary_error = {'Summary':{},'Python Error':{},'DL API Error':{},'Root Cause':{}}
            for key, value in error["summary_error"].items():
                if key == "Add:Inputs have incompatible shapes." or key == "Concatenate:A `Concatenate` layer requires inputs":
                    summary_error['Summary'][key] = ("ValueError", "Input Tensor Shape Mismatch Error", "APIs Call Sequence", value)
                if key == "Unimported API: name API is not defined":
                    summary_error['Summary'][key] = ("NameError", "Missing API Import Error", "Import Error", value)
                if key == "Invalid permutation argument `dims` for Permute Layer":
                    summary_error['Summary'][key] = ("ValueError", "Invalid API Argument Value Error", "Single API Call", value)
                if key == "Conv2D:Keyword argument not understood":
                    summary_error['Summary'][key] = ("TypeError", "Invalid API Argument Type Error", "Single API Call", value)
                if key == "Reshape:Invalid target shape for Reshape layer: dynamic tensor used instead of static shape.":
                    summary_error['Summary'][key] = ("TypeError", "Invalid API Argument Type Error", "Single API Call", value)
                if key == "Reshape:total size of new array must be unchanged":
                    summary_error['Summary'][key] = ("ValueError", "Invalid API Argument Value Error", "Single API Call", value)
                if key == "Lambda:The following Variables were created within a Lambda layer":
                    summary_error['Summary'][key] = ("ValueError", "Invalid API Argument Value Error", "Single API Call", value)    
                if key == "Conv2D:One of the dimensions in the output is <= 0 due to downsampling in":
                    summary_error['Summary'][key] = ("ValueError", "Invalid API Argument Value Error", "Single API Call", value)
                if "Negative dimension size caused by subtracting" in key:
                    summary_error['Summary'][key] = ("ValueError", "InputTensor- API Argument Mismatch Error", "Single API Call", value)
                if "is incompatible with the layer" in key:
                    summary_error['Summary']['incompatible with the layer'] = ("ValueError", "Input Tensor Shape Mismatch Error", "APIs Call Sequence", value)
                # 以上为 GPT-4O-mini 模型的错误信息
                if key == "missing required positional argument":
                    summary_error['Summary'][key] = ("TypeError", "Missing API Argument Error", "Single API Call", value)
                if key == "got an unexpected keyword argument":
                    summary_error['Summary'][key] = ("TypeError", "Unexpected API Argument Error", "Single API Call", value)
                if key == "cannot import name API from DLL":
                    summary_error['Summary'][key] = ("ImportError", "Non-Existent API Import Error", "Import Error", value)
                if key == "module API has no attribute API":
                    summary_error['Summary'][key] = ("AttributeError", "Non-Existent API Attribute Access Error", "Attribute Error", value)
                if key == "No module named":
                    summary_error['Summary'][key] = ("ModuleNotFoundError", "Non-Existent Module Import Error", "Import Error", value)
                if key == "Dimensions or shape mismatch":
                    summary_error['Summary'][key] = ("ValueError", "Input Tensor Shape Mismatch Error", "APIs Call Sequence", value) 
                if key == "received input tensors numbers mismatch":
                    summary_error['Summary'][key] = ("ValueError", "Input Tensor Numbers Mismatch Error", "APIs Call Sequence", value)
                if key == "Graph disconnected: cannot obtain value for tensor":
                    summary_error['Summary'][key] = ("ValueError", "Graph Disconnection Error", "APIs Call Sequence", value)
                if key == "got multiple values for argument":
                    summary_error['Summary'][key] = ("TypeError", "Redundant API Argument Error", "Single API Call", value)
                if key == "Conv2D:Depth of filter must not be 0":
                    summary_error['Summary'][key] = ("ValueError", "Input Tensor Shape Mismatch Error", "APIs Call Sequence", value)
                if key == "TFOpLambda:Failed to convert elements of":
                    summary_error['Summary'][key] = ("TypeError", "Invalid API Argument shape Error", "Single API Call", value)
                if key == "Lambda:Dimensions must be equal":
                    summary_error['Summary'][key] = ("ValueError", "Input Tensor Shape Mismatch Error", "APIs Call Sequence", value)
                if key == "Conv2D:Using a symbolic `tf.Tensor` as a Python `bool` is not allowed in Graph execution.": 
                    summary_error['Summary'][key] = ("OperatorNotAllowedInGraphError", "Invalid API Argument Type Error", "Single API Call", value)
                if key == "'return' outside function":
                    summary_error['Summary'][key] = ("SyntaxError", "Invalid Return Statement Error", "Syntax Error", value)
                if key == "referenced before assignment":
                    summary_error['Summary'][key] = ("UnboundLocalError", "Unassigned Variable Reference Error", "Syntax Error", value)
                if key == "invalid decimal literal":
                    summary_error['Summary'][key] = ("SyntaxError", "Invalid Decimal Literal Error", "Syntax Error", value)
                if key == "unknown activation function":
                    summary_error['Summary'][key] = ("ValueError", "Wrong API Argument Error", "Single API Call", value)
                if key == "Could not build a TypeSpec for KerasTensor":
                    summary_error['Summary'][key] = ("TypeError", "Invalid Input Tensor Type Error", "Single API Call", value)
                if key == "invalid syntax. missing comma":
                    summary_error['Summary'][key] = ("SyntaxError", "Missing Punctuation Error", "Syntax Error", value)
            
            # 统计错误信息
            for key, value in summary_error['Summary'].items():
                python_error_type = value[0]  # 获取Python错误类型
                dl_error_type = value[1]      # 获取DL错误类型
                root_cause = value[2]         # 获取错误根本原因
                count = value[3]              # 获取最后一个位置的数据，即数量
                # 如果 error_type 已经在 statistics 中，累加数量；否则，初始化数量
                if python_error_type in summary_error['Python Error']:
                    summary_error['Python Error'][python_error_type] += count
                else:
                    summary_error['Python Error'][python_error_type] = count
                # 如果 dl_error_type 已经在 statistics 中，累加数量；否则，初始化数量
                if dl_error_type in summary_error['DL API Error']:
                    summary_error['DL API Error'][dl_error_type] += count
                else:
                    summary_error['DL API Error'][dl_error_type] = count
                # 如果 root_cause 已经在 statistics 中，累加数量；否则，初始化数量
                if root_cause in summary_error['Root Cause']:
                    summary_error['Root Cause'][root_cause] += count
                else:
                    summary_error['Root Cause'][root_cause] = count


            # 保存处理后的错误信息
            # Dict2Json(save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l3.json').save(summary_error)

            dict2json(source_dict=summary_error, save_path=self.save_path+f'/{llm}/'+f'{llm}_messages_statistics_l3.json')
                  

if __name__ == '__main__':
    source_report_jsons = "./evaluation/dynamic_checking/report/DeepCodeRAG"
    save_path = "./evaluation/dynamic_checking/report/DeepCodeRAG"

    promptings = ["zeroshot_rag"]
    
    # llms = ["gpt_4o",
    #         "gpt_4o_mini",
    #         "codegemma_7b_it",
    #         "codellama_7b_instruct_hf",
    #         "deepseek_coder_v2_lite_instruct",
    #         "deepseek_v2_lite_chat",
    #         "gemma_2_9b_it",
    #         "meta_llama_3_1_8b_instruct"]

    llms = ["meta_llama_3_1_8b_instruct"]
    experiment_id = "models_experiment_0522_night"


    messages_statistics = MessagesStatistics(
        report_jsons=source_report_jsons,
        llms=llms, 
        promptings=promptings,
        experiment_id=experiment_id,
        save_path=save_path
    )
    # step 1
    message_counts=messages_statistics.main()
    # step 2
    messages_statistics.postprocess()