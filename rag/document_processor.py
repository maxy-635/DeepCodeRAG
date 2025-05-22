# ========== 文档处理模块 ==========
import os
from utils.utils import read_yaml_data
from process_api_signature import process_api_signature
# from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    文档处理器，用于加载和分割Document
    """
    def __init__(self):
        pass

    def doc2md(self, file_path):
        '''
        处理yaml文档，转换为markdown格式。
        此函数用于构造 API 向量库
        '''
        # 读取 Doc 文档
        yaml_data = read_yaml_data(file_path)

        # 转换为 markdown格式，易于LLM阅读
        md = f"**API Name**: `{yaml_data.get('2 api_name', '')}`"

        # 描述
        description = yaml_data.get("3 api_description", "")
        if description:
            md += f"**API Description**:{description}"

        # 签名
        # signatures = yaml_data.get('4 api_signature', [])
        # if signatures:
        #     md += "**API Signature**:"
        #     for sig in signatures:
        #         md += f"`{sig}`"
        #     md += ""

        # Details
        details = yaml_data.get("5 api_details", "")
        if details:
            md += f"**API Usage Details**:{details}"

        # 参数
        # parameters = yaml_data.get("6 api_parameters", [])
        # if parameters:
        #     md += "**API Parameters**:"
        #     for param in parameters:
        #         for name, desc in param.items():
        #             md += f"[name]: {desc.strip()}"
        #     md += ""
        # else:
        #     md += "**Parameters**:- None"

        # 示例
        # usage = yaml_data.get("7 api_usage_example", [])
        # if usage:
        #     md += "**API Usage Example**:```python"
        #     for line in usage:
        #         md += line.strip(" '") + ""
        #     md += "```"
        # else:
        #     md += "**API Usage Example**:- None"
        
        return md

    def doc2str(self, file_path):
        """
        处理yaml文档，分别将不同部分的信息转换为字符串格式。
        此函数用于构造 API 关键词检索数据库。
        返回API名称、描述、签名、详情、参数和示例的字符串形式的列表。
        """
        # 读取 Doc 文档
        yaml_data = read_yaml_data(file_path)

        # API 名称
        api_name_str = f"**API Name**: {yaml_data.get('2 api_name', '')}"

        # API 描述
        api_description = yaml_data.get("3 api_description", "")
        api_description_str = ""
        if api_description:
            api_description_str += f"**API Description**:{api_description}"

        # API 签名
        api_signatures = yaml_data.get('4 api_signature', [])
        api_signatures_str = ""
        if api_signatures:
            # 对 API签名的后处理
            # 注意：一般api_signatures长度不会超过1
            api_signatures = process_api_signature(api_signatures[0], n=5) # 只保留前5个参数
            api_signatures_str += f"**API Signature**:\n{api_signatures}"

        # API 详情
        api_details = yaml_data.get("5 api_details", "")
        api_details_str = ""
        if api_details:
            api_details_str += f"**API Usage Details**:\n{api_details}"
        else:
            api_details_str += "**API Usage Details**:None"
        
        # maxy0521 增补, 根据api_description和api_details综合生成api_usage_description_str
        # API Usage Description
        api_usage_description_str = ""
        # api_description 或 api_details 存在一个即可
        if api_description or api_details:
            api_usage_description_str += f"**API Usage Description**:\n{api_description} {api_details}"
        # 好像不存在下面的情况
        # if !api_description:
        #     api_usage_description_str += f"**API Usage Description**:\n{api_details}"
        # if !api_details:
        #     api_usage_description_str += f"**API Usage Description**:\n{api_description}"
        

        # API 参数
        api_parameters = yaml_data.get("6 api_parameters", [])
        api_parameters_str = ""
        if api_parameters:
            api_parameters_str += "**API Parameters**:"
            # 只取前5个参数, 避免过长.
            # 注意：即使api_parameters长度大于5, 也可以安全返回结果。Python 的列表切片机制是 容错的
            for param in api_parameters[:5]: 
                for name, desc in param.items():
                    api_parameters_str += f"\n{[name]}: {desc.strip()}"
        else:
            api_parameters_str += "**API Parameters**:None"

        # API 示例
        api_usage = yaml_data.get("7 api_usage_example", [])
        api_usage_str = ""
        if api_usage:
            api_usage_str += "**API Usage Example**:\n```python\n"
            for line in api_usage:
                api_usage_str += line.strip(" '") + "\n"
            api_usage_str += "```"
        else:
            api_usage_str += "**API Usage Example**:None"


        return api_name_str, api_description_str, api_signatures_str, api_details_str, api_usage_description_str, api_parameters_str, api_usage_str


if __name__ == "__main__":
    # 测试代码
    file_path = "./api_parser/tensorflow/apis_parsed_results/keras/tf.keras.layers.Conv2D_ae7e8808.yaml"
    # file_path = "api_parser/tensorflow/apis_parsed_results/keras/tf.keras.layers.Flatten_799c6e80.yaml"

    processor = DocumentProcessor()
    # processed_text = processor.doc2md(file_path)
    # print(processed_text)

    processed_text = processor.doc2str(file_path)
    for text in processed_text:
        print(text)