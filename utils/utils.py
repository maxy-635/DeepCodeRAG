import os
import re
import yaml
import json
from natsort import natsorted

def read_yaml_data(yaml_file):
    '''Read data from a YAML file.
    Args:
        yaml_file (str): Path to the YAML file.
    Returns:
        dict: Data read from the YAML file.
    '''
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data,Loader=yaml.FullLoader)#dict

    return data


def get_all_files(directory, file_type):
    """
    获取指定目录下的所有指定类型的文件。

    Args:
        directory (str): 要搜索的目录路径。
        file_type (str): 要搜索的文件类型（如 ".html"）。

    Returns:
        list: 按自然排序的文件路径列表。
    """
    def _get_files_in_directory(current_directory):
        # 获取指定目录中的所有文件和子文件夹
        files = os.listdir(current_directory)
        required_files = []

        # 遍历文件和子文件夹
        for file in files:
            # 构建完整的文件路径
            full_path = os.path.join(current_directory, file)
            # 判断是否为文件
            if os.path.isfile(full_path):
                # 检查文件是否是要求的文件类型
                if full_path.endswith(file_type):
                    # 将文件路径添加到列表
                    required_files.append(full_path)
            # 检查是否是目录
            elif os.path.isdir(full_path):
                # 如果是目录，递归调用函数，并将结果合并到当前列表
                required_files.extend(_get_files_in_directory(full_path))

        return required_files

    # 调用内部递归函数获取所有文件
    all_files = _get_files_in_directory(directory)

    # 按照文件名中的数字进行自然排序
    sorted_files = natsorted(all_files)

    return sorted_files


def get_llm_name(input):
    """
    根据model_id获取llm名称,便于保存文件
    注意 gpt-4o-mini-finetuned 对应 第二次微调出来的模型。ft:gpt-4o-mini-2024-07-18:personal::B1p5w0dT
    """
    # 提取模型名称
    output = re.sub(r"[^a-z0-9]+", "_", input.split("/")[-1].lower()).strip("_")

    return output



def dict2json(source_dict, save_path):
    """
    将字典保存为JSON文件
    """
    # 将字典保存为JSON文件
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(source_dict, json_file, indent=2)  # indent=2表示缩进为2个空格


def get_task_name(path)->list:
    """
    获取指定路径下的所有文件夹名称，并按照自然排序的方式进行排序
    目标是获得一批实验结果中的具体任务名称
    """
    def natural_sort_key(s):
        # 使用正则表达式拆分字符串中的数字和非数字部分
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    # 获取指定路径下的所有条目
    entries = os.listdir(path)
    # 过滤掉非目录项，只保留文件夹
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    subfolders.sort(key=natural_sort_key)
    return subfolders