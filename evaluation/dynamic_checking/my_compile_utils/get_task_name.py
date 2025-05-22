import os
import re


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


if __name__ == '__main__':
    # 示例用法
    path = "/Users/maxy/Documents/pythonProject/my_experiemnt/model-level-testing/DeepGraphCoT/response/llama/fewshot/models_20240510_test"
    subfolder_names = get_task_name(path)
    print(subfolder_names)
