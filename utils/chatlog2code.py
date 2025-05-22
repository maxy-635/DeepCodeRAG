"""
新增一个功能，将chatlog中的代码提取出来，然后保存到指定文件夹中
包含了之前match_and_save_code.py中的功能
"""

import yaml
from utils.utils import get_all_files
from utils.match_and_save_code import MatchSaveCode


class Chatlog2Code:
    """
    从chatlog中提取代码，并保存到指定文件夹中
    """

    def __init__(self, log_file_path, save_code_path):

        self.log_file_path = log_file_path
        self.save_code_path = save_code_path

    def read_yamldata_to_pycode(self):

        # 打开yaml文件
        file = open(self.log_file_path, "r", encoding="utf-8")
        file_data = file.read()
        file.close()

        # 将字符串转化为字典或列表
        task = yaml.load(file_data, Loader=yaml.FullLoader)
        data = task["response"]  # 既包含文本又包含代码

        # 将列表转换为字符串
        data_str = "\n".join(data)

        # 匹配，提取python代码
        match_saver = MatchSaveCode(
            target_folder=self.save_code_path, response=data_str
        )
        matched_code = match_saver.match_code()

        # 保存代码
        ## task_name此变量用于单独执行chatlog2code.py时的save_code方式采用.默认情况下，此变量不会发生作用。
        task_name = self.log_file_path.split("/")[-2]
        pyfile_name = (
            self.log_file_path.split("/")[-1]
            .replace("yaml", "py")
            .replace("log", "code")
        )
        match_saver.save_code(
            matched_code=matched_code, task_name=task_name, pyfile_name=pyfile_name
        )

        return matched_code


if __name__ == "__main__":
    # 用法示例
    chatlog_yaml_path = "/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG/chat_logs"
    save_code_path = "/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG/response"
    files_type = ".yaml"

    benchmark_yaml_files = get_all_files(
        directory=chatlog_yaml_path, file_type=files_type
    ).get_all_files_in_directory()

    for file in benchmark_yaml_files:
        get_yaml = Chatlog2Code(
            log_file_path=file, save_code_path=save_code_path
        )
        print(file)
        python_code = get_yaml.read_yamldata_to_pycode()
        print(python_code)
