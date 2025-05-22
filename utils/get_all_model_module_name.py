from utils.utils import get_all_files # 此处为pylint误报


class GetModelsFiles2Module:
    """
    用于获取 DeepCodeRAG 模型文件的类:批量读取DNN模型python文件的全路径，并转化为模块名
    """

    def __init__(self, directory, file_type, project_name):
        self.project_name = project_name
        self.directory = directory
        self.file_type = file_type

    def get_processing_models_files(self):

        model_files_names = get_all_files(self.directory, self.file_type)
        model_module_modified_files = []
        # 查找 "DeepCodeRAG" 的索引
        for model_file in model_files_names:
            index = model_file.find(self.project_name)
            # 提取包括 "DeepCodeRAG" 之后的字符
            if index != -1:
                # 当把"DeepCodeRAG"为project的根路径时，需要提取 "DeepCodeRAG" 之后的字符。例如：response.DeepCodeRAG.model_**
                extracted_string = model_file[
                    index + len(self.project_name) + 1 : -3
                ]  # 去掉 .py 后缀
                model_module_modified_file = extracted_string.replace(
                    "/", "."
                )  # 将 / 替换为 路径名转化为可直接导入的模块名
                model_module_modified_files.append(model_module_modified_file)
                # print("Extracted String:", modified_string)
            else:
                print("String 'DeepCodeRAG' not found in the path.")

        return model_module_modified_files


if __name__ == "__main__":
    # 用法示例
    models_files_path = "/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG/response"
    models_files_type = ".py"
    project_name = "DeepCodeRAG"
    get_module = GetModelsFiles2Module(
        directory=models_files_path,
        file_type=models_files_type,
        project_name=project_name,
    )
    for i in get_module.get_processing_models_files():
        print(i)
