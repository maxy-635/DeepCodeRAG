from utils.utils import get_all_files, read_yaml_data


class SecondPromptDesigner:
    """
    SecondPromptDesigner
    """

    def __init__(self):
        pass

    def prompt(self, task_requirement, api_docs, dll):

        prompt = f"""
As a developer specializing in deep learning, you are expected to complete the following DL code generation task: \n{task_requirement}\n
Please complete this task using the Functional APIs of {dll}, referring to the provided API documentation below: \n{api_docs}\n
Please import all necessary Functional APIs of {dll}, then complete python code in the 'dl_model()' function and return the constructed 'model'.
```python
def dl_model():

    return model
```
    """

        return prompt

    def prompt_v2(self, task_requirement, api_docs, dll):

        api_doc = "".join(api_docs) # 拼接每个 api_doc 为多段文字,使其可以显示出换行格式
        prompt = f"""
As a developer specializing in deep learning, you are expected to complete the following DL code generation task: \n# Task:\n{task_requirement}\n
Please complete this task according to the API usage constraint information in the following API document: \n{api_doc}\n
Please import all necessary Functional APIs of {dll}, then complete python code in the 'dl_model()' function and return the constructed 'model'.
```python
def dl_model():

    return model
```
    """

        return prompt



if __name__ == "__main__":
    BENCHMARK_YAML_PATH = "./benchmark/DeepEval"
    benchmark_yaml_files = get_all_files(BENCHMARK_YAML_PATH, ".yaml")

    # for i in range(len(benchmark_yaml_files)):
    task = read_yaml_data(benchmark_yaml_files[0])
    task_requirement = task['Requirement']
    prompter = SecondPromptDesigner()
    prompt = prompter.prompt_v2(task_requirement,api_docs=['api1\n','api2\n'], dll="Tensorflow")
    print(prompt)
