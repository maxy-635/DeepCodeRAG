class FirstPromptDesigner:
    """
    FirstPromptDesigner
    此提示词用于 第一次 LLM 推理, 让LLM 根据任务描述判别得到 candidate_apis
    """

    def __init__(self):
        pass

    def prompt(self, task_requirement, api_doc_chunks):
        """
        此提示词是为了适配方案1，LLM推理+ API 向量检索 + API 关键词检索
        """
        
        prompt = f"""
There is a deep learning code generation task described below:
# Task: \n{task_requirement}\n
From the following API list, select only the most relevant APIs that are necessary to complete the task above.
# API list:\n{api_doc_chunks}\n
Return ONLY valid Python code in the following exact format -- a list of dictionaries, each containing the selected API name:
```python
candidate_apis = [{{'api_name':'...'}}, ...]
```
"""
        return prompt
    
    def prompt_v2(self, task_requirement):
        """
        此提示词是为了适配方案2，LLM推理+ API 关键词检索
        """
        prompt = f"""
There is a deep learning code generation task described below, please provide the most relevant APIs that are necessary to complete the task:\n
# Task: "{task_requirement}"\n
Return ONLY valid Python code in the following exact format -- a list of dictionaries, each containing the selected API name:
```python
candidate_apis = [{{'api_name':'...'}}, ...]
```
"""
        return prompt
