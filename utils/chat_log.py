import os
import datetime
import yaml

class ChatLog:
    """
    将与LLM的对话内容保存到log文件中
    """
    def __init__(self, target_folder, user_prompt, llm_response):
        self.target_folder = target_folder
        self.prompt = user_prompt
        self.response = llm_response

    def save_chat_log(self, number):
        # 将对话内容写入log
        prompt = self.prompt.splitlines()
        response = self.response.splitlines()
        conversation = {'prompt': prompt, 'response': response}
        # 获取当前时间
        current_time = datetime.datetime.now()
        logfile_time = current_time.strftime("%Y_%m_%d_%H_%M")
        logfile_name ='log_' + logfile_time + '_' + str(number) + ".yaml"  # 格式化时间作为文件名+repaired
        file_name = os.path.join(self.target_folder, logfile_name) # 保存路径

        with open(file_name, "w", encoding="utf-8") as file:
            yaml.dump(conversation, file, allow_unicode=True)

        return conversation, file_name
