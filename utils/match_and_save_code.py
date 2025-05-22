import os
import datetime


class MatchSaveCode:
    """
    从LLM生成的代码中按照一定模式提取python代码,并保存到指定的文件夹中
    """

    def __init__(self, target_folder, response):
        self.response = response
        self.target_folder = target_folder

    def match_code(self):
        """
        从LLM生成的代码中按照一定模式提取python代码
        代码模式为:
        1. 代码段以```python开头,以```结尾
        2. 代码段中有def和return关键字
        符合以上两个条件的代码段即为我们需要的python代码,提取并返回,否则返回None
        """
        try:
            # 20240916 改：先提取 ```python *** ```之间的代码.
            completion = self.response
            if "```python" in completion:
                start_line = completion.index("```python")
                completion = completion[start_line:].strip()
                completion = completion.replace("```python", "")
                ending_line = completion.index("```")
                matched = completion[:ending_line].strip()

                # 20240918改：确定代码段中有def和return关键字，说明成功生成了一个完整的函数。
                def_flag = matched.find("def")
                return_flag = matched.find("return")
                if def_flag != -1 and return_flag != -1:
                    return matched
                else:
                    matched = None
                    print("match failed with no python code")
            else:
                matched = None
                print("match failed with no python code")

        except:
            print("match failed with no python code")
            matched = None

        return matched

    def save_code(self, matched_code, task_name, pyfile_name):
        """
        此函数有两种被调用的目的：
        1. 通过chatlog2code.py调用，将代码保存到指定的文件夹中，需要传入task_name变量，用于创建子文件夹
        2. 通过main.py调用，将代码保存到指定的文件夹中.默认为情况2.
        """
        # 情况1:当通过chatlog2code.py调用时，执行下面的代码
        # path = os.path.join(self.target_folder, task_name)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # file_name = os.path.join(path, pyfile_name)

        # 情况2:当通过main.py调用时，执行下面的代码
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        file_name = os.path.join(self.target_folder, pyfile_name)

        if matched_code == None:
            with open(file_name, "w") as file:
                pass  # 20240918改：如果没有匹配到代码，则生成空文件即可
        else:
            print("match succeed")
            with open(file_name, "w") as file:
                file.write(matched_code)


if __name__ == "__main__":
    code = """
```python
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
def dl_model():
    model = tf.keras.models.Sequential([
        Conv2D(filters = 32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPool2D(pool_size=(2,2)),
        Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        MaxPool2D(pool_size=(2,2)),
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dropout(rate=0.5),
        Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```
"""

    matcher = MatchSaveCode(target_folder="./", response=code)
    matched_code = matcher.match_code()
    print("提取的代码为：")
    print(matched_code)
