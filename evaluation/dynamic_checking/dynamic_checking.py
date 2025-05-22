"""
借用了DeepEvalRAG的代码，进行编译验证
"""

import os
import json
import importlib
import keras
import pandas as pd
from tabulate import tabulate
from utils.utils import get_task_name
from utils.df2excel import DataFrame2Excel
from utils.get_all_model_module_name import GetModelsFiles2Module
from utils.count_code_lines import count_code_lines


class Validation:
    """
    对DNN模型进行编译验证
    """

    def __init__(self):
        pass

    def compile_code(self, model_lib):
        """
        编译DNN模型
        """
        try:
            module = importlib.import_module(model_lib)
            print("module:", module)
            model = module.dl_model()  # 提示词中已经改为了dl_model()
            model.summary()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            compile_status = "compile success"
            compile_error = None
            error_type = None

        except Exception as error:
            compile_status = "compile failed"
            compile_error = error
            error_type = type(error).__name__  # 获取异常的类型名称

        verification = {
            "model_lib": str(model_lib),
            "compile_status": str(compile_status),
            "error_type": str(error_type),
            "compile_error": str(compile_error),
        }
        return verification

    def batch_compile_models(self, models_path):
        """
        获取所有模型库
        """
        model_lib_seqs = GetModelsFiles2Module(
            directory = models_path,
            file_type = ".py", 
            project_name="DeepCodeRAG"
        ).get_processing_models_files()
        verifications = []

        for model_lib in model_lib_seqs:
            try:
                verification = self.compile_code(model_lib)
                verifications.append(verification)
                print(f"-----编译验证成功: {model_lib} -----")

            except Exception as e:
                print(traceback.format_exc())
                # 1216修改
                # 有些python文件可能在import的时候就会报错，同样需要记录verification信息
                verifications.append({
                    "model_lib": str(model_lib),
                    "compile_status": "compile failed",
                    "compile_error": str(e),
                    })
                print(f"-----编译验证失败: {e} -----")

        return verifications

    def main(self, promptings, llms, experiment_batch):
        
        global df_data, pd_rows
        for llm in llms:
            dfs_last_row, dfs_last_col = [], []
            for prompting in promptings:
                models_path = os.path.join(
                    common_root_path, f"response/DeepCodeRAG/{llm}/{prompting}/{experiment_batch}"
                )
                verifications = self.batch_compile_models(models_path)  # 编译验证

                save_json_path = os.path.join(
                    common_root_path,
                    f"evaluation/dynamic_checking/report/DeepCodeRAG/{llm}/{experiment_batch}/{prompting}.json",
                )
                if not os.path.exists(os.path.dirname(save_json_path)):
                    os.makedirs(os.path.dirname(save_json_path))

                with open(save_json_path, "w") as f:  # 保存编译验证结果到json文件
                    json.dump(verifications, f, indent=2)  # indent=2表示缩进为2个空格

                tasks_names = get_task_name(models_path)
                count_dict_list = []

                for task in tasks_names:  # task是具体的benchmark任务名称，目前一共有10个任务
                    # 0728 修改，计算每个任务下的有效python文件数
                    # 计算每个任务下的有效python文件数
                    valid_pyfiles = 0
                    for pyfile in os.listdir(os.path.join(models_path, task)):
                        if pyfile.endswith(".py"):
                            if count_code_lines(os.path.join(models_path, task, pyfile))!= 0:
                                valid_pyfiles += 1

                    count_dict = {
                        "Benchmark": task,
                        "success": 0,
                        "failed": 0,
                        "valid_pyfiles": 0,
                        "rate(%)": 0,
                    }

                    for item in verifications:
                        if task + "." in item["model_lib"]:  # 注意这里必须加上"."，否则会出现HumanEval_DLtask_1和HumanEval_DLtask_10的问题
                            if item["compile_status"] == "compile success":
                                count_dict["success"] += 1
                            else:
                                count_dict["failed"] += 1

                    count_dict["valid_pyfiles"] = valid_pyfiles

                    # 1202修改：采用valid_pyfiles作为分母；
                    # 同时对 空文件对应的valid_pyfiles为0的情况进行处理，count_dict 中的 rate(%) 为0
                    if valid_pyfiles != 0:
                        count_dict["rate(%)"] = 100 * count_dict["success"] / valid_pyfiles
                    else:
                        count_dict["rate(%)"] = 0

                    count_dict_list.append(count_dict)

                df_data = pd.DataFrame(count_dict_list)
                # 增加一行，统计每列的总和
                Total = df_data.sum(axis=0)
                Total["Benchmark"] = "Total"
                df_data = pd.concat([df_data, Total.to_frame().T], ignore_index=True)

                # 提取最后一行的 success 和 failed 数据
                total_success = df_data.loc[df_data["Benchmark"] == "Total", "success"].values[0]
                # total_failed = df_data.loc[df_data["Benchmark"] == "Total", "failed"].values[0]

                # 采用valid_pyfiles作为分母
                total_rate = 100 * total_success / df_data.loc[df_data["Benchmark"] == "Total", "valid_pyfiles"]

                df_data.loc[df_data["Benchmark"] == "Total", "rate(%)"] = total_rate  # 更新 DataFrame 中的 rate 列

                print(tabulate(df_data, headers="keys", tablefmt="grid"))
                save_result_excel_path = os.path.join(
                    common_root_path,
                    f"evaluation/dynamic_checking/report/DeepCodeRAG/{llm}/{llm}_{experiment_batch}.xlsx",
                )
                DataFrame2Excel(df_data, save_result_excel_path).df2excel(sheet_name=prompting)

                total_row = df_data.iloc[-1].copy()  # 提取最后一行数据，total
                total_row["Benchmark"] = str(prompting)
                dfs_last_row.append(total_row)  # 保存每个提示词对应的total数据

                total_col = df_data.iloc[:, -1].copy()  # 提取最后一列数据，total
                total_col.name = str(prompting)
                dfs_last_col.append(total_col)

            first_col = df_data.iloc[:, 0].copy()  # 提取第一列数据，benchmark,task name
            dfs_last_col.insert(0, first_col)  # 将benchmark,task name插入到第一列

            # 所有提示词对应数据的总和保存为一个sheet(summary),
            summary_df = pd.DataFrame(dfs_last_row)
            summary_df.reset_index(drop=True, inplace=True)
            print(tabulate(summary_df, headers="keys", tablefmt="grid"))

            save_result_excel_path = os.path.join(
                common_root_path,
                f"evaluation/dynamic_checking/report/DeepCodeRAG/{llm}/{llm}_{experiment_batch}.xlsx",
            )
            DataFrame2Excel(summary_df, save_result_excel_path).df2excel(sheet_name="summary")

            # 0721 增加一个total_summary,提取每个sheet的最后一行数据，保存为一个sheet
            total_summary_df = pd.DataFrame(dfs_last_col).T[:-1]
            print(tabulate(total_summary_df, headers="keys", tablefmt="grid"))

            save_result_excel_path = os.path.join(
                common_root_path,
                f"evaluation/dynamic_checking/report/DeepCodeRAG/{llm}/{llm}_{experiment_batch}.xlsx",
            )
            
            DataFrame2Excel(total_summary_df, save_result_excel_path).df2excel(sheet_name="total_summary")
            print(f"Dynamic Checking for {llm} {experiment_batch} is finished.")


if __name__ == "__main__":
    common_root_path = "/Users/maxy/Documents/pythonProject/my_experiemnt/DeepCodeRAG"
    promptings =["zeroshot_rag"]

    # llms = ["gpt_4o",
    #         "gpt_4o_mini",
    #         "codegemma_7b_it",
    #         "codellama_7b_instruct_hf",
    #         "deepseek_coder_v2_lite_instruct",
    #         "deepseek_v2_lite_chat",
    #         "gemma_2_9b_it",
    #         "meta_llama_3_1_8b_instruct"]

    llms = ["meta_llama_3_1_8b_instruct"]
    # llms = ["gpt_4o_mini"]

    experiment_id = "models_DeepEvalRAG_0519_v3"

    compile_validation = Validation()
    compile_validation.main(promptings, llms, experiment_id)
