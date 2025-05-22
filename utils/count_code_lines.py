def count_code_lines(file_path):
    """
    Count the number of non-empty and non-comment lines in a Python file
    统计每个python文件中不包括空行和注释行 (""" """和#) 的代码行数，用于计算score
    """
    in_comment_block = False
    code_lines = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if in_comment_block:
                if '"""' in line or "'''" in line:
                    in_comment_block = False
                continue
            if line.startswith('"""') or line.startswith("'''"):
                in_comment_block = True
                # Check if the comment block ends on the same line
                if line.count('"""') == 2 or line.count("'''") == 2:
                    in_comment_block = False
                continue
            if not line or line.startswith("#"):
                continue
            code_lines += 1

    return code_lines


if __name__ == "__main__":

    from utils import get_all_files # 此处为pylint误报
    files = get_all_files("response/DeepCodeRAG", ".py")

    count = 0
    for file in files:
        total_code_lines = count_code_lines(file)
        if total_code_lines != 0:
            count += 1
        else:
            print('empty file',file)

    # print("Total number of 非空 Python files:", count)
    # print("空 Python files:", len(files) - count)
