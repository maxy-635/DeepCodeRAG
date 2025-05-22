import re

def process_api_signature(signature: str, n: int = 5) -> str:
    """
    处理函数签名，移除默认值并保留前 n 个参数。

    :param signature: 原始函数签名字符串
    :param n: 要保留的参数数量
    :return: 处理后的函数签名
    """
    # step1. 移除默认值
    def remove_defaults(signature: str) -> str:
        pattern = r'(\w+)\s*=\s*(?:\([^\)]*\)|"[^"]*"|\'[^\']*\'|[^,\)]*)'
        cleaned_signature = re.sub(pattern, r'\1', signature)
        cleaned_signature = re.sub(r'\s+', '', cleaned_signature)  # 去除多余空格
        cleaned_signature = re.sub(r',+', ',', cleaned_signature)  # 避免多个逗号
        cleaned_signature = re.sub(r'\(,', '(', cleaned_signature) # 修正 (,param)
        cleaned_signature = re.sub(r',\)', ')', cleaned_signature) # 修正 (param,)

        return cleaned_signature

    # 提取函数名和参数部分
    # 注意需要先判断是否有括号，比如@tf_contextlib.contextmanager就没有括号
    if '(' in signature and ')' in signature:

        func_name, args_str = signature.split('(', 1)
        args_str = args_str.rsplit(')', 1)[0]
        args_only = remove_defaults(f"({args_str})")

        # 拼接移除默认值后的签名
        cleaned_signature = f"{func_name}{args_only}"

        # step2. 保留前 n 个参数
        start = cleaned_signature.find('(')
        end = cleaned_signature.rfind(')')
        if start == -1 or end == -1 or end < start:
            return cleaned_signature  # 无效的函数签名

        prefix = cleaned_signature[:start+1]
        param_str = cleaned_signature[start+1:end]
        suffix = cleaned_signature[end:]

        # 分割参数
        params = [p.strip() for p in param_str.split(',')]
        kept_params = params[:n]
        
        # 重新拼接
        new_param_str = ', '.join(kept_params)

        output = f"{prefix}{new_param_str}{suffix}"

    else:
        # 如果没有括号，直接返回原始签名
        output = signature
    
    return output



if __name__ == "__main__":
    # 示例字符串
    signature = "tf.keras.layers.Conv2D(filters,kernel_size,strides=(1,1),padding='valid',data_format=None,dilation_rate=(1,1),groups=1,activation=None,use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs)"

    # 调用整合后的函数
    print(process_api_signature(signature, 5))
