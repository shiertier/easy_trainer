JOY2_QUESTION_PROMPT_STR = "describe the following image in detail:" # 提问

# LLM 设置
JOY2_TEMPERATURE_FLOAT = 0.5 # 控制生成文本的随机性
JOY2_TOP_K_INT = 10 # 限制采样池，在每一步选择最有可能的选项
JOY2_TOP_P_FLOAT = 0.9
JOY2_MAX_NEW_TOKENS_INT = 300  # 生成文本的最大长度

JOY2_IS_GREEDY_BOOL = False # 是否使用贪婪解码

JOY2_ADD_PREPEND_STR = "" # 添加到生成的文本前
JOY2_ADD_APPEND_STR = "" # 添加到生成的文本后