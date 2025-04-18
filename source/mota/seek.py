"""
seek.py - Mota API 核心函数模块

本模块提供 Mota 的核心 API 功能，使其他 Python 程序能够直接调用 Mota 与大语言模型交互，
而无需通过命令行界面。主要功能包括：

1. 统一的 LLM 调用接口：支持多种 LLM 提供商（OpenAI、Anthropic、GROQ 等）
2. 配置管理：加载和处理配置参数
3. 认证处理：安全获取 API 密钥
4. 提示词处理：格式化提示词模板
5. 知识检索：基于向量数据库的 RAG 实现
6. 响应解析：标准化不同 API 的响应格式

主要函数：
- seek: 核心 API 函数，提供与 LLM 交互的完整功能

典型用法：
>>> from mota import seek
>>> response = seek(provider="openai",
...                 model="gpt-4",
...                 prompt="你是一个专业助手",
...                 message="解释量子力学")
>>> print(response['content'])
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

# 从 core 模块导入所有核心功能
from mota.core import (
    setup_logging, load_config, get_api_key, format_prompt, extract_fields,
    get_llm_call_func, get_parser_func, retrieve_context_knowledge,
    logger
)
from edn_format import Keyword


def seek(
        # 必选参数
        provider: str = "openai",
        model: Optional[str] = None,
        prompt: str = "万能的专家系统，我需要帮助。",
        message: str = "",
        temperature: float = 0.7,
        stream: bool = True,
        config_path: Optional[str] = None,
        log_level: str = "INFO",
        log_output: str = "stdout",
        custom_params: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        custom_caller: Optional[Path] = None,
        custom_parser: Optional[Path] = None,
        knowledge_dir: Optional[str] = None,
        user_query: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Mota 核心 API 函数，提供与大语言模型交互的完整功能。

    参数:
        provider (str): LLM 提供商名称，如 "openai", "anthropic", "groq" 等
        model (Optional[str]): 模型名称，如果为 None 则从配置文件加载
        prompt (str): 系统提示词，用于指导模型行为
        message (str): 用户消息内容
        temperature (float): 温度参数，控制输出的随机性
        stream (bool): 是否启用流式响应
        config_path (Optional[str]): 配置文件路径，如果为 None 则使用默认配置
        log_level (str): 日志级别，可选值为 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        log_output (str): 日志输出目标，可以是 "stdout" 或文件路径
        custom_params (Optional[Dict[str, Any]]): 自定义请求参数，将与默认参数合并
        fields (Optional[List[str]]): 需要从响应中提取的字段列表
        custom_caller (Optional[str]): 自定义 LLM API 调用函数的模块路径
        custom_parser (Optional[str]): 自定义响应解析函数的模块路径
        knowledge_dir (Optional[str]): 知识库目录路径，用于 RAG 检索增强生成
        user_query (Optional[List[str]]): 附加的用户查询，将会附加到主要用户消息后

    返回:
        Dict[str, Any]: 包含 LLM 响应内容的字典，通常包含以下字段:
            - content (str): 模型生成的文本内容
            - model (str): 使用的模型名称
            - usage (Dict[str, int]): API 使用统计信息

    示例:
        >>> from mota import seek
        >>> response = seek(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     prompt="你是一个专业的Python教程助手",
        ...     message="解释装饰器的工作原理"
        ... )
        >>> print(response['content'])
    """
    try:
        # 设置日志
        setup_logging(log_level, log_output)
        logger.debug("日志系统已初始化")

        # 处理用户查询参数作为用户消息
        actual_user_message = message

        if user_query:
            logger.debug(f"检测到用户查询参数: {user_query}")
            # 将用户查询添加到用户消息后面
            actual_user_message = f"{actual_user_message} {
                ' '.join(user_query)}"
            logger.debug(f"合并后的用户消息: {actual_user_message}")

        # 加载配置
        config = load_config(config_path)
        logger.debug(f"加载配置: {config}")

        # 获取API密钥
        api_key = get_api_key(provider)
        logger.debug("成功获取到API密钥")

        # 配置请求参数，首先从配置文件获取基础参数
        request_params = {}

        # 从配置文件获取基础参数，注意处理可能的空值引用
        try:
            # 获取模型名称
            if Keyword('llm') in config and Keyword('providers') in config[Keyword('llm')] and Keyword(provider.lower()) in config[Keyword(
                    'llm')][Keyword('providers')] and Keyword('model') in config[Keyword('llm')][Keyword('providers')][Keyword(provider.lower())]:
                request_params["model"] = config[Keyword('llm')][Keyword(
                    'providers')][Keyword(provider.lower())][Keyword('model')]

            # 获取温度参数
            if Keyword('llm') in config and Keyword(
                    'temperature') in config[Keyword('llm')]:
                request_params["temperature"] = config[Keyword(
                    'llm')][Keyword('temperature')]

            # 获取流式响应设置
            if Keyword('llm') in config and Keyword(
                    'stream') in config[Keyword('llm')]:
                request_params["stream"] = config[Keyword(
                    'llm')][Keyword('stream')]

            # 获取最大令牌数
            if Keyword('llm') in config and Keyword(
                    'max_tokens') in config[Keyword('llm')]:
                request_params["max_tokens"] = config[Keyword(
                    'llm')][Keyword('max_tokens')]
        except Exception as e:
            logger.warning(f"从配置文件获取参数时出现异常: {e}，将使用默认值")

        # 添加用户消息参数
        request_params["message"] = actual_user_message

        # 将seek函数参数合并入request_params，这些参数优先级最高
        seek_params = {
            "model": model,
            "temperature": temperature,
            "stream": stream
        }
        for key, value in seek_params.items():
            if value is not None:
                request_params[key] = value
                logger.debug(f"使用seek函数参数 {key}={value}")

        # 解析自定义参数
        if custom_params:
            request_params.update(custom_params)
            logger.debug(f"合并自定义请求参数: {custom_params}")

        logger.debug(f"最终请求参数: {request_params}")

        # 格式化提示词
        formatted_prompt = prompt

        # 如果指定了知识库目录，则进行RAG检索
        if knowledge_dir:
            logger.info(f"检测到知识库目录: {knowledge_dir}，将进行RAG检索")
            # 构建查询字符串，结合系统提示词和用户消息
            query = f"{prompt} {actual_user_message}"
            # 调用RAG检索函数获取相关上下文
            try:
                context_knowledge = retrieve_context_knowledge(
                    knowledge_dir, query)
                # 将检索到的上下文合并为一个字符串
                context_text = "\n\n".join(context_knowledge)
                # 将上下文添加到提示词中
                formatted_prompt = f"{prompt}\n\n参考以下相关信息：\n\n{context_text}"
                logger.info("成功添加RAG检索结果到提示词")
                logger.debug(f"添加RAG后的提示词长度: {len(formatted_prompt)}")
            except Exception as e:
                logger.error(f"RAG检索失败: {e}")
                # 如果RAG检索失败，仍使用原始提示词继续
                logger.info("将使用原始提示词继续")

        # 格式化最终提示词
        formatted_prompt = format_prompt(formatted_prompt, **{})
        logger.debug(f"格式化后的提示词: {formatted_prompt}")

        # 调用 LLM API 的统一接口函数，根据配置参数和自定义函数实现调用逻辑
        llm_call_func = get_llm_call_func(custom_caller)
        response = llm_call_func(
            provider,
            api_key,
            formatted_prompt,
            request_params)

        logger.debug(f"API响应: {response}")

        # 获取解析器
        parse_func = get_parser_func(custom_parser)
        # 解析响应
        parsed_response = parse_func(response)
        logger.debug(f"解析后的响应: {parsed_response}")

        # 提取指定字段
        if fields:
            field_list = fields if isinstance(
                fields, list) else fields.split(',')
            extracted = extract_fields(parsed_response, field_list)
            return extracted
        else:
            return parsed_response

    except Exception as e:
        logger.error(f"调用失败: {e}")
        raise
