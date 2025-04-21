"""
core.py - Mota核心功能模块

本模块提供Mota与大语言模型(LLM)交互的核心功能实现，包括：
1. 配置管理：加载和处理EDN格式配置文件
2. 认证处理：多种API密钥获取方式
3. 提示词处理：格式化和处理提示词模板
4. API调用：统一的LLM API调用接口
5. 响应解析：标准化不同API的响应格式
6. 知识检索：基于向量数据库的RAG实现

主要功能组件：
- 日志系统配置
- 配置文件加载与解析
- API密钥安全管理
- 提示词模板格式化
- 自定义模块动态加载
- 响应解析与字段提取
- 知识库检索与上下文增强

本模块作为Mota的核心功能层，被main.py调用以实现完整的命令行工具功能。
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Callable, List
import edn_format
from pathlib import Path
import gnupg  # 用于处理加密的 .authinfo.gpg
import json
import typer

# 导入 LangChain 相关模块
from langchain_community.document_loaders import DirectoryLoader  # 用于递归加载目录下的各种文档格式
# 向量化嵌入模型，HuggingFaceEmbeddings通用性较好
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # 使用 FAISS 构建向量索引以便于高效检索

from mota.custom_interface import LLMCallerInterface, ResponseParserInterface
from mota.loader import load_module_from_path, find_implementor


# 初始化 logger
logger = logging.getLogger(__name__)

# 定义 OpenAI 客户端（仅在需要时初始化）
openai_client = None


class JsonDict:
    """
    自定义类型，用于将 JSON 格式的字符串解析为 Python 字典。

    该类通过静态方法实现 JSON 字符串的解析，支持空值处理和错误捕获。
    主要用于 Typer 命令行参数的类型转换，确保输入的 JSON 字符串能够正确转换为字典对象。
    """

    @staticmethod
    def __get_validators__():
        """
        为 Typer 的类型系统提供验证器。

        Yields:
            Callable: 返回解析 JSON 的静态方法 `parse_json`。
        """
        yield JsonDict.parse_json

    @staticmethod
    def parse_json(value: Optional[str]) -> Optional[Dict]:
        """
        将 JSON 格式的字符串解析为 Python 字典。

        Args:
            value (Optional[str]): 输入的 JSON 字符串，可能为 None。

        Returns:
            Optional[Dict]: 解析后的字典对象，如果输入为 None 则返回 None。

        Raises:
            typer.BadParameter: 如果输入的 JSON 字符串格式无效，抛出异常并附带错误信息。
        """
        if value is None:
            return None
        try:
            # 尝试解析 JSON 字符串为字典
            return json.loads(value)
        except json.JSONDecodeError as e:
            # 捕获 JSON 解析错误，抛出用户友好的错误信息
            raise typer.BadParameter(f"无效的 JSON 格式: {e}")
        except TypeError as e:
            # 捕获类型错误，例如输入非字符串类型
            raise typer.BadParameter(f"输入类型错误: {e}")


def setup_logging(level: str = "INFO", output: str = "stdout") -> None:
    """
    设置日志配置

    Args:
        level (str): 日志级别
        output (str): 日志输出目标
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {level}")

    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': numeric_level
    }

    if output.lower() == "stdout":
        logging_config['stream'] = sys.stdout
    else:
        logging_config['filename'] = output

    logging.root.handlers.clear()
    logging.basicConfig(**logging_config)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载EDN格式的配置文件

    Args:
        config_path (Optional[str]): 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典
    """
    default_config_path = Path(__file__).parent / "config" / "default.edn"

    try:
        with open(config_path or default_config_path, 'r') as f:
            return edn_format.loads(f.read())
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise


def get_api_key(provider: str,
                auth_source: str = "env",
                auth_path: Optional[str] = None) -> str:
    """
    获取API密钥

    Args:
        provider (str): LLM提供商名称
        auth_source (str): 认证源（env/command_line/authinfo/config）
        auth_path (Optional[str]): 认证文件路径

    Returns:
        str: API密钥
    """
    # 从环境变量获取
    if auth_source == "env":
        key = os.getenv(f"{provider.upper()}_API_KEY")
        if key:
            return key

    # 从命令行参数获取
    if auth_source == "command_line":
        # 假设通过命令行传入的密钥
        key = os.getenv(f"{provider.upper()}_API_KEY_CMD")
        if key:
            return key

    # 从 EMACS .authinfo 或 .authinfo.gpg 获取
    if auth_source in ["authinfo", "authinfo_gpg"]:
        auth_file = auth_path or os.path.expanduser("~/.authinfo")
        if auth_source == "authinfo_gpg":
            auth_file += ".gpg"
            gpg = gnupg.GPG()
            with open(auth_file, 'rb') as f:
                decrypted_data = gpg.decrypt_file(f)
                if decrypted_data.ok:
                    auth_data = decrypted_data.data.decode()
                else:
                    logger.error("解密 .authinfo.gpg 失败")
                    raise ValueError("无法解密 .authinfo.gpg 文件")
        else:
            with open(auth_file, 'r') as f:
                auth_data = f.read()

        for line in auth_data.splitlines():
            if line.startswith(f"{provider.lower()}-api-key"):
                return line.split()[1]

    # 从配置文件获取
    if auth_source == "config":
        config = load_config()
        return config.get(
            'llm',
            {}).get(
            'providers',
            {}).get(
            provider.lower(),
            {}).get(
                'api_key',
            '')

    raise ValueError(f"无法找到 {provider} 的API密钥")


def format_prompt(template: str, **kwargs: Any) -> str:
    """
    格式化提示词模板

    Args:
        template (str): 提示词模板
        kwargs (Any): 模板参数

    Returns:
        str: 格式化后的提示词
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"缺少模板参数: {e}")
        raise
    except Exception as e:
        logger.error(f"格式化提示词失败: {e}")
        raise


def default_parse(response: Any) -> Dict[str, Any]:
    """
    解析API响应

    Args:
        response (Any): API响应对象，可以是流式（Stream）或非流式（ChatCompletion）
        custom_parser (Optional[Callable]): 自定义解析函数

    Returns:
        Dict[str, Any]: 解析后的响应内容
    """
    # 默认解析逻辑
    try:
        # 处理 OpenAI 流式响应
        if hasattr(
                response,
                '__iter__') and not hasattr(
                response,
                'choices'):  # 检查是否为流式响应
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_content += chunk.choices[0].delta.content
            return {
                'content': full_content,
                # 流式响应可能无 model 属性
                'model': response.model if hasattr(response, 'model') else None,
                'usage': response.usage._asdict() if hasattr(response, 'usage') else {}
            }
        # 处理 OpenAI 非流式响应
        else:
            return {
                'content': response.choices[0].message.content,
                'model': response.model,
                'usage': response.usage._asdict() if hasattr(
                    response,
                    'usage') else {}}
    except Exception as e:
        logger.error(f"解析响应失败: {e}")
        raise


def extract_fields(response_dict: Dict[str, Any],
                   fields: List[str]) -> Dict[str, Any]:
    """
    从响应字典中提取指定字段

    Args:
        response_dict (Dict[str, Any]): 解析后的响应字典
        fields (List[str]): 需要提取的字段列表

    Returns:
        Dict[str, Any]: 提取的字段及其值
    """
    extracted = {}
    for field in fields:
        if field in response_dict:
            extracted[field] = response_dict[field]
        else:
            logger.warning(f"字段 {field} 在响应中不存在")
    return extracted


def default_llm_call(provider: str,
                     api_key: str,
                     formatted_prompt: str,
                     request_params: Dict[str,
                                          Any]) -> Any:
    """
    默认的 LLM API 调用函数，根据提供商名称调用相应的 LLM API.

    支持基于配置参数实现 OpenAI API 的调用。
    其他 LLM 提供商通过自定义模块实现。

    Args:
        provider (str): LLM 提供商名称，目前仅支持 "openai"。
        api_key (str): API 认证密钥。
        formatted_prompt (str): 格式化后的提示词。
        request_params (Dict[str, Any]): 请求参数，包括模型名称、温度、流模式、最大 token 数等。

    Returns:
        Any: LLM API 的响应对象。

    Raises:
        NotImplementedError: 当指定提供商的调用逻辑未实现时。
    """
    provider_lower = provider.lower()
    if provider_lower == "openai":
        from openai import OpenAI
        global openai_client
        if openai_client is None:
            openai_client = OpenAI(api_key=api_key)

        # 构建消息列表
        messages = [{"role": "system", "content": formatted_prompt}]

        # 添加用户消息到消息列表中
        user_message = request_params.get("message")
        messages.append({"role": "user", "content": user_message})

        return openai_client.chat.completions.create(
            model=request_params["model"],
            messages=messages,
            temperature=request_params["temperature"],
            stream=request_params["stream"],
            max_tokens=request_params["max_tokens"]
        )
    # 其他 LLM 提供商通过自定义模块实现
    else:
        logger.error(f"尚未实现 {provider} 提供商的API调用逻辑")
        raise NotImplementedError(f"{provider} 提供商的API调用逻辑未实现")


def load_custom_func(
        module_name: str,
        module_path: Path,
        interface_class: type,
        method_name: str
) -> Callable:
    """
    加载用户自定义函数。

    从指定的模块路径加载模块，并在其中查找实现了指定接口类的类。
    然后，实例化该类并返回其实例的指定方法。

    Args:
        module_name (str): 模块名，例如 "custom_caller"。
        module_path (Path): 模块路径，用户提供的自定义模块文件路径。
        interface_class (type): 要查找的接口类，例如 LLMCallerInterface。
        method_name (str): 要获取的方法名，例如 "call"。

    Returns:
        Callable: 实现了指定接口的类的实例的指定方法。

    Raises:
        ValueError: 如果在模块中未找到指定接口的实现类。
        AttributeError: 如果指定方法名不可调用。
        Exception: 当加载模块、查找实现类或获取方法失败时。
    """
    try:
        module = load_module_from_path(module_name, module_path)
        logger.debug(f"模块内容: {dir(module)}")  # 打印模块所有属性
        implementor_class = find_implementor(module, interface_class)
        if implementor_class is None:
            logger.error(f"错误：在提供的模块中未找到“{interface_class.__name__}”的实现类。")
            raise ValueError(f"未找到实现类：“{interface_class.__name__}”。")
        instance = implementor_class()
        func = getattr(instance, method_name)
        if not callable(func):
            raise AttributeError(f"“{method_name}”不是可调用方法")
        return func
    except Exception as e:
        logger.error(f"加载用户自定义函数失败: {e}")
        raise


def get_llm_call_func(custom_caller: Optional[Path]) -> Callable:
    """
    获取 LLM API 调用函数.

    如果用户提供了自定义的 LLM API 调用函数路径，则导入该函数。
    否则，使用默认的 LLM API 调用函数。

    Args:
        custom_caller (Optional[str]): 用户自定义函数路径。

    Returns:
        Callable: 用于调用 LLM API 的函数，必须实现 LLMCallerInterface 接口

    Raises:
        Exception: 当自定义函数导入失败时。
    """
    if custom_caller:
        try:
            func = load_custom_func(
                module_name="custom_caller",
                module_path=custom_caller,
                interface_class=LLMCallerInterface,
                method_name="call"
            )
            return func
        except Exception as e:
            logger.error(f"加载用户自定义LLM调用函数失败: {e}")
            raise
    else:
        return default_llm_call


def get_parser_func(custom_parser: Optional[Path]) -> Callable:
    """
    获取响应解析函数

    导入用户自定义的响应解析函数并验证其接口合规性。

    如果用户提供了自定义的响应解析函数路径，则导入该函数。否则，使用默认的响应解析函数。


    Args:
        custom_parser (Option[Path]): 用户自定义函数路径。

    Returns:
        Callable: 实现 ResponseParserInterface 的解析函数

    Raises:
        Exception: 当函数导入失败或接口不兼容时
    """
    if custom_parser:
        try:
            func = load_custom_func(
                module_name="custom_parser",
                module_path=custom_parser,
                interface_class=ResponseParserInterface,
                method_name="parse"
            )
            return func
        except Exception as e:
            logger.error(f"加载用户自定义LLM响应解析函数失败: {e}")
            raise
    else:
        return default_parse


def retrieve_context_knowledge(
        directory_path: str,
        query: str,
        top_k: int = 5) -> List[str]:
    """
    从指定目录递归加载文档，并基于问题描述检索出相关的上下文文本。

    参数:
        directory_path (str): 存放文档文件的根目录路径。该目录下会递归查找 DOCX、PDF、TXT 等文件。
        query (str): 用户输入的问题描述，用于检索相关的文档上下文。
        top_k (int): 检索返回的最相关文档数量，默认为 5。

    返回:
        List[str]: 一个字符串列表，每个字符串为一个检索出的文档的上下文文本内容。

    实现步骤:
        1. 使用 UnstructuredDirectoryLoader 递归加载指定目录下的所有支持的文档文件。
        2. 利用 OpenAIEmbeddings 将文档转换为向量表示。
        3. 使用 FAISS 向量存储构建文档索引。
        4. 基于用户输入的 query 进行相似度搜索，返回最相关的文档。
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        raise ValueError(f"指定的目录不存在：{directory_path}")

    # 1. 加载目录下的所有文档
    # 参数 recursive=True 表示递归读取子目录中的文件
    # 注意：DirectoryLoader 会自动根据文件后缀（如 .pdf, .docx, .txt）加载文档
    loader = DirectoryLoader(directory_path, recursive=True)
    logger.info("开始加载文档...")
    documents = loader.load()
    logger.info(f"共加载到 {len(documents)} 个文档。")

    if len(documents) == 0:
        raise ValueError(f"目录 {directory_path} 未加载到任何文档，请检查文件格式和内容")

    # 2. 初始化嵌入模型
    # 使用HuggingFaceEmbeddings的嵌入模型进行向量化
    # "all-MiniLM-L6-v2"（小）和"all-mpnet-base-v2"（大）
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # 3. 构建向量存储索引（FAISS）
    # FAISS.from_documents 会对每个文档调用嵌入模型，将文档转换为向量，并建立索引以支持快速检索
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 4. 根据用户的问题描述进行相似度搜索，返回 top_k 个最相关的文档
    logger.info("开始进行相似度搜索...")
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)

    # 提取检索到的文档内容（page_content 为文档文本内容）
    context_knowledge = [doc.page_content for doc in retrieved_docs]
    logger.debug("RAG检索到的上下文内容: %s", context_knowledge)
    logger.info("完成相似度搜索。")

    # 返回结果列表
    return context_knowledge
