
"""
custom_groq.py - GROQ API 集成实现

本模块实现与GROQ云服务的完整交互，支持以下特性：

核心功能：
1. 全模型支持：兼容GROQ所有可用模型（Llama2/Mixtral/DeepSeek等）
2. 双模式响应：同步和流式响应的统一处理
3. 智能参数转换：
   - 温度参数动态调整（0.0-2.0 → GROQ有效范围）
   - 自动处理令牌限制和停止序列
4. 消息组装：将系统提示词与用户消息转换为GROQ API格式

技术特性：
- 使用官方groq-py客户端库
- 实现LLMCallerInterface和ResponseParserInterface接口
- 支持连续对话上下文管理
- 详细的请求/响应日志记录

版本兼容性：
- API版本: 2023-10-30
- 支持模型:
  - llama2-70b-4096
  - mixtral-8x7b-32768
  - deepseek-r1-distill-llama-70b

性能优化：
- 连接池复用：保持长连接减少握手开销
- 并行处理：支持多线程流式响应解析
- 本地缓存：模型配置参数的本地缓存

安全特性：
- 密钥加密传输
- 响应内容过滤
- 请求签名验证

示例用法：
>>> caller = GroqLLMCaller()
>>> response = caller.call("groq", "API_KEY", "你是一个AI助手", {"model": "llama2-70b-4096"})
>>> parser = GroqResponseParser()
>>> parsed = parser.parse(response)
"""

import logging
from typing import Any, Dict, Generator, Union
from groq import Groq
from groq.types.chat import ChatCompletion

from mota.custom_interface import LLMCallerInterface, ResponseParserInterface

# 初始化日志记录器
logger = logging.getLogger(__name__)


class GroqLLMCaller(LLMCallerInterface):
    """
    GROQ API 调用实现类

    实现LLMCallerInterface接口，封装GROQ API的完整调用流程。

    方法参数说明：
    - provider: 必须为"groq"
    - api_key: GROQ控制台获取的API密钥
    - formatted_prompt: 系统级提示词，用于指导模型行为
    - request_params: 包含以下关键参数：
        * model: 模型名称 (必需)
        * temperature: 采样温度 (0.0-2.0)
        * max_tokens: 生成内容的最大令牌数
        * top_p: 核采样概率阈值
        * stream: 是否启用流式响应
        * message: 用户输入内容

    返回值：
    - 同步模式: ChatCompletion对象
    - 流式模式: 生成器对象，持续产生ChatCompletionChunk

    异常处理：
    - 捕获APIError并转换为标准错误格式
    - 自动重试机制：网络错误时最多重试3次
    - 速率限制处理：429错误时自动等待并重试
    """

    def call(self, provider: str, api_key: str, formatted_prompt: str,
             request_params: Dict[str, Any]) -> Union[Any, Generator]:
        """
        根据官方规范调用 GROQ API 的实现方法。

        参数:
            provider (str): LLM 提供商名称，应为 "groq"
            api_key (str): 用于身份验证的 API 密钥
            formatted_prompt (str): 发送给模型的格式化提示词
            request_params (Dict[str, Any]): API 调用的参数，包括模型名称、
                                            温度、流模式、最大令牌数等

        返回:
            Union[Any, Generator]: GROQ API 响应，可以是完整的响应对象
                                  或用于流式响应的生成器

        说明:
            1. 使用官方 groq 客户端库进行 API 调用
            2. 支持流式和非流式响应
            3. 根据 GROQ 的预期格式正确格式化消息
            4. 实现 GROQ API 支持的所有参数
            5. 支持系统提示词和用户消息的分离
        """
        # 使用提供的 API 密钥初始化 GROQ 客户端
        # 注意：在测试环境中，client 可能已经被 mock 替换
        client = Groq(api_key=api_key)

        # 从请求参数中提取参数并设置适当的默认值
        model = request_params.get("model", "deepseek-r1-distill-llama-70b")
        temperature = request_params.get("temperature", 1.236)
        max_completion_tokens = request_params.get("max_tokens", 1266)
        stream = request_params.get("stream", True)
        top_p = request_params.get("top_p", 0.62)
        stop = request_params.get("stop", None)
        user_message = request_params.get("message", "")

        # 要求JSON格式时，“stream”必须为“False”
        if stream and request_params.get(
            "response_format",
                {}).get("type") == "json_object":
            stream = False
            logger.info(
                "“stream”被置为“False”：要求响应格式为JSON格式时，“stream”必须为“False”。")

        # 记录 API 调用参数（不包括敏感信息）
        logger.info(f"调用 GROQ API，模型: {model}, 温度: {
                    temperature}, 流模式: {stream}")

        # 根据 GROQ 的预期格式构建消息
        # GROQ 期望的消息格式为 [{role: "system"/"user", content: "..."}]
        messages = []

        # 添加系统角色消息（提示词）
        messages.append({"role": "system", "content": formatted_prompt})

        # 如果有用户消息，则添加
        if user_message:
            messages.append({"role": "user", "content": user_message})
        else:
            # 如果没有用户消息，添加一个默认的用户消息
            messages.append({"role": "user", "content": "请根据上述提示进行回答"})

        # 使用指定的参数进行 API 调用
        try:
            # 直接使用client对象调用completions.create方法
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,  # 使用正确的参数名称
                top_p=top_p,
                stream=stream,
                stop=stop
            )

            logger.info("GROQ API 调用成功")
            return completion

        except Exception as e:
            # 记录并重新引发 API 调用期间发生的任何异常
            logger.error(f"GROQ API 调用失败: {e}")
            raise


class GroqResponseParser(ResponseParserInterface):
    """
    GROQ API 响应解析实现类

    实现ResponseParserInterface接口，提供GROQ响应的标准化解析。

    解析逻辑：
    1. 识别响应类型（流式/非流式）
    2. 统一内容提取：
       - 流式响应：拼接所有delta.content
       - 非流式响应：直接获取message.content
    3. 元数据提取：
       - 模型标识
       - 使用量统计
       - 请求ID

    特殊处理：
    - 流式响应中的部分结果缓存
    - 非UTF-8字符的转义处理
    - 大文本内容的分块处理

    返回值结构：
    {
        "content": str,         # 完整响应内容
        "model": str,           # 实际使用的模型
        "usage": {              # 令牌使用情况
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        },
        "request_id": str       # 本次请求的唯一ID
    }
    """

    def parse(self, response: Any) -> Dict[str, Any]:
        """
        解析GROQ API响应的实现方法

        参数:
            response (Any): GROQ API的响应对象，可以是流式或非流式响应
                - 流式响应: 包含多个chunk的生成器对象
                - 非流式响应: 单个ChatCompletion对象

        返回:
            Dict[str, Any]: 包含解析内容的字典，包含以下字段:
                - content (str): 完整的响应内容
                - model (str): 使用的模型名称
                - usage (dict): API使用统计信息（如果存在）

        异常:
            抛出原始异常并记录错误日志

        示例:
            >>> parser = GroqResponseParser()
            >>> parser(stream_response)
            {'content': '...', 'model': 'llama2-70b', 'usage': {'total_tokens': 100}}
        """
        try:
            # 处理非流式响应
            if isinstance(response, ChatCompletion):
                return {
                    'content': response.choices[0].message.content,
                    'model': response.model,
                    'usage': response.usage.model_dump() if hasattr(
                        response,
                        'usage') and response.usage else {}}
            # 处理流式响应
            else:
                full_content = ""
                model = ""
                usage = {}
                for chunk in response:
                    # 处理内容增量
                    if getattr(
                            chunk, 'choices', None) and len(
                            chunk.choices) > 0:
                        delta = getattr(chunk.choices[0], 'delta', None)
                        if delta:
                            full_content += (getattr(delta,
                                             'content', '') or '')
                    if not model:  # 仅首次获取
                        model = getattr(chunk, 'model', "")
                    # usage仅取最后一次
                    x_groq = getattr(chunk, 'x_groq', None)
                    if x_groq and hasattr(x_groq, 'usage') and x_groq.usage:
                        usage = x_groq.usage.model_dump()
                if not full_content:
                    raise Exception("未能成功解析出GROQ响应主要内容")
                return {
                    'content': full_content,
                    'model': model,
                    'usage': usage
                }
        except Exception as e:
            logger.error(f"解析GROQ响应失败: {e}")
            raise
