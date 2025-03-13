
"""
custom_groq.py - GROQ API 调用模块

本模块提供了用于调用 GROQ API 的实现类，实现了自定义接口。
用户可以根据 GROQ API 官方文档调整请求逻辑。
该模块实现了与 GROQ API 的标准交互，支持系统提示词和用户消息的分离。
"""

import logging
from typing import Any, Dict, Generator, Union
from groq import Groq

from mota.custom_interface import LLMCallerInterface, ResponseParserInterface

# 初始化日志记录器
logger = logging.getLogger(__name__)


class GroqLLMCaller(LLMCallerInterface):
    """
    GROQ API 调用实现类

    实现了 LLMCallerInterface 接口，提供 GROQ API 的调用功能。
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

        # 记录 API 调用参数（不包括敏感信息）
        logger.info(f"调用 GROQ API，模型: {model}, 温度: {temperature}, 流模式: {stream}")

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

    实现了 ResponseParserInterface 接口，提供 GROQ API 响应的解析功能。
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
            # 处理流式响应
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                full_content = ""
                model = ""
                usage = {}
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            full_content += delta.content
                    if not model and hasattr(chunk, 'model'):
                        model = chunk.model
                    # 累积usage统计信息（流式响应中每个chunk可能有部分统计）
                    if hasattr(chunk, 'usage') and chunk.usage:
                        chunk_usage = chunk.usage._asdict()
                        usage = {k: chunk_usage.get(k, 0) + usage.get(k, 0) for k in set(chunk_usage) | set(usage)}
                return {
                    'content': full_content,
                    'model': model,
                    'usage': usage
                }
            # 处理非流式响应
            else:
                return {
                    'content': response.choices[0].message.content,
                    'model': response.model,
                    'usage': response.usage._asdict() if (hasattr(response, 'usage') and response.usage) else {}
                }
        except Exception as e:
            logger.error(f"解析GROQ响应失败: {e}")
            raise
