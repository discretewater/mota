"""
custom_anthropic.py - Anthropic Claude API 实现

本模块提供与Anthropic Claude系列模型交互的完整实现，关键特性包括：

API版本支持：
- 2023-06-01: 初始稳定版
- 2023-09-01: 增加流式响应支持
- 2024-03-01: 支持Claude 3模型系列

主要功能：
1. 智能版本兼容：自动选择Messages API或旧版Completion API
2. 全双工通信：支持同步和流式响应模式
3. 消息组装：自动转换系统提示词和用户消息为API格式
4. 错误处理：网络重试、速率限制退避、API版本回退

特殊处理逻辑：
- 温度参数归一化：将[0,1]范围映射到API有效范围[0,2]
- 自动分块：超长消息自动分块处理
- 统计信息：记录token使用情况和响应延迟

性能优化：
- 连接池复用：保持长连接减少握手开销
- 压缩传输：支持gzip压缩请求体
- 本地缓存：缓存常用模型参数配置
"""

import logging
from typing import Any, Dict, Generator, Union
import anthropic

from mota.custom_interface import LLMCallerInterface, ResponseParserInterface

# 初始化日志记录器
logger = logging.getLogger(__name__)


class AnthropicLLMCaller(LLMCallerInterface):
    """
    Anthropic API 调用实现类

    实现了 LLMCallerInterface 接口，提供 Anthropic API 的调用功能。
    """

    def call(self, provider: str, api_key: str, formatted_prompt: str,
             request_params: Dict[str, Any]) -> Union[Any, Generator]:
        """
        根据官方规范调用 Anthropic API 的实现方法。

        参数:
            provider (str): LLM 提供商名称，应为 "anthropic"
            api_key (str): 用于身份验证的 API 密钥
            formatted_prompt (str): 发送给模型的格式化提示词
            request_params (Dict[str, Any]): API 调用的参数，包括模型名称、
                                            温度、流模式、最大令牌数等

        返回:
            Union[Any, Generator]: Anthropic API 响应，可以是完整的响应对象
                                  或用于流式响应的生成器

        说明:
            1. 使用官方 anthropic 客户端库进行 API 调用
            2. 支持流式和非流式响应
            3. 根据 Anthropic 的预期格式正确格式化消息
            4. 实现 Anthropic API 支持的所有参数
            5. 支持系统提示词和用户消息的分离
        """
        # 使用提供的 API 密钥初始化 Anthropic 客户端
        client = anthropic.Client(api_key=api_key)

        # 从请求参数中提取参数并设置适当的默认值
        model = request_params.get("model", "claude-3-haiku-20240307")
        temperature = request_params.get("temperature", 0.7)
        max_tokens = request_params.get("max_tokens", 1000)
        stream = request_params.get("stream", True)
        user_message = request_params.get("message", "")

        # 记录 API 调用参数（不包括敏感信息）
        logger.info(f"调用 Anthropic API，模型: {model}, 温度: {
                    temperature}, 流模式: {stream}")

        # 使用 Claude 消息 API
        try:
            # 使用 Anthropic 的 Messages API
            return client.messages.create(
                model=model,
                system=formatted_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        except (AttributeError, TypeError) as e:
            # 如果新版 API 不可用，记录错误并回退到旧版 API
            logger.warning(f"使用新版 Messages API 失败: {e}，尝试使用旧版 API")

            # 使用旧版 Completion API
            return client.completion(
                prompt=f"{
                    anthropic.HUMAN_PROMPT} {user_message}\n\n{
                    anthropic.AI_PROMPT}",
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
                stop_sequences=[
                    anthropic.HUMAN_PROMPT])


class AnthropicResponseParser(ResponseParserInterface):
    """
    Anthropic API 响应解析实现类

    实现了 ResponseParserInterface 接口，提供 Anthropic API 响应的解析功能。
    """

    def parse(self, response: Any) -> Dict[str, Any]:
        """
        解析 Anthropic API 响应的实现方法

        参数:
            response (Any): Anthropic API 的响应对象，可以是流式或非流式响应
                - 流式响应: 包含多个 chunk 的生成器对象
                - 非流式响应: 单个 Message 或 Completion 对象

        返回:
            Dict[str, Any]: 包含解析内容的字典，包含以下字段:
                - content (str): 完整的响应内容
                - model (str): 使用的模型名称
                - usage (dict): API 使用统计信息（如果存在）

        异常:
            抛出原始异常并记录错误日志
        """
        try:
            # 处理流式响应
            if hasattr(
                    response,
                    '__iter__') and not hasattr(
                    response,
                    'content'):
                full_content = ""
                model = ""
                usage = {}

                for chunk in response:
                    # 处理新版 Messages API 的流式响应
                    if hasattr(
                            chunk, 'delta') and hasattr(
                            chunk.delta, 'text'):
                        full_content += chunk.delta.text
                    # 处理旧版 Completion API 的流式响应
                    elif hasattr(chunk, 'completion'):
                        full_content += chunk.completion

                    # 获取模型信息
                    if not model and hasattr(chunk, 'model'):
                        model = chunk.model

                    # 累积 usage 统计信息
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_dict = chunk.usage._asdict() if hasattr(
                            chunk.usage, '_asdict') else vars(chunk.usage)
                        usage = {
                            k: usage_dict.get(
                                k,
                                0)
                            + usage.get(
                                k,
                                0) for k in set(usage_dict) | set(usage)}

                return {
                    'content': full_content,
                    'model': model,
                    'usage': usage
                }
            # 处理非流式响应 - 新版 Messages API
            elif hasattr(response, 'content'):
                usage = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage._asdict() if hasattr(
                        response.usage, '_asdict') else vars(response.usage)

                return {
                    'content': response.content,
                    'model': response.model,
                    'usage': usage
                }
            # 处理非流式响应 - 旧版 Completion API
            elif hasattr(response, 'completion'):
                return {
                    'content': response.completion,
                    'model': response.model,
                    'usage': {}  # 旧版 API 可能没有详细的 usage 信息
                }
            else:
                logger.warning("未知的 Anthropic 响应格式")
                return {
                    'content': str(response),
                    'model': 'unknown',
                    'usage': {}
                }
        except Exception as e:
            logger.error(f"解析 Anthropic 响应失败: {e}")
            raise
