"""
custom_interface.py - LLM API 接口协议定义

本模块定义了两个核心接口：
1. LLMCallerInterface: LLM API调用接口协议
   - 规范API调用方法的参数和返回值
   - 确保不同提供商实现统一的调用方式
2. ResponseParserInterface: 响应解析接口协议
   - 标准化不同API返回数据的解析流程
   - 支持流式响应和普通响应的统一处理

接口设计原则：
- 类型安全：使用ABC抽象基类和类型注解
- 扩展性：通过继承实现新提供商的支持
- 兼容性：同时支持同步和异步调用模式
- 可测试性：明确的接口定义便于mock测试

重要提示：
所有自定义LLM实现类必须严格遵循本模块定义的接口协议
"""

from typing import Any, Dict, Union, Generator
from abc import ABC, abstractmethod


class LLMCallerInterface(ABC):
    """
    LLM API 调用接口协议

    定义了调用 LLM API 的标准接口，所有自定义 LLM 调用实现都应遵循此接口。
    """

    @abstractmethod
    def call(self, provider: str, api_key: str, formatted_prompt: str,
             request_params: Dict[str, Any]) -> Union[Any, Generator]:
        """
        调用 LLM API 的标准接口方法

        参数:
            provider (str): LLM 提供商名称，如 "openai", "anthropic", "groq" 等
            api_key (str): 用于身份验证的 API 密钥
            formatted_prompt (str): 发送给模型的格式化提示词
            request_params (Dict[str, Any]): API 调用的参数，包括模型名称、
                                           温度、流模式、最大令牌数等

        返回:
            Union[Any, Generator]: LLM API 响应，可以是完整的响应对象
                                  或用于流式响应的生成器
        """
        ...


class ResponseParserInterface(ABC):
    """
    LLM API 响应解析接口协议

    定义了解析 LLM API 响应的标准接口，所有自定义响应解析器都应遵循此接口。
    """

    def parse(self, response: Any) -> Dict[str, Any]:
        """
        解析 LLM API 响应的标准接口方法

        参数:
            response (Any): LLM API 的响应对象，可以是流式或非流式响应

        返回:
            Dict[str, Any]: 包含解析内容的字典，通常包含以下字段:
                - content (str): 完整的响应内容
                - model (str): 使用的模型名称
                - usage (dict): API 使用统计信息（如果存在）
        """
        ...
