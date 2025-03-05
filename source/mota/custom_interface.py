"""
custom_interface.py - LLM API 调用接口定义

本模块定义了用于调用各种 LLM API 的标准接口。
包括 LLM API 调用接口和响应解析接口，为不同的 LLM 提供商提供统一的调用方式。
"""

from typing import Any, Dict, Union, Generator, Protocol, runtime_checkable


@runtime_checkable
class LLMCallerInterface(Protocol):
    """
    LLM API 调用接口协议
    
    定义了调用 LLM API 的标准接口，所有自定义 LLM 调用实现都应遵循此接口。
    """
    
    def __call__(self, provider: str, api_key: str, formatted_prompt: str, 
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


@runtime_checkable
class ResponseParserInterface(Protocol):
    """
    LLM API 响应解析接口协议
    
    定义了解析 LLM API 响应的标准接口，所有自定义响应解析器都应遵循此接口。
    """
    
    def __call__(self, response: Any) -> Dict[str, Any]:
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
