"""
test_seek.py - Mota API 核心函数测试

本测试模块验证 seek 函数的功能，包括：
1. 基本调用功能：验证 seek 函数能够正确调用 LLM API
2. 参数处理：测试各种参数组合的正确处理
3. 错误处理：验证异常情况的正确处理
4. 集成测试：验证与其他模块的集成

测试策略：
- 模拟测试：使用 Mock 对象隔离外部依赖
- 参数测试：验证不同参数组合的行为
- 异常测试：验证错误处理机制
"""

import pytest
from unittest.mock import patch, MagicMock
from edn_format import Keyword

from mota.seek import seek


class FakeResponse:
    """模拟 LLM API 响应对象"""

    def __init__(self, content="测试内容", model="test-model"):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.model = model
        self.usage = MagicMock(_asdict=lambda: {"total_tokens": 10})


@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
@patch("mota.seek.get_parser_func")
def test_seek_basic_functionality(
    mock_get_parser_func, mock_get_llm_call_func,
    mock_get_api_key, mock_load_config
):
    """测试 seek 函数的基本功能"""
    # 设置模拟对象
    mock_config = {
        Keyword("logging"): {"level": "DEBUG"},
        Keyword("llm"): {
            Keyword("providers"): {
                Keyword("openai"): {
                    Keyword("model"): "gpt-4"
                }
            },
            Keyword("temperature"): 0.7,
            Keyword("stream"): True,
            Keyword("max_tokens"): 1000
        }
    }
    mock_load_config.return_value = mock_config
    mock_get_api_key.return_value = "test-api-key"

    # 模拟 LLM 调用函数
    mock_llm_call = MagicMock(return_value=FakeResponse())
    mock_get_llm_call_func.return_value = mock_llm_call

    # 模拟响应解析函数
    mock_parser = MagicMock(
        return_value={
            "content": "测试响应",
            "model": "gpt-4",
            "usage": {
                "total_tokens": 10}})
    mock_get_parser_func.return_value = mock_parser

    # 调用 seek 函数
    result = seek(
        provider="openai",
        model="gpt-4",
        prompt="测试提示词",
        message="测试消息"
    )

    # 验证结果
    assert result["content"] == "测试响应"
    assert result["model"] == "gpt-4"
    assert result["usage"]["total_tokens"] == 10

    # 验证调用
    mock_llm_call.assert_called_once()
    mock_parser.assert_called_once()


@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
@patch("mota.seek.get_parser_func")
def test_seek_with_custom_params(
    mock_get_parser_func, mock_get_llm_call_func,
    mock_get_api_key, mock_load_config
):
    """测试 seek 函数使用自定义参数"""
    # 设置模拟对象
    mock_config = {
        Keyword("logging"): {"level": "DEBUG"},
        Keyword("llm"): {
            Keyword("providers"): {
                Keyword("openai"): {
                    Keyword("model"): "gpt-4"
                }
            },
            Keyword("temperature"): 0.7,
            Keyword("stream"): True,
            Keyword("max_tokens"): 1000
        }
    }
    mock_load_config.return_value = mock_config
    mock_get_api_key.return_value = "test-api-key"

    # 模拟 LLM 调用函数
    mock_llm_call = MagicMock(return_value=FakeResponse())
    mock_get_llm_call_func.return_value = mock_llm_call

    # 模拟响应解析函数
    mock_parser = MagicMock(
        return_value={
            "content": "测试响应",
            "model": "gpt-4",
            "usage": {
                "total_tokens": 10}})
    mock_get_parser_func.return_value = mock_parser

    # 自定义参数
    custom_params = {
        "temperature": 0.9,
        "top_p": 0.95
    }

    # 调用 seek 函数
    result = seek(
        provider="openai",
        model="gpt-4",
        prompt="测试提示词",
        message="测试消息",
        custom_params=custom_params
    )

    # 验证结果
    assert result["content"] == "测试响应"

    # 验证调用参数
    call_args = mock_llm_call.call_args[0]
    request_params = call_args[3]  # 第四个参数是 request_params
    assert request_params["temperature"] == 0.9
    assert request_params["top_p"] == 0.95


@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
@patch("mota.seek.get_parser_func")
def test_seek_with_fields_extraction(
    mock_get_parser_func, mock_get_llm_call_func,
    mock_get_api_key, mock_load_config
):
    """测试 seek 函数的字段提取功能"""
    # 设置模拟对象
    mock_config = {
        Keyword("logging"): {"level": "DEBUG"},
        Keyword("llm"): {
            Keyword("providers"): {
                Keyword("openai"): {
                    Keyword("model"): "gpt-4"
                }
            },
            Keyword("temperature"): 0.7,
            Keyword("stream"): True,
            Keyword("max_tokens"): 1000
        }
    }
    mock_load_config.return_value = mock_config
    mock_get_api_key.return_value = "test-api-key"

    # 模拟 LLM 调用函数
    mock_llm_call = MagicMock(return_value=FakeResponse())
    mock_get_llm_call_func.return_value = mock_llm_call

    # 模拟响应解析函数
    mock_parser = MagicMock(return_value={
        "content": "测试响应",
        "model": "gpt-4",
        "usage": {"total_tokens": 10}
    })
    mock_get_parser_func.return_value = mock_parser

    # 调用 seek 函数，只提取 content 字段
    result = seek(
        provider="openai",
        model="gpt-4",
        prompt="测试提示词",
        message="测试消息",
        fields=["content"]
    )

    # 验证结果只包含 content 字段
    assert "content" in result
    assert "model" not in result
    assert "usage" not in result
    assert result["content"] == "测试响应"


@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
@patch("mota.seek.get_parser_func")
@patch("mota.seek.retrieve_context_knowledge")
def test_seek_with_knowledge_dir(
    mock_retrieve_context_knowledge,
    mock_get_parser_func, mock_get_llm_call_func,
    mock_get_api_key, mock_load_config
):
    """测试 seek 函数的知识库检索功能"""
    # 设置模拟对象
    mock_config = {
        Keyword("logging"): {"level": "DEBUG"},
        Keyword("llm"): {
            Keyword("providers"): {
                Keyword("openai"): {
                    Keyword("model"): "gpt-4"
                }
            },
            Keyword("temperature"): 0.7,
            Keyword("stream"): True,
            Keyword("max_tokens"): 1000
        }
    }
    mock_load_config.return_value = mock_config
    mock_get_api_key.return_value = "test-api-key"

    # 模拟 LLM 调用函数
    mock_llm_call = MagicMock(return_value=FakeResponse())
    mock_get_llm_call_func.return_value = mock_llm_call

    # 模拟响应解析函数
    mock_parser = MagicMock(
        return_value={
            "content": "测试响应",
            "model": "gpt-4",
            "usage": {
                "total_tokens": 10}})
    mock_get_parser_func.return_value = mock_parser

    # 模拟知识库检索结果
    mock_retrieve_context_knowledge.return_value = [
        "知识库内容1",
        "知识库内容2"
    ]

    # 调用 seek 函数
    result = seek(  # noqa: F841
        provider="openai",
        model="gpt-4",
        prompt="测试提示词",
        message="测试消息",
        knowledge_dir="/path/to/knowledge"
    )

    # 验证知识库检索被调用
    mock_retrieve_context_knowledge.assert_called_once_with(
        "/path/to/knowledge",
        "测试提示词 测试消息"
    )

    # 验证 LLM 调用中包含了知识库内容
    call_args = mock_llm_call.call_args[0]
    formatted_prompt = call_args[2]  # 第三个参数是 formatted_prompt
    assert "参考以下相关信息" in formatted_prompt
    assert "知识库内容1" in formatted_prompt
    assert "知识库内容2" in formatted_prompt


@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
def test_seek_error_handling(
    mock_get_llm_call_func, mock_get_api_key, mock_load_config
):
    """测试 seek 函数的错误处理"""
    # 设置模拟对象
    mock_config = {
        Keyword("logging"): {"level": "DEBUG"},
        Keyword("llm"): {
            Keyword("providers"): {
                Keyword("openai"): {
                    Keyword("model"): "gpt-4"
                }
            },
            Keyword("temperature"): 0.7,
            Keyword("stream"): True,
            Keyword("max_tokens"): 1000
        }
    }
    mock_load_config.return_value = mock_config
    mock_get_api_key.return_value = "test-api-key"

    # 模拟 LLM 调用函数抛出异常
    mock_get_llm_call_func.return_value = MagicMock(
        side_effect=Exception("API 调用失败"))

    # 验证异常被正确传递
    with pytest.raises(Exception) as excinfo:
        seek(
            provider="openai",
            model="gpt-4",
            prompt="测试提示词",
            message="测试消息"
        )

    assert "API 调用失败" in str(excinfo.value)
