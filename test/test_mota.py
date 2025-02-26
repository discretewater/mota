"""
Mota测试模块
"""

import pytest
import os
from pathlib import Path
import logging
from collections.abc import Mapping
from edn_format import Keyword
from unittest.mock import patch, MagicMock, mock_open
from mota.main import (
    setup_logging,
    load_config,
    get_api_key,
    format_prompt,
    parse_response,
    extract_fields
)


def test_setup_logging():
    """测试日志设置"""
    setup_logging("DEBUG", "stdout")
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_load_config():
    """测试配置加载"""
    config = load_config()
    assert isinstance(config, Mapping)
    assert Keyword("logging") in config
    assert Keyword("llm") in config


def test_get_api_key_env():
    """测试从环境变量获取API密钥"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        key = get_api_key("openai", auth_source="env")
        assert key == "test_key"


def test_get_api_key_command_line():
    """测试从命令行参数获取API密钥"""
    with patch.dict(os.environ, {"OPENAI_API_KEY_CMD": "cmd_key"}):
        key = get_api_key("openai", auth_source="command_line")
        assert key == "cmd_key"


def test_get_api_key_authinfo():
    """测试从.authinfo文件获取API密钥"""
    mock_authinfo_content = "openai-api-key authinfo_key"
    with patch("builtins.open", mock_open(read_data=mock_authinfo_content)):
        key = get_api_key("openai", auth_source="authinfo", auth_path="dummy_path")
        assert key == "authinfo_key"


def test_format_prompt():
    """测试提示词格式化"""
    template = "Hello, {name}!"
    result = format_prompt(template, name="World")
    assert result == "Hello, World!"

    with pytest.raises(KeyError):
        format_prompt(template, wrong_param="World")


def test_parse_response():
    """测试响应解析"""
    # 模拟响应对象
    class MockResponse:
        class Choice:
            class Message:
                content = "Test content"
            message = Message()
        choices = [Choice()]
        model = "test-model"
        usage = type("Usage", (), {"_asdict": lambda self: {"total_tokens": 10}})()

    response = MockResponse()
    result = parse_response(response)

    assert result["content"] == "Test content"
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 10


def test_parse_response_custom_parser():
    """测试自定义响应解析器"""
    def custom_parser(response):
        return {"custom_field": "custom_value"}

    result = parse_response(None, custom_parser)
    assert result["custom_field"] == "custom_value"


def test_extract_fields():
    """测试从响应字典中提取指定字段"""
    response_dict = {
        "content": "Test content",
        "model": "test-model",
        "usage": {"total_tokens": 10}
    }
    fields = ["content", "model"]
    extracted = extract_fields(response_dict, fields)
    assert extracted == {
        "content": "Test content",
        "model": "test-model"
    }

    # 测试不存在的字段
    fields = ["content", "nonexistent"]
    extracted = extract_fields(response_dict, fields)
    assert extracted == {
        "content": "Test content"
    }
