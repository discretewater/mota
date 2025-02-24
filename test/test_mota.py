"""
Mota测试模块
"""

import pytest
from pathlib import Path
import logging
from source.mota import (
    setup_logging,
    load_config,
    get_api_key,
    format_prompt,
    parse_response
)


def test_setup_logging():
    """测试日志设置"""
    setup_logging("DEBUG", "stdout")
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_load_config():
    """测试配置加载"""
    config = load_config()
    assert isinstance(config, dict)
    assert "logging" in config
    assert "llm" in config


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


def test_custom_response_parser():
    """测试自定义响应解析器"""
    def custom_parser(response):
        return {"custom_field": "custom_value"}

    result = parse_response(None, custom_parser)
    assert result["custom_field"] == "custom_value"
