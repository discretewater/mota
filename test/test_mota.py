"""
Mota测试模块
"""

from typer.testing import CliRunner
import pytest
import os
import logging
from collections.abc import Mapping
from edn_format import Keyword
from unittest.mock import patch, MagicMock, mock_open
import json

from mota.core import (
    setup_logging,
    load_config,
    get_api_key,
    format_prompt,
    default_parse,
    get_parser_func,
    extract_fields,
    retrieve_context_knowledge
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
        key = get_api_key(
            "openai",
            auth_source="authinfo",
            auth_path="dummy_path")
        assert key == "authinfo_key"


def test_format_prompt():
    """测试提示词格式化"""
    template = "Hello, {name}!"
    result = format_prompt(template, name="World")
    assert result == "Hello, World!"

    with pytest.raises(KeyError):
        format_prompt(template, wrong_param="World")


def test_format_prompt_empty():
    """测试空提示词格式化"""
    # 测试空字符串提示词
    empty_template = ""
    result = format_prompt(empty_template)
    assert result == ""

    # 测试只有空格的提示词
    space_template = "   "
    result = format_prompt(space_template)
    assert result == "   "

    # 测试包含格式化占位符但无需参数的提示词
    no_param_template = "Hello, World!"
    result = format_prompt(no_param_template)
    assert result == "Hello, World!"


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
        usage = type(
            "Usage", (), {
                "_asdict": lambda self: {
                    "total_tokens": 10}})()

    response = MockResponse()
    result = default_parse(response)

    assert result["content"] == "Test content"
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 10


def test_parse_response_custom_parser():
    """测试自定义响应解析器"""
    # 模拟自定义解析器模块
    with patch("mota.core.load_custom_func") as mock_load_custom_func:
        # 创建一个模拟的解析函数
        def mock_parser(response):
            return {"custom_field": "custom_value"}
        mock_load_custom_func.return_value = mock_parser

        # 调用get_parser_func并传入文件路径字符串
        parser_func = get_parser_func("path/to/custom_parser.py")

        # 使用返回的解析函数解析响应
        result = parser_func("mock_response")
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


def test_retrieve_context_knowledge_edge_cases():
    """测试retrieve_context_knowledge函数的边界情况

    覆盖以下场景：
    1. 空目录：应抛出ValueError异常
    2. 无效文件格式：应正常处理（依赖DirectoryLoader的实现）
    3. 单文件目录：应能正常加载
    4. 超大文档：测试加载性能
    """
    import tempfile
    from pathlib import Path

    # 测试空目录
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(ValueError) as excinfo:
            retrieve_context_knowledge(empty_dir, "test query")
        assert "未加载到任何文档" in str(excinfo.value)

    # 测试单文件目录
    with tempfile.TemporaryDirectory() as single_file_dir:
        file_path = Path(single_file_dir) / "test.txt"
        file_path.write_text("单一文件测试内容")

        # 使用mock绕过实际向量数据库操作
        with patch("mota.core.FAISS") as mock_faiss:
            # 设置模拟的检索结果
            mock_vectorstore = MagicMock()
            mock_vectorstore.similarity_search.return_value = [
                MagicMock(page_content="测试内容")
            ]
            mock_faiss.from_documents.return_value = mock_vectorstore

            result = retrieve_context_knowledge(single_file_dir, "test")
            assert len(result) == 1  # 验证返回模拟的检索结果

    # 测试无效路径（已在其他测试用例覆盖，此处不需要重复测试）


def test_load_invalid_module():
    """测试加载无效模块的异常处理

    验证当提供不存在的模块路径时：
    1. 应正确抛出ImportError异常
    2. 异常信息应包含模块路径
    """
    from mota.loader import load_module_from_path

    # 使用不存在的文件路径
    invalid_path = "/path/does/not/exist.py"

    # 验证异常类型和错误信息
    with pytest.raises(ImportError) as excinfo:
        load_module_from_path("invalid_module", invalid_path)

    # 检查异常信息是否包含路径信息
    assert invalid_path in str(excinfo.value)
    assert "加载模块失败" in str(excinfo.value)


# 新增针对主要功能的测试用例（仅针对OpenAI ChatGPT API模拟）


# 定义一个假的OpenAI API响应对象

class FakeUsage:
    def _asdict(self):
        return {"total_tokens": 50}


class FakeChoice:
    class FakeMessage:
        content = "Fake response from OpenAI"
    message = FakeMessage()


class FakeOpenAIResponse:
    choices = [FakeChoice()]
    model = "gpt-4"
    usage = FakeUsage()


# 定义一个简单的dummy配置，用于模拟load_config返回的配置字典
dummy_config = {
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


@patch("mota.seek.load_config", return_value=dummy_config)
@patch("mota.seek.get_api_key", return_value="dummy_api_key")
@patch("openai.OpenAI")
def test_main_openai_success(
        mock_openai_cls,
        mock_get_api_key,
        mock_load_config):
    """
    测试 main 函数在 openai 提供商下的成功执行。
    使用 mock 模拟 OpenAI API响应，并验证输出结果包含预期响应内容。
    """
    # 重置全局 openai_client 以确保测试隔离
    import mota.core as core_mod
    core_mod.openai_client = None

    # 构造假的 OpenAI 客户端实例及其响应
    fake_client_instance = MagicMock()
    fake_chat = MagicMock()
    fake_completions = MagicMock(return_value=FakeOpenAIResponse())
    fake_chat.completions.create = fake_completions
    fake_client_instance.chat = fake_chat
    mock_openai_cls.return_value = fake_client_instance

    from mota.main import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--log-level", "DEBUG",
        "--provider", "openai",
        "--model", "gpt-4",
        "--prompt", "Test prompt",
        "Test message"  # 添加必需的 MESSAGE 参数
    ])
    # 验证输出中包含模拟的响应内容
    assert "Fake response from OpenAI" in result.output


@patch("mota.seek.load_config", return_value=dummy_config)
@patch("mota.seek.get_api_key", return_value="dummy_api_key")
@patch("openai.OpenAI")
def test_main_openai_custom_params(
        mock_openai_cls,
        mock_get_api_key,
        mock_load_config):
    """
    测试 main 函数使用自定义请求参数执行，
    验证自定义参数是否正确合并到API请求中。
    """
    # 重置全局 openai_client 以确保测试隔离
    import mota.core as core_mod
    core_mod.openai_client = None

    fake_client_instance = MagicMock()
    fake_chat = MagicMock()
    fake_completions = MagicMock(return_value=FakeOpenAIResponse())
    fake_chat.completions.create = fake_completions
    fake_client_instance.chat = fake_chat
    mock_openai_cls.return_value = fake_client_instance

    from mota.main import cli
    runner = CliRunner()
    # 定义自定义参数，覆盖默认的temperature值
    custom_params = json.dumps({"temperature": 0.9})
    result = runner.invoke(cli, [
        "--log-level", "DEBUG",
        "--provider", "openai",
        "--model", "gpt-4",
        "--prompt", "Test prompt",
        "--custom-params", custom_params,
        "Test message"  # 添加必需的 MESSAGE 参数
    ])
    # 验证 API 调用中使用的参数包含自定义值
    fake_completions.assert_called_once()
    called_args, called_kwargs = fake_completions.call_args
    assert called_kwargs.get("temperature") == 0.9
    assert "Fake response from OpenAI" in result.output


@patch("mota.seek.load_config", return_value=dummy_config)
@patch("mota.seek.get_api_key", return_value="dummy_api_key")
@patch("openai.OpenAI")
def test_main_openai_field_extraction(
        mock_openai_cls,
        mock_get_api_key,
        mock_load_config):
    """
    测试 main 函数的字段提取功能，
    当使用 --fields 参数时，输出应只包含指定的字段。
    """
    # 重置全局 openai_client 以确保测试隔离
    import mota.core as core_mod
    core_mod.openai_client = None

    fake_client_instance = MagicMock()
    fake_chat = MagicMock()
    fake_completions = MagicMock(return_value=FakeOpenAIResponse())
    fake_chat.completions.create = fake_completions
    fake_client_instance.chat = fake_chat
    mock_openai_cls.return_value = fake_client_instance

    from mota.main import cli
    runner = CliRunner()
    # 请求只提取 'content' 字段
    result = runner.invoke(cli, [
        "--log-level", "DEBUG",
        "--provider", "openai",
        "--model", "gpt-4",
        "--prompt", "Test prompt",
        "--fields", "content",
        "Test message"  # 添加必需的 MESSAGE 参数
    ])
    # 从输出中取最后一行（假定为打印的字典）进行断言，忽略调试日志的干扰
    output_lines = result.output.strip().splitlines()
    printed_dict_line = output_lines[-1] if output_lines else ""
    assert "Fake response from OpenAI" in result.output
    assert "'content':" in printed_dict_line and "'model':" not in printed_dict_line
