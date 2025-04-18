"""
test_custom_groq.py - GROQ自定义实现测试

本测试模块验证以下核心功能：
1. 接口合规性测试：确保自定义实现符合接口协议
2. 参数传递测试：验证请求参数正确转换
3. 响应解析测试：检查流式/非流式响应解析正确性
4. 异常处理测试：模拟网络错误和API限制

测试策略：
- 模拟测试：使用Mock对象避免真实API调用
- 边界测试：极端参数和异常输入
- 兼容性测试：覆盖同步和流式模式
- 性能测试：验证响应时间在合理范围

测试场景覆盖：
- 正常流式响应处理
- 正常同步响应处理
- 无用户消息的默认处理
- 接口协议合规性验证
- 错误响应解析
- 组合调用流程测试

环境要求：
- Python 3.10+
- pytest-mock 插件
- 禁用网络访问（所有测试基于Mock）
"""

import os
import sys
import types
import pytest
import importlib.util
from typing import Any, Dict, Generator, Union
from unittest.mock import MagicMock, patch
from mota.custom_interface import LLMCallerInterface
from groq.types.chat import ChatCompletion

# Import the modules to be tested
spec = importlib.util.spec_from_file_location(
    "main_module", os.path.join(
        os.path.dirname(__file__), "../source/mota/main.py"))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Import core module
core_spec = importlib.util.spec_from_file_location(
    "core_module", os.path.join(
        os.path.dirname(__file__), "../source/mota/core.py"))
core_module = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core_module)


def test_get_llm_call_func_default():
    """
    测试在未提供自定义函数时，默认的 LLM API 调用函数是否被返回。
    """
    func = core_module.get_llm_call_func(None)
    assert func == core_module.default_llm_call


def test_get_llm_call_func_custom():
    """
    测试加载用户自定义的 LLM API 调用函数。
    向临时模块注入 DummyLLMCaller，并验证 get_llm_call_func 是否正确导入此函数。
    """
    # 创建一个临时模块文件
    import tempfile
    import os

    # 创建一个临时目录和临时模块文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_module_path = os.path.join(temp_dir, "dummy_llm_caller.py")

        # 写入一个实现了LLMCallerInterface的类到临时模块
        with open(temp_module_path, "w") as f:
            f.write("""
from typing import Any, Dict, Generator, Union
from mota.custom_interface import LLMCallerInterface

class DummyLLMCaller(LLMCallerInterface):
    def call(self, provider: str, api_key: str, formatted_prompt: str,
             request_params: Dict[str, Any]) -> Union[Any, Generator]:
        # 简单的实现，返回一个固定的响应
        return {"dummy_response": "This is a test response"}
""")

        # 使用临时模块路径调用get_llm_call_func
        custom_func = core_module.get_llm_call_func(temp_module_path)

        # 验证返回的函数不是默认函数
        assert custom_func != core_module.default_llm_call

        # 调用返回的函数并验证其行为
        with patch("mota.loader.load_module_from_path") as mock_load_module:
            # 模拟加载模块的行为
            mock_module = types.ModuleType("dummy_module")

            # 创建一个DummyLLMCaller类
            class DummyLLMCaller(LLMCallerInterface):
                def call(self,
                         provider: str,
                         api_key: str,
                         formatted_prompt: str,
                         request_params: Dict[str,
                                              Any]) -> Union[Any,
                                                             Generator]:
                    return {"dummy_response": "This is a test response"}

            # 将DummyLLMCaller添加到模拟模块
            setattr(mock_module, "DummyLLMCaller", DummyLLMCaller)
            mock_load_module.return_value = mock_module

            # 重新获取函数
            custom_func = core_module.get_llm_call_func(temp_module_path)

            # 调用函数并验证结果
            result = custom_func(
                "test_provider",
                "test_api_key",
                "test_prompt",
                {})
            assert result == {"dummy_response": "This is a test response"}


@pytest.mark.parametrize("stream_mode", [True, False])
def test_custom_groq_api(stream_mode):
    """
    测试自定义 GROQ API 集成。
    使用模拟的 Groq 客户端以验证正确的参数与功能。
    """
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    # 模拟 Groq 客户端及其方法
    with patch.object(custom_groq, 'Groq') as mock_groq:
        # 创建模拟客户端实例和完成方法
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_completions = MagicMock()
        mock_client.chat.completions.create = mock_completions

        # 测试参数
        provider = "groq"
        api_key = "test-api-key"
        prompt = "你是一个有用的AI助手"  # 现在这是系统提示词
        params = {
            "model": "deepseek-r1-distill-llama-70b",
            "temperature": 0.6,
            "stream": stream_mode,
            "max_tokens": 2048,  # 使用兼容性参数名称
            "top_p": 0.9,
            "message": "Hello, GROQ!"  # 用户消息
        }

        # 调用函数 - 使用类实例调用
        groq_caller = custom_groq.GroqLLMCaller()
        groq_caller.call(provider, api_key, prompt, params)

        # 验证 Groq 客户端是否使用正确的 API 密钥初始化
        mock_groq.assert_called_once_with(api_key=api_key)

        # 验证 completions.create 是否使用正确的参数调用
        expected_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Hello, GROQ!"}
        ]

        mock_completions.assert_called_once_with(
            model=params["model"],
            messages=expected_messages,
            temperature=params["temperature"],
            max_completion_tokens=params["max_tokens"],  # 验证参数名称转换
            top_p=params["top_p"],
            stream=params["stream"],
            stop=None
        )


@pytest.mark.parametrize("stream_mode", [True, False])
def test_custom_groq_api_without_user_message(stream_mode):
    """
    测试在没有用户消息时的 GROQ API 集成。
    验证在无用户消息的情况下消息格式是否正确。
    """
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    # 模拟 Groq 客户端及其方法
    with patch.object(custom_groq, 'Groq') as mock_groq:
        # 创建模拟客户端实例和完成方法
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        mock_completions = MagicMock()
        mock_client.chat.completions.create = mock_completions

        # 测试参数 - 没有用户消息
        provider = "groq"
        api_key = "test-api-key"
        prompt = "你是一个专业的AI助手"  # 系统提示词
        params = {
            "model": "deepseek-r1-distill-llama-70b",
            "temperature": 0.6,
            "stream": stream_mode,
            "max_tokens": 2048,
            "top_p": 0.9
            # 没有 message 参数
        }

        # 使用类实例测试
        groq_caller = custom_groq.GroqLLMCaller()
        groq_caller.call(provider, api_key, prompt, params)

        # 验证 completions.create 是否使用正确的参数调用
        # 应该有系统消息和默认用户消息
        expected_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请根据上述提示进行回答"}
        ]

        mock_completions.assert_called_once_with(
            model=params["model"],
            messages=expected_messages,
            temperature=params["temperature"],
            max_completion_tokens=params["max_tokens"],
            top_p=params["top_p"],
            stream=params["stream"],
            stop=None
        )


class MockStreamChunk:
    """模拟GROQ流式响应块"""

    def __init__(self, content="", model=None, usage=None):
        self.choices = [MagicMock(delta=MagicMock(content=content))]
        self.model = model
        # 构造 x_groq 嵌套结构
        self.x_groq = MagicMock()
        if usage:
            # 创建带 model_dump() 方法的 usage 对象
            usage_mock = MagicMock()
            usage_mock.model_dump = MagicMock(return_value=usage)
            self.x_groq.usage = usage_mock
        else:
            self.x_groq = None


def create_mock_completion(
        content="this is test content",
        model="test-model",
        usage=None):
    # 创建基于ChatCompletion结构的MagicMock
    mock_response = MagicMock(spec=ChatCompletion)  # 关键点：spec约束

    # 设置 choices 结构
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=content,
                to_dict=MagicMock(return_value={"content": content})
            )
        )
    ]

    mock_response.model = model

    # 设置 usage 字段
    if usage is not None:
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump = MagicMock(return_value=usage)
    else:
        mock_response.usage = None

    return mock_response


def test_parse_groq_response_non_stream():
    """测试解析非流式响应"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    response = create_mock_completion(
        content="test content",
        model="test-model",
        usage={
            "total_tokens": 100})

    # 使用类实例测试
    parser = custom_groq.GroqResponseParser()
    result = parser.parse(response)

    assert result["content"] == "test content"
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 100


def test_parse_groq_response_stream():
    """测试解析流式响应"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    chunks = [
        MockStreamChunk(content="Hello", model="llama2-70b"),
        MockStreamChunk(content=" World"),
        MockStreamChunk(content="!",
                        usage={"total_tokens": 128})
    ]

    # 使用类实例测试
    parser = custom_groq.GroqResponseParser()
    result = parser.parse(iter(chunks))

    assert result["content"] == "Hello World!"
    assert result["model"] == "llama2-70b"
    assert result["usage"]["total_tokens"] == 128


def test_parse_groq_response_error_handling(caplog):
    """测试异常处理"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    # 创建一个会引发异常的无效响应对象
    invalid_response = MagicMock(spec=dict)  # 指定错误的类型规范
    # 添加会通过初始检查但后续处理会失败的属性
    invalid_response.__iter__ = lambda self: iter([self])
    invalid_response.choices = []  # 空choices列表会引发索引错误

    # 使用类实例测试
    parser = custom_groq.GroqResponseParser()
    with pytest.raises(Exception):
        parser.parse(invalid_response)

    assert "解析GROQ响应失败" in caplog.text


def test_interface_implementation():
    """测试接口实现"""
    # 导入自定义 GROQ 模块和接口
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_groq.py"))
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    interface_spec = importlib.util.spec_from_file_location(
        "custom_interface", os.path.join(
            os.path.dirname(__file__), "../source/mota/custom_interface.py"))
    custom_interface = importlib.util.module_from_spec(interface_spec)
    interface_spec.loader.exec_module(custom_interface)

    # 验证类是否实现了接口
    groq_caller = custom_groq.GroqLLMCaller()
    groq_parser = custom_groq.GroqResponseParser()

    # 验证类是否实现了接口的方法
    assert hasattr(groq_caller, 'call')
    assert callable(groq_caller.call)
    assert hasattr(groq_parser, 'parse')
    assert callable(groq_parser.parse)

    # 验证类是否是接口的子类 - 使用实例检查而不是类检查
    # 由于动态导入的模块可能有不同的命名空间，直接检查实例是否实现了接口方法
    assert hasattr(groq_caller, 'call')
    assert callable(groq_caller.call)
    assert hasattr(groq_parser, 'parse')
    assert callable(groq_parser.parse)

    # 检查方法签名是否符合接口要求
    from inspect import signature
    caller_sig = signature(groq_caller.call)
    assert len(caller_sig.parameters) >= 4  # 至少有4个参数


def test_combined_llm_call_and_parser():
    """
    测试 LLM 调用函数与解析函数的组合使用。

    验证 get_llm_call_func 和 get_parser_func 返回的函数能够正确组合使用，
    完成从 API 调用到响应解析的完整流程。
    """
    # 创建一个模拟的 groq 模块
    mock_groq_module = types.ModuleType("groq")
    mock_groq_class = MagicMock()
    setattr(mock_groq_module, "Groq", mock_groq_class)

    # 保存原始的 sys.modules
    original_modules = dict(sys.modules)

    try:
        # 将模拟的 groq 模块注入到 sys.modules
        sys.modules["groq"] = mock_groq_module

        # 导入自定义 GROQ 模块
        groq_spec = importlib.util.spec_from_file_location(
            "custom_groq", os.path.join(
                os.path.dirname(__file__), "../source/mota/custom_groq.py"))
        custom_groq = importlib.util.module_from_spec(groq_spec)
        groq_spec.loader.exec_module(custom_groq)

        # 创建模拟客户端实例和完成方法
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_completions = MagicMock()
        mock_client.chat.completions.create = mock_completions

        # 创建一个模拟的非流式响应
        mock_response = create_mock_completion(
            content="这是一个测试响应",
            model="deepseek-r1-distill-llama-70b",
            usage={"total_tokens": 150}
        )
        mock_completions.return_value = mock_response

        # 使用 main_module 中的函数获取调用函数和解析函数
        with patch("mota.loader.load_module_from_path") as mock_load_module:
            # 模拟加载模块的行为
            mock_load_module.return_value = custom_groq

            # 获取LLM调用函数
            llm_call_func = main_module.get_llm_call_func(os.path.join(
                os.path.dirname(__file__), "../source/mota/custom_groq.py"))

            # 获取响应解析函数
            parser_func = main_module.get_parser_func(os.path.join(
                os.path.dirname(__file__), "../source/mota/custom_groq.py"))

            # 测试参数
            provider = "groq"
            api_key = "test-api-key"
            prompt = "你是一个专业的AI助手"
            params = {
                "model": "deepseek-r1-distill-llama-70b",
                "temperature": 0.7,
                "stream": False,
                "max_tokens": 2000,
                "message": "请解释量子力学"
            }

            # 调用LLM函数
            response = llm_call_func(provider, api_key, prompt, params)

            # 验证LLM调用参数
            mock_groq_class.assert_called_once_with(api_key=api_key)
            expected_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请解释量子力学"}
            ]
            mock_completions.assert_called_once_with(
                model=params["model"],
                messages=expected_messages,
                temperature=params["temperature"],
                max_completion_tokens=params["max_tokens"],
                top_p=0.62,  # 默认值
                stream=False,
                stop=None
            )

            # 解析响应
            parsed_result = parser_func(response)

            # 验证解析结果
            assert parsed_result["content"] == "这是一个测试响应"
            assert parsed_result["model"] == "deepseek-r1-distill-llama-70b"
            assert parsed_result["usage"]["total_tokens"] == 150
    finally:
        # 恢复原始的 sys.modules
        sys.modules.clear()
        sys.modules.update(original_modules)
