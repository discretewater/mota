#!/usr/bin/env python3
"""
Test the LLM API calling functions' unified interface

This test file verifies the functionality of default_llm_call and get_llm_call_func functions,
ensuring that parameters are correctly passed and expected results are returned when using
either default or custom LLM API calling functions.
"""

import os
import sys
import types
import pytest
import importlib.util
from unittest.mock import MagicMock, patch

# Define a dummy custom LLM calling function for testing
def dummy_llm_caller(provider, api_key, formatted_prompt, request_params):
    return {
        "provider": provider,
        "api_key": api_key,
        "prompt": formatted_prompt,
        "params": request_params
    }

# Import the module to be tested (source/mota/main.py)
spec = importlib.util.spec_from_file_location("main_module", os.path.join(os.path.dirname(__file__), "../source/mota/main.py"))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

def test_get_llm_call_func_default():
    """
    Test that the default LLM API calling function is returned when no custom function is provided.
    """
    func = main_module.get_llm_call_func(None)
    assert func == main_module.default_llm_call

def test_get_llm_call_func_custom():
    """
    Test loading a user-defined LLM API calling function.
    Inject dummy_llm_caller into a temporary module and verify that get_llm_call_func
    correctly imports this function.
    """
    temp_module = types.ModuleType("temp_module")
    temp_module.dummy_llm_caller = dummy_llm_caller
    sys.modules["temp_module"] = temp_module

    func = main_module.get_llm_call_func("temp_module:dummy_llm_caller")
    result = func("test_provider", "test_key", "Hello, world!", {
        "model": "dummy-model",
        "temperature": 0.5,
        "stream": False,
        "max_tokens": 100
    })
    assert result["provider"] == "test_provider"
    assert result["api_key"] == "test_key"
    assert result["prompt"] == "Hello, world!"
    assert result["params"]["model"] == "dummy-model"

@pytest.mark.parametrize("stream_mode", [True, False])
def test_custom_groq_api(stream_mode):
    """
    测试自定义 GROQ API 集成。
    模拟 Groq 客户端以验证正确的参数传递和功能。
    """
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
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
        groq_caller(provider, api_key, prompt, params)
        
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
        
        # 测试向后兼容性 - 使用函数别名
        mock_groq.reset_mock()
        mock_completions.reset_mock()
        custom_groq.call_groq_api(provider, api_key, prompt, params)
        
        # 验证函数别名是否正常工作
        mock_groq.assert_called_once_with(api_key=api_key)
        mock_completions.assert_called_once()


@pytest.mark.parametrize("stream_mode", [True, False])
def test_custom_groq_api_without_user_message(stream_mode):
    """
    测试没有用户消息的 GROQ API 集成。
    验证在没有用户消息的情况下消息格式是否正确。
    """
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
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
        groq_caller(provider, api_key, prompt, params)
        
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
        
        # 测试向后兼容性 - 使用函数别名
        mock_groq.reset_mock()
        mock_completions.reset_mock()
        custom_groq.call_groq_api(provider, api_key, prompt, params)
        
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
        self.usage = usage

class MockCompletion:
    """模拟GROQ非流式响应"""
    def __init__(self, content="test content", model="test-model", usage=None):
        self.choices = [MagicMock(message=MagicMock(content=content))]
        self.model = model
        self.usage = usage

def test_parse_groq_response_non_stream():
    """测试解析非流式响应"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    mock_usage = MagicMock()
    mock_usage._asdict.return_value = {"total_tokens": 100}
    response = MockCompletion(usage=mock_usage)
    
    # 使用类实例测试
    parser = custom_groq.GroqResponseParser()
    result = parser(response)
    
    assert result["content"] == "test content"
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 100
    
    # 测试向后兼容性 - 使用函数别名
    result = custom_groq.parse_groq_response(response)
    
    assert result["content"] == "test content"
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 100

def test_parse_groq_response_stream():
    """测试解析流式响应"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)

    chunks = [
        MockStreamChunk(content="Hello", model="llama2-70b", 
                      usage=MagicMock(_asdict=lambda: {"total_tokens": 50})),
        MockStreamChunk(content=" World", 
                      usage=MagicMock(_asdict=lambda: {"total_tokens": 75})),
        MockStreamChunk(content="!", 
                      usage=MagicMock(_asdict=lambda: {"total_tokens": 100}))
    ]
    
    # 使用类实例测试
    parser = custom_groq.GroqResponseParser()
    result = parser(iter(chunks))
    
    assert result["content"] == "Hello World!"
    assert result["model"] == "llama2-70b"
    assert result["usage"]["total_tokens"] == 225  # 50 + 75 + 100
    
    # 测试向后兼容性 - 使用函数别名
    result = custom_groq.parse_groq_response(iter(chunks))
    
    assert result["content"] == "Hello World!"
    assert result["model"] == "llama2-70b"
    assert result["usage"]["total_tokens"] == 225  # 50 + 75 + 100

def test_parse_groq_response_error_handling(caplog):
    """测试异常处理"""
    # 导入自定义 GROQ 模块
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
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
        parser(invalid_response)
    
    assert "解析GROQ响应失败" in caplog.text
    
    # 测试向后兼容性 - 使用函数别名
    caplog.clear()
    with pytest.raises(Exception):
        custom_groq.parse_groq_response(invalid_response)
    
    assert "解析GROQ响应失败" in caplog.text
def test_interface_implementation():
    """测试接口实现"""
    # 导入自定义 GROQ 模块和接口
    groq_spec = importlib.util.spec_from_file_location(
        "custom_groq", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_groq.py")
    )
    custom_groq = importlib.util.module_from_spec(groq_spec)
    groq_spec.loader.exec_module(custom_groq)
    
    interface_spec = importlib.util.spec_from_file_location(
        "custom_interface", 
        os.path.join(os.path.dirname(__file__), "../source/mota/custom_interface.py")
    )
    custom_interface = importlib.util.module_from_spec(interface_spec)
    interface_spec.loader.exec_module(custom_interface)
    
    # 验证类是否实现了接口
    groq_caller = custom_groq.GroqLLMCaller()
    groq_parser = custom_groq.GroqResponseParser()
    
    assert isinstance(groq_caller, custom_interface.LLMCallerInterface)
    assert isinstance(groq_parser, custom_interface.ResponseParserInterface)
    
    # 验证函数别名是否实现了接口
    assert isinstance(custom_groq.call_groq_api, custom_interface.LLMCallerInterface)
    assert isinstance(custom_groq.parse_groq_response, custom_interface.ResponseParserInterface)
