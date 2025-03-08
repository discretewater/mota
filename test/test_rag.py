"""
RAG功能测试模块

测试Mota的检索增强生成(Retrieval Augmented Generation)功能，
验证知识库检索和上下文增强的正确性。
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from edn_format import Keyword

from mota.main import (
    retrieve_context_knowledge,
    setup_logging
)


@pytest.fixture
def setup_test_environment():
    """设置测试环境"""
    # 配置日志
    setup_logging("DEBUG", "stdout")
    # 创建测试目录路径
    test_knowledge_dir = Path(__file__).parent / "fixture" / "knowledge"
    return str(test_knowledge_dir)


def test_retrieve_context_knowledge_directory_exists(setup_test_environment):
    """测试知识库目录存在时的检索功能"""
    knowledge_dir = setup_test_environment

    # 模拟DirectoryLoader和FAISS
    with patch("mota.main.DirectoryLoader") as mock_loader, \
            patch("mota.main.HuggingFaceEmbeddings") as mock_embeddings, \
            patch("mota.main.FAISS") as mock_faiss:

        # 设置模拟对象的行为
        mock_documents = [MagicMock(page_content=f"测试文档内容 {i}") for i in range(3)]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_documents
        mock_loader.return_value = mock_loader_instance

        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = MagicMock()
        mock_retrieved_docs = [MagicMock(page_content=f"检索到的文档 {i}") for i in range(2)]
        mock_vectorstore.similarity_search.return_value = mock_retrieved_docs
        mock_faiss.from_documents.return_value = mock_vectorstore

        # 调用被测试的函数
        query = "量子力学是什么？"
        result = retrieve_context_knowledge(knowledge_dir, query, top_k=2)

        # 验证函数行为
        mock_loader.assert_called_once_with(knowledge_dir, recursive=True)
        mock_loader_instance.load.assert_called_once()
        mock_embeddings.assert_called_once_with(model_name="all-mpnet-base-v2")
        mock_faiss.from_documents.assert_called_once_with(mock_documents, mock_embeddings_instance)
        mock_vectorstore.similarity_search.assert_called_once_with(query, k=2)

        # 验证返回结果
        assert len(result) == 2
        assert result[0] == "检索到的文档 0"
        assert result[1] == "检索到的文档 1"


def test_retrieve_context_knowledge_directory_not_exists():
    """测试知识库目录不存在时的异常处理"""
    non_existent_dir = "/path/does/not/exist"

    # 验证函数是否抛出预期的异常
    with pytest.raises(ValueError) as excinfo:
        retrieve_context_knowledge(non_existent_dir, "测试查询")

    # 验证异常消息
    assert "指定的目录不存在" in str(excinfo.value)


def test_retrieve_context_knowledge_with_real_files(setup_test_environment):
    """测试使用实际文件的知识库检索功能"""
    knowledge_dir = setup_test_environment

    # 确保测试目录存在
    if not os.path.exists(knowledge_dir):
        pytest.skip(f"测试目录不存在: {knowledge_dir}")

    # 模拟FAISS和嵌入模型，但使用实际的DirectoryLoader加载文件
    with patch("mota.main.HuggingFaceEmbeddings") as mock_embeddings, \
            patch("mota.main.FAISS") as mock_faiss:

        # 设置模拟对象的行为
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = MagicMock()
        # 创建模拟的检索结果，使用实际文件的内容片段
        mock_retrieved_docs = [
            MagicMock(page_content="量子力学（quantum mechanics）是物理学的分支学科。"),
            MagicMock(page_content="量子理论的重要应用包括宇宙学、量子化学、量子光学")
        ]
        mock_vectorstore.similarity_search.return_value = mock_retrieved_docs
        mock_faiss.from_documents.return_value = mock_vectorstore

        # 调用被测试的函数
        query = "量子力学的应用"
        result = retrieve_context_knowledge(knowledge_dir, query)

        # 验证返回结果
        assert len(result) == 2
        assert "量子力学" in result[0]
        assert "量子理论的重要应用" in result[1]


@patch("mota.main.DirectoryLoader")
@patch("mota.main.HuggingFaceEmbeddings")
@patch("mota.main.FAISS")
def test_retrieve_context_knowledge_integration(mock_faiss, mock_embeddings, mock_loader, setup_test_environment):
    """测试知识库检索的集成功能"""
    knowledge_dir = setup_test_environment

    # 设置模拟对象
    mock_documents = [MagicMock(page_content=f"量子力学文档 {i}") for i in range(5)]
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_documents
    mock_loader.return_value = mock_loader_instance

    mock_embeddings_instance = MagicMock()
    mock_embeddings.return_value = mock_embeddings_instance

    mock_vectorstore = MagicMock()
    mock_retrieved_docs = [MagicMock(page_content=f"相关量子力学内容 {i}") for i in range(3)]
    mock_vectorstore.similarity_search.return_value = mock_retrieved_docs
    mock_faiss.from_documents.return_value = mock_vectorstore

    # 调用被测试的函数，使用不同的top_k值
    query = "量子纠缠是什么？"
    result = retrieve_context_knowledge(knowledge_dir, query, top_k=3)

    # 验证函数行为和结果
    mock_loader.assert_called_once_with(knowledge_dir, recursive=True)
    mock_faiss.from_documents.assert_called_once()
    mock_vectorstore.similarity_search.assert_called_once_with(query, k=3)

    assert len(result) == 3
    for i in range(3):
        assert result[i] == f"相关量子力学内容 {i}"


@patch("mota.main.parse_response")
@patch("mota.main.load_config")
@patch("mota.main.get_api_key")
@patch("mota.main.get_llm_call_func")
@patch("mota.main.retrieve_context_knowledge")
def test_main_with_knowledge_dir(mock_retrieve, mock_get_llm_call, mock_get_api_key, mock_load_config, mock_parse_response):
    """测试主函数中的知识库检索集成"""
    from typer.testing import CliRunner
    from mota.main import cli

    # 设置模拟对象
    mock_load_config.return_value = {
        Keyword('logging'): {Keyword('level'): "DEBUG"},
        Keyword('llm'): {
            Keyword('providers'): {
                Keyword('groq'): {
                    Keyword('model'): "deepseek-r1-distill-llama-70b"
                }
            },
            Keyword('temperature'): 0.7,
            Keyword('stream'): True,
            Keyword('max_tokens'): 1000
        }
    }
    mock_get_api_key.return_value = "test-api-key"

    # 模拟知识库检索结果
    mock_retrieve.return_value = ["量子力学是物理学的分支", "量子理论有广泛的应用"]

    # 模拟LLM调用函数和响应
    mock_llm_call = MagicMock()
    mock_response = MagicMock()
    mock_llm_call.return_value = mock_response
    mock_get_llm_call.return_value = mock_llm_call

    # 模拟响应解析
    mock_parsed_response = {"content": "这是关于量子力学的回答"}
    mock_parse_response.return_value = mock_parsed_response
    mock_llm_call.return_value = mock_response

    # 使用CliRunner执行命令
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--log-level", "DEBUG",
        "--provider", "groq",
        "--prompt", "请用中文回答",
        "--knowledge-dir", "test/fixture/knowledge",
        "解释量子力学。"
    ])

    # 验证知识库检索函数被调用
    # 直接检查调用参数
    assert mock_retrieve.called, "retrieve_context_knowledge 函数未被调用"
    knowledge_dir_arg = mock_retrieve.call_args[0][0]
    query_arg = mock_retrieve.call_args[0][1]

    assert "test/fixture/knowledge" in knowledge_dir_arg
    assert "请用中文回答 解释量子力学。" in query_arg

    # 验证LLM调用函数接收到了增强的提示词
    prompt_arg = mock_llm_call.call_args[0][2]
    assert "请用中文回答" in prompt_arg
    assert "参考以下相关信息" in prompt_arg
    assert "量子力学是物理学的分支" in prompt_arg
    assert "量子理论有广泛的应用" in prompt_arg

    # 验证输出包含解析后的响应
    assert "这是关于量子力学的回答" in result.output
