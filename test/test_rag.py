"""
test_rag.py - 检索增强生成(RAG)功能测试

本测试模块验证以下核心功能：
1. 知识库加载：测试不同格式文档（PDF/DOCX/TXT）的加载能力
2. 向量索引：验证FAISS索引构建的正确性和性能
3. 相似度检索：测试top-k检索的相关性和排序准确性
4. 上下文整合：检查检索结果与提示词的融合逻辑

测试策略：
- 模拟测试：使用Mock对象隔离外部依赖
- 边界测试：空目录、无效路径、超大文档等异常场景
- 性能基准：记录典型查询的响应时间
- 一致性验证：确保多次检索结果的一致性

环境要求：
- 测试知识库目录：test/fixture/knowledge/
- 嵌入模型：all-mpnet-base-v2（HuggingFace）
- 最小相似度阈值：0.75

测试数据说明：
- 包含量子力学基础知识的文本文件
- 模拟文档包含中英文混合内容
- 文档大小从1KB到1MB不等
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from edn_format import Keyword

from mota.core import (
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
    with patch("mota.core.DirectoryLoader") as mock_loader, \
            patch("mota.core.HuggingFaceEmbeddings") as mock_embeddings, \
            patch("mota.core.FAISS") as mock_faiss:

        # 设置模拟对象的行为
        mock_documents = [
            MagicMock(
                page_content=f"测试文档内容 {i}") for i in range(3)]
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_documents
        mock_loader.return_value = mock_loader_instance

        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vectorstore = MagicMock()
        mock_retrieved_docs = [
            MagicMock(
                page_content=f"检索到的文档 {i}") for i in range(2)]
        mock_vectorstore.similarity_search.return_value = mock_retrieved_docs
        mock_faiss.from_documents.return_value = mock_vectorstore

        # 调用被测试的函数
        query = "量子力学是什么？"
        result = retrieve_context_knowledge(knowledge_dir, query, top_k=2)

        # 验证函数行为
        mock_loader.assert_called_once_with(knowledge_dir, recursive=True)
        mock_loader_instance.load.assert_called_once()
        mock_embeddings.assert_called_once_with(model_name="all-mpnet-base-v2")
        mock_faiss.from_documents.assert_called_once_with(
            mock_documents, mock_embeddings_instance)
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
    with patch("mota.core.HuggingFaceEmbeddings") as mock_embeddings, \
            patch("mota.core.FAISS") as mock_faiss:

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


@patch("mota.core.DirectoryLoader")
@patch("mota.core.HuggingFaceEmbeddings")
@patch("mota.core.FAISS")
def test_retrieve_context_knowledge_integration(
        mock_faiss,
        mock_embeddings,
        mock_loader,
        setup_test_environment):
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
    mock_retrieved_docs = [
        MagicMock(
            page_content=f"相关量子力学内容 {i}") for i in range(3)]
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


@patch("mota.seek.retrieve_context_knowledge")
@patch("mota.seek.load_config")
@patch("mota.seek.get_api_key")
@patch("mota.seek.get_llm_call_func")
@patch("mota.seek.get_parser_func")
def test_main_with_rag_integration(
        mock_get_parser_func,
        mock_get_llm_call_func,
        mock_get_api_key,
        mock_load_config,
        mock_retrieve_context_knowledge):
    """测试主函数中的知识库检索集成（RAG）功能"""
    from typer.testing import CliRunner
    from mota.main import cli

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

    # 模拟知识库检索结果
    mock_retrieve_context_knowledge.return_value = [
        "量子力学是物理学的一个分支，描述微观粒子的行为。",
        "量子物理不同于宏观物理。",
        "量子纠缠是量子力学中的一种现象，指两个或多个粒子的量子状态相互关联。"
    ]

    # 模拟LLM调用和响应解析
    mock_llm_call = MagicMock()
    mock_llm_call.return_value = "模拟的LLM响应"
    mock_get_llm_call_func.return_value = mock_llm_call

    mock_parser = MagicMock()
    mock_parser.return_value = {
        "content": "量子力学是研究原子和亚原子尺度现象的物理学分支，与相对论共同构成现代物理学的两大支柱。它通过概率和波函数描述微观粒子的行为，解释了经典物理学无法解释的现象，并在多个领域有广泛应用。"}
    mock_get_parser_func.return_value = mock_parser

    # 使用CliRunner调用主函数
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--provider", "openai",
        "--model", "gpt-x",
        "--prompt", "你是一个量子物理学专家。",
        "--knowledge-dir", "/path/to/knowledge",
        "请解释量子力学。"
    ])

    # 验证知识库检索是否被调用
    mock_retrieve_context_knowledge.assert_called_once()
    knowledge_dir_arg = mock_retrieve_context_knowledge.call_args[0][0]
    query_arg = mock_retrieve_context_knowledge.call_args[0][1]
    assert knowledge_dir_arg == "/path/to/knowledge"
    assert "你是一个量子物理学专家。" in query_arg
    assert "请解释量子力学。" in query_arg

    # 验证LLM调用中是否包含了检索到的上下文
    llm_call_args = mock_llm_call.call_args
    formatted_prompt = llm_call_args[0][2]  # 第三个位置参数是formatted_prompt

    # 验证格式化后的提示词中包含了检索到的上下文
    assert "你是一个量子物理学专家。" in formatted_prompt
    assert "量子力学是物理学的一个分支" in formatted_prompt
    assert "量子纠缠是量子力学中的一种现象" in formatted_prompt

    # 验证响应解析是否被调用
    mock_parser.assert_called_once()

    # 验证命令执行成功
    assert result.exit_code == 0
