"""
Mota - 全能大语言模型API交互工具

本模块提供与多种大语言模型(LLM)API交互的统一接口，支持以下功能：
1. 多提供商API集成：OpenAI、Anthropic、Gemini、GROQ、GROK、DeepSeek、Mistral、OpenRouter等
2. 动态插件机制：支持加载自定义API调用器和响应解析器
3. 检索增强生成(RAG)：集成知识库检索功能
4. 统一配置管理：通过EDN格式配置文件管理所有参数
5. 安全认证管理：支持多种密钥获取方式（环境变量、配置文件、authinfo文件等）

主要功能模块：
- 配置加载与验证
- API密钥安全管理
- 提示词模板格式化
- 上下文知识检索
- 多提供商API统一调用接口
- 响应解析与结果提取

典型使用场景：
- 快速切换不同LLM服务提供商
- 开发自定义LLM集成插件
- 构建基于知识库的智能问答系统
- 统一管理多个API密钥和配置参数

版权声明：
本程序遵循GNU通用公共许可证(GPL)第三版
版权所有 (C) 2024 Mota开发团队
"""

from typing import Optional, List
import typer
from edn_format import Keyword

# 从core模块导入所有核心功能
from mota.core import (
    setup_logging, load_config, get_api_key, format_prompt, extract_fields,
    get_llm_call_func, get_parser_func, retrieve_context_knowledge,
    logger
)


# 创建一个不使用子命令的 Typer 应用
cli = typer.Typer(help="Mota - LLM API Interaction Tool", add_completion=False)


@cli.command()
def main(
    # 必选参数
    provider: str = typer.Option("openai", help="LLM 提供商",
                                 case_sensitive=False,
                                 show_choices=True,
                                 show_default=True,
                                 rich_help_panel=None,
                                 prompt="请选择 LLM 提供商",
                                 metavar="PROVIDER",
                                 callback=None,
                                 is_eager=False,
                                 hidden=False,
                                 show_envvar=False,
                                 flag_value=None),
    model: Optional[str] = typer.Option(None, help="模型名称", show_default=True),
    prompt: str = typer.Option("万能的专家系统，我需要帮助。", help="系统提示词", show_default=True),
    message: str = typer.Argument(..., help="用户消息 (必选参数)"),
    temperature: float = typer.Option(0.7, help="温度", show_default=True),
    stream: bool = typer.Option(True, help="启用流模式", show_default=True),
    config_path: Optional[str] = typer.Option(None, help="配置文件路径"),
    log_level: str = typer.Option("INFO", help="日志级别", show_default=True),
    log_output: str = typer.Option("stdout", help="日志输出目标", show_default=True),
    custom_params: Optional[str] = typer.Option(None, help="自定义聊天请求参数，使用JSON格式"),
    fields: Optional[str] = typer.Option(None, help="需要提取的响应字段，使用逗号分隔"),
    custom_caller: Optional[str] = typer.Option(None, help="用户自定义 LLM API 调用函数的模块路径，格式为 module:function", show_default=False),
    custom_parser: Optional[str] = typer.Option(None, help="自定义响应解析函数路径，格式为 模块名:函数名", show_default=False),
    knowledge_dir: Optional[str] = typer.Option(None, help="知识库目录路径，用于RAG检索增强生成", show_default=False),
    user_query: Optional[List[str]] = typer.Argument(None, help="附加的用户查询，将会附加到主要用户消息后")
) -> None:
    """
    程序入口点，与LLM进行对话
    """
    try:
        # 设置日志
        setup_logging(log_level, log_output)
        logger.debug("日志系统已初始化")

        # 处理用户查询参数作为用户消息
        actual_user_message = message

        if user_query:
            logger.debug(f"检测到用户查询参数: {user_query}")
            # 将用户查询添加到用户消息后面
            actual_user_message = f"{actual_user_message} {' '.join(user_query)}"
            logger.debug(f"合并后的用户消息: {actual_user_message}")

        # 加载配置
        config = load_config(config_path)
        logger.debug(f"加载配置: {config}")

        # 获取API密钥
        api_key = get_api_key(provider)
        logger.debug(f"获取到的API密钥: {api_key}")

        # 配置请求参数
        request_params = {
            "model": model or config[Keyword('llm')][Keyword('providers')][Keyword(provider.lower())][Keyword('model')],
            "temperature": temperature or config[Keyword('llm')][Keyword('temperature')],
            "stream": stream if Keyword('stream') not in config[Keyword('llm')] else config[Keyword('llm')][Keyword('stream')],
            "max_tokens": config[Keyword('llm')][Keyword('max_tokens')],
            "message": actual_user_message  # 添加用户消息参数
        }

        # 解析自定义参数
        if custom_params:
            import json
            user_params = json.loads(custom_params)
            request_params.update(user_params)
            logger.debug(f"合并自定义请求参数: {user_params}")

        logger.debug(f"最终请求参数: {request_params}")

        # 格式化提示词
        formatted_prompt = prompt

        # 如果指定了知识库目录，则进行RAG检索
        if knowledge_dir:
            logger.info(f"检测到知识库目录: {knowledge_dir}，将进行RAG检索")
            # 构建查询字符串，结合系统提示词和用户消息
            query = f"{prompt} {actual_user_message}"
            # 调用RAG检索函数获取相关上下文
            try:
                context_knowledge = retrieve_context_knowledge(knowledge_dir, query)
                # 将检索到的上下文合并为一个字符串
                context_text = "\n\n".join(context_knowledge)
                # 将上下文添加到提示词中
                formatted_prompt = f"{prompt}\n\n参考以下相关信息：\n\n{context_text}"
                logger.info("成功添加RAG检索结果到提示词")
                logger.debug(f"添加RAG后的提示词长度: {len(formatted_prompt)}")
            except Exception as e:
                logger.error(f"RAG检索失败: {e}")
                # 如果RAG检索失败，仍使用原始提示词继续
                logger.info("将使用原始提示词继续")

        # 格式化最终提示词
        formatted_prompt = format_prompt(formatted_prompt, **{})
        logger.debug(f"格式化后的提示词: {formatted_prompt}")

        # 调用 LLM API 的统一接口函数，根据配置参数和自定义函数实现调用逻辑
        llm_call_func = get_llm_call_func(custom_caller)
        response = llm_call_func(provider, api_key, formatted_prompt, request_params)

        logger.debug(f"API响应: {response}")

        # 获取解析器
        parse_func = get_parser_func(custom_parser)
        # 解析响应
        parsed_response = parse_func(response)
        logger.debug(f"解析后的响应: {parsed_response}")

        # 提取指定字段
        if fields:
            field_list = fields.split(',')
            extracted = extract_fields(parsed_response, field_list)
            print(extracted)
        else:
            print(parsed_response)

    except Exception as e:
        logger.error(f"聊天失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli()
