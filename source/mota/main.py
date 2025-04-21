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
from pathlib import Path
import typer

# 从core模块导入所有核心功能
from mota.core import (  # noqa: F401
    get_llm_call_func, get_parser_func,
    logger, JsonDict
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
    prompt: str = typer.Option(
        "万能的专家系统，我需要帮助。",
        help="系统提示词",
        show_default=True),
    message: str = typer.Argument(..., help="用户消息 (必选参数)"),
    temperature: float = typer.Option(0.7, help="温度", show_default=True),
    stream: bool = typer.Option(True, help="启用流模式", show_default=True),
    config_path: Optional[str] = typer.Option(None, help="配置文件路径"),
    log_level: str = typer.Option("INFO", help="日志级别", show_default=True),
    log_output: str = typer.Option("stdout", help="日志输出目标", show_default=True),
    custom_params: Optional[str] = typer.Option(
        None, parser=JsonDict.parse_json, help="自定义聊天请求参数，使用JSON格式"),
    fields: Optional[str] = typer.Option(None, help="需要提取的响应字段，使用逗号分隔"),
    custom_caller: Optional[Path] = typer.Option(
        None, help="用户自定义 LLM API 调用函数的模块路径，格式为 module:function", show_default=False),
    custom_parser: Optional[Path] = typer.Option(
        None, help="自定义响应解析函数路径，格式为 模块名:函数名", show_default=False),
    knowledge_dir: Optional[str] = typer.Option(
        None, help="知识库目录路径，用于RAG检索增强生成", show_default=False),
    user_query: Optional[List[str]] = typer.Argument(
        None, help="附加的用户查询，将会附加到主要用户消息后")
) -> None:
    """
    程序入口点，与LLM进行对话
    """
    try:
        # 调用核心API函数
        from mota.seek import seek
        result = seek(
            provider=provider,
            model=model,
            prompt=prompt,
            message=message,
            temperature=temperature,
            stream=stream,
            config_path=config_path,
            log_level=log_level,
            log_output=log_output,
            custom_params=custom_params,
            fields=fields.split(',') if fields else None,
            custom_caller=custom_caller,
            custom_parser=custom_parser,
            knowledge_dir=knowledge_dir,
            user_query=user_query
        )

        # 输出结果
        print(result)

    except Exception as e:
        logger.error(f"聊天失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli()
