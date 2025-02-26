#!/usr/bin/env python3
"""
Mota - A Comprehensive LLM API Interaction Tool

This module provides a unified interface for interacting with various LLM APIs
including OpenAI, Anthropic, Google's Gemini, GROQ, GROK, DeepSeek, Mistral,
OpenRouter, and others. It supports configuration management, authentication handling,
and customizable API interactions.

This program is licensed under the GNU General Public License (GPL) version 3.
Copyright (C) [year] [your name]
"""

import os
import sys
import logging
import typing
from typing import Optional, Dict, Any, Callable
import typer
import edn_format
from pathlib import Path
from dotenv import dotenv_values
import keyring  # 用于处理 .authinfo 文件
import gnupg  # 用于处理加密的 .authinfo.gpg

# Initialize typer app
app = typer.Typer(help="Mota - LLM API Interaction Tool")

# Initialize logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", output: str = "stdout") -> None:
    """
    设置日志配置

    Args:
        level (str): 日志级别
        output (str): 日志输出目标
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"无效的日志级别: {level}")

    logging_config = {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'level': numeric_level
    }

    if output.lower() == "stdout":
        logging_config['stream'] = sys.stdout
    else:
        logging_config['filename'] = output

    logging.root.handlers.clear()
    logging.basicConfig(**logging_config)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载EDN格式的配置文件

    Args:
        config_path (Optional[str]): 配置文件路径

    Returns:
        Dict[str, Any]: 配置字典
    """
    default_config_path = Path(__file__).parent / "config" / "default.edn"

    try:
        with open(config_path or default_config_path, 'r') as f:
            return edn_format.loads(f.read())
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        raise


def get_api_key(provider: str,
                auth_source: str = "env",
                auth_path: Optional[str] = None) -> str:
    """
    获取API密钥

    Args:
        provider (str): LLM提供商名称
        auth_source (str): 认证源（env/command_line/authinfo/config）
        auth_path (Optional[str]): 认证文件路径

    Returns:
        str: API密钥
    """
    # 从环境变量获取
    if auth_source == "env":
        key = os.getenv(f"{provider.upper()}_API_KEY")
        if key:
            return key

    # 从命令行参数获取
    if auth_source == "command_line":
        # 假设通过命令行传入的密钥
        key = os.getenv(f"{provider.upper()}_API_KEY_CMD")
        if key:
            return key

    # 从 EMACS .authinfo 或 .authinfo.gpg 获取
    if auth_source in ["authinfo", "authinfo_gpg"]:
        auth_file = auth_path or os.path.expanduser("~/.authinfo")
        if auth_source == "authinfo_gpg":
            auth_file += ".gpg"
            gpg = gnupg.GPG()
            with open(auth_file, 'rb') as f:
                decrypted_data = gpg.decrypt_file(f)
                if decrypted_data.ok:
                    auth_data = decrypted_data.data.decode()
                else:
                    logger.error("解密 .authinfo.gpg 失败")
                    raise ValueError("无法解密 .authinfo.gpg 文件")
        else:
            with open(auth_file, 'r') as f:
                auth_data = f.read()

        for line in auth_data.splitlines():
            if line.startswith(f"{provider.lower()}-api-key"):
                return line.split()[1]

    # 从配置文件获取
    if auth_source == "config":
        config = load_config()
        return config.get('llm', {}).get('providers', {}).get(provider.lower(), {}).get('api_key', '')

    raise ValueError(f"无法找到 {provider} 的API密钥")


def format_prompt(template: str, **kwargs: Any) -> str:
    """
    格式化提示词模板

    Args:
        template (str): 提示词模板
        kwargs (Any): 模板参数

    Returns:
        str: 格式化后的提示词
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"缺少模板参数: {e}")
        raise
    except Exception as e:
        logger.error(f"格式化提示词失败: {e}")
        raise


def parse_response(response: Any,
                   custom_parser: Optional[Callable] = None) -> Dict[str, Any]:
    """
    解析API响应

    Args:
        response (Any): API响应对象
        custom_parser (Optional[Callable]): 自定义解析函数

    Returns:
        Dict[str, Any]: 解析后的响应内容
    """
    if custom_parser:
        return custom_parser(response)

    # 默认解析逻辑
    try:
        return {
            'content': response.choices[0].message.content,
            'model': response.model,
            'usage': response.usage._asdict() if hasattr(
                response,
                'usage') else {}
        }
    except Exception as e:
        logger.error(f"解析响应失败: {e}")
        raise


def extract_fields(response_dict: Dict[str, Any],
                   fields: typing.List[str]) -> Dict[str, Any]:
    """
    从响应字典中提取指定字段

    Args:
        response_dict (Dict[str, Any]): 解析后的响应字典
        fields (List[str]): 需要提取的字段列表

    Returns:
        Dict[str, Any]: 提取的字段及其值
    """
    extracted = {}
    for field in fields:
        if field in response_dict:
            extracted[field] = response_dict[field]
        else:
            logger.warning(f"字段 {field} 在响应中不存在")
    return extracted


@app.command()
def chat(
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
    prompt_text: str = typer.Argument(..., help="聊天提示"),
    temperature: float = typer.Option(0.7, help="温度", show_default=True),
    stream: bool = typer.Option(True, help="启用流模式", show_default=True),
    config_path: Optional[str] = typer.Option(None, help="配置文件路径"),
    log_level: str = typer.Option("INFO", help="日志级别", show_default=True),
    log_output: str = typer.Option("stdout", help="日志输出目标", show_default=True),
    custom_params: Optional[str] = typer.Option(None, help="自定义聊天请求参数，使用JSON格式"),
    fields: Optional[str] = typer.Option(None, help="需要提取的响应字段，使用逗号分隔")
) -> None:
    """
    与LLM进行对话
    """
    try:
        # 设置日志
        setup_logging(log_level, log_output)
        logger.debug("日志系统已初始化")

        # 加载配置
        config = load_config(config_path)
        logger.debug(f"加载配置: {config}")

        # 获取API密钥
        api_key = get_api_key(provider)
        logger.debug(f"获取到的API密钥: {api_key}")

        # 配置请求参数
        request_params = {
            "model": model or config['llm']['providers'][provider.lower()]['model'],
            "temperature": temperature or config['llm']['temperature'],
            "stream": stream if 'stream' not in config['llm'] else config['llm']['stream'],
            "max_tokens": config['llm']['max_tokens']
        }

        # 解析自定义参数
        if custom_params:
            import json
            user_params = json.loads(custom_params)
            request_params.update(user_params)
            logger.debug(f"合并自定义请求参数: {user_params}")

        logger.debug(f"最终请求参数: {request_params}")

        # 格式化提示词
        formatted_prompt = format_prompt(prompt_text, **{})
        logger.debug(f"格式化后的提示词: {formatted_prompt}")

        # TODO: 根据提供商实现具体的API调用逻辑
        # 这里以OpenAI为例
        if provider.lower() == "openai":
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=request_params["model"],
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=request_params["temperature"],
                stream=request_params["stream"],
                max_tokens=request_params["max_tokens"]
            )
        elif provider.lower() == "anthropic":
            import anthropic
            client = anthropic.Client(api_key)
            response = client.completion(
                prompt=formatted_prompt,
                model=request_params["model"],
                temperature=request_params["temperature"],
                stop_sequences=["\n"],
                max_tokens_to_sample=request_params["max_tokens"]
            )
        # TODO: 添加其他提供商的API调用逻辑
        else:
            logger.error(f"尚未实现 {provider} 提供商的API调用逻辑")
            raise NotImplementedError(f"{provider} 提供商的API调用逻辑未实现")

        logger.debug(f"API响应: {response}")

        # 解析响应
        parsed_response = parse_response(response)
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


def main():
    """
    程序入口点
    """
    app()


if __name__ == "__main__":
    main()
