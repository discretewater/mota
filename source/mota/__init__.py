"""
Mota - 大语言模型交互核心包

本包提供与各种大语言模型(LLM)交互的核心功能，包含以下主要模块：
- main: 主程序模块，提供命令行接口和核心工作流程
- loader: 动态模块加载器，支持插件式扩展
- custom_interface: 定义LLM交互接口协议
- custom_*: 各LLM提供商的具体实现

主要特性：
1. 统一接口：通过标准化接口集成多种LLM服务
2. 插件架构：支持动态加载自定义实现
3. 可扩展性：易于添加新的LLM提供商支持
4. 配置驱动：通过EDN文件统一管理所有配置

版本要求：
- Python 3.10+
- 依赖管理通过Poetry

典型导入方式：
>>> from mota.main import cli
>>> from mota.loader import load_module_from_path
"""

__version__ = "1.0.0"
__all__ = [
    'main',
    'loader',
    'custom_interface',
    'custom_groq',
    'custom_anthropic',
    'seek']

from .seek import seek
