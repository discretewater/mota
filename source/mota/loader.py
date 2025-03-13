"""
loader.py - 模块加载器

本模块提供了动态加载 Python 模块和查找接口实现类的功能。
主要用于加载用户自定义的 LLM API 调用和响应解析实现。

主要功能：
1. load_module_from_path: 根据文件路径动态加载 Python 模块
2. find_implementor: 在模块中查找实现了指定接口的类

典型使用场景：
- 加载用户自定义的 LLM API 调用实现
- 加载用户自定义的响应解析器
- 动态加载插件或扩展模块

注意：
- 该模块使用 Python 的 importlib 实现动态加载
- 支持从任意路径加载 Python 模块
- 支持接口实现的自动发现
"""

import importlib.util
import sys
from typing import Type
from abc import ABC
import os


def load_module_from_path(module_name: str, file_path: str):
    """
    根据文件路径加载Python模块。

    Args:
        module_name (str): 指定加载模块的名称
        file_path (str): 用户提供的Python文件路径

    Returns:
        module: 加载成功的模块对象

    Raises:
        FileNotFoundError: 如果文件路径无效
        ImportError: 如果模块加载失败
    """
    try:
        if not os.path.exists(file_path):
            raise ImportError(f"文件路径不存在: {file_path}")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"无法从路径加载模块: {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        if spec.loader is None:
            raise ImportError(f"模块加载器为None: {file_path}")
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
        sys.exit(1)
    except Exception as e:
        print(f"错误：加载模块失败 - {str(e)}")
        sys.exit(1)


def find_implementor(module, interface: Type[ABC]) -> Type[ABC] | None:
    """
    在模块中查找继承了指定接口的类。

    Args:
        module: 已加载的模块对象
        interface: 要检查的接口类型

    Returns:
        继承了 interface 的类，如果未找到则返回 None
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # 检查是否为类并且是 interface 的子类
        if isinstance(attr, type) and issubclass(attr, interface):
            return attr
    return None
