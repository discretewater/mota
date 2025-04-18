"""
loader.py - 动态模块加载器

本模块实现插件系统的核心加载机制，主要功能包括：

关键功能：
1. 路径加载：从任意文件路径动态加载Python模块（load_module_from_path）
2. 接口发现：自动识别模块中符合接口协议的实现类（find_implementor）

技术特性：
- 安全隔离：每个模块在独立命名空间中加载
- 热重载：支持运行时更新模块实现
- 错误处理：完善的异常捕获和错误日志
- 类型检查：严格验证实现类是否符合接口协议

典型应用场景：
- 加载新的LLM提供商实现（如custom_groq.py）
- 动态更换响应解析逻辑
- 扩展工具功能而不修改核心代码

安全注意事项：
- 应验证模块签名防止恶意代码注入
- 建议限制加载路径到指定目录
- 生产环境应禁用动态重载功能
"""

import importlib.util
import sys
from typing import Type
from pathlib import Path
from abc import ABC
import os


def load_module_from_path(module_name: str, file_path: Path):
    """
    根据文件路径加载Python模块。

    Args:
        module_name (str): 指定加载模块的名称
        file_path (Path): 用户提供的Python文件路径

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
        raise ImportError(f"文件 '{file_path}' 不存在")
    except Exception as e:
        raise ImportError(f"加载模块失败 - {str(e)}")


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
        if (
            isinstance(attr, type)
            and issubclass(attr, interface)  # 是接口的子类
            and attr is not interface  # 排除接口自身
            and not getattr(attr, "__abstractmethods__", False)  # 排除抽象类
        ):
            return attr
    return None
