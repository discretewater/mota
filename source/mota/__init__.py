
"""
* 概览
** 简介
Mota 是用于大语言模型(LLM)交互的核心Python包，提供标准化接口和扩展框架。支持主流LLM服务集成，采用插件架构实现灵活扩展。

** 设计
- 模块化架构分离接口定义与具体实现
- 通过抽象基类强制接口一致性
- 配置驱动模式支持运行时动态调整
- 插件系统基于动态导入和自动发现机制

** 实现
- 使用EDN格式统一管理配置(API密钥/模型参数等)
- 基于importlib的动态模块加载器
- 类型注解强化接口约束

** 建议
- 异步IO支持
- 优先通过继承custom_interface基类开发新适配器
- 复杂业务逻辑建议实现为独立插件
- 生产环境应禁用自动依赖安装功能

** 待办
- [ ] 添加性能监控指标
- [ ] 开发测试套件框架

* 用例
** 命令行交互
# 执行对话任务
$ mota --log-level 'DEBUG' --provider=groq --custom-caller=source/mota/custom_groq.py --custom-parser=source/mota/custom_groq.py --prompt "无与伦比的科技大师，你好！我需要你的帮助。" "请解释欧拉公式。"

** API调用示例
>>> from mota import seek
>>> response = seek(provider="openai",
...                 model="gpt-4",
...                 prompt="你是一个专业助手",
...                 message="解释量子力学")
>>> print(response['content'])

** 其它调用示例
>>> from mota.main import cli
>>> from mota.core import load_custom_func

* 版权许可
本程序遵循GNU Affero通用公共许可证第三版(AGPLv3)。完整授权条款参见：
http://www.gnu.org/licenses/agpl-3.0.html
"""

__version__ = "1.0.0"
__all__ = [
    'core',
    'LLMCallerInterface',
    'ResponseParserInterface',
    'seek'
]

from . import core
from .custom_interface import LLMCallerInterface, ResponseParserInterface
from .seek import seek
