# Mota

Mota 是一个用于与各种主要大语言模型（LLM）API服务交互的综合工具。它通过插件式架构支持多个 LLM 提供商，包括 OpenAI、Anthropic、GROQ 等，并提供统一的配置管理、认证处理和可扩展的 API 交互接口。

## 特性

- **插件式架构**：通过动态加载自定义模块实现 LLM 服务扩展
  - 支持自定义 LLM 调用器（Caller）和响应解析器（Parser）
  - 提供标准接口协议（LLMCallerInterface/ResponseParserInterface）
  - 已内置 Anthropic、GROQ 等厂商的完整实现示例
  
- **检索增强生成（RAG）**：
  - 集成 LangChain 文档加载和向量检索
  - 支持 DOCX/PDF/TXT 等多种文档格式
  - 基于 FAISS 实现高效语义搜索

- **统一配置管理**：
  - 使用 EDN 格式统一管理所有参数
  - 支持多级配置继承和覆盖
  - 提供默认配置文件（config/default.edn）

- **多源认证管理**：
  - 支持环境变量、命令行参数、.authinfo 文件
  - 提供 GPG 加密凭证保护
  - 自动选择最优认证源

- **企业级功能**：
  - 双模式响应处理（流式/非流式）
  - 智能参数转换与校验

- **开发者友好**：
  - 详尽的日志记录与调试信息
  - 类型注解完善的 Python API
  - 100% 测试覆盖率的核心模块
  - 模块化的功能组件设计

## 安装

### 通过 PyPI 安装
```bash
pip install mota
```

### 从源码安装（开发模式）
```bash
git clone https://github.com/username/mota.git
cd mota
pip install -e .
```

## 使用

### 基础示例
```bash
# 使用GROQ
mota --log-level 'DEBUG' --provider=groq --custom-caller=source/mota/custom_groq.py --custom-parser=source/mota/custom_groq.py --knowledge-dir test/fixture/knowledge --prompt "无与伦比的科技大师，你好！我需要你的帮助。" "解释量子力学。"
```

### Python API 使用示例
```python
from mota import seek

response = seek(
    provider="groq",
    custom_caller="source/mota/custom_groq.py",
    custom_parser="source/mota/custom_groq.py",
    model="deepseek-r1-distill-llama-70b",
    prompt="无与伦比的科技大师，你好！我需要你的帮助。",
    message="详细说明Transformer架构。",
    fields=["content", "usage"]
)

print(">>> ", response)
```

### Seek API 参数说明

| 参数             | 类型                  | 默认值     | 说明                                                                 |
|------------------|-----------------------|------------|---------------------------------------------------------------------|
| provider         | str                   | "openai"   | 支持的LLM提供商: openai/anthropic/groq/mistral/deepseek/openrouter |
| model            | Optional[str]         | None       | 当为None时自动使用配置文件中的默认模型                             |
| stream           | bool                  | True       | 流式响应模式，建议在CLI中关闭，在API中启用                         |
| knowledge_dir    | Optional[str]         | None       | 启用RAG检索时需指向包含.txt/.pdf/.docx等文件的目录                 |
| fields           | Optional[List[str]]   | None       | 支持嵌套字段提取，如 ["content", "usage.total_tokens"]             |
| custom_caller    | Optional[str]         | None       | 格式："/path/to/module.py:ClassName"                               |
| user_query       | Optional[List[str]]   | None       | 支持追加多个查询参数，自动拼接至主消息后                           |

### 异步调用与错误处理
```python
import asyncio
from mota import seek
from mota.core import LLMError

# 异步调用示例
async def async_seek():
    response = await seek(
        provider="groq",
        message="异步编程的优势",
        stream=False,
        async_mode=True
    )
    print(response['content'])

asyncio.run(async_seek())

# 错误处理示例
try:
    response = seek(provider="openai", model="gpt-5")  # 不存在的模型
except LLMError as e:
    print(f"API错误代码: {e.code}")
    print(f"错误详情: {e.details}")
except Exception as e:
    print(f"系统错误: {str(e)}")
```

### 高级功能示例
```bash
# 使用 GROQ 并加载自定义实现（开发调试）
mota --log-level DEBUG \
  --provider groq \
  --custom-caller source/mota/custom_groq.py \
  --custom-parser source/mota/custom_groq.py \
  --knowledge-dir ./knowledge_base \
  --prompt "你是一位量子物理专家，请用中文回答：" \
  "请详细解释薛定谔方程"

# 使用 Anthropic 结合 RAG 检索
mota --provider anthropic \
  --custom-caller source/mota/custom_anthropic.py \
  --knowledge-dir ./tech_docs \
  --temperature 0.3 \
  "如何实现分布式系统的一致性？"
```

### 企业级部署
```bash
# 使用加密认证信息（需要提前配置 GPG）
mota --provider groq \
  --auth-source authinfo_gpg \
  --auth-path ~/.authinfo.gpg \
  --model deepseek-r1-distill-llama-70b \
  "分析以下财务报表：<附加财务数据>"
```

### 核心命令行选项

| 选项                | 说明                                                                 |
|---------------------|--------------------------------------------------------------------|
| `--provider`        | 指定 LLM 提供商 (`groq`, `anthropic`, `openai` 等)，默认: openai     |
| `--model`           | 选择特定模型 (如 `claude-3-haiku-20240307`)                          |
| `--temperature`     | 控制生成随机性 (0.0~2.0)，默认: 0.7                                 |
| `--stream`          | 启用/禁用流式响应，默认: True                                      |
| `--knowledge-dir`   | 指定知识库目录实现 RAG 检索                                        |
| `--custom-caller`   | 自定义 LLM 调用模块路径 (需实现 `LLMCallerInterface`)               |
| `--custom-parser`   | 自定义响应解析模块路径 (需实现 `ResponseParserInterface`)           |
| `--log-level`       | 设置日志级别 (`DEBUG`/`INFO`/`WARNING`/`ERROR`)，默认: INFO         |

### 高级选项
| 选项                | 说明                                                                 |
|---------------------|--------------------------------------------------------------------|
| `--config-path`     | 指定自定义 EDN 配置文件路径                                         |
| `--auth-source`     | 认证源选择 (`env`/`command_line`/`authinfo`/`config`)               |
| `--auth-path`       | 指定认证文件路径 (配合 `--auth-source` 使用)                        |
| `--custom-params`   | 附加 API 参数 (JSON 格式)，如 `{"max_tokens": 2048}`               |
| `--fields`          | 提取响应字段 (逗号分隔)，如 `content,usage.total_tokens`           |

## 配置与认证

### 配置文件
- 默认路径：`source/mota/config/default.edn`
- 支持多级配置继承与合并
- 主要配置项：
  ```clojure
  :llm {
    :default_provider "openai"
    :temperature 0.7
    :stream true
    :providers {
      :groq {:model "deepseek-r1-distill-llama-70b"}
      :anthropic {:model "claude-3-haiku-20240307"}
    }
  }
  ```

### 认证管理
1. **环境变量**  
   设置 `GROQ_API_KEY` 或 `ANTHROPIC_API_KEY` 等环境变量

2. **加密存储**  
   使用 GPG 加密的认证文件 (~/.authinfo.gpg)：
   ```
   groq-api-key xxxxx
   anthropic-api-key xxxxx
   ```

3. **动态注入**  
   通过命令行临时指定：
   ```bash
   export ANTHROPIC_API_KEY=$(keyring get anthropic api-key)
   mota --provider anthropic ...
   ```

4. **混合模式**  
   自动按优先级选择认证源：环境变量 > 命令行 > 配置文件 > 系统密钥环

## 扩展开发

### 实现自定义 LLM 调用器
1. 创建新模块（如 `custom_llm.py`）
2. 实现 `LLMCallerInterface` 接口：
   ```python
   from mota.custom_interface import LLMCallerInterface

   class CustomLLMCaller(LLMCallerInterface):
       def call(self, provider, api_key, prompt, params):
           # 实现 API 调用逻辑
           return api_response
   ```

### 实现自定义响应解析器
1. 创建新模块（如 `custom_parser.py`）
2. 实现 `ResponseParserInterface` 接口：
   ```python
   from mota.custom_interface import ResponseParserInterface

   class CustomParser(ResponseParserInterface):
       def parse(self, response):
           # 实现响应解析逻辑
           return {
               "content": "...",
               "model": "...",
               "usage": {...}
           }
   ```

### 测试与调试
```bash
# 运行单元测试
pytest -v --cov=mota --cov-report=html

# 启动开发服务器
MOTA_DEV=1 mota --log-level DEBUG ...
```

## 许可证
本项目采用 GNU 通用公共许可证 v3.0 (GPL-3.0-or-later)，保留对代码进行商业使用的限制。
