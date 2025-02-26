# Mota

Mota 是一个用于与各种主要的大语言模型（LLM）API服务交互的综合工具。它支持多个 LLM 提供商，包括 OpenAI、Anthropic、Gemini、GROQ、GROK、DeepSeek、Mistral、OpenRouter 等，提供统一的接口进行配置管理、认证处理和自定义 API 交互。

## 特性

- **支持多个 LLM 提供商**：包括 OpenAI、Anthropic、Gemini、GROQ、GROK、DeepSeek、Mistral、OpenRouter 等。
- **命令行界面**：使用 `typer` 和 `typing` 包处理命令行参数，提供丰富的帮助选项和默认值。
- **日志记录**：通过 `logging` 包处理日志，支持配置日志级别和输出目的地。
- **配置文件**：支持使用 EDN 格式的配置文件，通过 `edn_format` 包进行处理，配置文件中定义了程序的默认参数。
- **多种认证方式**：支持从环境变量、命令行参数、EMACS `.authinfo` 文件以及配置文件中读取各服务的 API 密钥。
- **自定义聊天参数**：除了提供基本的默认参数外，用户还可以自定义聊天请求的参数，如温度、流模式等。
- **健壮的异常处理**：程序具备健壮的异常处理机制，确保在遇到错误时能够提供有用的日志信息。
- **作为库使用**：精心设计的公开函数，使得 Mota 可以作为 Python 库集成到其他项目中。
- **调试支持**：在调试模式下，程序能够输出详细的响应报文和 HTTP 状态码等信息。
- **全面的测试用例**：为关键函数和所有公开函数提供了测试用例，确保代码的可靠性和稳定性。

## 安装

```bash
pip install mota
```

## 使用

```bash
mota chat --provider openai --model gpt-4 "Hello, how are you?"
```

### 命令行选项

- `--provider`: 选择 LLM 提供商（默认：openai）
- `--model`: 指定模型名称
- `--temperature`: 温度参数（默认：0.7）
- `--stream`: 启用流模式（默认：True）
- `--config-path`: 配置文件路径
- `--log-level`: 日志级别（默认：INFO）
- `--log-output`: 日志输出目标（默认：stdout）
- `--custom-params`: 自定义聊天请求参数，使用 JSON 格式
- `--fields`: 需要提取的响应字段，使用逗号分隔

## 配置

Mota 使用 EDN 格式的配置文件，默认路径为 `source/mota/config/default.edn`。您可以通过 `--config-path` 参数指定自定义配置文件路径。

## 认证

Mota 支持多种认证方式来获取各服务的 API 密钥：

1. **环境变量**：通过设置环境变量，例如 `OPENAI_API_KEY`。
2. **命令行参数**：通过命令行参数传入 API 密钥。
3. **EMACS `.authinfo` 文件**：在家目录下的 `.authinfo` 或 `.authinfo.gpg` 文件中管理 API 密钥。
4. **配置文件**：在 EDN 配置文件中指定 API 密钥。

## 开发

### 测试

```bash
pytest
```

### 贡献

欢迎贡献代码！请提交 PR 或 issue 与我们讨论。

## 许可证

本项目采用 GNU 通用公共许可证 v3.0。
