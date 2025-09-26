# Mota

Mota is a comprehensive tool for interacting with various major Large Language Model (LLM) API services. It supports multiple LLM providers, including OpenAI, Anthropic, GROQ, and others, through a plugin-based architecture. It also provides unified configuration management, authentication handling, and an extensible API interaction interface.

## Features

- **Plugin-Based Architecture**: Extends LLM services by dynamically loading custom modules.
  - Supports custom LLM Callers and Response Parsers.
  - Provides standard interface protocols (LLMCallerInterface/ResponseParserInterface).
  - Includes complete implementation examples for vendors like Anthropic and GROQ.

- **Retrieval-Augmented Generation (RAG)**:
  - Integrates LangChain for document loading and vector retrieval.
  - Supports multiple document formats such as DOCX, PDF, and TXT.
  - Implements efficient semantic search based on FAISS.

- **Unified Configuration Management**:
  - Manages all parameters uniformly using the EDN format.
  - Supports multi-level configuration inheritance and overriding.
  - Provides a default configuration file (config/default.edn).

- **Multi-Source Authentication Management**:
  - Supports environment variables, command-line arguments, and .authinfo files.
  - Provides GPG-encrypted credential protection.
  - Automatically selects the optimal authentication source.

- **Enterprise-Grade Features**:
  - Dual-mode response handling (streaming/non-streaming).
  - Intelligent parameter conversion and validation.

- **Developer-Friendly**:
  - Detailed logging and debugging information.
  - Well-typed Python API with complete type annotations.
  - 100% test coverage for core modules.
  - Modular design of functional components.

## Installation

### Installation via PyPI
```bash
pip install mota
```

### Installation from Source (Development Mode)
```bash
git clone https://codeberg.org/WIZARDELF/mota.git
cd mota
pip install -e .
```

## Usage

### Basic Example
```bash
# Using GROQ
mota --log-level 'DEBUG' --provider=groq --custom-caller=source/mota/custom_groq.py --custom-parser=source/mota/custom_groq.py --knowledge-dir test/fixture/knowledge --prompt "Unparalleled tech master, hello! I need your help." "Explain quantum mechanics."
```

### Python API Usage Example
```python
from mota import seek

response = seek(
    provider="groq",
    custom_caller="source/mota/custom_groq.py",
    custom_parser="source/mota/custom_groq.py",
    model="deepseek-r1-distill-llama-70b",
    prompt="Unparalleled tech master, hello! I need your help.",
    message="Explain the Transformer architecture in detail.",
    fields=["content", "usage"]
)

print(">>> ", response)
```

### Seek API Parameter Description

| Parameter        | Type                  | Default    | Description                                                                                             |
|------------------|-----------------------|------------|---------------------------------------------------------------------------------------------------------|
| provider         | str                   | "openai"   | Supported LLM providers: openai/anthropic/groq/mistral/deepseek/openrouter                                |
| model            | Optional[str]         | None       | When None, the default model from the configuration file is used automatically.                         |
| stream           | bool                  | True       | Streaming response mode. It is recommended to disable it in the CLI and enable it in the API.           |
| knowledge_dir    | Optional[str]         | None       | When RAG is enabled, this must point to a directory containing files like .txt, .pdf, .docx, etc.       |
| fields           | Optional[List[str]]   | None       | Supports nested field extraction, e.g., ["content", "usage.total_tokens"].                               |
| custom_caller    | Optional[str]         | None       | Format: "/path/to/module.py:ClassName"                                                                  |
| user_query       | Optional[List[str]]   | None       | Supports appending multiple query parameters, which are automatically concatenated to the main message. |

### Asynchronous Calls and Error Handling
```python
import asyncio
from mota import seek
from mota.core import LLMError

# Asynchronous call example
async def async_seek():
    response = await seek(
        provider="groq",
        message="The advantages of asynchronous programming",
        stream=False,
        async_mode=True
    )
    print(response['content'])

asyncio.run(async_seek())

# Error handling example
try:
    response = seek(provider="openai", model="gpt-5")  # Non-existent model
except LLMError as e:
    print(f"API Error Code: {e.code}")
    print(f"Error Details: {e.details}")
except Exception as e:
    print(f"System Error: {str(e)}")
```

### Advanced Feature Examples
```bash
# Using GROQ and loading a custom implementation (for development and debugging)
mota --log-level DEBUG \
  --provider groq \
  --custom-caller source/mota/custom_groq.py \
  --custom-parser source/mota/custom_groq.py \
  --knowledge-dir ./knowledge_base \
  --prompt "You are a quantum physics expert, please answer in English:" \
  "Please explain the Schrödinger equation in detail"

# Using Anthropic with RAG retrieval
mota --provider anthropic \
  --custom-caller source/mota/custom_anthropic.py \
  --knowledge-dir ./tech_docs \
  --temperature 0.3 \
  "How to achieve consistency in a distributed system?"
```

### Enterprise-Level Deployment
```bash
# Using encrypted authentication information (requires GPG pre-configuration)
mota --provider groq \
  --auth-source authinfo_gpg \
  --auth-path ~/.authinfo.gpg \
  --model deepseek-r1-distill-llama-70b \
  "Analyze the following financial statement: <attach financial data>"
```

### Core Command-Line Options

| Option              | Description                                                                      |
|---------------------|----------------------------------------------------------------------------------|
| `--provider`        | Specify the LLM provider (`groq`, `anthropic`, `openai`, etc.). Default: openai.   |
| `--model`           | Select a specific model (e.g., `claude-3-haiku-20240307`).                         |
| `--temperature`     | Control generation randomness (0.0-2.0). Default: 0.7.                           |
| `--stream`          | Enable/disable streaming response. Default: True.                                |
| `--knowledge-dir`   | Specify the knowledge base directory for RAG retrieval.                          |
| `--custom-caller`   | Path to the custom LLM caller module (must implement `LLMCallerInterface`).      |
| `--custom-parser`   | Path to the custom response parser module (must implement `ResponseParserInterface`). |
| `--log-level`       | Set the log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Default: INFO.               |

### Advanced Options
| Option              | Description                                                                      |
|---------------------|----------------------------------------------------------------------------------|
| `--config-path`     | Specify the path to a custom EDN configuration file.                             |
| `--auth-source`     | Select the authentication source (`env`/`command_line`/`authinfo`/`config`).     |
| `--auth-path`       | Specify the authentication file path (used with `--auth-source`).                |
| `--custom-params`   | Additional API parameters (JSON format), e.g., `{"max_tokens": 2048}`.            |
| `--fields`          | Extract response fields (comma-separated), e.g., `content,usage.total_tokens`.   |

## Configuration and Authentication

### Configuration File
- Default path: `source/mota/config/default.edn`
- Supports multi-level configuration inheritance and merging.
- Main configuration items:
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

### Authentication Management
1. **Environment Variables**  
   Set environment variables like `GROQ_API_KEY` or `ANTHROPIC_API_KEY`.

2. **Encrypted Storage**  
   Use a GPG-encrypted authentication file (~/.authinfo.gpg):
   ```
   groq-api-key xxxxx
   anthropic-api-key xxxxx
   ```

3. **Dynamic Injection**  
   Specify temporarily via the command line:
   ```bash
   export ANTHROPIC_API_KEY=$(keyring get anthropic api-key)
   mota --provider anthropic ...
   ```

4. **Hybrid Mode**  
   Automatically selects the authentication source by priority: Environment Variables > Command Line > Configuration File > System Keyring.

## Extension Development

### Implementing a Custom LLM Caller
1. Create a new module (e.g., `custom_llm.py`).
2. Implement the `LLMCallerInterface` interface:
   ```python
   from mota.custom_interface import LLMCallerInterface

   class CustomLLMCaller(LLMCallerInterface):
       def call(self, provider, api_key, prompt, params):
           # Implement API call logic
           return api_response
   ```

### Implementing a Custom Response Parser
1. Create a new module (e.g., `custom_parser.py`).
2. Implement the `ResponseParserInterface` interface:
   ```python
   from mota.custom_interface import ResponseParserInterface

   class CustomParser(ResponseParserInterface):
       def parse(self, response):
           # Implement response parsing logic
           return {
               "content": "...",
               "model": "...",
               "usage": {...}
           }
   ```

### Testing and Debugging
```bash
# Run unit tests
pytest -v --cov=mota --cov-report=html

# Start development server
MOTA_DEV=1 mota --log-level DEBUG ...
```

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0-or-later), with restrictions on commercial use of the code.
