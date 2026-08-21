# Codex 接入 Qwen3.8-27B

当前 Codex 使用 OpenAI Responses API。本项目的 SGLang 服务原生提供
`/v1/responses`，并已启用 Qwen reasoning 和 tool-call parser。

## 1. 加载 API Key

在运行 Codex 的机器上设置：

```bash
export QWEN_API_KEY="your-generated-api-key"
```

如果 Codex 与服务运行在同一台机器，可以从本项目配置加载：

```bash
set -a
source .env.8000
set +a
```

不要把 `.env.8000` 或 API Key 提交到 Git。

## 2. Codex provider 配置

将以下内容加入 Codex 配置文件，并把地址替换为实际服务地址：

```toml
model = "qwen3.8-27b"
model_provider = "qwen-local"
model_context_window = 262144
model_auto_compact_token_limit = 240000

[model_providers.qwen-local]
name = "Qwen3.8-27B Local"
base_url = "http://127.0.0.1:8000/v1"
env_key = "QWEN_API_KEY"
wire_api = "responses"
```

如果 Codex 在另一台机器运行，请使用 VPN、SSH tunnel 或受保护的内网地址。不要把
无 TLS 的推理端口直接暴露到公网。

## 3. 临时命令行测试

```bash
codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -m qwen3.8-27b \
  -c 'model_provider="qwen-local"' \
  -c 'model_providers.qwen-local={ name="Qwen Local", base_url="http://127.0.0.1:8000/v1", env_key="QWEN_API_KEY", wire_api="responses" }' \
  -c model_context_window=262144 \
  -c model_auto_compact_token_limit=240000 \
  'Use the shell tool to execute pwd, then report the output.'
```

成功时，Codex 应产生结构化 command execution，而不是在最终文本中打印
`<tool_call>` XML。

## 4. 为什么使用 240K 自动压缩阈值

262,144 是模型上下文总窗口，不等于可以把 262,144 tokens 全部用于历史消息。
以下内容也会占用上下文：

- Codex system/developer instructions
- 工具 JSON schema
- 文件内容和命令输出
- 工具调用及返回值
- 模型最终回答

240K 阈值预留约 22K tokens，降低接近窗口上限时请求失败的概率。

## 5. 常见问题

### 模型输出 `<think>` 或 `<tool_call>`，但 Codex 不执行工具

确认 `serve.sh` 包含：

```text
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

修改后重启服务。

### Codex 提示找不到模型 metadata

自定义模型不在 Codex 内置 metadata 列表时，Codex 可能使用 fallback metadata。
显式设置 `model_context_window` 和 `model_auto_compact_token_limit`，并以实际工具调用
测试为准。

### 256K 请求很慢

256K 能力主要用于容量。接近窗口上限的 prefill 需要读取和处理数十万 tokens，首次
响应达到分钟级属于预期行为。短上下文 decode 速度与极限 prefill 吞吐是两个指标。
