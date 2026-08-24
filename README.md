# Qwen3.8-27B FP8 on 8× RTX 4090

在 8 张 RTX 4090 24GB 上，使用原生 `uv` 环境和 SGLang 部署单路
Qwen3.8-27B FP8 服务。服务使用全部 8 张 GPU、128K 上下文、DFlash2 speculative
decoding、前缀缓存、OpenAI Responses API 和 Codex 工具调用。

> Native `uv` deployment of Qwen3.8-27B FP8 on 8× RTX 4090. One TP8+EP2
> SGLang service with a 128K context window, DFlash2 acceleration, prefix caching,
> OpenAI Responses API, and Codex tool calling. No Docker or systemd required.

## 最终架构

```text
GPU 0,1,2,3,4,5,6,7 ── TP8 + EP2 ── :8001 ── 128K context
```

服务使用单一进程、CUDA context 和端口；API Key 保存在忽略的 `.env.8001` 中。

| 项目 | 配置 |
|---|---|
| GPU | 8× RTX 4090 24GB |
| 模型 | 官方 Qwen3.8-27B-FP8，约 29GB |
| 服务数量 | 1 |
| 并行方式 | TP8 + EP2 |
| 上下文 | 131,072 tokens |
| 推理加速 | DFlash2 + speculative verify CUDA graph + radix cache |
| API | Completions、Chat Completions、Responses |
| 运行环境 | Python 3.12 + uv |
| 容器/系统服务 | 不需要 |

## 实测结果

测试机器没有 NVLink，并且 8 张卡分布在两组 PCIe/NUMA 域中。

| 配置 | 约 61K 上下文 | 约 112K 上下文 |
|---|---:|---:|
| MTP 基线 | 26.92 tok/s | 15.73 tok/s |
| DFlash2（4 卡 TP4+EP1） | 56.97 tok/s | 30.27 tok/s |
| DFlash2（8 卡 TP8+EP2，最终） | **57.54 tok/s** | **34.22 tok/s** |

短请求的高 decode 速度不能代表 Codex 的长会话体验。当前默认将全部 8 张卡用于
单路 128K 服务，并以 DFlash2 提升真实长上下文 decode。对刚使用过的 61K 历史，
radix cache 命中后 TTFT 为 0.472s、decode 为 55.83 tok/s。

4 卡 DFlash2 在约 61K 时几乎追平 8 卡，但约 112K 时 8 卡快约 13%，因此默认保留
8 卡给 Codex 长会话。

对刚使用过的约 61K 历史，radix cache 命中后 TTFT 为 **0.472s**、decode 为
**55.83 tok/s**。完整验收记录见 [docs/EXPERIMENTS.zh-CN.md](docs/EXPERIMENTS.zh-CN.md)。

## 环境要求

- Linux x86_64
- 8× NVIDIA GPU，推荐每张至少 24GB
- 可工作的 NVIDIA 驱动和 CUDA runtime
- Python 3.12
- Git、curl、OpenSSL
- 模型文件存储空间约 35GB

本项目实测环境：

- Python 3.12.14
- PyTorch 2.13.0+cu130
- CUDA 13.0 runtime
- SGLang `0.5.6.post3.dev9122+g0e4a09480`
- SGLang commit `0e4a09480`

不同驱动、CUDA、SGLang commit 和 GPU 拓扑可能产生不同结果。

## 快速开始

### 1. 安装 uv 环境

```bash
git clone https://github.com/ykk648/qwen3.8-27b-8x4090-sglang.git
cd qwen3.8-27b-8x4090-sglang
./install.sh
```

`install.sh` 使用 `uv` 创建 `.venv`。Python 包默认通过阿里云 PyPI 镜像下载；
SGLang 固定到已验证的 Git commit。安装时设置 `SGLANG_BUILD_RUST_EXTS=none`，因此
不要求额外安装 Rust/Cargo。

安装脚本还会为固定的 SGLang commit 应用一个小型 Responses API 兼容补丁，使 Codex
code mode 的 `custom_tool_call` 和 `custom_tool_call_output` 能保留为 Qwen 可理解的
工具历史。上游 SGLang main 在本项目发布时尚未原生支持这两种输入项。

### 2. 下载官方 FP8 模型和 DFlash2 草稿模型

```bash
./download-model.sh
```

默认从 ModelScope 下载：

```text
Qwen/Qwen3.8-27B-FP8
```

默认保存到：

```text
./models/Qwen3.8-27B-FP8
```

也可以复用已有模型目录，不执行下载。

### 3. 生成单路 8 卡配置

使用默认模型目录：

```bash
./configure.sh
```

使用已有模型目录：

```bash
./configure.sh /data/models/Qwen3.8-27B-FP8
```

脚本会生成 `.env.8001`：GPU 0–7、`TP8 + EP2`、端口 8001、128K 上下文。该文件
被 `.gitignore` 排除，并保存自动生成的随机 API Key。不要把它提交到 Git 或发送到公开聊天中。

如果机器的 PCIe/NUMA 拓扑不同，请修改 `.env.8001` 中的 `CUDA_VISIBLE_DEVICES`。可以用
以下命令查看拓扑：

```bash
nvidia-smi topo -m
```

### 4. 启动服务

启动默认单路服务：

```bash
./start-all.sh
```

这只是普通的 `nohup` 命令行进程，不会安装 systemd 服务。

### 5. 等待并检查服务

```bash
tail -f logs/sglang-8001.log
./status.sh
./healthcheck.sh 8001
```

首次启动需要加载权重和捕获 MTP CUDA graph。出现 `/v1/models` 响应后即可使用。

### 6. 测试 Responses API

```bash
./test-responses.py --instance 8001
```

### 7. 停止服务

```bash
./stop-all.sh
```

或只停止一路：

```bash
./stop-instance.sh 8001
```

## API 调用

安全加载 API Key：

```bash
set -a
source .env.8000
set +a
```

列出模型：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $QWEN_API_KEY"
```

Responses API：

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $QWEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "input": "Write a Python binary search function.",
    "max_output_tokens": 512
  }'
```

不要把服务端口直接暴露到公网。跨机器使用时，建议放在防火墙、VPN、SSH tunnel 或
带 TLS 的反向代理后面。

## Codex 接入

SGLang main 原生提供 `/v1/responses`。本项目同时启用了：

```text
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

如果缺少这两个解析器，模型输出的 `<think>` 和 `<tool_call>` 可能会被当成普通文本，
Codex 将无法执行结构化工具调用。

完整配置和验证步骤见 [docs/CODEX.md](docs/CODEX.md)。建议设置：

```toml
model_context_window = 262144
model_auto_compact_token_limit = 240000
```

240K 自动压缩阈值为系统提示、工具定义、工具结果和最终输出保留约 22K token 空间。

## 性能测试

```bash
./benchmark.py --instance 8000 --runs 3 --output-tokens 512
```

测试脚本使用固定 prompt、`temperature=0`、流式返回和并发 1，并从首 token 后计算
decode tok/s。不同 prompt、采样参数、输出长度和计时口径不能直接横向比较。

## 当前 128K + DFlash2 参数

```text
TP_SIZE=8
EP_SIZE=2
CONTEXT_LENGTH=131072
MAX_TOTAL_TOKENS=150000
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.90
MAX_RUNNING_REQUESTS=1
DISABLE_RADIX_CACHE=0
MAX_MAMBA_CACHE_SIZE=5
SPECULATIVE_ALGORITHM=DFLASH
DFLASH_BLOCK_SIZE=8
DFLASH_DRAFT_KV_CACHE_DTYPE=fp8_e4m3
DFLASH_DRAFT_WINDOW_SIZE=2048
```

旧的 256K 双路与 MTP 方案仅保留为历史实验；当前默认配置以 Codex 长会话的实际响应速度为
目标。

## 已知限制

- 为单用户、低并发、单流 decode 优化，不是高并发网关配置。
- `MAX_RUNNING_REQUESTS=1` 会限制每路并发。
- 为适配 24GB 卡，关闭了 prefill CUDA graph，保留 DFlash2 target/draft verify CUDA graph。
- 默认开启 radix cache。DFlash2 下连续的约 61K 历史命中 60,928 个 token，TTFT 为
  0.472s；它复用未变化的历史前缀，DFlash2 则负责提高长上下文的逐 token decode。
- 4090 没有 NVLink；GPU 分组应根据实际 PCIe/NUMA 拓扑调整。
- SGLang 使用开发 commit，未来升级前应重新跑 256K 和 Codex 工具调用测试。

## 项目边界

本仓库不包含模型权重，也不修改模型许可证。下载和使用模型前，请阅读对应模型页面的
许可证与使用条款。本项目只提供部署脚本、配置和实验记录。

## License

MIT
