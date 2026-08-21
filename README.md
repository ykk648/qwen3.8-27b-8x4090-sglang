# Qwen3.8-27B FP8 on 8× RTX 4090

在 8 张 RTX 4090 24GB 上，使用原生 `uv` 环境和 SGLang 部署两路独立的
Qwen3.8-27B FP8 服务。每路使用 4 张 GPU、支持 256K 上下文、MTP speculative
decoding、OpenAI Responses API 和 Codex 工具调用。

> Native `uv` deployment of Qwen3.8-27B FP8 on 8× RTX 4090. Two independent
> TP4 SGLang replicas, 256K context per replica, MTP acceleration, OpenAI
> Responses API, and Codex tool calling. No Docker or systemd required.

## 最终架构

```text
GPU 0,1,2,3 ── TP4 replica A ── :8000 ── 256K context
GPU 4,5,6,7 ── TP4 replica B ── :8001 ── 256K context
```

两路服务共享同一份模型文件，但拥有独立进程、CUDA context、KV cache 和端口。
默认使用相同 API Key，也可以手动设置为不同密钥。

| 项目 | 配置 |
|---|---|
| GPU | 8× RTX 4090 24GB |
| 模型 | 官方 Qwen3.8-27B-FP8，约 29GB |
| 服务数量 | 2 |
| 每路并行方式 | TP4 + EP1 |
| 每路上下文 | 262,144 tokens |
| 推理加速 | MTP + speculative decode CUDA graph |
| API | Completions、Chat Completions、Responses |
| 运行环境 | Python 3.12 + uv |
| 容器/系统服务 | 不需要 |

## 实测结果

测试机器没有 NVLink，并且 8 张卡分布在两组 PCIe/NUMA 域中。

| 配置 | 热 TTFT | 短请求输出速度 |
|---|---:|---:|
| SGLang main，TP4 + MTP | 0.204 s | **139.16 tok/s** |
| SGLang main，TP8 + MTP | 0.203 s | 135.53 tok/s |

TP4 在单用户、并发 1 的测试中比 TP8 快约 2.6%，因此最终把 8 张 GPU 拆成两路
TP4，而不是启动一路 TP8。

256K 不是仅验证启动：真实的 258,000-token prompt 加 32-token 输出成功完成。

| Prompt | Output | 总耗时 | Prefill 吞吐 |
|---:|---:|---:|---:|
| 258,000 tokens | 32 tokens | 187.08 s | 1,379.12 tok/s |

短请求的 139.16 tok/s 是 decode 速度；接近 256K 的首次 prefill 约需三分钟，两者
不能混为同一个指标。完整实验记录见 [docs/EXPERIMENTS.zh-CN.md](docs/EXPERIMENTS.zh-CN.md)。

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

### 2. 下载官方 FP8 模型

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

### 3. 生成双路配置

使用默认模型目录：

```bash
./configure.sh
```

使用已有模型目录：

```bash
./configure.sh /data/models/Qwen3.8-27B-FP8
```

脚本会生成：

- `.env.8000`：GPU 0–3，端口 8000
- `.env.8001`：GPU 4–7，端口 8001

两个文件均被 `.gitignore` 排除，并保存自动生成的随机 API Key。不要把它们提交到
Git 或发送到公开聊天中。

如果机器的 PCIe/NUMA 拓扑不同，请修改两个文件中的 `CUDA_VISIBLE_DEVICES`。可以用
以下命令查看拓扑：

```bash
nvidia-smi topo -m
```

### 4. 启动服务

启动两路：

```bash
./start-all.sh
```

只启动一路：

```bash
./start-instance.sh 8000
./start-instance.sh 8001
```

这只是普通的 `nohup` 命令行进程，不会安装 systemd 服务。

### 5. 等待并检查服务

```bash
tail -f logs/sglang-8000.log
./status.sh
./healthcheck.sh 8000
./healthcheck.sh 8001
```

首次启动需要加载权重和捕获 MTP CUDA graph。出现 `/v1/models` 响应后即可使用。

### 6. 测试 Responses API

```bash
./test-responses.py --instance 8000
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

## 关键 256K 参数

```text
CONTEXT_LENGTH=262144
MAX_TOTAL_TOKENS=300000
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.85
MAX_RUNNING_REQUESTS=1
```

第一次实验让 SGLang 自动创建约 781K token pool，并使用 32768 的 prefill chunk，
服务可以启动，但真实 258K prefill 因缺少额外 workspace 而 OOM。限制 token pool、
降低 chunk size 和静态显存比例后，258K 请求稳定完成。

## 已知限制

- 为单用户、低并发、单流 decode 优化，不是高并发网关配置。
- `MAX_RUNNING_REQUESTS=1` 会限制每路并发。
- 为适配 24GB 卡，关闭了 prefill CUDA graph，但保留 MTP decode CUDA graph。
- 默认关闭 radix cache，避免长上下文缓存占用额外显存。
- 4090 没有 NVLink；GPU 分组应根据实际 PCIe/NUMA 拓扑调整。
- SGLang 使用开发 commit，未来升级前应重新跑 256K 和 Codex 工具调用测试。

## 项目边界

本仓库不包含模型权重，也不修改模型许可证。下载和使用模型前，请阅读对应模型页面的
许可证与使用条款。本项目只提供部署脚本、配置和实验记录。

## License

MIT
