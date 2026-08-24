# Qwen3.8-27B 部署与优化实验

本文记录官方 Qwen3.8-27B-FP8 权重在 8× RTX 4090 24GB、无 NVLink 环境中的
vLLM/SGLang 对比、MTP 优化、256K 上下文和 Codex 接入结果。

## 测试目标

- 单用户、并发 1
- 优先单请求 decode 速度
- 使用官方 FP8 权重
- 原生 uv 环境，不依赖 Docker 或 systemd
- 最终支持 256K 上下文和 Codex Responses API

## 测试口径

短请求测试使用固定 prompt，每次强制生成 512 tokens，开启流式返回，并发 1。每组
运行三次，以热请求中位数作为主要结果。输出速度从首个 token 到请求结束计算，不把
TTFT 计入 decode tok/s。

不同模型文件、prompt、采样参数、输出长度和计时方式的数字不能直接比较。

## 第一轮：原生 vLLM 与 SGLang

| Runtime | 配置 | 输出速度 |
|---|---|---:|
| vLLM | TP2 | 33.0 tok/s |
| vLLM | TP4 | 49.1 tok/s |
| vLLM | TP8 | 55.0 tok/s |
| vLLM | TP8 + MTP3 | 73.8 tok/s |
| SGLang 0.5.16 | TP4 | 57.36 tok/s |
| SGLang 0.5.16 | TP8 + EP2 | 62.34 tok/s |

TP1 在 24GB 卡上加载失败。约 29GB 的 FP8 模型文件并不等于 29GB 就能运行；还需要
KV cache、SSM state、激活、CUDA graph 和临时 workspace。

TP4 到 TP8 的提升很小，说明无 NVLink 的 PCIe tensor-parallel collective 已成为
明显瓶颈。

## SGLang 0.5.16 的 MTP 问题

稳定版在捕获 speculative decode CUDA graph 时出现 fixed buffer 长度错误。禁用
decode CUDA graph 后虽然可以运行，但 MTP 只有约 21 tok/s。MTP forward 更重，
没有 CUDA graph 时 kernel launch 和调度开销抵消了多 token 接受收益。

## 第二轮：SGLang main

升级到 commit `0e4a09480` 后，旧版 speculative mask buffer 错误消失。新版默认还会
尝试 prefill CUDA graph，在 24GB 卡上可能 OOM，因此最终配置：

- 禁用 prefill CUDA graph
- 保留 speculative decode CUDA graph
- 开启 MTP
- 并发限制为 1

| 配置 | 热 TTFT | 输出速度 |
|---|---:|---:|
| TP4 + EP1 + MTP | 0.204 s | **139.16 tok/s** |
| TP8 + EP2 + MTP | 0.203 s | 135.53 tok/s |

TP4 比 TP8 快约 2.6%。最终架构因此选择两路 TP4，而不是一路 TP8。

## vLLM 参数扫描

| 配置 | 输出速度 |
|---|---:|
| TP8 + MTP3 + FP8 KV + 131K | 82.45 tok/s |
| TP8 + MTP4 + FP8 KV + 131K | 92.98 tok/s |
| TP8 + MTP4 + BF16 KV + FlashAttention + 64K | 120.84 tok/s |
| TP4 + MTP4 + BF16 KV + FlashAttention + 64K | 115.52 tok/s |

BF16 KV + 64K 快速配置明显快于完整 131K FP8-KV 配置，但上下文容量不同，不能把
它看成无代价升级。

## draft vocabulary/sampler 补丁

从单卡定制项目的思路中移植了约 40K draft vocabulary 和 sampler 补丁：

| 测试 | 未打补丁 | 打补丁 | 提升 |
|---|---:|---:|---:|
| Greedy | 115.52 tok/s | 121.37 tok/s | 5.1% |
| T=1.0、top-p=0.95、top-k=20 | 90.24 tok/s | 93.89 tok/s | 4.0% |

补丁有效，但收益低于升级 SGLang main，因此没有作为最终默认部署。

## 为什么单卡 3090 定制项目也能达到 100+ tok/s

这类项目通常不是官方 FP8 权重直接启动，而是组合使用：

- W4A16/INT4 权重量化
- 单卡运行，消除 TP all-reduce
- 缩小 MTP draft vocabulary
- 定制 sampler
- attention 和 CUDA graph 补丁
- 定制 vLLM fork

主要收益来自避免多卡通信、减少权重带宽和提高 speculative decoding 接受率。代价是
使用非官方量化权重、更短默认上下文、复杂补丁以及潜在质量变化。

## 256K 第一次尝试为什么 OOM

模型原生声明 `max_position_embeddings=262144`，但第一次配置让 runtime 自动创建了
约 781,558-token pool，并设置：

```text
chunked_prefill_size=32768
mem_fraction_static=0.90
```

服务能够启动，但真实 258K prompt 在 GDN prefill 阶段申请额外 192MiB workspace
时 OOM。根因是 token pool 预留过大，几乎没有运行时余量。

最终稳定配置：

```text
CONTEXT_LENGTH=262144
MAX_TOTAL_TOKENS=300000
CHUNKED_PREFILL_SIZE=8192
MEM_FRACTION_STATIC=0.85
MAX_RUNNING_REQUESTS=1
```

真实请求结果：

| Prompt tokens | Output tokens | 总耗时 | Prefill 吞吐 |
|---:|---:|---:|---:|
| 258,000 | 32 | 187.08 s | 1,379.12 tok/s |

请求结束后，每张活动 GPU 约使用 14.5–15GB。低显存占用来自 FP8 权重 TP4 分片、
混合注意力架构和受控的 300K token pool，不代表完整模型可以单卡运行。

## Codex Responses API

SGLang main 暴露 `/v1/responses`，但必须启用：

```text
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
```

缺少 parser 时，模型会把 `<think>` 和 `<tool_call>` 作为普通文本返回。增加 parser
后，真实 Codex CLI 测试成功产生结构化 command execution、执行 shell 命令并把结果
返回模型，证明 Responses 流式事件和工具调用闭环可用。

## 历史方案：256K 双路 TP4

```text
Replica A: GPU 0-3, TP4, port 8000, 256K
Replica B: GPU 4-7, TP4, port 8001, 256K
```

当时选择双路 TP4 的原因：

1. 单流速度略高于 TP8。
2. 避免跨两组 PCIe/NUMA 域的通信。
3. 8 张 GPU 可以同时服务两个独立任务。
4. 每路都保留完整 256K 上下文和 Codex 工具能力。

## 结论

- 多卡数量增加不保证单流速度线性提升。
- MTP 的实际收益高度依赖 CUDA graph 和 draft token 接受率。
- 256K 必须通过真实长 prompt 验证，不能只看启动日志。
- 上下文容量、KV 精度和速度之间存在明确取舍。
- 自定义 runtime 补丁可以更快，但维护成本和模型一致性也必须计入选择。

## 当前部署：128K 单路 8 卡 + DFlash2（2026-08-24）

Codex 的真实长会话暴露出一个此前短请求基准没有反映的问题：随着实际历史变长，逐 token
输出速度会持续下降。256K 表示容量可达，不代表 128K 以上仍有可交互的 decode 速度。

当前生产配置改为单路 `:8001`，使用全部 8 张 GPU：

```text
TP_SIZE=8
EP_SIZE=2
CONTEXT_LENGTH=131072
MAX_TOTAL_TOKENS=150000
MEM_FRACTION_STATIC=0.90
MAX_RUNNING_REQUESTS=1
DISABLE_RADIX_CACHE=0
MAX_MAMBA_CACHE_SIZE=5
```

官方 FP8-MoE 权重不能使用 `TP8 + EP1`：expert 的分块尺寸不满足 FP8 block 对齐。可用的
8 卡拓扑是 `TP8 + EP2`，使每个 expert 的有效 TP 分片保持为 4。

### MTP 基线：真实长上下文 decode

固定输出 256 tokens、并发 1、MTP3 和 decode CUDA graph 开启：

| 实际 prompt tokens | TTFT | Decode 吞吐 |
|---:|---:|---:|
| 约 60,958 | 36.82 s | 26.92 tok/s |
| 约 111,528 | 67.24 s | 15.73 tok/s |

这说明 8 卡单路解决的是容量与缓存空间，不是长上下文 attention 的线性加速；无 NVLink 的
PCIe 通信也限制了 TP8 的收益。MTP 日志中的接受长度为 3.92–4.00、接受率 0.97–1.00，已
接近 MTP3 上限，因此继续增加同一模型的 MTP 参数预期收益很小。

### DFlash2：当前 Codex 优化方案

`z-lab/Qwen3.8-27B-DFlash2` 是 Qwen3.8-27B 的 5 层 block-diffusion 草稿模型，不是
独立聊天模型。它以 8-token verify block、2048-token 草稿滑动窗口和 FP8 草稿 KV 运行。
服务仍为 TP8 + EP2、128K、并发 1，MTP 不与 DFlash2 同时启用。

与 MTP 使用相同的固定输出 256 tokens 口径：

| 实际 prompt tokens | MTP decode | DFlash2 decode | DFlash2 TTFT |
|---:|---:|---:|---:|
| 约 60,958 | 26.92 tok/s | **57.54 tok/s** | 35.70 s |
| 约 111,528 | 15.73 tok/s | **34.22 tok/s** | 31.26 s |

在 112K 上下文，DFlash2 比 MTP 快 2.18 倍。日志的 DFlash2 接受长度为 7.25/8、接受率
0.89；这是真实 decode 加速，而不是只减少 prefill。

### 前缀缓存

启用 radix cache 并将 `MAX_MAMBA_CACHE_SIZE` 设为 5 后，服务可以稳定启动。对完全相同的
65,536-token 历史连续请求两次：

| 请求 | TTFT | Decode 吞吐 |
|---|---:|---:|
| 冷请求 | 38.55 s | 25.71 tok/s |
| 缓存命中 | 5.40 s | 25.70 tok/s |

MTP 下缓存的冷/热对比表明缓存机制可用。DFlash2 下，先后插入一个 112K 压测会挤掉较小的
61K 前缀；但对紧接着重复的 61K 历史，缓存命中 60,928 tokens，TTFT 为 **0.472s**，decode
为 **55.83 tok/s**。这就是 Codex 连续会话的目标场景：缓存负责消除重复历史 prefill，
DFlash2 负责提高活跃长上下文的生成速度。
