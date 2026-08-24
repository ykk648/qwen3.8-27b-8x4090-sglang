# Qwen3.8-27B：Codex 最终部署验收

本文只记录当前推荐的 Codex 部署方案及其验收数据。测试机器为 8× RTX 4090 24GB、无
NVLink；目标是单用户在长会话中保持可交互速度，而不是追求短 prompt 的峰值数字。

## 最终方案

```text
GPU 0,1,2,3,4,5,6,7
        └── TP8 + EP2 ── SGLang :8001 ── Qwen3.8-27B-FP8
                              ├── DFlash2 speculative decoding
                              └── radix prefix cache
```

| 项目 | 最终配置 |
|---|---|
| 服务端口 | `8001` |
| 模型 | 官方 `Qwen3.8-27B-FP8` |
| 草稿模型 | `z-lab/Qwen3.8-27B-DFlash2` |
| GPU 并行 | `TP8 + EP2` |
| 最大上下文 | 131,072 tokens |
| 并发 | 1 |
| DFlash2 | 8-token verify block、2048-token draft window、FP8 draft KV |
| 缓存 | radix cache 开启，`MAX_MAMBA_CACHE_SIZE=5` |
| API | Completions、Chat Completions、Responses、Codex 工具历史兼容 |

`TP8 + EP1` 不能加载官方 FP8-MoE 权重，因为 expert 分片不满足 FP8 block 对齐；因此 8
卡的正确拓扑是 `TP8 + EP2`。

## Codex 长会话验收

固定输出 256 tokens、并发 1、流式返回。decode 吞吐从首 token 到请求结束计，不把 TTFT
计入 tok/s。

| 实际 prompt tokens | MTP 基线 | DFlash2 4 卡 | DFlash2 8 卡（最终） |
|---:|---:|---:|---:|
| 约 60,958 | 26.92 tok/s | 56.97 tok/s | **57.54 tok/s** |
| 约 111,528 | 15.73 tok/s | 30.27 tok/s | **34.22 tok/s** |

结论：DFlash2 使约 112K 活跃上下文的 decode 从 15.73 提升到 **34.22 tok/s**，比 MTP
快 2.18 倍。4 卡在约 61K 时几乎追平 8 卡，但在约 112K 时慢约 13%；为了 Codex 的长会话
体验，最终保留全部 8 张卡。

## 多轮历史缓存验收

连续对相同的约 61K 历史发起请求时，radix cache 命中 60,928 个 token：

| 场景 | TTFT | Decode 吞吐 |
|---|---:|---:|
| 首次读取历史（DFlash2） | 35.70 s | 57.54 tok/s |
| 紧接着复用同一历史 | **0.472 s** | 55.83 tok/s |

缓存负责消除 Codex 多轮请求中未变化历史的重复 prefill；DFlash2 负责提高已经处于长上下文时的
逐 token 生成速度。二者缺一不可。

## Codex 兼容性

- `/v1/responses` 已通过实际冒烟测试，返回 `reasoning` 与 `message` 输出。
- 服务保留 `qwen3` reasoning parser、`qwen3_coder` tool-call parser，以及本项目的
  `custom_tool_call` / `custom_tool_call_output` 兼容补丁。
- API Key 只保存在被忽略的 `.env.8001`，不写入仓库或本文档。

## 使用边界

- 本方案针对单用户、低并发；`MAX_RUNNING_REQUESTS=1` 是有意设置。
- 128K 是可用容量，不代表任意 128K 活跃历史都能达到短请求速度；实测约 112K 时为
  34.22 tok/s。
- 草稿 CUDA graph 的首次捕获约需数分钟，服务常驻后不影响日常使用。

## 历史结论（一笔带过）

此前尝试过双路 TP4、256K、vLLM、MTP 和若干 sampler 补丁。它们证明模型可以运行，但真实
Codex 长历史下的 decode 仍会明显下降，因此不再作为默认部署。当前以 128K、单路 8 卡、
DFlash2 和 radix cache 为唯一推荐方案。
