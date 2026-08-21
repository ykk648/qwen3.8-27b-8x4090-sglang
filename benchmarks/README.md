# Raw benchmark results

These JSONL files are the raw outputs behind the summarized tables in
`docs/EXPERIMENTS.zh-CN.md`.

- `sglang-main/`: SGLang main with MTP decode CUDA graph
- `sglang-stable/`: SGLang 0.5.16 baseline and eager MTP tests
- `vllm/`: vLLM TP, MTP, KV dtype, and context-length scans
- `vllm-patched/`: draft-vocabulary and sampler patch tests

Each short-prompt benchmark used concurrency 1, streaming output, a fixed
prompt, and three 512-token runs. Compare results only when model weights,
sampling settings, output length, and timing method are equivalent.
