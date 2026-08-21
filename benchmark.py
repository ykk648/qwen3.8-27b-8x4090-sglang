#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


PROMPT = "Explain how a B-tree database index works, including insertion, page splits, and lookup complexity."


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def run_once(base_url: str, api_key: str, model: str, output_tokens: int) -> dict[str, float]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": PROMPT,
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_token_at = None
    completion_tokens = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("choices") and event["choices"][0].get("text"):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("The server returned no output tokens")
    decode_time = finished - first_token_at
    return {
        "ttft_s": first_token_at - started,
        "output_tps": completion_tokens / decode_time,
        "completion_tokens": float(completion_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="8000", choices=["8000", "8001"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = read_env(root / f".env.{args.instance}")
    base_url = f"http://127.0.0.1:{config['PORT']}"
    results = [
        run_once(base_url, config["QWEN_API_KEY"], config.get("MODEL_NAME", "qwen3.8-27b"), args.output_tokens)
        for _ in range(args.runs)
    ]
    print(
        json.dumps(
            {
                "runs": args.runs,
                "median_ttft_s": statistics.median(item["ttft_s"] for item in results),
                "median_output_tps": statistics.median(item["output_tps"] for item in results),
                "mean_output_tps": statistics.mean(item["output_tps"] for item in results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
