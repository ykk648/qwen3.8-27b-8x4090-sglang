#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="8000", choices=["8000", "8001"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = read_env(root / f".env.{args.instance}")
    payload = json.dumps(
        {
            "model": config.get("MODEL_NAME", "qwen3.8-27b"),
            "input": "Reply with exactly RESPONSES_OK.",
            "max_output_tokens": 128,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{config['PORT']}/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {config['QWEN_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)

    output_types = [item.get("type") for item in result.get("output", [])]
    print(json.dumps({"status": result.get("status"), "output_types": output_types}))


if __name__ == "__main__":
    main()
