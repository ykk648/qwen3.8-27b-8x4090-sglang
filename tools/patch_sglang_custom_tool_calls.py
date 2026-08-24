#!/usr/bin/env python3
"""Add Codex custom-tool history support to the pinned SGLang Responses API."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MARKER = "# Codex custom-tool compatibility patch."
ANCHOR = '        if msg_type == "function_call_output":\n'
PATCH = """        # Codex custom-tool compatibility patch.
        # Codex code mode records tools as free-form custom calls rather than
        # JSON-schema function calls. Qwen's chat template only has a function
        # call representation, so retain the raw custom input in one `input`
        # argument and map its output back to a tool message.
        if msg_type == "custom_tool_call":
            raw_input = message.get("input", "")
            if not isinstance(raw_input, str):
                raw_input = orjson.dumps(raw_input).decode("utf-8")
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": orjson.dumps({"input": raw_input}).decode(
                                "utf-8"
                            ),
                        },
                    }
                ],
            }
        if msg_type == "custom_tool_call_output":
            out = message.get("output", "")
            if isinstance(out, list):
                out = "".join(
                    part.get("text", "") for part in out if isinstance(part, dict)
                )
            elif not isinstance(out, str):
                out = orjson.dumps(out).decode("utf-8")
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id"),
                "content": out,
            }
"""


def serving_responses_path() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("SGLang is not installed in the active Python environment")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    path = package_root / "srt/entrypoints/openai/serving_responses.py"
    if not path.is_file():
        raise RuntimeError(f"SGLang Responses implementation not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    path = serving_responses_path()
    content = path.read_text()
    is_patched = MARKER in content

    if args.check:
        if not is_patched:
            raise SystemExit(f"missing patch: {path}")
        print(f"patch present: {path}")
        return

    if is_patched:
        print(f"patch already present: {path}")
        return
    if ANCHOR not in content:
        raise SystemExit(f"unsupported SGLang source layout: {path}")

    path.write_text(content.replace(ANCHOR, PATCH + ANCHOR, 1))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
