#!/usr/bin/env python3
"""Run the fixed 200-item objective quality/equivalence pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from run_performance import Server, STUDY


SELECTION = json.loads((STUDY / "processed/condition_selection.json").read_text())
ITEMS = [json.loads(line) for line in (STUDY / "prompts/quality_items_v3.jsonl").read_text().splitlines()]


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
            if item.get("status") == "ok":
                result.add(item["run_key"])
        except json.JSONDecodeError:
            pass
    return result


def chat(server: Server, item: dict, seed: int) -> dict:
    payload = {
        "model": "local-gemma4", "messages": [
            {"role": "system", "content": "Follow the user's request exactly. Provide the requested final format."},
            {"role": "user", "content": item["prompt"]},
        ],
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": seed,
        "max_tokens": item["max_tokens"], "stream": False,
        "reasoning_budget_tokens": 0,
    }
    if item.get("family") == "mmlu_pro":
        payload["grammar"] = 'root ::= "Final answer: " [A-J]'
    start = time.monotonic_ns()
    response = server.post("/v1/chat/completions", payload, timeout=600)
    end = time.monotonic_ns()
    output = response["choices"][0]["message"].get("content") or ""
    return {
        "output": output, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_token_ids": server.tokenize(output), "wall_ms": (end - start) / 1e6,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "usage": response.get("usage"), "response_id": response.get("id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("12b", "26b"), required=True)
    parser.add_argument("--conditions", help="comma list: off_a,off_b,nN")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-per-family", action="store_true")
    parser.add_argument("--item-id")
    parser.add_argument("--port", type=int, default=18131)
    args = parser.parse_args()

    selected = SELECTION[args.model]["quality_conditions"]
    allowed = ["off_a", "off_b", *selected]
    conditions = args.conditions.split(",") if args.conditions else allowed
    if not set(conditions) <= set(allowed):
        raise ValueError(f"conditions must be within {allowed}")
    order = allowed[:]
    random.Random(20260809 + (120 if args.model == "12b" else 260)).shuffle(order)
    order_path = STUDY / f"raw/quality/{args.model}_condition_order.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    if not order_path.exists():
        order_path.write_text(json.dumps({"model": args.model, "order": order}, indent=2) + "\n")
    else:
        order = json.loads(order_path.read_text())["order"]
    order = [condition for condition in order if condition in conditions]

    raw_path = STUDY / f"raw/quality/{args.model}_v6.jsonl"
    done = completed(raw_path)
    if args.item_id:
        items = [item for item in ITEMS if item["item_id"] == args.item_id]
        if not items:
            raise ValueError(f"unknown item id: {args.item_id}")
    elif args.smoke_per_family:
        items = [next(item for item in ITEMS if item["family"] == family)
                 for family in ("gsm8k", "mmlu_pro", "ifeval", "mbpp")]
    else:
        items = ITEMS[:args.limit] if args.limit else ITEMS
    for condition in order:
        n_max = 0 if condition.startswith("off_") else int(condition[1:])
        server = Server(args.model, n_max, args.port, reasoning_format="deepseek")
        log_path = STUDY / f"logs/quality/{args.model}_{condition}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path = log_path.with_name(f"{log_path.stem}_resume_{int(time.time())}.log")
        try:
            server.start(log_path)
            chat(server, {"prompt": "Reply with exactly OK", "max_tokens": 16}, seed=1)
            item_order = items[:]
            random.Random(20260809 + sum(map(ord, condition))).shuffle(item_order)
            for item in item_order:
                key = f"{args.model}:{condition}:{item['item_id']}"
                if key in done:
                    continue
                base = {
                    "run_key": key, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model_short": args.model, "condition": condition, "n_max": n_max,
                    "item_id": item["item_id"], "family": item["family"],
                    "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                    "seed": 20260809 + int(hashlib.sha256(item["item_id"].encode()).hexdigest()[:8], 16),
                    "temperature": 0.0, "top_k": 1, "context": 8192,
                    "max_tokens": item["max_tokens"], "log": str(log_path), "command": server.command(),
                }
                try:
                    result = chat(server, item, base["seed"])
                    result.update(base)
                    result["status"] = "ok" if result["output"] else "invalid_empty_output"
                except Exception as exc:
                    result = dict(base, status="failed_request", error=repr(exc))
                append_jsonl(raw_path, result)
                if result["status"] == "ok":
                    done.add(key)
                print(f"{key} {result['status']} {result.get('wall_ms')}", flush=True)
        except Exception as exc:
            append_jsonl(raw_path, {
                "run_key": f"{args.model}:{condition}:server:{int(time.time())}",
                "model_short": args.model, "condition": condition, "status": "failed_server",
                "error": repr(exc), "log": str(log_path), "command": server.command(),
            })
            print(f"FAILED {args.model} {condition}: {exc!r}", flush=True)
        finally:
            print(f"stopped {args.model} {condition} exit={server.stop()}", flush=True)


if __name__ == "__main__":
    main()
