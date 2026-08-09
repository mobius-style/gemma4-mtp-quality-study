#!/usr/bin/env python3
"""Run the triggered exploratory p-min sweep at preselected higher depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

from run_performance import PROTOCOL, STUDY, Server


DEPTHS = {"12b": (6, 12), "26b": (6, 8)}
PMINS = (0.25, 0.50, 0.75, 0.90)


def append(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("12b", "26b"), required=True)
    parser.add_argument("--port", type=int, default=18141)
    args = parser.parse_args()
    prompts = json.loads((STUDY / f"prompts/performance_{args.model}.json").read_text())["prompts"]
    raw_path = STUDY / f"raw/pmin/{args.model}.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok":
                done.add(row["run_key"])
    configs = [(depth, pmin) for depth in DEPTHS[args.model] for pmin in PMINS]
    random.Random(20260809 + (1212 if args.model == "12b" else 2626)).shuffle(configs)
    for depth, pmin in configs:
        label = f"n{depth}_p{str(pmin).replace('.', '')}"
        if all(f"{args.model}:{label}:{prompt['prompt_id']}" in done for prompt in prompts):
            continue
        server = Server(args.model, depth, args.port, p_min=pmin)
        log_path = STUDY / f"logs/pmin/{args.model}_{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path = log_path.with_name(f"{log_path.stem}_resume_{int(time.time())}.log")
        try:
            server.start(log_path)
            server.completion("Warm-up request. Continue with numbered diagnostic notes.\n1.", 1, 64)
            order = prompts[:]
            random.Random(20260809 + depth + int(100 * pmin)).shuffle(order)
            for prompt in order:
                key = f"{args.model}:{label}:{prompt['prompt_id']}"
                if key in done:
                    continue
                base = {
                    "run_key": key, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model_short": args.model, "condition": label, "n_max": depth, "p_min": pmin,
                    "prompt_id": prompt["prompt_id"], "prompt_sha256": prompt["sha256"],
                    "length_class": prompt["length_class"], "input_tokens_expected": prompt["input_tokens"],
                    "requested_output_tokens": 256, "seed": prompt["seed"], "temperature": 0.0,
                    "top_k": 1, "command": server.command(), "log": str(log_path),
                }
                try:
                    result = server.completion(prompt["content"], prompt["seed"])
                    result.update(base)
                    result["status"] = "ok" if result["output"] else "invalid_empty_output"
                    proposed = result.get("draft_tokens_proposed") or 0
                    accepted = result.get("draft_tokens_accepted") or 0
                    result["acceptance_rate"] = accepted / proposed if proposed else None
                except Exception as exc:
                    result = dict(base, status="failed_request", error=repr(exc))
                append(raw_path, result)
                if result["status"] == "ok":
                    done.add(key)
                print(f"{key} {result['status']} {result.get('decode_tok_s')}", flush=True)
        except Exception as exc:
            append(raw_path, {"run_key": f"{args.model}:{label}:server:{int(time.time())}",
                              "model_short": args.model, "condition": label, "status": "failed_server",
                              "error": repr(exc), "command": server.command(), "log": str(log_path)})
        finally:
            print(f"stopped {args.model} {label} exit={server.stop()}", flush=True)


if __name__ == "__main__":
    main()
