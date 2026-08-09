#!/usr/bin/env python3
"""Run a 20-item MMLU-Pro option-order perturbation diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import pandas as pd

from run_performance import STUDY, Server
from run_quality import chat

LETTERS = "ABCDEFGHIJ"


def build_items() -> list[dict]:
    path = STUDY / "prompts/mmlu_option_perturbation.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text().splitlines()]
    originals = [json.loads(line) for line in (STUDY / "prompts/quality_items_v3.jsonl").read_text().splitlines()
                 if json.loads(line)["family"] == "mmlu_pro"][:20]
    frame = pd.read_parquet(STUDY / "datasets/source/mmlu_pro_test.parquet")
    by_id = {int(row.question_id): row for row in frame.itertuples()}
    output = []
    for index, original in enumerate(originals):
        row = by_id[original["source_index"]]
        offset = 1 + (index * 7) % 9
        options = list(original["options"])
        rotated = options[offset:] + options[:offset]
        old_answer = int(original["answer_index"])
        new_answer = (old_answer - offset) % len(options)
        prompt = "Answer the multiple-choice question. Return exactly one line and nothing else: `Final answer: X`, where X is one option letter. Do not explain your reasoning.\n\n"
        prompt += row.question + "\n" + "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(rotated))
        output.append({
            "item_id": f"perturb-{index:03d}", "original_item_id": original["item_id"],
            "family": "mmlu_pro", "prompt": prompt, "reference": LETTERS[new_answer],
            "old_reference": original["reference"], "rotation": offset, "options": rotated,
            "max_tokens": 128,
        })
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output))
    return output


def append(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("12b", "26b"), required=True)
    parser.add_argument("--port", type=int, default=18151)
    args = parser.parse_args()
    items = build_items()
    raw_path = STUDY / f"raw/negative_control/{args.model}.jsonl"
    done = set()
    if raw_path.exists():
        done = {row["run_key"] for row in map(json.loads, raw_path.read_text().splitlines()) if row.get("status") == "ok"}
    for condition in ("off", "n6"):
        depth = 0 if condition == "off" else 6
        server = Server(args.model, depth, args.port, reasoning_format="deepseek")
        log = STUDY / f"logs/negative_control/{args.model}_{condition}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            server.start(log)
            order = items[:]; random.Random(20260809 + depth).shuffle(order)
            for item in order:
                key = f"{args.model}:{condition}:{item['item_id']}"
                if key in done: continue
                seed = 20260809 + int(hashlib.sha256(item["item_id"].encode()).hexdigest()[:8], 16)
                try:
                    result = chat(server, item, seed)
                    result.update({"run_key": key, "model_short": args.model, "condition": condition,
                                   "n_max": depth, "item_id": item["item_id"], "family": "mmlu_pro",
                                   "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                                   "seed": seed, "status": "ok" if result["output"] else "invalid_empty_output",
                                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": str(log)})
                except Exception as exc:
                    result = {"run_key": key, "model_short": args.model, "condition": condition,
                              "item_id": item["item_id"], "status": "failed_request", "error": repr(exc)}
                append(raw_path, result)
                print(key, result["status"], flush=True)
        finally:
            print(f"stopped {args.model} {condition} exit={server.stop()}", flush=True)


if __name__ == "__main__":
    main()
