#!/usr/bin/env python3
"""Fetch pinned official benchmark files and create the frozen 200-item sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
import urllib.request

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/source"
VENDOR = ROOT / "vendor/instruction_following_eval"
SEED = 20260809

GOOGLE_REV = "015539128d9a7dbe14b5f5308a198a15da808949"
GSM_REV = "3101c7d5072418e28b9008a6636bde82a006892c"
MMLU_REV = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

FILES = {
    "gsm8k_test.jsonl": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM_REV}/grade_school_math/data/test.jsonl",
    "mmlu_pro_test.parquet": f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{MMLU_REV}/data/test-00000-of-00001.parquet?download=true",
    "ifeval_input_data.jsonl": f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/instruction_following_eval/data/input_data.jsonl",
    "sanitized-mbpp.json": f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/mbpp/sanitized-mbpp.json",
}

VENDOR_FILES = {
    "evaluation_lib.py", "instructions.py", "instructions_registry.py", "instructions_util.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose(items: list, n: int, offset: int) -> list:
    indices = random.Random(SEED + offset).sample(range(len(items)), n)
    return [items[index] for index in sorted(indices)]


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        fetch(url, SOURCE / name)
    VENDOR.mkdir(parents=True, exist_ok=True)
    init = VENDOR / "__init__.py"
    if not init.exists():
        init.write_text("\"\"\"Pinned official IFEval scorer package.\"\"\"\n")
    for name in sorted(VENDOR_FILES):
        url = f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/instruction_following_eval/{name}"
        fetch(url, VENDOR / name)

    gsm_all = jsonl(SOURCE / "gsm8k_test.jsonl")
    mmlu_all = pd.read_parquet(SOURCE / "mmlu_pro_test.parquet").to_dict("records")
    ifeval_all = jsonl(SOURCE / "ifeval_input_data.jsonl")
    mbpp_raw = json.loads((SOURCE / "sanitized-mbpp.json").read_text())
    mbpp_all = [item for item in mbpp_raw if 11 <= int(item["task_id"]) <= 510]

    output: list[dict] = []
    for index, source in enumerate(choose(gsm_all, 60, 1)):
        answer = source["answer"].rsplit("####", 1)[-1].strip()
        output.append({
            "item_id": f"gsm8k-{index:03d}", "family": "gsm8k",
            "source_index": gsm_all.index(source), "source_revision": GSM_REV,
            "prompt": "Solve the following grade-school mathematics problem. Show concise reasoning, then end with exactly `Final answer: <number>`.\n\n" + source["question"],
            "reference": answer, "max_tokens": 512,
        })

    letters = "ABCDEFGHIJ"
    for index, source in enumerate(choose(mmlu_all, 60, 2)):
        options = list(source["options"])
        prompt = "Answer the multiple-choice question. Return exactly one line and nothing else: `Final answer: X`, where X is one option letter. Do not explain your reasoning.\n\n"
        prompt += source["question"] + "\n" + "\n".join(f"{letters[i]}. {option}" for i, option in enumerate(options))
        reference = str(source.get("answer") or letters[int(source["answer_index"])]).strip().upper()
        output.append({
            "item_id": f"mmlu_pro-{index:03d}", "family": "mmlu_pro",
            "source_index": int(source["question_id"]), "source_revision": MMLU_REV,
            "category": source["category"], "prompt": prompt,
            "reference": reference, "answer_index": int(source["answer_index"]),
            "options": options, "max_tokens": 128,
        })

    for index, source in enumerate(choose(ifeval_all, 40, 3)):
        output.append({
            "item_id": f"ifeval-{index:03d}", "family": "ifeval",
            "source_index": int(source["key"]), "source_revision": GOOGLE_REV,
            "prompt": source["prompt"], "reference": {
                "key": source["key"], "instruction_id_list": source["instruction_id_list"],
                "kwargs": source["kwargs"],
            }, "max_tokens": 768,
        })

    for index, source in enumerate(choose(mbpp_all, 40, 4)):
        tests = list(source["test_list"])
        prompt = (
            "Write Python code that solves the task below. Return only one fenced Python code block and no explanation. "
            "The code must pass all supplied tests.\n\nTask: " + source["prompt"] +
            "\n\nTests:\n" + "\n".join(tests)
        )
        output.append({
            "item_id": f"mbpp-{index:03d}", "family": "mbpp",
            "source_index": int(source["task_id"]), "source_revision": GOOGLE_REV,
            "prompt": prompt, "reference": tests,
            "test_imports": list(source.get("test_imports", [])), "max_tokens": 512,
        })

    counts = {family: sum(item["family"] == family for item in output)
              for family in ("gsm8k", "mmlu_pro", "ifeval", "mbpp")}
    assert counts == {"gsm8k": 60, "mmlu_pro": 60, "ifeval": 40, "mbpp": 40}
    out_path = ROOT / "prompts/quality_items_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = [json.loads(line) for line in out_path.read_text().splitlines()]
        if existing != output:
            raise RuntimeError("frozen quality sample already exists and differs")
    else:
        out_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output))

    manifest = {
        "sampling_seed": SEED, "counts": counts,
        "sample_sha256": sha256(out_path),
        "sources": [
            {"filename": name, "url": url, "sha256": sha256(SOURCE / name), "bytes": (SOURCE / name).stat().st_size}
            for name, url in FILES.items()
        ],
        "vendor": [
            {"filename": name, "revision": GOOGLE_REV, "sha256": sha256(VENDOR / name)}
            for name in sorted(VENDOR_FILES)
        ],
    }
    manifest["prompt_protocol_revision"] = "v3_post_smoke_mmlu_direct_answer_fix"
    manifest_path = ROOT / "datasets/manifest_v3.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
