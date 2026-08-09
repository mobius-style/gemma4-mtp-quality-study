#!/usr/bin/env python3
"""Analyze option-order and internal-duplication negative controls."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_quality import FINAL_OPTION, score_mmlu  # noqa: E402


def parsed(output: str) -> str | None:
    matches = FINAL_OPTION.findall(output)
    return matches[-1].upper() if matches else None


def main() -> None:
    items = [json.loads(line) for line in (ROOT / "prompts/quality_items_v3.jsonl").read_text().splitlines()]
    perturb = [json.loads(line) for line in (ROOT / "prompts/mmlu_option_perturbation.jsonl").read_text().splitlines()]
    perturb_index = {item["item_id"]: item for item in perturb}
    original_index = {item["item_id"]: item for item in items}
    summaries = []
    validation = {"valid": True, "errors": []}
    for model in ("12b", "26b"):
        raw = [json.loads(line) for line in (ROOT / f"raw/negative_control/{model}.jsonl").read_text().splitlines()]
        if len(raw) != 40 or len({row["run_key"] for row in raw}) != 40:
            validation["errors"].append(f"{model}: expected 40 unique rows")
        primary = [json.loads(line) for line in (ROOT / f"raw/quality/{model}_v6.jsonl").read_text().splitlines()]
        primary_index = {(row["condition"], row["item_id"]): row for row in primary}
        for condition in ("off", "n6"):
            group = [row for row in raw if row["condition"] == condition]
            correct = 0
            same_semantic_option = 0
            exact_between_mtp = 0
            for row in group:
                item = perturb_index[row["item_id"]]
                prediction = parsed(row["output"])
                correct += bool(score_mmlu(row["output"], item["reference"])["correct"])
                original_condition = "off_a" if condition == "off" else "n6"
                original = original_index[item["original_item_id"]]
                original_prediction = parsed(primary_index[(original_condition, item["original_item_id"])]["output"])
                if prediction and original_prediction:
                    original_text = original["options"][ord(original_prediction) - ord("A")]
                    perturbed_text = item["options"][ord(prediction) - ord("A")]
                    same_semantic_option += original_text == perturbed_text
                if condition == "n6":
                    off = next(candidate for candidate in raw if candidate["condition"] == "off" and candidate["item_id"] == row["item_id"])
                    exact_between_mtp += row["output"] == off["output"]
            summaries.append({
                "model_short": model, "condition": condition, "n": len(group),
                "perturbed_accuracy": correct / len(group),
                "semantic_option_stability_vs_original_order": same_semantic_option / len(group),
                "exact_mtp_vs_off_rate": None if condition == "off" else exact_between_mtp / len(group),
            })

    substantive_prompts = []
    for item in items:
        prompt = item["prompt"]
        if item["family"] in {"gsm8k", "mmlu_pro"}:
            prompt = prompt.split("\n\n", 1)[-1]
        elif item["family"] == "mbpp" and "Task: " in prompt:
            prompt = prompt.split("Task: ", 1)[-1]
        substantive_prompts.append(prompt)
    normalized = [re.findall(r"[\w']+", prompt.lower()) for prompt in substantive_prompts]
    owners = {}
    for item, tokens in zip(items, normalized):
        for index in range(max(0, len(tokens) - 12)):
            owners.setdefault(tuple(tokens[index:index + 13]), set()).add(item["item_id"])
    cross_item_ngrams = {" ".join(ngram): sorted(ids) for ngram, ids in owners.items() if len(ids) > 1}
    gsm_leak = []
    for item in items:
        if item["family"] == "gsm8k":
            question_tokens = set(re.findall(r"-?\d+(?:[.,]\d+)?", item["prompt"].replace(",", "")))
            if item["reference"].replace(",", "") in question_tokens:
                gsm_leak.append(item["item_id"])
    diagnostics = {
        "option_order_summary": summaries,
        "internal_exact_prompt_duplicates": len(items) - len({item["prompt"] for item in items}),
        "cross_item_shared_13gram_count": len(cross_item_ngrams),
        "cross_item_shared_13gram_examples": dict(list(cross_item_ngrams.items())[:20]),
        "gsm_reference_number_already_in_prompt": gsm_leak,
        "dataset_canary_field_available": False,
        "training_corpus_contamination_status": "unresolved; no training-corpus access or benchmark canaries",
    }
    validation["valid"] = not validation["errors"]
    pd.DataFrame(summaries).to_csv(ROOT / "processed/negative_control_summary.csv", index=False)
    (ROOT / "analysis/contamination_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "analysis/negative_control_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"validation": validation, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
