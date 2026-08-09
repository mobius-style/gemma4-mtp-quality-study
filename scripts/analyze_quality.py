#!/usr/bin/env python3
"""Score quality outputs and perform paired equivalence/non-inferiority analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_quality import score_item, self_test  # noqa: E402

FAMILIES = ("gsm8k", "mmlu_pro", "ifeval", "mbpp")
EXPECTED_CONDITIONS = {"12b": ("off_a", "off_b", "n6", "n3"),
                       "26b": ("off_a", "off_b", "n6", "n4")}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def common_prefix(a: list[int], b: list[int]) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def paired_bootstrap_macro(rows: list[dict], seed: int, n: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    differences = np.zeros(n)
    for family in FAMILIES:
        group = [row for row in rows if row["family"] == family]
        delta = np.asarray([int(row["mtp_correct"]) - int(row["baseline_correct"]) for row in group])
        differences += np.mean(rng.choice(delta, (n, len(delta)), replace=True), axis=1) / len(FAMILIES)
    return {
        "difference": statistics.fmean(int(row["mtp_correct"]) - int(row["baseline_correct"]) for row in rows),
        "macro_difference": statistics.fmean(
            statistics.fmean(int(row["mtp_correct"]) - int(row["baseline_correct"])
                             for row in rows if row["family"] == family)
            for family in FAMILIES
        ),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def main() -> None:
    self_test()
    items = {item["item_id"]: item for item in
             map(json.loads, (ROOT / "prompts/quality_items_v3.jsonl").read_text().splitlines())}
    all_scored = []
    pair_rows = []
    quality_summary = []
    paired_summary = []
    equivalence_summary = []
    validation = {"valid": True, "models": {}}

    for model_index, model in enumerate(("12b", "26b")):
        raw = [json.loads(line) for line in (ROOT / f"raw/quality/{model}_v6.jsonl").read_text().splitlines()]
        errors = []
        if len(raw) != 800 or len({row["run_key"] for row in raw}) != 800:
            errors.append("expected 800 unique rows")
        for row in raw:
            if row.get("status") != "ok" or not row.get("output"):
                errors.append(f"invalid output: {row.get('run_key')}")
                continue
            if hashlib.sha256(row["output"].encode()).hexdigest() != row["output_sha256"]:
                errors.append(f"hash mismatch: {row['run_key']}")
            if row["prompt_sha256"] != hashlib.sha256(items[row["item_id"]]["prompt"].encode()).hexdigest():
                errors.append(f"prompt drift: {row['run_key']}")
            score = score_item(row["output"], items[row["item_id"]])
            all_scored.append({
                "model_short": model, "condition": row["condition"], "item_id": row["item_id"],
                "family": row["family"], "correct": bool(score["correct"]),
                "finish_reason": row["finish_reason"], "wall_ms": row["wall_ms"],
                "output_tokens": row.get("usage", {}).get("completion_tokens"),
                "output_sha256": row["output_sha256"], "score_detail": score,
            })
        for condition in EXPECTED_CONDITIONS[model]:
            group = [row for row in all_scored if row["model_short"] == model and row["condition"] == condition]
            for family in FAMILIES:
                family_group = [row for row in group if row["family"] == family]
                quality_summary.append({
                    "model_short": model, "condition": condition, "family": family,
                    "n": len(family_group), "correct": sum(row["correct"] for row in family_group),
                    "accuracy": statistics.fmean(row["correct"] for row in family_group),
                    "length_truncations": sum(row["finish_reason"] == "length" for row in family_group),
                })
            family_acc = [item["accuracy"] for item in quality_summary
                          if item["model_short"] == model and item["condition"] == condition]
            quality_summary.append({
                "model_short": model, "condition": condition, "family": "macro",
                "n": len(group), "correct": sum(row["correct"] for row in group),
                "accuracy": statistics.fmean(family_acc),
                "length_truncations": sum(row["finish_reason"] == "length" for row in group),
            })

        index = {(row["condition"], row["item_id"]): row for row in raw}
        score_index = {(row["condition"], row["item_id"]): row for row in all_scored if row["model_short"] == model}
        baseline_stable = {}
        model_test_indices = []
        for item_id in items:
            a, b = index[("off_a", item_id)], index[("off_b", item_id)]
            baseline_stable[item_id] = a["output"] == b["output"] and a["output_token_ids"] == b["output_token_ids"]
        for condition in EXPECTED_CONDITIONS[model][2:]:
            current_pairs = []
            for item_id, item in items.items():
                base = index[("off_a", item_id)]
                mtp = index[(condition, item_id)]
                base_score = score_index[("off_a", item_id)]
                mtp_score = score_index[(condition, item_id)]
                prefix = common_prefix(base["output_token_ids"], mtp["output_token_ids"])
                pair = {
                    "model_short": model, "condition": condition, "item_id": item_id,
                    "family": item["family"], "baseline_stable": baseline_stable[item_id],
                    "byte_equal": base["output"] == mtp["output"],
                    "token_equal": base["output_token_ids"] == mtp["output_token_ids"],
                    "common_token_prefix": prefix,
                    "first_divergent_token": None if base["output_token_ids"] == mtp["output_token_ids"] else prefix,
                    "normalized_char_edit_similarity": Levenshtein.normalized_similarity(base["output"], mtp["output"]),
                    "length_difference_tokens": len(mtp["output_token_ids"]) - len(base["output_token_ids"]),
                    "baseline_correct": base_score["correct"], "mtp_correct": mtp_score["correct"],
                }
                pair_rows.append(pair)
                current_pairs.append(pair)

            for family in (*FAMILIES, "all"):
                group = current_pairs if family == "all" else [row for row in current_pairs if row["family"] == family]
                stable = [row for row in group if row["baseline_stable"]]
                divergent = [row for row in group if row["first_divergent_token"] is not None]
                equivalence_summary.append({
                    "model_short": model, "condition": condition, "family": family, "n": len(group),
                    "baseline_stable_n": len(stable),
                    "byte_equal_n": sum(row["byte_equal"] for row in group),
                    "byte_equal_rate": statistics.fmean(row["byte_equal"] for row in group),
                    "stable_subset_byte_equal_rate": statistics.fmean(row["byte_equal"] for row in stable) if stable else None,
                    "token_equal_rate": statistics.fmean(row["token_equal"] for row in group),
                    "median_first_divergence": statistics.median(row["first_divergent_token"] for row in divergent) if divergent else None,
                    "mean_edit_similarity": statistics.fmean(row["normalized_char_edit_similarity"] for row in group),
                    "median_length_difference_tokens": statistics.median(row["length_difference_tokens"] for row in group),
                })

            stats = paired_bootstrap_macro(current_pairs, 20260809 + model_index * 100 + int(condition[1:]))
            tests = []
            for family in FAMILIES:
                group = [row for row in current_pairs if row["family"] == family]
                regressions = sum(row["baseline_correct"] and not row["mtp_correct"] for row in group)
                improvements = sum(not row["baseline_correct"] and row["mtp_correct"] for row in group)
                discordant = regressions + improvements
                pvalue = binomtest(min(regressions, improvements), discordant, 0.5).pvalue if discordant else 1.0
                tests.append({
                    "model_short": model, "condition": condition, "family": family, "n": len(group),
                    "both_correct": sum(row["baseline_correct"] and row["mtp_correct"] for row in group),
                    "regressions": regressions, "improvements": improvements,
                    "both_incorrect": sum(not row["baseline_correct"] and not row["mtp_correct"] for row in group),
                    "net_change": improvements - regressions, "mcnemar_exact_p": pvalue,
                })
            for item in tests:
                model_test_indices.append(len(paired_summary))
                paired_summary.append(item)
            paired_summary.append({
                "model_short": model, "condition": condition, "family": "macro", "n": 200,
                "both_correct": None, "regressions": sum(item["regressions"] for item in tests),
                "improvements": sum(item["improvements"] for item in tests),
                "both_incorrect": None, "net_change": sum(item["net_change"] for item in tests),
                "macro_accuracy_difference": stats["macro_difference"],
                "paired_bootstrap_ci_low": stats["ci_low"], "paired_bootstrap_ci_high": stats["ci_high"],
                "noninferiority_margin": -0.05, "noninferiority_pass": stats["ci_low"] > -0.05,
            })

        adjusted = holm([paired_summary[index]["mcnemar_exact_p"] for index in model_test_indices])
        for index, adj in zip(model_test_indices, adjusted):
            paired_summary[index]["holm_adjusted_p_within_model"] = adj

        validation["models"][model] = {
            "errors": errors, "valid_rows": len([row for row in raw if row.get("status") == "ok"]),
            "baseline_byte_equal": sum(baseline_stable.values()), "baseline_total": len(baseline_stable),
            "baseline_unstable_items": [item for item, stable in baseline_stable.items() if not stable],
        }
        validation["valid"] &= not errors

    processed = ROOT / "processed"
    processed.mkdir(exist_ok=True)
    flat_scores = [{**{k: v for k, v in row.items() if k != "score_detail"},
                    "score_detail": json.dumps(row["score_detail"], ensure_ascii=False, sort_keys=True)} for row in all_scored]
    pd.DataFrame(flat_scores).to_csv(processed / "quality_item_scores.csv", index=False)
    (processed / "quality_item_scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_scored))
    pd.DataFrame(quality_summary).to_csv(processed / "quality_summary.csv", index=False)
    pd.DataFrame(paired_summary).to_csv(processed / "paired_quality_changes.csv", index=False)
    pd.DataFrame(equivalence_summary).to_csv(processed / "equivalence_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(processed / "equivalence_pairs.csv", index=False)
    (processed / "quality_analysis.json").write_text(json.dumps({
        "quality_summary": quality_summary, "paired_summary": paired_summary,
        "equivalence_summary": equivalence_summary,
    }, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "analysis/quality_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print(pd.DataFrame([row for row in quality_summary if row["family"] == "macro"]).to_string(index=False))
    print(pd.DataFrame([row for row in paired_summary if row["family"] == "macro"]).to_string(index=False))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
