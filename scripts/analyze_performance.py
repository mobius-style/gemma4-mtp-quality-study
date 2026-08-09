#!/usr/bin/env python3
"""Validate and summarize the append-only performance sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEPTHS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
TARGETS = [256, 1024, 4000]


def bootstrap_median(values: list[float], seed: int, n: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values)
    samples = np.median(rng.choice(array, (n, len(array)), replace=True), axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def load_and_validate(model: str) -> tuple[list[dict], list[str]]:
    path = ROOT / f"raw/performance/{model}.jsonl"
    all_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in all_rows if row.get("status") == "ok"]
    errors = []
    expected = {f"{model}:{'off' if n == 0 else f'n{n}'}:{target}:{task}"
                for n in EXPECTED_DEPTHS for target in TARGETS for task in range(3)}
    keys = [row["run_key"] for row in rows]
    if set(keys) != expected:
        errors.append(f"run key mismatch: missing={sorted(expected-set(keys))}, extra={sorted(set(keys)-expected)}")
    if len(keys) != len(set(keys)):
        errors.append("duplicate successful run keys")
    for row in rows:
        for field in ("output", "prompt_tokens", "output_tokens", "decode_tok_s", "decode_ms",
                      "prompt_tok_s", "wall_ms", "time_to_first_content_ms"):
            if row.get(field) in (None, ""):
                errors.append(f"{row['run_key']}: missing {field}")
        if row.get("output_tokens") != 256:
            errors.append(f"{row['run_key']}: output token count {row.get('output_tokens')}")
        if row.get("input_tokens_expected") != row.get("prompt_tokens"):
            errors.append(f"{row['run_key']}: input token mismatch")
        if hashlib.sha256(row["output"].encode()).hexdigest() != row.get("output_sha256"):
            errors.append(f"{row['run_key']}: output hash mismatch")
    expected_main = "offloaded 49/49 layers to GPU" if model == "12b" else "offloaded 31/31 layers to GPU"
    for n in EXPECTED_DEPTHS:
        condition = "off" if n == 0 else f"n{n}"
        candidates = sorted((ROOT / "logs/performance").glob(f"{model}_{condition}*.log"))
        text = "\n".join(item.read_text(errors="replace") for item in candidates)
        if expected_main not in text:
            errors.append(f"{model} {condition}: full main-model GPU placement not found")
        if n and "offloaded 5/5 layers to GPU" not in text:
            errors.append(f"{model} {condition}: full draft GPU placement not found")
    return rows, errors


def main() -> None:
    processed = ROOT / "processed"
    processed.mkdir(exist_ok=True)
    all_rows = []
    validation = {"valid": True, "models": {}}
    summaries = []
    selections = {}
    equivalence = []
    rng_seed = 20260809
    for model_index, model in enumerate(("12b", "26b")):
        rows, errors = load_and_validate(model)
        validation["models"][model] = {"valid_rows": len(rows), "errors": errors}
        validation["valid"] &= not errors
        all_rows.extend(rows)
        baseline = {row["prompt_id"]: row for row in rows if row["condition"] == "off"}
        model_summaries = []
        for n in EXPECTED_DEPTHS:
            condition = "off" if n == 0 else f"n{n}"
            group = [row for row in rows if row["condition"] == condition]
            speeds = [row["decode_tok_s"] for row in group]
            aggregate = sum(row["output_tokens"] for row in group) / (sum(row["decode_ms"] for row in group) / 1000)
            proposed = sum(row.get("draft_tokens_proposed", 0) or 0 for row in group)
            accepted = sum(row.get("draft_tokens_accepted", 0) or 0 for row in group)
            ratios = [] if n == 0 else [row["decode_tok_s"] / baseline[row["prompt_id"]]["decode_tok_s"] for row in group]
            ci = (None, None) if not ratios else bootstrap_median(ratios, rng_seed + model_index * 100 + n)
            exact = None if n == 0 else sum(row["output"] == baseline[row["prompt_id"]]["output"] for row in group)
            summary = {
                "model_short": model, "condition": condition, "n_max": n,
                "requests": len(group), "aggregate_decode_tok_s": aggregate,
                "mean_decode_tok_s": statistics.fmean(speeds), "median_decode_tok_s": statistics.median(speeds),
                "sd_decode_tok_s": statistics.pstdev(speeds),
                "cv_decode_tok_s": statistics.pstdev(speeds) / statistics.fmean(speeds),
                "median_paired_speedup": None if not ratios else statistics.median(ratios),
                "median_paired_speedup_ci_low": ci[0], "median_paired_speedup_ci_high": ci[1],
                "aggregate_prompt_tok_s": sum(row["prompt_tokens"] for row in group) / (sum(row["prompt_ms"] for row in group) / 1000),
                "median_wall_ms": statistics.median(row["wall_ms"] for row in group),
                "median_ttfc_ms": statistics.median(row["time_to_first_content_ms"] for row in group),
                "draft_proposed": proposed, "draft_accepted": accepted,
                "acceptance_rate": accepted / proposed if proposed else None,
                "exact_output_matches": exact, "exact_output_match_rate": None if exact is None else exact / len(group),
            }
            summaries.append(summary)
            model_summaries.append(summary)
            if n:
                for row in group:
                    equivalence.append({
                        "model_short": model, "condition": condition, "prompt_id": row["prompt_id"],
                        "byte_equal": row["output"] == baseline[row["prompt_id"]]["output"],
                        "baseline_sha256": baseline[row["prompt_id"]]["output_sha256"],
                        "mtp_sha256": row["output_sha256"],
                    })
        candidates = [item for item in model_summaries if item["n_max"] > 0]
        maximum = max(candidates, key=lambda item: item["aggregate_decode_tok_s"])
        strict = [item for item in candidates
                  if item["aggregate_decode_tok_s"] >= 0.95 * maximum["aggregate_decode_tok_s"]
                  and item["cv_decode_tok_s"] <= 0.10]
        low_variance = max((item for item in candidates if item["cv_decode_tok_s"] <= 0.10),
                           key=lambda item: item["aggregate_decode_tok_s"])
        selections[model] = {
            "maximum_throughput": maximum["condition"],
            "strict_conservative": min(strict, key=lambda item: item["n_max"])["condition"] if strict else None,
            "exploratory_low_variance_fallback": low_variance["condition"],
            "quality_conditions": [maximum["condition"], low_variance["condition"]],
            "selection_status": "protocol fallback: no setting met both 95%-of-max and CV<=10%" if not strict else "pre-specified rule met",
        }

    pd.DataFrame([{k: v for k, v in row.items() if k not in {"output", "raw_terminal_event", "command"}}
                  for row in all_rows]).to_csv(processed / "performance_runs.csv", index=False)
    pd.DataFrame(summaries).to_csv(processed / "performance_summary.csv", index=False)
    pd.DataFrame(equivalence).to_csv(processed / "performance_exact_equivalence.csv", index=False)
    (processed / "performance_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (processed / "condition_selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (ROOT / "analysis/performance_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"validation": validation, "selections": selections}, indent=2))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
