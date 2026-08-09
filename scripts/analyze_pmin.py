#!/usr/bin/env python3
"""Validate and summarize the triggered exploratory p-min sweep."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = {"12b": (6, 12), "26b": (6, 8)}
PMINS = (0.25, 0.50, 0.75, 0.90)


def summarize(model: str, depth: int, pmin: float, rows: list[dict]) -> dict:
    speeds = [row["decode_tok_s"] for row in rows]
    proposed = sum(row.get("draft_tokens_proposed", 0) or 0 for row in rows)
    accepted = sum(row.get("draft_tokens_accepted", 0) or 0 for row in rows)
    return {
        "model_short": model, "n_max": depth, "p_min": pmin, "requests": len(rows),
        "aggregate_decode_tok_s": sum(row["output_tokens"] for row in rows) / (sum(row["decode_ms"] for row in rows) / 1000),
        "mean_decode_tok_s": statistics.fmean(speeds), "median_decode_tok_s": statistics.median(speeds),
        "sd_decode_tok_s": statistics.pstdev(speeds),
        "cv_decode_tok_s": statistics.pstdev(speeds) / statistics.fmean(speeds),
        "acceptance_rate": accepted / proposed if proposed else None,
        "draft_proposed": proposed, "draft_accepted": accepted,
    }


def main() -> None:
    summaries = []
    validation = {"valid": True, "models": {}}
    for model in ("12b", "26b"):
        raw = [json.loads(line) for line in (ROOT / f"raw/pmin/{model}.jsonl").read_text().splitlines()]
        perf = [json.loads(line) for line in (ROOT / f"raw/performance/{model}.jsonl").read_text().splitlines()]
        errors = []
        if len(raw) != 72 or len({row["run_key"] for row in raw}) != 72:
            errors.append("expected 72 unique p-min rows")
        for depth in DEPTHS[model]:
            base = [row for row in perf if row.get("status") == "ok" and row["condition"] == f"n{depth}"]
            summaries.append(summarize(model, depth, 0.0, base))
            for pmin in PMINS:
                label = f"n{depth}_p{str(pmin).replace('.', '')}"
                group = [row for row in raw if row.get("status") == "ok" and row["condition"] == label]
                if len(group) != 9:
                    errors.append(f"{label}: expected 9 valid rows, found {len(group)}")
                for row in group:
                    if row["input_tokens_expected"] != row.get("prompt_tokens") or row.get("output_tokens") != 256:
                        errors.append(f"token drift: {row['run_key']}")
                summaries.append(summarize(model, depth, pmin, group))
                logs = "\n".join(path.read_text(errors="replace") for path in
                                   (ROOT / "logs/pmin").glob(f"{model}_{label}*.log"))
                main_marker = "offloaded 49/49 layers to GPU" if model == "12b" else "offloaded 31/31 layers to GPU"
                if main_marker not in logs or "offloaded 5/5 layers to GPU" not in logs:
                    errors.append(f"GPU placement missing: {label}")
        validation["models"][model] = {"valid_rows": len([row for row in raw if row.get("status") == "ok"]), "errors": errors}
        validation["valid"] &= not errors
    for model in ("12b", "26b"):
        baseline = next(row for row in summaries if row["model_short"] == model and row["n_max"] == 6 and row["p_min"] == 0)
        for row in summaries:
            if row["model_short"] == model:
                row["speed_ratio_to_n6_p0"] = row["aggregate_decode_tok_s"] / baseline["aggregate_decode_tok_s"]
    pd.DataFrame(summaries).to_csv(ROOT / "processed/pmin_summary.csv", index=False)
    (ROOT / "processed/pmin_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (ROOT / "analysis/pmin_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    if not validation["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
