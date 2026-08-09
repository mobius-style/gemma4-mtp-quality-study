#!/usr/bin/env python3
"""Machine acceptance gate for the completed study package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check(condition: bool, name: str, detail: object, checks: list[dict]) -> None:
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(float(actual) - expected) <= tolerance


def main() -> None:
    checks: list[dict] = []

    frozen = subprocess.run(
        ["sha256sum", "-c", "FREEZE.sha256"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    check(frozen.returncode == 0, "frozen_protocol_hashes", frozen.stdout + frozen.stderr, checks)

    validation_files = [
        "performance_validation.json", "quality_validation.json",
        "pmin_validation.json", "negative_control_validation.json",
    ]
    for name in validation_files:
        payload = json.loads((ROOT / "analysis" / name).read_text())
        check(payload.get("valid") is True, f"upstream_{name}", payload, checks)

    expected_raw = {
        "performance_12b": (ROOT / "raw/performance/12b.jsonl", 81),
        "performance_26b": (ROOT / "raw/performance/26b.jsonl", 81),
        "quality_12b_v6": (ROOT / "raw/quality/12b_v6.jsonl", 800),
        "quality_26b_v6": (ROOT / "raw/quality/26b_v6.jsonl", 800),
        "pmin_12b": (ROOT / "raw/pmin/12b.jsonl", 72),
        "pmin_26b": (ROOT / "raw/pmin/26b.jsonl", 72),
        "negative_12b": (ROOT / "raw/negative_control/12b.jsonl", 40),
        "negative_26b": (ROOT / "raw/negative_control/26b.jsonl", 40),
    }
    for name, (path, expected) in expected_raw.items():
        rows = jsonl(path)
        valid = sum(row.get("status") == "ok" for row in rows)
        check(valid == expected, f"raw_count_{name}", {"valid": valid, "expected": expected}, checks)
        models = {row.get("model_short") for row in rows if row.get("model_short")}
        check(models <= {"12b", "26b"}, f"model_scope_{name}", sorted(models), checks)

    perf = pd.read_csv(ROOT / "processed/performance_summary.csv")
    quality = pd.read_csv(ROOT / "processed/quality_summary.csv")
    paired = pd.read_csv(ROOT / "processed/paired_quality_changes.csv")
    equiv = pd.read_csv(ROOT / "processed/equivalence_summary.csv")
    length_bins = pd.read_csv(ROOT / "processed/length_divergence_by_bin.csv")
    depth_rank = pd.read_csv(ROOT / "processed/depth_rank_uncertainty.csv")
    within_family = pd.read_csv(ROOT / "processed/length_divergence_within_family.csv")
    stratified_length = pd.read_csv(ROOT / "processed/length_divergence_stratified.csv")
    plateau_12b = pd.read_csv(ROOT / "processed/12b_performance_plateau.csv")

    check(len(perf) == 18, "performance_grid_rows", len(perf), checks)
    for model, off, peak in (("12b", 89.195875, 226.393358), ("26b", 169.583143, 322.525690)):
        off_value = perf[(perf.model_short == model) & (perf.n_max == 0)].aggregate_decode_tok_s.iloc[0]
        peak_row = perf[perf.model_short == model].sort_values("aggregate_decode_tok_s").iloc[-1]
        check(close(off_value, off), f"numeric_anchor_{model}_off", float(off_value), checks)
        check(int(peak_row.n_max) == 6 and close(peak_row.aggregate_decode_tok_s, peak),
              f"numeric_anchor_{model}_peak", peak_row.to_dict(), checks)

    anchors = {
        ("12b", "n6"): (-0.0083333333333333, -0.0416666666666666, 0.0270833333333333),
        ("12b", "n3"): (-0.0041666666666667, -0.0333333333333333, 0.025),
        ("26b", "n6"): (-0.0083333333333333, -0.0291666666666666, 0.0083333333333333),
        ("26b", "n4"): (-0.0083333333333333, -0.0291666666666666, 0.0083333333333333),
    }
    for key, expected in anchors.items():
        row = paired[(paired.model_short == key[0]) & (paired.condition == key[1]) &
                     (paired.family == "macro")].iloc[0]
        actual = (row.macro_accuracy_difference, row.paired_bootstrap_ci_low,
                  row.paired_bootstrap_ci_high)
        check(all(close(a, b) for a, b in zip(actual, expected)) and bool(row.noninferiority_pass),
              f"quality_anchor_{key[0]}_{key[1]}", actual, checks)

    eq_expected = {("12b", "n6"): (112, 200), ("26b", "n6"): (118, 200)}
    for key, expected in eq_expected.items():
        row = equiv[(equiv.model_short == key[0]) & (equiv.condition == key[1]) &
                    (equiv.family == "all")].iloc[0]
        actual = (int(row.byte_equal_n), int(row.n))
        check(actual == expected, f"equivalence_anchor_{key[0]}_{key[1]}", actual, checks)

    macro = quality[quality.family == "macro"]
    check(len(macro) == 8, "quality_macro_rows", len(macro), checks)

    length_anchors = {
        ("12b", "n6", "1-8"): (60, 61),
        ("12b", "n6", "129+"): (24, 98),
        ("26b", "n6", "1-8"): (61, 61),
        ("26b", "n6", "129+"): (22, 88),
    }
    for key, expected in length_anchors.items():
        row = length_bins[
            (length_bins.model_short == key[0]) &
            (length_bins.condition == key[1]) &
            (length_bins.off_output_token_bin == key[2])
        ].iloc[0]
        actual = (int(row.exact_matches), int(row.n))
        check(actual == expected, f"length_anchor_{key[0]}_{key[1]}_{key[2]}", actual, checks)

    rank_anchors = {("12b", 6): 0.51262, ("26b", 6): 0.93909}
    for key, expected in rank_anchors.items():
        row = depth_rank[(depth_rank.model_short == key[0]) & (depth_rank.n_max == key[1])].iloc[0]
        check(close(row.bootstrap_probability_fastest, expected, 1e-8) and
              int(row.bootstrap_replicates) == 100_000,
              f"depth_rank_anchor_{key[0]}_n{key[1]}", row.to_dict(), checks)

    within_family_anchors = {
        ("12b", "n6", "gsm8k"): (-0.3290696351, -0.552172, -0.084923),
        ("12b", "n6", "ifeval"): (-0.5664790402, -0.724657, -0.271146),
        ("12b", "n6", "mbpp"): (-0.7011901634, -0.830107, -0.509387),
        ("26b", "n6", "gsm8k"): (-0.3212286270, -0.547194, -0.080585),
        ("26b", "n6", "ifeval"): (-0.5125826064, -0.693688, -0.270793),
        ("26b", "n6", "mbpp"): (-0.6427095868, -0.810148, -0.391192),
    }
    for key, expected in within_family_anchors.items():
        row = within_family[
            (within_family.model_short == key[0]) &
            (within_family.condition == key[1]) &
            (within_family.family == key[2])
        ].iloc[0]
        actual = (row.spearman_rho, row.bootstrap_ci95_low, row.bootstrap_ci95_high)
        check(all(close(a, b, 1e-5) for a, b in zip(actual, expected)) and
              bool(row.identifiable_within_family) and row.spearman_p_bh_all_identifiable <= 0.0135,
              f"within_family_anchor_{key[0]}_{key[2]}", row.to_dict(), checks)

    stratified_anchors = {
        ("12b", "n6"): (-0.4898424833, -0.610, -0.343),
        ("26b", "n6"): (-0.4502879752, -0.581, -0.297),
    }
    for key, expected in stratified_anchors.items():
        row = stratified_length[
            (stratified_length.model_short == key[0]) &
            (stratified_length.condition == key[1])
        ].iloc[0]
        actual = (row.within_family_rank_correlation, row.bootstrap_ci95_low,
                  row.bootstrap_ci95_high)
        check(all(close(a, b, 1e-3) for a, b in zip(actual, expected)) and
              int(row.n_informative) == 140 and
              int(row.bootstrap_replicates) == 20_000 and
              int(row.permutation_replicates) == 100_000 and
              row.permutation_p_bh_four_conditions <= 1.00001e-5,
              f"stratified_length_anchor_{key[0]}_{key[1]}", row.to_dict(), checks)

    check(set(plateau_12b.n_max.astype(int)) == {4, 6, 8, 12},
          "12b_plateau_depths", plateau_12b.to_dict(orient="records"), checks)
    check(plateau_12b.objective_quality_200_items_evaluated.sum() == 1 and
          bool(plateau_12b[plateau_12b.n_max == 6]
               .objective_quality_200_items_evaluated.iloc[0]) and
          not plateau_12b.strict_conservative_rule_eligible.any(),
          "12b_plateau_quality_boundary", plateau_12b.to_dict(orient="records"), checks)

    png = sorted((ROOT / "figures").glob("*.png"))
    svg = sorted((ROOT / "figures").glob("*.svg"))
    check(len(png) == 11 and all(p.stat().st_size > 20_000 for p in png),
          "publication_png_figures", [p.name for p in png], checks)
    check(len(svg) == 11 and all(p.stat().st_size > 10_000 for p in svg),
          "publication_svg_figures", [p.name for p in svg], checks)

    required = [
        "README.md", "RESEARCH_REPORT.md", "EXECUTIVE_SUMMARY.md", "REPRODUCE.md",
        "PAPER_OUTLINE.md", "CLAIMS_MATRIX.md", "LITERATURE_SEARCH.md", "DATA_DICTIONARY.md",
        "analysis/GATE_REPORT.md", "analysis/CONFIRMATORY_DECISION.md",
        "paper/PAPER.md", "paper/PAPER.pdf", "paper/references.bib",
        "tables/environment.md", "tables/recommended_operating_points.md",
        "tables/acceptance_by_depth.md", "tables/first_divergence_summary.md",
        "tables/dense_vs_moe_n6.md", "tables/performance_exact_equivalence_by_depth.md",
        "tables/length_divergence_by_bin.md", "tables/depth_rank_uncertainty.md",
        "processed/length_divergence_by_bin.csv", "processed/divergence_survival.csv",
        "processed/depth_rank_uncertainty.csv", "processed/review_followup_analysis.json",
        "processed/length_divergence_within_family.csv",
        "processed/length_divergence_stratified.csv",
        "processed/12b_performance_plateau.csv",
        "tables/length_divergence_within_family.md",
        "tables/length_divergence_stratified.md", "tables/12b_performance_plateau.md",
        "scripts/analyze_review_followup.py", "REVIEW_RESPONSE_2026-08-09.md",
        "REVIEW_RESPONSE_ROUND2_2026-08-09.md",
        "REVIEW_RESPONSE_ROUND3_2026-08-09.md",
        "UPSTREAM_ISSUE_25618_COMMENT_DRAFT.md",
    ]
    for rel in required:
        path = ROOT / rel
        check(path.is_file() and path.stat().st_size > 100, f"deliverable_{rel}",
              path.stat().st_size if path.exists() else None, checks)

    paper = (ROOT / "paper/PAPER.md").read_text()
    required_headings = [
        "# Abstract", "# 1. Introduction", "# 3. Related work",
        "# 5. Methods", "# 6. Performance results",
        "# 7. Deterministic-equivalence results",
        "# 8. Objective task-quality results", "# 14. Limitations",
        "# 15. Reproducibility", "# 16. Ethics", "# 17. Conclusion",
        "# References",
    ]
    missing = [heading for heading in required_headings if heading not in paper]
    check(not missing, "paper_required_sections", missing, checks)
    abstract = paper.split("# Abstract", 1)[1].split("# 1. Introduction", 1)[0]
    abstract_word_count = len(abstract.split())
    check(180 <= abstract_word_count <= 250, "abstract_word_count",
          abstract_word_count, checks)
    check("31B" not in paper and "31b" not in paper, "excluded_31b", "no 31B condition in paper", checks)

    result = {
        "study_date": "2026-08-09",
        "definition_of_done": {
            "performance_grid": "18/18 model-condition summaries",
            "quality_requests": "1600/1600 valid",
            "objective_items_per_model_condition": 200,
            "figures": "11 PNG + 11 SVG",
            "paper": "Markdown + PDF",
        },
        "checks": checks,
        "passed": all(item["pass"] for item in checks),
        "failed": [item["name"] for item in checks if not item["pass"]],
    }
    output = ROOT / "analysis/VALIDATION.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
