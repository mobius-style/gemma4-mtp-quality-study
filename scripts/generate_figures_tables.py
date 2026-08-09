#!/usr/bin/env python3
"""Generate publication figures and compact Markdown tables from processed data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.dpi": 300})
COLORS = {"12b": "#276FBF", "26b": "#C44536"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def md_table(frame: pd.DataFrame, path: Path, digits: int = 3) -> None:
    work = frame.copy()
    for column in work.select_dtypes(include=["float"]).columns:
        work[column] = work[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    columns = list(work.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in work.astype(str).itertuples(index=False):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    perf = pd.read_csv(ROOT / "processed/performance_summary.csv")
    quality = pd.read_csv(ROOT / "processed/quality_summary.csv")
    equiv = pd.read_csv(ROOT / "processed/equivalence_summary.csv")
    pairs = pd.read_csv(ROOT / "processed/equivalence_pairs.csv")
    paired = pd.read_csv(ROOT / "processed/paired_quality_changes.csv")
    pmin = pd.read_csv(ROOT / "processed/pmin_summary.csv")
    perf_exact = pd.read_csv(ROOT / "processed/performance_exact_equivalence.csv")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for model in ("12b", "26b"):
        group = perf[perf.model_short == model].sort_values("n_max")
        ax.errorbar(group.n_max, group.mean_decode_tok_s, yerr=group.sd_decode_tok_s,
                    marker="o", capsize=3, label=model.upper(), color=COLORS[model])
    ax.set(xlabel="Maximum draft depth N (0 = MTP off)", ylabel="Decode throughput (tokens/s)")
    ax.set_xticks([0, 1, 2, 3, 4, 6, 8, 12, 16]); ax.legend(frameon=False)
    save(fig, "01_depth_vs_throughput")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for model in ("12b", "26b"):
        group = perf[(perf.model_short == model) & (perf.n_max > 0)].sort_values("n_max")
        ax.plot(group.n_max, 100 * group.acceptance_rate, marker="o", label=model.upper(), color=COLORS[model])
    ax.set(xlabel="Maximum draft depth N", ylabel="Accepted draft tokens (%)", ylim=(0, 100))
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16]); ax.legend(frameon=False)
    save(fig, "02_depth_vs_acceptance")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for model in ("12b", "26b"):
        group = perf[(perf.model_short == model) & (perf.n_max > 0)].sort_values("n_max")
        ax.plot(group.n_max, 100 * group.exact_output_match_rate, marker="o", label=model.upper(), color=COLORS[model])
    ax.set(xlabel="Maximum draft depth N", ylabel="Exact match to OFF (%)", ylim=(-2, 102))
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16]); ax.legend(frameon=False)
    save(fig, "03_depth_vs_exact_match_performance_prompts")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
    for ax, model in zip(axes, ("12b", "26b")):
        group = quality[(quality.model_short == model) & (quality.family == "macro")].copy()
        group["n"] = group.condition.map(lambda x: 0 if x.startswith("off") else int(x[1:]))
        group = group[group.condition != "off_b"].sort_values("n")
        ax.plot(group.n, 100 * group.accuracy, marker="o", color=COLORS[model])
        ax.axhline(100 * group[group.n == 0].accuracy.iloc[0] - 5, ls="--", lw=1, color="0.5", label="-5 pp margin")
        ax.set_title(model.upper()); ax.set_xlabel("N (0 = MTP off)"); ax.set_xticks(group.n)
    axes[0].set_ylabel("Macro objective score (%)"); axes[0].legend(frameon=False)
    save(fig, "04_selected_depth_vs_quality")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for model in ("12b", "26b"):
        q = quality[(quality.model_short == model) & (quality.family == "macro") & (quality.condition != "off_b")]
        for row in q.itertuples():
            n = 0 if row.condition == "off_a" else int(row.condition[1:])
            speed = perf[(perf.model_short == model) & (perf.n_max == n)].aggregate_decode_tok_s.iloc[0]
            ax.scatter(speed, 100 * row.accuracy, color=COLORS[model], s=42)
            ax.annotate(f"{model.upper()} {row.condition.replace('_a','')}", (speed, 100 * row.accuracy), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Aggregate decode throughput (tokens/s)", ylabel="Macro objective score (%)")
    save(fig, "05_throughput_quality_pareto")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True, sharey=True)
    for ax, (model, condition) in zip(axes.flat, [("12b", "n3"), ("12b", "n6"), ("26b", "n4"), ("26b", "n6")]):
        values = pairs[(pairs.model_short == model) & (pairs.condition == condition)].first_divergent_token.dropna()
        ax.hist(values, bins=np.arange(0, max(130, values.max() + 16), 16), color=COLORS[model], alpha=.85)
        ax.set_title(f"{model.upper()} {condition}"); ax.set_xlabel("First divergent token"); ax.set_ylabel("Sequences")
    save(fig, "06_first_divergence_distribution")

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    for model in ("12b", "26b"):
        group = perf[(perf.model_short == model) & (perf.n_max > 0)].sort_values("n_max")
        ax.plot(group.n_max, group.median_paired_speedup, marker="o", color=COLORS[model], label=model.upper())
    ax.axhline(1, color="0.5", lw=1); ax.set(xlabel="Maximum draft depth N", ylabel="Median paired speedup (x)")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16]); ax.legend(frameon=False)
    save(fig, "07_dense_vs_moe_speedup")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=False)
    for ax, model in zip(axes, ("12b", "26b")):
        for depth in sorted(pmin[pmin.model_short == model].n_max.unique()):
            group = pmin[(pmin.model_short == model) & (pmin.n_max == depth)].sort_values("p_min")
            ax.plot(group.p_min, group.aggregate_decode_tok_s, marker="o", label=f"n={depth}")
        ax.set_title(model.upper()); ax.set_xlabel("p-min"); ax.set_ylabel("Aggregate decode tokens/s"); ax.legend(frameon=False)
    save(fig, "08_pmin_sweep")

    perf_out = perf[["model_short", "n_max", "aggregate_decode_tok_s", "median_paired_speedup",
                     "cv_decode_tok_s", "acceptance_rate", "median_wall_ms", "median_ttfc_ms", "exact_output_match_rate"]]
    md_table(perf_out, TAB / "performance_by_depth.md")
    md_table(quality[quality.family != "macro"][["model_short", "condition", "family", "correct", "n", "accuracy", "length_truncations"]], TAB / "quality_by_family.md")
    md_table(quality[quality.family == "macro"][["model_short", "condition", "accuracy", "length_truncations"]], TAB / "quality_macro.md")
    md_table(paired, TAB / "paired_quality_changes.md")
    md_table(equiv[equiv.family == "all"], TAB / "equivalence_summary.md")
    md_table(pmin, TAB / "pmin_summary.md")

    acceptance = perf[perf.n_max > 0][["model_short", "n_max", "draft_proposed",
                                       "draft_accepted", "acceptance_rate"]]
    md_table(acceptance, TAB / "acceptance_by_depth.md")
    exact_by_depth = (perf_exact.groupby(["model_short", "condition"], as_index=False)
                      .agg(requests=("byte_equal", "size"), exact_matches=("byte_equal", "sum"),
                           exact_match_rate=("byte_equal", "mean")))
    exact_by_depth["n_max"] = exact_by_depth.condition.str[1:].astype(int)
    md_table(exact_by_depth[["model_short", "n_max", "requests", "exact_matches", "exact_match_rate"]],
             TAB / "performance_exact_equivalence_by_depth.md")
    divergence = (pairs[pairs.first_divergent_token.notna()]
                  .groupby(["model_short", "condition"], as_index=False)
                  .agg(divergent_sequences=("first_divergent_token", "size"),
                       mean_first_divergence=("first_divergent_token", "mean"),
                       median_first_divergence=("first_divergent_token", "median"),
                       p25_first_divergence=("first_divergent_token", lambda x: x.quantile(.25)),
                       p75_first_divergence=("first_divergent_token", lambda x: x.quantile(.75))))
    md_table(divergence, TAB / "first_divergence_summary.md")

    comparison_rows = []
    for model in ("12b", "26b"):
        off = perf[(perf.model_short == model) & (perf.n_max == 0)].iloc[0]
        n6 = perf[(perf.model_short == model) & (perf.n_max == 6)].iloc[0]
        comparison_rows.append({
            "model_short": model,
            "off_tok_s": off.aggregate_decode_tok_s,
            "n6_tok_s": n6.aggregate_decode_tok_s,
            "aggregate_speedup": n6.aggregate_decode_tok_s / off.aggregate_decode_tok_s,
            "paired_median_speedup": n6.median_paired_speedup,
            "n6_acceptance": n6.acceptance_rate,
            "n6_cv": n6.cv_decode_tok_s,
        })
    md_table(pd.DataFrame(comparison_rows), TAB / "dense_vs_moe_n6.md")


if __name__ == "__main__":
    main()
