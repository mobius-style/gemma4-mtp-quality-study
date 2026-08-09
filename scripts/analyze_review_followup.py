#!/usr/bin/env python3
"""Post-review exploratory analyses for response length and depth ranking.

These analyses were not part of the frozen confirmatory protocol.  They are
therefore emitted into separately named artifacts and must not be interpreted
as confirmatory tests of an independent per-token divergence mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
SEED = 20260809
BOOTSTRAP_REPLICATES = 100_000
WITHIN_FAMILY_BOOTSTRAP_REPLICATES = 20_000
WITHIN_FAMILY_PERMUTATION_REPLICATES = 100_000
LENGTH_BINS = [(1, 8, "1-8"), (9, 32, "9-32"), (33, 64, "33-64"),
               (65, 128, "65-128"), (129, np.inf, "129+")]
HORIZONS = (8, 16, 32, 64, 128, 256)
QUALITY_CONDITIONS = {"12b": ("n3", "n6"), "26b": ("n4", "n6")}

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
})
COLORS = {"12b": "#276FBF", "26b": "#C44536"}
LINESTYLES = {"n3": "--", "n4": "--", "n6": "-"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted.tolist()
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def load_quality_pairs_with_lengths() -> pd.DataFrame:
    pairs = pd.read_csv(PROCESSED / "equivalence_pairs.csv")
    frames: list[pd.DataFrame] = []
    for model, conditions in QUALITY_CONDITIONS.items():
        raw_name = "12b_v6.jsonl" if model == "12b" else "26b_v6.jsonl"
        raw = read_jsonl(ROOT / "raw" / "quality" / raw_name)
        off_lengths = {
            row["item_id"]: len(row["output_token_ids"])
            for row in raw
            if row["condition"] == "off_a" and row["status"] == "ok"
        }
        group = pairs[(pairs.model_short == model) & pairs.condition.isin(conditions)].copy()
        group["off_output_tokens"] = group.item_id.map(off_lengths)
        if group.off_output_tokens.isna().any():
            missing = group[group.off_output_tokens.isna()].item_id.tolist()
            raise RuntimeError(f"Missing OFF lengths for {model}: {missing}")
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def markdown_table(frame: pd.DataFrame, path: Path, digits: int = 3) -> None:
    work = frame.copy()
    for column in work.select_dtypes(include=["float"]).columns:
        work[column] = work[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    columns = list(work.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in work.astype(str).itertuples(index=False):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def km_equal_survival(lengths: np.ndarray, exact: np.ndarray, horizon: int) -> tuple[float, int]:
    """Descriptive Kaplan-Meier estimate of no-divergence through horizon.

    Divergent pairs have an observed event time. Exact pairs are censored at
    their natural OFF-response length; hence censoring may be informative.
    """
    event_times = lengths[~exact]
    survival = 1.0
    for time in np.sort(np.unique(event_times[event_times <= horizon])):
        at_risk = int(np.sum(lengths >= time))
        events = int(np.sum(event_times == time))
        if at_risk:
            survival *= 1.0 - events / at_risk
    return survival, int(np.sum(lengths >= horizon))


def quality_length_analysis() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pairs = load_quality_pairs_with_lengths()
    bin_rows: list[dict] = []
    survival_rows: list[dict] = []
    correlation_rows: list[dict] = []

    for model, conditions in QUALITY_CONDITIONS.items():
        for condition in conditions:
            group = pairs[(pairs.model_short == model) & (pairs.condition == condition)].copy()
            lengths = group.off_output_tokens.to_numpy(dtype=int)
            exact = group.byte_equal.to_numpy(dtype=bool)
            rho, p_value = spearmanr(lengths, exact.astype(int))
            correlation_rows.append({
                "model_short": model,
                "condition": condition,
                "n": int(len(group)),
                "exact_matches": int(exact.sum()),
                "exact_match_rate": float(exact.mean()),
                "median_off_tokens_exact": float(np.median(lengths[exact])),
                "median_off_tokens_divergent": float(np.median(lengths[~exact])),
                "spearman_rho_length_vs_exact": float(rho),
                "spearman_p_unadjusted": float(p_value),
            })

            for low, high, label in LENGTH_BINS:
                mask = lengths >= low
                if np.isfinite(high):
                    mask &= lengths <= high
                selected = exact[mask]
                bin_rows.append({
                    "model_short": model,
                    "condition": condition,
                    "off_output_token_bin": label,
                    "n": int(mask.sum()),
                    "exact_matches": int(selected.sum()),
                    "exact_match_rate": float(selected.mean()) if selected.size else np.nan,
                })

            for horizon in HORIZONS:
                survival, at_risk = km_equal_survival(lengths, exact, horizon)
                survival_rows.append({
                    "model_short": model,
                    "condition": condition,
                    "token_horizon": horizon,
                    "descriptive_equal_survival": survival,
                    "at_risk_at_horizon": at_risk,
                })

    return pd.DataFrame(bin_rows), pd.DataFrame(survival_rows), correlation_rows


def bootstrap_spearman_ci(
    lengths: np.ndarray,
    exact: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    samples = rng.integers(
        0, len(lengths), size=(WITHIN_FAMILY_BOOTSTRAP_REPLICATES, len(lengths))
    )
    ranked_length = rankdata(lengths[samples], axis=1)
    ranked_length -= ranked_length.mean(axis=1, keepdims=True)
    sampled_exact = exact[samples].astype(float)
    sampled_exact -= sampled_exact.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(ranked_length ** 2, axis=1) * np.sum(sampled_exact ** 2, axis=1)
    )
    valid = denominator > 0
    correlations = (
        np.sum(ranked_length[valid] * sampled_exact[valid], axis=1) / denominator[valid]
    )
    return (
        float(np.quantile(correlations, 0.025)),
        float(np.quantile(correlations, 0.975)),
        int(valid.sum()),
    )


def within_family_residual_correlation(group: pd.DataFrame) -> tuple[float, list[str], int]:
    ranked_parts: list[np.ndarray] = []
    exact_parts: list[np.ndarray] = []
    identifying_families: list[str] = []
    for family, family_group in group.groupby("family", sort=True):
        lengths = family_group.off_output_tokens.to_numpy(dtype=float)
        exact = family_group.byte_equal.to_numpy(dtype=float)
        if len(np.unique(lengths)) < 2:
            continue
        ranked = rankdata(lengths, method="average") / (len(lengths) + 1)
        ranked_parts.append(ranked - ranked.mean())
        exact_parts.append(exact - exact.mean())
        identifying_families.append(family)
    ranked_all = np.concatenate(ranked_parts)
    exact_all = np.concatenate(exact_parts)
    correlation = float(np.corrcoef(ranked_all, exact_all)[0, 1])
    return correlation, identifying_families, int(len(ranked_all))


def stratified_bootstrap_correlation(
    group: pd.DataFrame,
    identifying_families: list[str],
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    numerator = np.zeros(WITHIN_FAMILY_BOOTSTRAP_REPLICATES)
    ranked_ss = np.zeros(WITHIN_FAMILY_BOOTSTRAP_REPLICATES)
    exact_ss = np.zeros(WITHIN_FAMILY_BOOTSTRAP_REPLICATES)
    for family in identifying_families:
        family_group = group[group.family == family]
        lengths = family_group.off_output_tokens.to_numpy(dtype=float)
        exact = family_group.byte_equal.to_numpy(dtype=float)
        samples = rng.integers(
            0, len(lengths), size=(WITHIN_FAMILY_BOOTSTRAP_REPLICATES, len(lengths))
        )
        ranked = rankdata(lengths[samples], axis=1) / (len(lengths) + 1)
        ranked -= ranked.mean(axis=1, keepdims=True)
        sampled_exact = exact[samples]
        sampled_exact -= sampled_exact.mean(axis=1, keepdims=True)
        numerator += np.sum(ranked * sampled_exact, axis=1)
        ranked_ss += np.sum(ranked ** 2, axis=1)
        exact_ss += np.sum(sampled_exact ** 2, axis=1)
    denominator = np.sqrt(ranked_ss * exact_ss)
    valid = denominator > 0
    correlations = numerator[valid] / denominator[valid]
    return (
        float(np.quantile(correlations, 0.025)),
        float(np.quantile(correlations, 0.975)),
        int(valid.sum()),
    )


def stratified_permutation_p(
    group: pd.DataFrame,
    identifying_families: list[str],
    observed: float,
    rng: np.random.Generator,
) -> float:
    extreme = 0
    complete = 0
    chunk_size = 5_000
    while complete < WITHIN_FAMILY_PERMUTATION_REPLICATES:
        chunk = min(chunk_size, WITHIN_FAMILY_PERMUTATION_REPLICATES - complete)
        numerator = np.zeros(chunk)
        ranked_ss = np.zeros(chunk)
        exact_ss = np.zeros(chunk)
        for family in identifying_families:
            family_group = group[group.family == family]
            lengths = family_group.off_output_tokens.to_numpy(dtype=float)
            exact = family_group.byte_equal.to_numpy(dtype=float)
            ranked = rankdata(lengths, method="average") / (len(lengths) + 1)
            ranked -= ranked.mean()
            permutations = np.argsort(rng.random((chunk, len(exact))), axis=1)
            permuted_exact = exact[permutations]
            permuted_exact -= permuted_exact.mean(axis=1, keepdims=True)
            numerator += permuted_exact @ ranked
            ranked_ss += np.sum(ranked ** 2)
            exact_ss += np.sum(permuted_exact ** 2, axis=1)
        correlations = numerator / np.sqrt(ranked_ss * exact_ss)
        extreme += int(np.sum(np.abs(correlations) >= abs(observed)))
        complete += chunk
    return (extreme + 1) / (WITHIN_FAMILY_PERMUTATION_REPLICATES + 1)


def within_family_length_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = load_quality_pairs_with_lengths()
    family_rows: list[dict] = []
    stratified_rows: list[dict] = []

    for model_index, (model, conditions) in enumerate(QUALITY_CONDITIONS.items()):
        for condition_index, condition in enumerate(conditions):
            group = pairs[(pairs.model_short == model) & (pairs.condition == condition)].copy()
            for family_index, (family, family_group) in enumerate(group.groupby("family", sort=True)):
                lengths = family_group.off_output_tokens.to_numpy(dtype=int)
                exact = family_group.byte_equal.to_numpy(dtype=bool)
                identifiable = len(np.unique(lengths)) > 1 and len(np.unique(exact)) > 1
                if identifiable:
                    rho, p_value = spearmanr(lengths, exact.astype(int))
                    rng = np.random.default_rng(
                        SEED + 10_000 + model_index * 1_000 + condition_index * 100 + family_index
                    )
                    ci_low, ci_high, valid_bootstraps = bootstrap_spearman_ci(lengths, exact, rng)
                else:
                    rho, p_value = np.nan, np.nan
                    ci_low, ci_high, valid_bootstraps = np.nan, np.nan, 0
                family_rows.append({
                    "model_short": model,
                    "condition": condition,
                    "family": family,
                    "n": int(len(family_group)),
                    "exact_matches": int(exact.sum()),
                    "exact_match_rate": float(exact.mean()),
                    "min_off_tokens": int(lengths.min()),
                    "median_off_tokens": float(np.median(lengths)),
                    "max_off_tokens": int(lengths.max()),
                    "median_off_tokens_exact": (
                        float(np.median(lengths[exact])) if exact.any() else np.nan
                    ),
                    "median_off_tokens_divergent": (
                        float(np.median(lengths[~exact])) if (~exact).any() else np.nan
                    ),
                    "spearman_rho": float(rho),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "spearman_p_unadjusted": float(p_value),
                    "bootstrap_replicates": WITHIN_FAMILY_BOOTSTRAP_REPLICATES,
                    "valid_bootstrap_replicates": valid_bootstraps,
                    "identifiable_within_family": identifiable,
                })

            observed, identifying_families, n_informative = within_family_residual_correlation(group)
            rng_boot = np.random.default_rng(
                SEED + 20_000 + model_index * 1_000 + condition_index * 100
            )
            ci_low, ci_high, valid_bootstraps = stratified_bootstrap_correlation(
                group, identifying_families, rng_boot
            )
            rng_permutation = np.random.default_rng(
                SEED + 30_000 + model_index * 1_000 + condition_index * 100
            )
            permutation_p = stratified_permutation_p(
                group, identifying_families, observed, rng_permutation
            )
            stratified_rows.append({
                "model_short": model,
                "condition": condition,
                "identifying_families": ";".join(identifying_families),
                "n_informative": n_informative,
                "within_family_rank_correlation": observed,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "permutation_p_unadjusted": permutation_p,
                "bootstrap_replicates": WITHIN_FAMILY_BOOTSTRAP_REPLICATES,
                "valid_bootstrap_replicates": valid_bootstraps,
                "permutation_replicates": WITHIN_FAMILY_PERMUTATION_REPLICATES,
            })

    family_frame = pd.DataFrame(family_rows)
    family_frame["spearman_p_bh_all_identifiable"] = bh_adjust(
        family_frame.spearman_p_unadjusted.tolist()
    )
    stratified_frame = pd.DataFrame(stratified_rows)
    stratified_frame["permutation_p_bh_four_conditions"] = bh_adjust(
        stratified_frame.permutation_p_unadjusted.tolist()
    )
    return family_frame, stratified_frame


def depth_rank_analysis() -> tuple[pd.DataFrame, list[dict]]:
    runs = pd.read_csv(PROCESSED / "performance_runs.csv")
    rng = np.random.default_rng(SEED)
    probability_rows: list[dict] = []
    contrasts: list[dict] = []

    for model in ("12b", "26b"):
        group = runs[(runs.model_short == model) & (runs.n_max > 0)].copy()
        pivot = group.pivot(index="n_max", columns="prompt_id", values="decode_ms").sort_index()
        if pivot.isna().any().any() or pivot.shape[1] != 9:
            raise RuntimeError(f"Expected complete 9-prompt depth grid for {model}, got {pivot.shape}")
        depths = pivot.index.to_numpy(dtype=int)
        decode_ms = pivot.to_numpy(dtype=float)
        samples = rng.integers(0, pivot.shape[1], size=(BOOTSTRAP_REPLICATES, pivot.shape[1]))
        sampled_total_ms = decode_ms[:, samples].sum(axis=2)
        sampled_rates = 256.0 * pivot.shape[1] * 1000.0 / sampled_total_ms
        winner_index = np.argmax(sampled_rates, axis=0)
        for index, depth in enumerate(depths):
            probability_rows.append({
                "model_short": model,
                "n_max": int(depth),
                "bootstrap_probability_fastest": float(np.mean(winner_index == index)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })

        aggregate_rates = 256.0 * pivot.shape[1] * 1000.0 / decode_ms.sum(axis=1)
        order = np.argsort(aggregate_rates)[::-1]
        best_i, runner_i = int(order[0]), int(order[1])
        best_depth, runner_depth = int(depths[best_i]), int(depths[runner_i])
        prompt_ratios = decode_ms[runner_i] / decode_ms[best_i]
        sampled_ratios = np.median(prompt_ratios[samples], axis=1)
        contrasts.append({
            "model_short": model,
            "aggregate_best_n": best_depth,
            "aggregate_runner_up_n": runner_depth,
            "paired_median_speed_ratio_best_over_runner": float(np.median(prompt_ratios)),
            "bootstrap_ci95_low": float(np.quantile(sampled_ratios, 0.025)),
            "bootstrap_ci95_high": float(np.quantile(sampled_ratios, 0.975)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "matched performance prompt",
        })

    return pd.DataFrame(probability_rows), contrasts


def performance_plateau_analysis() -> pd.DataFrame:
    perf = pd.read_csv(PROCESSED / "performance_summary.csv")
    quality = pd.read_csv(PROCESSED / "quality_summary.csv")
    group = perf[(perf.model_short == "12b") & (perf.n_max > 0)].copy()
    maximum = float(group.aggregate_decode_tok_s.max())
    group["share_of_observed_maximum"] = group.aggregate_decode_tok_s / maximum
    plateau = group[group.share_of_observed_maximum >= 0.95].copy()
    evaluated_conditions = set(
        quality[(quality.model_short == "12b") & (quality.family == "macro")]
        .condition.str.replace("n", "", regex=False)
        .loc[lambda series: series.str.fullmatch(r"\d+")]
        .astype(int)
    )
    plateau["objective_quality_200_items_evaluated"] = plateau.n_max.isin(evaluated_conditions)
    plateau["strict_conservative_rule_eligible"] = plateau.cv_decode_tok_s <= 0.10
    plateau["interpretation"] = plateau.apply(
        lambda row: (
            "observed maximum; quality evaluated"
            if int(row.n_max) == 6
            else "performance-only plateau candidate; 200-item quality not evaluated"
        ),
        axis=1,
    )
    return plateau[[
        "model_short",
        "n_max",
        "aggregate_decode_tok_s",
        "share_of_observed_maximum",
        "median_paired_speedup",
        "cv_decode_tok_s",
        "acceptance_rate",
        "objective_quality_200_items_evaluated",
        "strict_conservative_rule_eligible",
        "interpretation",
    ]].sort_values("n_max")


def make_figures(bin_frame: pd.DataFrame, rank_frame: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    x = np.arange(len(LENGTH_BINS))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for model, conditions in QUALITY_CONDITIONS.items():
        for condition in conditions:
            group = bin_frame[(bin_frame.model_short == model) & (bin_frame.condition == condition)]
            ax.plot(
                x,
                100 * group.exact_match_rate,
                marker="o",
                color=COLORS[model],
                linestyle=LINESTYLES[condition],
                label=f"{model.upper()} {condition}",
            )
    ax.set(
        xlabel="OFF response length (tokens; post-hoc bins)",
        ylabel="Exact byte-match rate (%)",
        ylim=(-3, 103),
        xticks=x,
        xticklabels=[item[2] for item in LENGTH_BINS],
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_length_vs_exact_match.png", bbox_inches="tight")
    fig.savefig(FIGURES / "09_length_vs_exact_match.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    for ax, model in zip(axes, ("12b", "26b")):
        group = rank_frame[rank_frame.model_short == model].sort_values("n_max")
        ax.bar(group.n_max.astype(str), 100 * group.bootstrap_probability_fastest, color=COLORS[model])
        ax.set_title(model.upper())
        ax.set_xlabel("MTP depth N")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Bootstrap probability of fastest aggregate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_depth_rank_uncertainty.png", bbox_inches="tight")
    fig.savefig(FIGURES / "10_depth_rank_uncertainty.svg", bbox_inches="tight")
    plt.close(fig)


def make_within_family_figure(family_frame: pd.DataFrame) -> None:
    selected = family_frame[
        (family_frame.condition == "n6") & family_frame.identifiable_within_family
    ].copy()
    family_order = ["gsm8k", "ifeval", "mbpp"]
    selected["family_order"] = selected.family.map({name: i for i, name in enumerate(family_order)})
    selected = selected.sort_values(["family_order", "model_short"], ascending=[True, False])
    selected["label"] = selected.apply(
        lambda row: f"{row.family.upper()} — {row.model_short.upper()} (n={int(row.n)})", axis=1
    )

    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for index, row in enumerate(selected.itertuples(index=False)):
        ax.errorbar(
            row.spearman_rho,
            index,
            xerr=[[row.spearman_rho - row.bootstrap_ci95_low],
                  [row.bootstrap_ci95_high - row.spearman_rho]],
            fmt="o",
            color=COLORS[row.model_short],
            capsize=3,
        )
    ax.axvline(0, color="0.5", lw=1)
    ax.set(
        xlabel="Within-family Spearman correlation: OFF length vs exact equality",
        yticks=y,
        yticklabels=selected.label,
        xlim=(-1.0, 0.25),
    )
    ax.invert_yaxis()
    ax.text(
        0.01,
        -0.18,
        "Points = N=6 estimates; bars = 95% pair-bootstrap CI (20,000 resamples). "
        "MMLU-Pro omitted: OFF length is constant.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "11_within_family_length_association.png", bbox_inches="tight")
    fig.savefig(FIGURES / "11_within_family_length_association.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    bin_frame, survival_frame, correlations = quality_length_analysis()
    family_frame, stratified_frame = within_family_length_analysis()
    rank_frame, contrasts = depth_rank_analysis()
    plateau_frame = performance_plateau_analysis()

    bin_frame.to_csv(PROCESSED / "length_divergence_by_bin.csv", index=False)
    survival_frame.to_csv(PROCESSED / "divergence_survival.csv", index=False)
    rank_frame.to_csv(PROCESSED / "depth_rank_uncertainty.csv", index=False)
    family_frame.to_csv(PROCESSED / "length_divergence_within_family.csv", index=False)
    stratified_frame.to_csv(PROCESSED / "length_divergence_stratified.csv", index=False)
    plateau_frame.to_csv(PROCESSED / "12b_performance_plateau.csv", index=False)
    payload = {
        "analysis_status": "post-review exploratory; not in the frozen confirmatory protocol",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "length_analysis": {
            "baseline_length_definition": "number of token IDs in the OFF-A response",
            "important_caveat": (
                "Response length is strongly confounded with task family and output form; "
                "natural response endings also make censoring potentially informative."
            ),
            "correlations": correlations,
            "within_family_analysis": {
                "status": "second-round post-review exploratory analysis",
                "family_specific": family_frame.to_dict(orient="records"),
                "task_stratified": stratified_frame.to_dict(orient="records"),
                "interpretation_boundary": (
                    "Within-family association reduces between-family task-form confounding "
                    "but does not randomize response length or isolate a causal per-token hazard."
                ),
            },
        },
        "depth_rank_analysis": {
            "important_caveat": (
                "Prompt bootstrap estimates workload-sampling uncertainty only; it does not "
                "represent repeated-run, thermal, or system-state variability."
            ),
            "top_two_contrasts": contrasts,
            "12b_observed_performance_plateau": plateau_frame.to_dict(orient="records"),
            "plateau_interpretation_boundary": (
                "N=8 and N=12 were not run on the 200-item objective-quality suite; "
                "the plateau is a performance description, not a quality-constrained equivalence set."
            ),
        },
    }
    (PROCESSED / "review_followup_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")

    markdown_table(bin_frame, TABLES / "length_divergence_by_bin.md")
    markdown_table(rank_frame, TABLES / "depth_rank_uncertainty.md")
    markdown_table(family_frame, TABLES / "length_divergence_within_family.md")
    markdown_table(stratified_frame, TABLES / "length_divergence_stratified.md")
    markdown_table(plateau_frame, TABLES / "12b_performance_plateau.md")
    make_figures(bin_frame, rank_frame)
    make_within_family_figure(family_frame)

    for row in correlations:
        print(
            f"{row['model_short']} {row['condition']}: exact={row['exact_matches']}/{row['n']}, "
            f"rho={row['spearman_rho_length_vs_exact']:.3f}, "
            f"p={row['spearman_p_unadjusted']:.3g}"
        )
    for row in contrasts:
        print(
            f"{row['model_short']}: best n={row['aggregate_best_n']} vs "
            f"runner n={row['aggregate_runner_up_n']}, median ratio="
            f"{row['paired_median_speed_ratio_best_over_runner']:.3f}, "
            f"CI [{row['bootstrap_ci95_low']:.3f}, {row['bootstrap_ci95_high']:.3f}]"
        )
    for row in stratified_frame.itertuples(index=False):
        print(
            f"{row.model_short} {row.condition} within-family: "
            f"r={row.within_family_rank_correlation:.3f}, "
            f"CI [{row.bootstrap_ci95_low:.3f}, {row.bootstrap_ci95_high:.3f}], "
            f"permutation p={row.permutation_p_unadjusted:.3g}, "
            f"BH p={row.permutation_p_bh_four_conditions:.3g}"
        )
    print("12b observed >=95% performance plateau:")
    print(plateau_frame.to_string(index=False))


if __name__ == "__main__":
    main()
