# Data dictionary

## Raw evidence

- `raw/performance/{12b,26b}.jsonl`: one row per fixed performance request;
  includes run key, prompt/condition identifiers, token counts, timings,
  output, output hash, draft proposal/accept counts, launch command, and status.
- `raw/quality/*_v6.jsonl`: analyzed quality responses. Earlier versions are
  preserved smoke/harness iterations and are not inputs to final scoring.
- `raw/pmin/{12b,26b}.jsonl`: exploratory p-min requests; p=0 references come
  from the primary performance files.
- `raw/negative_control/{12b,26b}.jsonl`: rotated-option MMLU controls.
- `logs/`: full `llama-server` stdout/stderr by phase and condition.

## Key processed fields

| Field | Meaning |
|---|---|
| `aggregate_decode_tok_s` | Sum output tokens divided by sum decode seconds |
| `mean_decode_tok_s` | Arithmetic mean of request decode rates |
| `median_paired_speedup` | Median, over matching prompts, of MTP rate / OFF rate |
| `cv_decode_tok_s` | Population SD divided by mean across nine requests |
| `acceptance_rate` | Total accepted draft tokens / total proposed draft tokens |
| `byte_equal_rate` | Share whose UTF-8 output strings equal OFF exactly |
| `token_equal_rate` | Share whose target-token ID sequences equal OFF exactly |
| `first_divergent_token` | Zero-based common-prefix length when token arrays differ |
| `normalized_char_edit_similarity` | RapidFuzz normalized Levenshtein similarity, 0–1 |
| `accuracy` | Deterministic objective score proportion; macro is unweighted across four families |
| `regressions` | OFF correct and MTP incorrect paired items |
| `improvements` | OFF incorrect and MTP correct paired items |
| `paired_bootstrap_ci_*` | 2.5% and 97.5% quantiles from task-stratified paired bootstrap |
| `noninferiority_pass` | True only when CI lower bound is greater than -0.05 |
| `off_output_token_bin` | Post-review bin of the natural OFF-A response token-ID count |
| `descriptive_equal_survival` | Descriptive no-divergence survival estimate; natural endings are censored and may be informative |
| `at_risk_at_horizon` | Pairs whose OFF response continues to the stated token horizon |
| `bootstrap_probability_fastest` | Share of 100,000 matched-prompt resamples in which a depth has the highest aggregate rate |
| `within_family_rank_correlation` | Correlation after fractional-ranking OFF length and centering both length rank and equality within each variable-length task family |
| `spearman_p_bh_all_identifiable` | BH-adjusted exploratory p-value across the 12 family comparisons with both length and equality variation |
| `permutation_p_bh_four_conditions` | BH-adjusted task-stratified Monte Carlo p-value across four retained model/condition comparisons |
| `share_of_observed_maximum` | Aggregate decode rate divided by the maximum rate in the tested positive-N grid |

## Post-review exploratory outputs

- `processed/length_divergence_by_bin.csv`: exact-match counts by model,
  retained condition, and natural OFF-output-length bin.
- `processed/divergence_survival.csv`: descriptive no-divergence estimates at
  fixed token horizons. This is not a causal per-token hazard estimate.
- `processed/depth_rank_uncertainty.csv`: prompt-bootstrap probability that
  each positive N is fastest.
- `processed/review_followup_analysis.json`: seeds, top-two depth contrasts,
  Spearman associations, and mandatory interpretation caveats.
- `processed/length_divergence_within_family.csv`: family-specific correlations,
  pair-bootstrap intervals, and BH-adjusted exploratory p-values.
- `processed/length_divergence_stratified.csv`: pooled within-family centered-rank
  correlations, stratified-bootstrap intervals, and within-family permutation tests.
- `processed/12b_performance_plateau.csv`: 12B depths at least 95% as fast as
  the observed maximum, with CV and objective-quality coverage flags.

All time fields ending in `_ms` are milliseconds. Rates are decimal
proportions unless the presentation table explicitly multiplies by 100.
