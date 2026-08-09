# Research-methodology gate report

Final gate date: 2026-08-09. Verdict: **PASS for a scoped exploratory working
paper; HOLD for universal, causal, or exact-quality claims.**

## Gate 1 — Study freeze and deviations: PASS

`PREREGISTRATION.md` and `configs/protocol.json` were hashed before new model
outputs. `sha256sum -c FREEZE.sha256` passes. Selection fallback and all
quality-harness corrections are timestamped in `analysis/DEVIATIONS.md`; older
smoke raw files remain preserved and excluded from the analyzed v6 dataset.

## Gate 2 — Claim status: PASS

- Descriptive: throughput, latency, acceptance, equality, and score estimates.
- Inferential: paired bootstrap non-inferiority and exact McNemar/Holm tests.
- Exploratory: low-variance fallback, p-min sweep, option perturbation,
  post-review depth ranking, global/within-family length association, and the
  12B performance-only plateau.
- Prohibited: universal optimality, architecture causality, uncontaminated
  absolute capability, BF16 generalization, and semantic identity.

The manuscript and `CLAIMS_MATRIX.md` retain these labels.

## Gate 3 — Quantitative inference: PASS with pilot limitation

Effect sizes, 95% confidence intervals, paired regression/improvement counts,
exact McNemar tests, and within-model Holm correction are present. The -5
macro-point non-inferiority margin and 10,000-resample paired bootstrap were
frozen before quality outputs. N=200 was not claimed to have pre-established
power for small effects.

## Gate 4 — Scorer reliability: PASS / not applicable for raters

No human or LLM subjective judge was used. Deterministic scorers passed eight
known pass/fail fixtures before full scoring. The official IFEval code was
pinned; MBPP execution was sandboxed. Inter-rater kappa is not applicable.

## Gate 5 — Contamination: LIMITED, explicitly bounded

The fixed sample uses public benchmarks. Exact duplicates, shared normalized
13-grams, a GSM reference-number diagnostic, and MMLU option-order perturbation
were evaluated. There are no canaries or training-corpus access, so
contamination remains unresolved and absolute capability/SOTA claims are
blocked. The estimand is the paired MTP intervention.

## Gate 6 — Falsifiers and negative controls: PASS

- RQ1 falsifier: speedup below 1.20x. Both models exceed it.
- RQ3 falsifier: one stable-baseline MTP mismatch. Many occur; exact
  equivalence is rejected.
- RQ4 rejection condition: lower 95% CI at or below -5 points. All retained
  conditions clear it, with exploratory status.
- Negative control: independent OFF repeat (12B 200/200 stable; 26B 199/200).
- Perturbation control: 20 rotated-option MMLU items per model under OFF/N=6.

## Painful reviewer attacks

1. **“One GPU, one commit, and Q4_0 cannot establish a general MTP property.”**
   Accepted. Every conclusion is scoped to this stack; higher precision,
   another runtime commit, another GPU, and concurrency are replication tasks.
2. **“Public tasks may be contaminated and 200 items cannot exclude small
   regressions.”** Accepted. The manuscript prohibits absolute quality claims,
   reports uncertainty, and treats non-inferiority as exploratory.
3. **“Thermal drift may explain nearby performance differences.”** Partly
   unresolved because NVML was unavailable. Randomized condition order,
   warm-up exclusion, paired prompts, and CV reporting mitigate but do not
   eliminate this threat. N=6's large OFF-relative gain is robust to this
   concern; small p-min gains are not promoted.
4. **“The apparent length effect may only reflect task-family composition.”**
   Partly resolved. Negative associations remain separately within GSM8K,
   IFEval, and MBPP, and task-stratified pooled CIs exclude zero. Length is not
   randomized, so within-family complexity, response structure, stopping, and
   truncation continue to block a causal per-token claim.
5. **“12B N=6 is not a unique performance optimum.”** Accepted. N={4,6,8,12}
   are all within the frozen 95%-of-maximum speed threshold. Only N=6 has the
   paired 200-item quality evidence, so the set is a performance-tuning band,
   not a quality-equivalent deployment set.

## Final submission boundary

The artifact is ready as an independent working-paper draft and reproducible
local evidence package. Public scholarly submission remains contingent on a
human author supplying authorship metadata, verifying venue/license rules, and
accepting accountability. Claims of “same output,” “all quantizations,”
“caused by MoE,” or “universally optimal” remain HOLD.
