# Working-paper outline populated from the completed study

1. **Abstract** — report N=6 maxima, paired speedups, 56–59% exact-match rates,
   objective score changes and CIs, within-family length association, 12B
   performance plateau, and scope.
2. **Introduction** — separate throughput from behavioral and task-level
   equivalence; motivate quantized consumer-GPU validation.
3. **Background** — autoregressive decoding, speculative verification, MTP,
   finite-precision execution, and the three meanings of lossless.
4. **Related work** — Leviathan, Chen, Gloeckle, Gemma 4 report/model card,
   `llama.cpp`; source audit in `LITERATURE_SEARCH.md`.
5. **Research questions** — RQ1 acceleration, RQ2 depth, RQ3 exactness, RQ4
   objective non-inferiority, RQ5 model-level association.
6. **Experimental setup** — RTX 5070 Ti, pinned commit/hashes, QAT Q4_0,
   context 8192, concurrency one, N grid, randomized order, warm-up exclusion.
7. **Performance** — 12B 89.20→226.39 tok/s; 26B 169.58→322.53; both observed
   N=6 maxima; latency, TTFC, acceptance, CV, prompt-bootstrap rank uncertainty,
   and the performance-only N={4,6,8,12} 12B plateau.
8. **Deterministic equivalence** — stable OFF repeat, 56%/59% N=6 equality,
   first divergence, edit similarity, global and within-family post-review length
   gradients; exactness falsified without asserting a causal per-token law.
9. **Task quality** — four-family N=200 paired pilot, scoring, regressions and
   improvements, bootstrap CIs, McNemar/Holm, non-inferiority interpretation.
10. **Dense vs. MoE comparison** — larger relative gain for 12B, higher
    acceptance and faster OFF baseline for 26B; systems hypothesis, explicitly non-causal.
11. **Pareto and operating points** — N=6 quality-constrained maximum; N=3/N=4
    exploratory lower-variance alternatives; no strict conservative winner.
12. **p-min** — +1.70% 12B and +2.47% 26B best exploratory gains; no quality
    rerun, no recommendation change.
13. **Negative controls / contamination** — OFF repeat, option rotation,
    n-gram diagnostics, unresolved public-task exposure.
14. **Discussion** — high value for interactive single-stream decode; OFF for
    hash-stable or audit-reproducible output.
15. **Limitations and threats** — one stack, telemetry loss, pilot size,
    permissive-margin risk, near-depth uncertainty, residual within-family
    length confounding,
    constrained MMLU, no open-ended/Japanese judge, no BF16 mechanism test.
16. **Reproducibility and ethics** — raw/log separation, hashes, sandbox,
    no human subjects, AI-assistance disclosure.
17. **Conclusion** — precise local answer, no universal “optimal/lossless” claim.
18. **Appendices** — exact commands, source revisions, deviations, evidence map.

## Explicit TODO items (future work; not required for the completed core paper)

- **TODO—Replication:** test the same pairs with a BF16 or higher-precision target.
- **TODO—Replication:** rerun on a newer pinned `llama.cpp` commit and another GPU.
- **TODO—Expansion:** preregister a larger/private objective suite for a tighter margin.
- **TODO—Expansion:** add blinded, human-calibrated Japanese and open-ended evaluation.
- **TODO—Serving study:** test high concurrency, longer contexts, power, and thermals
  after repairing NVML.
