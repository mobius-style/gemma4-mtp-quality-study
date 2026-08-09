# Claims matrix

Labels: **Supported** means supported only within the frozen local scope;
**Partially supported** means evidence is directionally relevant but incomplete;
**Unsupported** means the observed data reject or do not establish the claim;
**Requires replication** means a new condition is necessary.

| Candidate claim | Status | Exact evidence / boundary |
|---|---|---|
| MTP accelerates 12B on this stack | Supported | N=6 aggregate 226.39 vs OFF 89.20 tok/s; paired median 2.497x, 95% CI [2.266,3.054]; 81/81 performance runs valid |
| MTP accelerates 26B-A4B on this stack | Supported | N=6 aggregate 322.53 vs 169.58 tok/s; paired median 1.860x, CI [1.701,2.202]; 81/81 valid |
| N=6 is the fastest observed depth in the fixed grid | Supported | Maximum aggregate rate over N={1,2,3,4,6,8,12,16} for both models |
| N=6 is reliably faster than neighboring depths in the workload population | Partially supported | Prompt bootstrap: N=6 fastest in 51.3% of 12B and 93.9% of 26B-A4B resamples; best/runner paired-ratio intervals include 1; bootstrap omits fresh-run variability |
| 12B has an observed >=95%-of-maximum performance plateau at N={4,6,8,12} | Supported as post-review description | Aggregate rates are 221.69, 226.39, 220.67, and 223.30 tok/s, all within 2.6% of the observed maximum |
| All 12B plateau depths are quality-equivalent or interchangeable | Unsupported | Only N=6 received the paired 200-item objective-quality run; N=4/N=8/N=12 cannot inherit its non-inferiority result |
| N=6 is universally optimal | Unsupported | Only one GPU/runtime/quantization/workload/grid was tested |
| MTP preserves exact ordinary greedy output | Unsupported | N=6 byte/token equality 112/200 (12B), 118/200 overall and 117/199 stable subset (26B) |
| Divergence is associated with response length in this suite | Supported as post-review association | N=6 exact match is 60/61 and 61/61 at 1--8 OFF tokens, versus 24/98 and 22/88 at 129+; Spearman rho=-0.703 for both models |
| The negative length association remains within variable-length task families | Supported as second-review exploratory association | N=6 rho is -0.329/-0.566/-0.701 for 12B and -0.321/-0.513/-0.643 for 26B-A4B in GSM8K/IFEval/MBPP; all six bootstrap CIs exclude 0; MMLU-Pro has no length variation |
| Between-family task composition fully explains the length association | Unsupported | Within-family centered-rank correlations are -0.490 [-0.610,-0.343] for 12B and -0.450 [-0.581,-0.297] for 26B-A4B at N=6 |
| Short outputs are generally behaviorally identical under MTP | Unsupported | Short bins are dominated by constrained-answer tasks; task family, output form, and length are confounded |
| Response length causally increases divergence or follows an independent constant per-token probability | Unsupported | Length was not randomized; within-family complexity, response form, stopping, truncation, mechanism, and censoring remain unresolved |
| Divergence is only ordinary-runtime nondeterminism | Unsupported | OFF repeat exact 200/200 for 12B and 199/200 for 26B; MTP divergence is far more frequent |
| MTP reduces objective quality | Unsupported | Directional macro change -0.83 pt, but 95% CIs cross zero and Holm-adjusted family p-values are all 1.0 |
| MTP is non-inferior within -5 macro points | Supported, exploratory | N=6 lower CI -4.17 pt (12B), -2.92 pt (26B), both above frozen -5 pt margin; pilot N=200 |
| MTP has exactly identical semantic quality | Unsupported | No exhaustive semantic test; objective suite is limited and exact strings often differ |
| N=6 is the quality-constrained fastest observed point | Supported, exploratory | N=6 is the grid maximum and clears the frozen paired non-inferiority rule; near-depth population ordering remains uncertain |
| A strict conservative deployment point was found | Unsupported | No N simultaneously reached 95% of maximum and CV≤10% |
| N=3/N=4 are lower-variance alternatives | Supported | 12B N=3 CV 7.01%; 26B N=4 CV 9.83%; both quality comparisons clear the margin |
| 12B receives a larger relative speedup than 26B-A4B | Supported as association | N=6 paired medians 2.497x vs 1.860x under this test |
| MoE causes the smaller relative speedup | Unsupported | Architecture is confounded with size, training, active parameters, baseline speed, and kernels; the faster OFF baseline is a hypothesis, not an isolated mechanism |
| 26B-A4B has higher N=6 acceptance | Supported as association | 70.56% vs 57.69% accepted/proposed |
| p-min materially improves the recommendation | Partially supported | Best +1.70% (12B) and +2.47% (26B), but small vs variation and no paired quality run |
| 26B N=6 p-min=.50 is ready for deployment | Requires replication | 330.50 tok/s exploratory maximum; quality and independent performance confirmation absent |
| Quantization causes the exact-output divergence | Requires replication | Q4_0 was tested; no matched BF16/higher-precision control or kernel trace |
| Results generalize to RTX 3090 or other GPUs | Requires replication | Only RTX 5070 Ti measured |
| Results generalize to Ollama | Requires replication | Ollama inference was explicitly not used |
| Results generalize to production concurrency | Requires replication | `parallel=1`; no queueing or multi-user load |
| The models have uncontaminated benchmark scores | Unsupported | Public datasets; no training-corpus access/canaries; contamination status unresolved |
| There was no measurable task-family effect | Supported within pilot | All exact McNemar tests Holm-adjusted to p=1.0; this is absence of detected effect, not proof of equality |
| The work is a confirmatory trial | Unsupported | Prospectively frozen local exploratory plan informed by preliminary data; harness corrections disclosed |
| The study is reproducible on the same artifacts | Supported | Pinned hashes, prompts, scripts, raw JSONL, logs, processed outputs, manifests, validation report |
