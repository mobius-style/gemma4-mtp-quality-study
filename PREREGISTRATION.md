# Timestamped analysis plan

Status: frozen before collection of the new study outputs.

## Research type and central claim

This is an empirical systems study. Its central question is whether Gemma 4
MTP in this fixed quantized runtime provides a useful throughput gain without a
meaningful loss on a paired objective task suite, while separately testing
whether ordinary greedy output is reproduced exactly.

The study distinguishes algorithmic losslessness, deterministic behavioral
equivalence, and empirical task-quality equivalence. Evidence for one is not
treated as evidence for another.

## Scope and units

- Models: Gemma 4 12B IT QAT Q4_0 dense and Gemma 4 26B-A4B IT QAT Q4_0 MoE.
- Runtime: fixed local `llama.cpp` commit and binary recorded in
  `environment.json`.
- Hardware: one RTX 5070 Ti, concurrency one, text-only.
- Main sampling unit for quality: prompt/task item, not generated token.
- Performance sampling unit: one request, blocked by model and input-length
  class.

## Research questions and falsifiers

1. RQ1 acceleration: MTP is practically useful only if a tested setting raises
   paired median decode throughput by at least 20% over OFF without increasing
   failure rate. An observed gain below 20% falsifies practical acceleration
   for that model and workload.
2. RQ2 depth: the maximum-throughput point is the measured N with the highest
   aggregate decode rate. A conservative point is the smallest N within 95%
   of that maximum with request-level CV no greater than 10%. No untested N is
   called optimal.
3. RQ3 deterministic equivalence: equivalence requires 100% byte equality and
   100% target-token equality against a reproducible OFF baseline. One verified
   mismatch falsifies exact equivalence for the tested stack.
4. RQ4 objective quality: the primary non-inferiority margin is -5 macro
   percentage points across four task families. Non-inferiority passes only if
   the paired-bootstrap 95% lower confidence bound is greater than -0.05.
5. RQ5 model association: dense-vs-MoE differences are reported as
   model-level associations. No architecture-causal claim is permitted.

## Phase 1: N-only performance sweep

- Conditions: OFF and `spec-draft-n-max` in {1,2,3,4,6,8,12,16}.
- `spec-draft-p-min=0` for all MTP conditions.
- Input-length targets: 256, 1024, and 4000 tokens.
- Three distinct fixed prompts per length and condition; 256 requested output
  tokens with EOS ignored.
- Temperature 0, top-k 1, fixed per-prompt seeds, context 8192, parallel 1,
  Flash Attention on, F16 target/draft KV, batch/ubatch 512, target and draft
  fully requested on GPU.
- Condition order is deterministically randomized with seed 20260809 and
  recorded. One warm-up request per server start is excluded.
- Primary performance endpoint: paired median decode-tokens/s ratio to OFF.
- Secondary endpoints: aggregate decode-tokens/s, prompt-tokens/s, wall
  latency, streaming time-to-first-content, acceptance, proposal count,
  accepted count, and run-to-run spread.
- Failed runs remain in raw data. No outlier is excluded from the primary
  summary. A run is invalid only for nonzero exit, empty output, parameter/hash
  drift, or unexpected GPU placement.

## Phase 2 and 3: held-out equivalence and objective quality pilot

The fixed sample contains 200 items selected deterministically without model
outputs: 60 GSM8K, 60 MMLU-Pro, 40 IFEval, and 40 MBPP items. Sampling seed is
20260809. Public test/validation splits and exact source revisions/hashes are
recorded when acquired.

For each model, quality conditions are selected by this rule after Phase 1:

1. throughput condition: maximum aggregate decode tokens/s across all length
   blocks;
2. conservative condition: smallest N within 95% of the maximum and CV <=10%;
3. if both rules select the same N, retain the next smaller N by throughput
   rank so that two MTP conditions remain.

The quality prompts are not used to choose N. Each retained condition and OFF
receives identical prompts and output budgets. OFF is independently repeated
on all items to test ordinary-runtime determinism.

Objective scoring:

- GSM8K: exact normalized final numeric answer; parse failure is incorrect.
- MMLU-Pro: exact final option label; parse failure is incorrect.
- IFEval: official prompt-level strict and loose instruction checks.
- MBPP: official tests executed with time and memory limits inside a
  network-isolated Bubblewrap sandbox; syntax/timeout/sandbox failure is
  incorrect.
- Primary quality endpoint: unweighted macro-average of the four task-family
  scores.
- Secondary endpoints: family scores, paired regressions, paired improvements,
  net paired change, McNemar exact test, and parse-failure rate.
- Confidence intervals: prompt-level paired bootstrap with 10,000 resamples,
  seed 20260809. Secondary paired tests use Holm correction within each model.

The fixed N=200 is a resource-bounded pilot. It is not justified as having 80%
power for arbitrary small effects. The non-inferiority conclusion is allowed
only when the observed paired CI meets the frozen -5 point rule; otherwise the
finding is explicitly underpowered/inconclusive. No post-hoc power is used.

## Equivalence metrics

For every OFF-vs-MTP pair record byte equality, target-token equality, common
token-prefix length, first divergent target-token index, normalized character
edit similarity, output-length difference, and semantic task score. A sequence
with one early divergence is one divergent sequence, not hundreds of errors.

## Optional p-min phase

Run only if at least one N>=6 reaches 90% of the maximum N-only throughput for
either model. For the two best eligible higher N values, test p-min in
{0.25,0.50,0.75,0.90}; do not alter the N-only results. These results remain
exploratory and cannot redefine the frozen primary endpoint.

## Reliability, contamination, and ethics gates

- No subjective human or LLM judge is used, so inter-rater kappa is not an
  applicable primary gate. Deterministic scorers are unit-tested against known
  pass/fail fixtures before scoring model outputs.
- Public benchmark contamination cannot be ruled out from open model weights.
  Therefore absolute capability/SOTA claims are prohibited. The paired
  intervention comparison is the estimand; dataset source dates, known canary
  availability, exact-phrase diagnostics, and answer/option perturbation tests
  are reported where feasible.
- No human participants, private logs, or personal data are collected. The
  paper must state that no IRB-regulated human-subject data were used and must
  disclose AI assistance in software, analysis, and drafting.

## Stopping and deviations

The fixed study stops after all planned valid runs or after a documented hard
failure that makes a phase infeasible. Results are not inspected to increase N
until favorable. Every implementation or protocol deviation is appended with
timestamp, reason, and effect to `analysis/DEVIATIONS.md`; it cannot be silently
folded into confirmatory wording.

## Pre-specified rejection conditions and reviewer attacks

- Reject exact-equivalence if any verified MTP pair diverges while its two OFF
  controls match.
- Do not claim quality non-inferiority unless its paired CI clears -5 points.
- Do not call an N optimal outside the tested grid and workloads.
- Painful reviewer attack 1: one GPU/runtime/quantization cannot establish a
  general speculative-decoding property. The claim is restricted to this
  stack.
- Painful reviewer attack 2: public tasks may be contaminated and N=200 may be
  underpowered. Absolute model-quality claims are prohibited and uncertainty
  is reported.
