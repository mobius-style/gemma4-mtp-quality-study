# Deviations from the frozen protocol

## 2026-08-09 — conservative-condition fallback

The frozen conservative selection rule required the smallest MTP depth within
95% of maximum aggregate throughput and request-level CV <=10%. No tested depth
met both requirements for either model: the maximum was n=6 for both models;
the nearest low-variance candidates were n=3 for 12B and n=4 for 26B. Before
examining any quality outputs, the quality pilot therefore retained:

- the pre-specified maximum-throughput point (n=6), and
- the highest-throughput condition with CV <=10% (12B n=3; 26B n=4).

This fallback weakens the label “conservative deployment point”: it is a
low-variance comparison candidate, not the frozen 95%-of-maximum conservative
winner. The paper reports the original rule as returning no eligible setting
and labels the fallback exploratory. No quality result was available when this
decision was made.

## 2026-08-09 — output-budget correction after quality smoke test

The four-family smoke test showed that the first MMLU-Pro response reached the
initial 384-token cap before emitting its required final option label. Counting
that infrastructure truncation as a wrong answer would confound quality with an
arbitrary generation ceiling. Before running the full suite, prompt text and
selected items were held fixed while maximum output budgets were changed to
512 for GSM8K, 1024 for MMLU-Pro, 768 for IFEval, and 512 for MBPP. The original
v1 prompt file and eight 12B smoke outputs remain preserved; the full study uses
new `quality_items_v2.jsonl` and `*_v2.jsonl` files. This budget correction was
made because of finish reason `length`, not because of correctness results.

The first v2 MMLU-Pro smoke response still reached 1024 tokens because the model
repeated arithmetic rather than obeying “reason briefly.” Before any full-suite
run, the MMLU-Pro prompt was therefore changed to require exactly one
`Final answer: X` line with no explanation and its cap was reduced to 128. The
item identities, option order, answers, and all non-MMLU prompts stayed fixed.
The v1/v2 prompts and smoke outputs remain preserved; the full analysis uses
the explicitly versioned v3 files. This is a harness-validity correction, and
the study remains exploratory rather than externally preregistered.

The v3 smoke log then showed that server-level `--reasoning off` still supplied
the Gemma chat template with an unrestricted thought channel
(`reasoning budget: tokens=-1`). The request payload was corrected to set the
runtime's documented `reasoning_budget_tokens=0`; the full run is stored as
`*_v4.jsonl`. This implements the intended, already stated non-thinking quality
condition rather than changing the task or sampling policy.

Because `reasoning-format=none` leaves Gemma channel markers in
`message.content`, the parser still exposed the forced thought prefix even with
a zero budget. The full-run server therefore uses `reasoning-format=deepseek`,
which separates any reasoning field from final `message.content`; this is the
runtime's documented parsing mode. The final analysis input is `*_v5.jsonl`.

The v5 MMLU smoke still used the entire cap for an explanation despite the
direct-answer instruction. To remove this known parser/truncation confound, the
MMLU-Pro requests alone use a llama.cpp grammar constrained to exactly
`Final answer: [A-J]`. This is a constrained-choice accuracy measurement, not a
chain-of-thought MMLU-Pro reproduction, and is described as such. The full run
uses `*_v6.jsonl`; all earlier smoke files remain excluded but preserved.
