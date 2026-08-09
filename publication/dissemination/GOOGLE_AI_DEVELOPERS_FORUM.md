# Manual publication package — Google AI Developers Forum

Status: hold until the Zenodo DOI, GitHub release, and Hugging Face Dataset are public.

## Suggested title

Gemma 4 MTP on an RTX 5070 Ti: throughput gains, frequent greedy-output divergence, and paired quality evidence

## Post body

I have published an independent quality–throughput study of QAT Q4_0 Gemma 4
12B IT and 26B-A4B IT using multi-token prediction in a pinned CUDA
`llama.cpp` build on one RTX 5070 Ti.

In the fixed tested grid, N=6 produced the highest measured aggregate decode
throughput for both models: 226.39 versus 89.20 tokens/s for 12B, and 322.53
versus 169.58 tokens/s for 26B-A4B. Exact greedy-output identity was much
weaker: 112/200 and 118/200 N=6 outputs were byte-identical to ordinary
decoding. On the paired 200-item objective suite, the macro-score change was
-0.83 percentage points for both models; both confidence intervals remained
above the prospectively frozen -5-point pilot margin.

The scope is deliberately narrow. This does not establish universal N=6
optimality, exact behavioral identity, or generalization to other GPUs,
runtimes, precisions, or serving workloads. The public artifact excludes
verbatim benchmark items and raw generated text while retaining IDs, hashes,
processed measurements, scripts, pinned sources, figures, tables, and audit
records.

- Paper and archive: https://doi.org/10.5281/zenodo.21860461
- Source and public release asset: https://github.com/mobius-style/gemma4-mtp-quality-study/releases/tag/v1.0.0
- Sanitized processed Dataset: https://huggingface.co/datasets/moebiusT7/gemma4-mtp-quality-study

Author: Taiko Toeda, Independent Researcher  
ORCID: https://orcid.org/0009-0001-7267-0201

