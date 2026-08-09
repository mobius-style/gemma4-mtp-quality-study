# Gemma 4 MTP Quality–Throughput Study

An independent paired benchmark of QAT Q4_0 Gemma 4 12B IT and 26B-A4B IT on
one RTX 5070 Ti using a pinned CUDA `llama.cpp` runtime.

Author: Taiko Toeda, Independent Researcher  
ORCID: https://orcid.org/0009-0001-7267-0201  
Version: 1.0.0  
DOI: [10.5281/zenodo.21860461](https://doi.org/10.5281/zenodo.21860461)

## Result boundary

N=6 was the fastest observed depth in the fixed tested grid for both models.
It increased aggregate decode throughput from 89.20 to 226.39 tokens/s for 12B
and from 169.58 to 322.53 tokens/s for 26B-A4B. At N=6, exact byte equality
with ordinary decoding was 112/200 and 118/200, respectively. The paired
objective macro-score change was -0.83 percentage points for both models, with
95% confidence intervals above the prospectively frozen -5-point pilot margin.

These results apply only to the measured GPU, runtime commit, quantization,
models, settings, and workloads. N=6 is an evidence-backed high-throughput
default for this stack, not a universal optimum. See `CLAIMS_MATRIX.md` before
reusing any claim.

## Public artifact boundary

This public package excludes verbatim benchmark items, prompts, raw request
records, generated output text, complete server logs, vendored third-party
source, local paths, and the private internal archive. It retains processed
measurements, item and prompt IDs, hashes, scripts, pinned sources, figures,
tables, and validation records. See
`publication/DATA_REDISTRIBUTION_AUDIT.md` and
`THIRD_PARTY_DATA_LICENSES.md`.

## Start here

- Paper source: `paper/PAPER.md`
- Final PDF: `paper/PAPER.pdf` (added only after DOI reservation)
- Claim boundaries: `CLAIMS_MATRIX.md`
- Reproduction: `REPRODUCE.md`
- Third-party licenses: `THIRD_PARTY_DATA_LICENSES.md`
- AI disclosure: `publication/AI_ASSISTANCE_DISCLOSURE.md`
- Human-only llama.cpp evidence index:
  `publication/llamacpp_issue/HUMAN_ONLY_EVIDENCE_INDEX.md`

## Citation

Toeda, T. (2026). *Multi-Token Prediction on a Consumer GPU: A Quality–Throughput Study of Quantized Gemma 4* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.21860461

Machine-readable citation metadata is in `CITATION.cff`.

## Licensing

Original paper/prose/figures/processed results are CC BY 4.0. Original study
code is Apache-2.0. Third-party materials retain their own licenses; verbatim
benchmark materials are not redistributed. See `LICENSE.md`.

