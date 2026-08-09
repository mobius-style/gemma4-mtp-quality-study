---
pretty_name: Gemma 4 MTP Quality-Throughput Study
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
- ja
tags:
- benchmarking
- gemma
- llama-cpp
- speculative-decoding
- multi-token-prediction
- quantized-inference
size_categories:
- n<1K
---

# Gemma 4 MTP Quality–Throughput Study

Author: Taiko Toeda, Independent Researcher  
ORCID: https://orcid.org/0009-0001-7267-0201  
Version: 1.0.0  
DOI: `10.5281/zenodo.21860461`

This Dataset repository distributes the sanitized processed measurements from
an independent paired benchmark of QAT Q4_0 Gemma 4 12B IT and 26B-A4B IT on
one RTX 5070 Ti using pinned CUDA `llama.cpp` commit
`7ba604f1cb61cd14898138e9abc0b4ff2601f180`.

## Included data

- request-level timing, throughput, token acceptance, condition, task/prompt
  identifiers, and hashes;
- item-level correctness, finish reason, timing, output length, and output
  hashes;
- paired equivalence and quality summaries;
- aggregate performance, uncertainty, and post-review descriptive outputs;
- figures, tables, manifests, and field documentation.

Verbatim benchmark items, prompts, reference answers, generated output text,
raw request records, complete server logs, local paths, model weights, and
vendored third-party source are excluded. See
`publication/DATA_REDISTRIBUTION_AUDIT.md` and
`THIRD_PARTY_DATA_LICENSES.md` in the linked publication package.

## Result boundary

N=6 was the fastest observed depth in the fixed tested grid for both models.
The paired objective-quality evidence cleared the prospectively frozen
-5-percentage-point pilot margin, while exact ordinary-decoding output equality
was only 56% and 59% at N=6. These findings do not establish universal
optimality, behavioral identity, or cross-platform generalization.

## Links

- Paper and archival record: `https://doi.org/10.5281/zenodo.21860461`
- Lightweight source repository: `https://github.com/mobius-style/gemma4-mtp-quality-study`
- Public reproducibility archive: `https://github.com/mobius-style/gemma4-mtp-quality-study/releases/download/v1.0.0/gemma4-mtp-quality-study-PUBLIC_REPRODUCIBILITY_ARCHIVE-v1.0.0.zip`

## Licensing and attribution

The original processed measurements and documentation in this Dataset
repository are CC BY 4.0. Project-authored code in the linked source repository
is Apache-2.0. Third-party benchmark content is not redistributed and is not
relicensed. Cite the DOI record and consult the third-party license inventory.

Generative AI assisted scripting, research engineering, analysis,
visualizations, language preparation, and dissemination preparation. AI is not
an author; the human author is responsible for the published work.

