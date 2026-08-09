# Literature search and source audit

Search date: 2026-08-09 (Asia/Tokyo). This search was scoped to primary
research papers, official model cards, official benchmark repositories, and
the upstream `llama.cpp` repository. Search themes were speculative decoding,
multi-token prediction, Gemma 4 MTP, deterministic equivalence under
quantization, and the four objective benchmark families used here.

## Included sources

| Topic | Primary source | Use in this study |
|---|---|---|
| Speculative decoding | Leviathan, Kalman, and Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192 | Algorithmic distribution-preservation claim and original acceleration framing |
| Speculative sampling | Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv:2302.01318 | Independent formulation using modified rejection sampling |
| Multi-token prediction | Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, arXiv:2404.19737 | MTP training and inference background |
| Gemma 4 | Gemma Team, *Gemma 4 Technical Report*, arXiv:2607.02770 | Model-family and dense/MoE background |
| Gemma 4 MTP | Google DeepMind, official Gemma 4 26B-A4B MTP model card | Vendor claim and drafter description |
| Runtime | `ggml-org/llama.cpp`, `docs/speculative.md`, pinned locally at commit `7ba604f1...` | CLI semantics and implementation context |
| Quantized divergence | `llama.cpp` issue #25618 | Prior implementation-level report; treated as issue evidence, not peer-reviewed proof |
| Drafter artifact | Unsloth Gemma 4 QAT GGUF MTP repository, pinned revision recorded in the dataset/environment manifests | Exact local Q4_0 drafter provenance |
| GSM8K | Cobbe et al., arXiv:2110.14168; official OpenAI repository | Exact-answer mathematics task |
| MMLU-Pro | Wang et al., arXiv:2406.01574; official TIGER-Lab repository/dataset | Constrained-choice knowledge/reasoning task |
| IFEval | Zhou et al., arXiv:2311.07911; official Google Research implementation | Deterministically verifiable instruction following |
| MBPP | Austin et al., arXiv:2108.07732; official Google Research data | Executable Python task |

## Exclusion and interpretation rules

- Secondary blog posts, leaderboards, social-media reports, and vendor
  paraphrases were not used to establish scientific claims.
- The Google model card's “up to 3x” and “exact same quality” wording is a
  vendor statement about the intended method, not a result reproduced by this
  paper.
- GitHub issue #25618 is evidence that quantized greedy divergence has been
  observed elsewhere. It does not identify the root cause and is not elevated
  to a general theorem.
- No primary paper found in this scoped search reports this exact combination
  of Gemma 4 QAT Q4_0, `llama.cpp`, RTX 5070 Ti, the tested N grid, and paired
  objective-quality scoring. This is a scoped search result, not a novelty
  claim.

## Stable links

- https://arxiv.org/abs/2211.17192
- https://arxiv.org/abs/2302.01318
- https://arxiv.org/abs/2404.19737
- https://arxiv.org/abs/2607.02770
- https://huggingface.co/google/gemma-4-26B-A4B-it-assistant
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/issues/25618
- https://arxiv.org/abs/2110.14168
- https://arxiv.org/abs/2406.01574
- https://arxiv.org/abs/2311.07911
- https://arxiv.org/abs/2108.07732

