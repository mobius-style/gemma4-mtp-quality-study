# Manual publication package — Medium

Status: hold until the Zenodo DOI, GitHub release, and Hugging Face Dataset are public.

Author metadata: Taiko Toeda — Independent Researcher — ORCID 0009-0001-7267-0201

## Title

What Gemma 4 Multi-Token Prediction Changed on One RTX 5070 Ti

## Subtitle

A frozen quality–throughput study separates algorithmic losslessness, exact greedy behavior, and paired task-quality evidence.

## Article

Multi-token prediction can raise local language-model decoding throughput by
drafting several future tokens and verifying them in parallel. “Lossless,”
however, can describe different things: an algorithmic property, exact
finite-precision greedy behavior, or empirical task quality. I measured those
separately for QAT Q4_0 Gemma 4 12B IT and 26B-A4B IT on one RTX 5070 Ti using
a pinned CUDA `llama.cpp` runtime.

Across the fixed depth grid, N=6 was the observed throughput maximum for both
models. Aggregate decode throughput rose from 89.20 to 226.39 tokens/s for 12B
and from 169.58 to 322.53 tokens/s for 26B-A4B. The near-depth ranking is not a
universal optimum: the 12B measurements in particular show an observed
performance plateau at N={4,6,8,12}, and only N=6 among those four depths
received the paired objective-quality run.

Exact output identity told a different story. At N=6, only 112/200 12B outputs
and 118/200 26B-A4B outputs were byte-identical to ordinary decoding. Yet the
paired objective macro-score change was -0.83 percentage points for both
models, with 95% confidence intervals above the prospectively frozen -5-point
pilot margin. That supports a scoped operating conclusion for this stack: MTP
was substantially faster and the pilot did not detect degradation beyond its
frozen margin, but MTP was not behaviorally identical.

The study is intentionally limited to one GPU, one pinned runtime, Q4_0 target
and drafter artifacts, parallel=1, and the measured workloads. It does not
generalize to BF16, other GPUs, other runtimes, or production concurrency.

The public artifact uses a conservative redistribution boundary. It omits
verbatim benchmark questions, prompts, raw responses, complete logs, vendored
third-party source, model weights, and local paths. It retains task IDs,
hashes, processed measurements, source revisions, scripts, figures, tables,
licenses, and validation records.

- Paper and archival record: https://doi.org/10.5281/zenodo.21860461
- GitHub source and release archive: https://github.com/happy-HHH/gemma4-mtp-quality-study/releases/tag/v1.0.0
- Hugging Face processed Dataset: https://huggingface.co/datasets/moebiusT7/gemma4-mtp-quality-study

Generative AI assisted scripting, research engineering, analysis,
visualizations, manuscript language preparation, and dissemination
preparation. AI is not an author. I am responsible for the research design,
measurements accepted for publication, interpretation, citations, claims, and
the final publication decision.

