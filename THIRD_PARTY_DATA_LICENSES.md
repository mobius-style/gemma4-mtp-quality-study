# Third-party data and code license inventory

Audit date: 2026-08-09. The exact artifacts used in the experiment are pinned
in `datasets/manifest_v3.json`. This inventory distinguishes the source-code
license from the dataset/item license and applies the more source-specific
notice where the downloaded mirror declares one.

| Component | Exact experimental source | Pinned revision | Upstream license verified for the used artifact | Verbatim prompts/questions redistributed? | Scorer/source redistributed? | Public treatment |
|---|---|---|---|---:|---:|---|
| GSM8K test set | `openai/grade-school-math`, `grade_school_math/data/test.jsonl` | `3101c7d5072418e28b9008a6636bde82a006892c` | MIT repository license, copyright OpenAI 2021 | No | No upstream code | IDs, hashes, source URL, and derived scores only; reproduce by pinned download |
| IFEval input data | `google-research/google-research`, `instruction_following_eval/data/input_data.jsonl` | `015539128d9a7dbe14b5f5308a198a15da808949` | Google Research states datasets in the repository are CC BY 4.0 | No | No vendored upstream code | IDs, hashes, source URL, and derived scores only; reproduce by pinned download |
| IFEval evaluator source | `google-research/google-research`, `instruction_following_eval` | `015539128d9a7dbe14b5f5308a198a15da808949` | Apache License 2.0 for Google Research source files | Not applicable | No | Public scripts retrieve/verify the pinned upstream source; local vendored copy excluded |
| MBPP sanitized data | `google-research/google-research`, `mbpp/sanitized-mbpp.json` | `015539128d9a7dbe14b5f5308a198a15da808949` | Google Research states datasets in the repository are CC BY 4.0 | No | No upstream code | IDs, hashes, source URL, and derived scores only; reproduce by pinned download |
| MMLU-Pro test parquet | `TIGER-Lab/MMLU-Pro` Hugging Face dataset mirror, `data/test-00000-of-00001.parquet` | `b189ec765aa7ed75c8acfea42df31fdae71f97be` | MIT in the exact Hugging Face dataset metadata at the pinned mirror revision; the separate GitHub code repository uses Apache-2.0 | No | No upstream code | IDs, hashes, source URL, and derived scores only; reproduce by pinned download |

## Attribution and source links

- GSM8K: https://github.com/openai/grade-school-math
- Google Research repository: https://github.com/google-research/google-research
- MMLU-Pro code repository: https://github.com/TIGER-AI-Lab/MMLU-Pro
- Exact MMLU-Pro dataset mirror: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be

The internal frozen archive retains the exact downloaded benchmark files,
prompts, raw model records, and logs for private evidentiary preservation. The
public archive is intentionally not byte-identical to that internal archive.

