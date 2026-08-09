# Data redistribution audit

Audit date: 2026-08-09

## Decision

The internal Round-3 archive is preserved privately and unchanged. The public
reproducibility archive is a separately generated allowlisted package. Even
where an upstream license permits redistribution, the public package takes the
conservative uniform approach of omitting verbatim benchmark content and raw
generated text.

## Exclusions and retained substitutes

| Internal candidate | Embedded content or risk | Public decision | Retained public substitute |
|---|---|---|---|
| `datasets/source/` | Complete third-party benchmark files | Exclude | Exact source URLs, pinned revisions, byte counts, and SHA-256 in `datasets/manifest_v3.json` |
| `prompts/` | Verbatim benchmark questions, references, fixtures, and synthetic performance prompts | Exclude | Task IDs, source indices/revisions where applicable, prompt SHA-256, and regeneration code |
| `raw/` | Commands, local paths, prompt text, generated output text, token IDs, response IDs, and request metadata | Exclude | Sanitized processed request metrics, output hashes, item IDs, condition IDs, and internal archive hash |
| `logs/` | Complete server logs, local paths, host/runtime details, and possible prompt/output echoes | Exclude | Environment summary, runtime/model hashes, aggregate and request-level timing metrics |
| `vendor/` | Verbatim third-party IFEval evaluator source | Exclude | Pinned upstream revision/hash inventory and retrieval code |
| `processed/quality_item_scores.jsonl` | Duplicated item-level records including score-detail payloads | Exclude | Sanitized CSV without `score_detail` |
| `processed/quality_item_scores.csv` | `score_detail` can contain parsed and reference-normalized answers | Redact field | Item ID, family, condition, correctness, finish reason, timing, token count, and output SHA-256 |
| `processed/performance_runs.csv` | Local absolute log path | Redact field | All measurement columns except `log` |
| `configs/environment.json` | Hostname and local executable/model paths | Redact values | Hardware/software facts, hashes, byte sizes, and portable placeholders |
| Launch scripts | Local username and absolute study/runtime paths | Portability transform in public copy | `STUDY_ROOT`/`LLAMA_CPP_ROOT` environment variables; transformation disclosed in public manifest |
| `UPSTREAM_ISSUE_25618_COMMENT_DRAFT.md` | AI-written issue prose prohibited for posting | Private-only; exclude | Evidence-navigation matrix only under `publication/llamacpp_issue/` |
| Internal FULL ZIP | Contains all excluded classes above | Never publish | Separately generated `PUBLIC_REPRODUCIBILITY_ARCHIVE` plus both archive hashes in the release manifest |

## Reproducibility impact

No scientific measurement is recalculated or altered by sanitization. A
replicator can fetch every benchmark source at its pinned revision, verify its
SHA-256, regenerate prompts, and rerun the supplied project-authored scripts.
The public package exposes task identifiers, output hashes, scores, timings,
model/runtime hashes, analysis code, tables, figures, and validation results.

