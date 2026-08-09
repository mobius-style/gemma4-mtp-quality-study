# Human-only evidence navigation matrix

| Measurement | Processed source | Row/table identifier | Private raw evidence path | Paper location | Figure/table | Runtime/model identity |
|---|---|---|---|---|---|---|
| 12B N=6 exact equality | `processed/equivalence_summary.csv` | `12b`, `n6` | `raw/quality/12b_v6.jsonl` | Section 7 | Figure 6; `tables/equivalence_summary.md` | llama.cpp `7ba604f1cb61cd14898138e9abc0b4ff2601f180`; target `faff1a63…`; drafter `fcb35dea…` |
| 26B-A4B N=6 exact equality | `processed/equivalence_summary.csv` | `26b`, `n6` | `raw/quality/26b_v6.jsonl` | Section 7 | Figure 6; `tables/equivalence_summary.md` | llama.cpp `7ba604f1cb61cd14898138e9abc0b4ff2601f180`; target `4c856523…`; drafter `7272d975…` |
| Forced-256-token equivalence | `processed/performance_exact_equivalence.csv` | model/condition rows | `raw/performance/12b.jsonl`; `raw/performance/26b.jsonl` | Section 7 | Figure 3 | Same pinned runtime/model hashes |
| First divergent token | `processed/equivalence_pairs.csv` | `first_divergent_token` | `raw/quality/*_v6.jsonl` | Section 7 | Figure 6; `tables/first_divergence_summary.md` | Same pinned runtime/model hashes |
| Length/equality association | `processed/length_divergence_by_bin.csv`; `processed/length_divergence_within_family.csv`; `processed/length_divergence_stratified.csv` | model/length-bin/family rows | `raw/quality/*_v6.jsonl` | Sections 5.8 and 7 | Figures 9 and 11 | Same pinned runtime/model hashes |
| 12B throughput grid | `processed/performance_summary.csv` | `12b`, all `n_max` | `raw/performance/12b.jsonl` | Section 6 | Figure 1; `tables/performance_by_depth.md` | Same pinned runtime/model hashes |
| 26B-A4B throughput grid | `processed/performance_summary.csv` | `26b`, all `n_max` | `raw/performance/26b.jsonl` | Section 6 | Figure 1; `tables/performance_by_depth.md` | Same pinned runtime/model hashes |
| N=6 paired objective quality | `processed/paired_quality_changes.csv`; `processed/quality_summary.csv` | model `n6` rows | `raw/quality/*_v6.jsonl` | Section 8 | Figure 4; `tables/paired_quality_changes.md` | Same pinned runtime/model hashes |
| OFF repeat negative control | `processed/equivalence_summary.csv`; `processed/negative_control_summary.csv` | `off_repeat`; model rows | `raw/quality/*_v6.jsonl`; `raw/negative_control/*.jsonl` | Sections 7 and 12 | `tables/equivalence_summary.md` | Same pinned runtime/model hashes |
| Public artifact and hashes | `publication/PUBLICATION_MANIFEST.md`; `PUBLIC_MANIFEST.sha256` | release record | Private paths not public | Appendix B | Not applicable | Public URLs inserted after DOI publication |

