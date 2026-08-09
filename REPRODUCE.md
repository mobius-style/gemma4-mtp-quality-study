# Public reproduction guide

The public package intentionally omits third-party benchmark text, raw model
responses, and complete logs. Reproduction starts by retrieving the pinned
official sources and verifying their recorded SHA-256 values.

```bash
export STUDY_ROOT="$PWD"
export LLAMA_CPP_ROOT=/absolute/path/to/pinned/llama.cpp-build-root
python3 -m venv "$STUDY_ROOT/.venv"
"$STUDY_ROOT/.venv/bin/pip" install -r "$STUDY_ROOT/configs/requirements.txt"
"$STUDY_ROOT/.venv/bin/python" "$STUDY_ROOT/scripts/fetch_and_prepare_data.py"
sha256sum "$STUDY_ROOT"/datasets/source/*
```

Supply target and drafter GGUF files whose SHA-256 values match
`configs/environment.json`, then replace the portable model-path placeholders
in that local configuration. A different model digest, runtime commit, GPU, or
backend is a replication condition rather than a direct reproduction.

To repeat inference, use the launchers after reviewing all paths and resource
requirements:

```bash
bash "$STUDY_ROOT/scripts/run_performance_sweep.sh"
bash "$STUDY_ROOT/scripts/run_quality_bench.sh"
bash "$STUDY_ROOT/scripts/run_pmin_sweep.sh"
```

To rebuild the existing derived results from a lawful local copy of the
private raw evidence, use the analysis commands documented in the internal
record. The published processed outputs, tables, figures, and validation JSON
are included so the reported results remain inspectable without redistributing
benchmark text or generated responses.

