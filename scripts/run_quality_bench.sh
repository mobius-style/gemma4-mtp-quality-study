#!/usr/bin/env bash
set -euo pipefail
study_root=${STUDY_ROOT:?Set STUDY_ROOT to the public study root}
"$study_root/.venv/bin/python" "$study_root/scripts/score_quality.py" --self-test
"$study_root/.venv/bin/python" "$study_root/scripts/run_quality.py" --model 12b --port 18131
"$study_root/.venv/bin/python" "$study_root/scripts/run_quality.py" --model 26b --port 18132
