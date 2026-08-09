#!/usr/bin/env bash
set -euo pipefail
study_root=${STUDY_ROOT:?Set STUDY_ROOT to the public study root}
python3 "$study_root/scripts/run_performance.py" --model 12b --port 18121
python3 "$study_root/scripts/run_performance.py" --model 26b --port 18122
