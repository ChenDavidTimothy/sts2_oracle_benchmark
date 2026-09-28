#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
python -m sts2.branch_bench --config configs/a3_branches.yaml --device auto --output runs/a3_branches.csv
