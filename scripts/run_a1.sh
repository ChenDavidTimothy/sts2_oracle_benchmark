#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
python -m sts2.bench --config configs/a1_adaptive_sweep.yaml --device auto
python scripts/plot_results.py runs/a1_adaptive_sweep/metrics_all.csv
python scripts/plot_adaptive_results.py runs/a1_adaptive_sweep/metrics_all.csv
