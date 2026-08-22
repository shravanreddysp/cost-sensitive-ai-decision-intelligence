#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python run_experiment.py --bootstrap 1000 --cost-ratios 2 3 5 8 10 --central-ratio 5 --risk-aversion 0.25
