#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pyenv local 3.10.12
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements/torch-cu124.txt
pip install -r requirements/base.txt
pip install -e .
python scripts/check_env.py

