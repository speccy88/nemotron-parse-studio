#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
venv_dir=${NEMOTRON_VENV_DIR:-.venv}
if [ ! -x "$venv_dir/bin/python" ]; then "$PYTHON" -m venv "$venv_dir"; fi
requirements=requirements-lite.txt
if [ "${1:-}" = "--model" ]; then requirements=requirements-model.txt; fi
"$venv_dir/bin/python" -m pip install -r "$requirements"
printf '\nThe app starts its private worker automatically; Backend/run.sh starts stdio mode. Model weights are a separate explicit download.\n'
