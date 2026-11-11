#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -n "${NEMOTRON_PYTHON:-}" ]; then
    exec "$NEMOTRON_PYTHON" studio_worker.py --stdio "$@"
elif [ -x ../.venv-model/bin/python ]; then
    exec ../.venv-model/bin/python studio_worker.py --stdio "$@"
elif [ -x .venv/bin/python ]; then
    exec .venv/bin/python studio_worker.py --stdio "$@"
else
    printf 'Run Backend/setup.sh first.\n' >&2
    exit 1
fi
