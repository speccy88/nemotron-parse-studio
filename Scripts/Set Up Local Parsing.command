#!/bin/zsh
set -euo pipefail

release_dir="${0:A:h}"
backend="$release_dir/Nemotron Parse Studio.app/Contents/Resources/Backend"
runtime="$HOME/Library/Application Support/NemotronParseStudio/Runtime"
venv_dir="$runtime/Backend/.venv"

if [[ ! -f "$backend/setup.sh" ]]; then
    print -u2 "Extract the release ZIP and run setup next to Nemotron Parse Studio.app."
    exit 1
fi

python="${PYTHON:-python3.12}"
if [[ ! -x "$venv_dir/bin/python" ]] && ! command -v "$python" >/dev/null 2>&1; then
    for candidate in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 "$HOME/.local/bin/python3.12"; do
        if [[ -x "$candidate" ]]; then
            python="$candidate"
            break
        fi
    done
    if ! command -v "$python" >/dev/null 2>&1; then
        print -u2 "Install Python 3.12 for Apple silicon, then run setup again."
        exit 1
    fi
fi

mkdir -p "$runtime/Backend"
PYTHON="$python" NEMOTRON_VENV_DIR="$venv_dir" "$backend/setup.sh" --model
if [[ "${1:-}" != "--python-only" ]]; then
    "$venv_dir/bin/python" "$backend/download_model.py" --output "$runtime/Models"
fi
print "Setup complete. Open Nemotron Parse Studio.app."
