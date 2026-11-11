#!/usr/bin/env python3
"""Explicitly download the pinned NVIDIA model (~2.6 GB); never run on app launch."""
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

REVISION = "a9a1febb1b9bf497fc67bd1d23fa33e170b249db"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "Models"))
args = parser.parse_args()
path = snapshot_download("nvidia/Nemotron-3.5-Parse", revision=REVISION,
                         local_dir=args.output,
                         allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors", "LICENSE", "THIRD_PARTY_NOTICES.md", "TOKENIZER_ATTRIBUTION.md"])
print(path)
