#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run Nemotron 3.5 Parse on an image with Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, GenerationConfig


DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Model directory (defaults to the folder containing this script).",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8744)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-overlay", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()

    image = Image.open(args.image).convert("RGB")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    inputs = processor(
        images=[image],
        text=args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(args.device)

    generation_config = GenerationConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.repetition_penalty = args.repetition_penalty

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=generation_config)

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    print(generated_text)

    if args.save_overlay is not None:
        from postprocessing import extract_classes_bboxes, transform_bbox_to_original

        _classes, bboxes, _texts = extract_classes_bboxes(generated_text)
        bboxes = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes]
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(
                (bbox[0], bbox[1], max(bbox[0], bbox[2]), max(bbox[1], bbox[3])),
                outline="red",
                width=2,
            )
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.save_overlay)


if __name__ == "__main__":
    main()
