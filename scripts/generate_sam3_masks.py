"""Generate per-person instance masks with SAM 3 text prompts.

The script is designed for large graduation photos. By default it runs SAM 3
on a resized copy of each image, then restores each binary mask to the original
image size before saving. This avoids the worst memory blow-up from returning
many full-resolution masks at once.

Example:
    python scripts/generate_sam3_masks.py material/*.jpg \
        --checkpoint /path/to/sam3.pt \
        --output-dir output/sam3_masks
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_SAM3_REPO = Path(__file__).resolve().parents[2] / "sam3"
DEFAULT_BPE_PATH = DEFAULT_SAM3_REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


@dataclass
class Sam3Instance:
    id: int
    score: float
    bbox: list[int]
    box_xyxy: list[float]
    area: int
    mask_path: str


@dataclass
class Sam3ImageResult:
    image: str
    prompt: str
    width: int
    height: int
    inference_width: int
    inference_height: int
    checkpoint: str | None
    instances: list[Sam3Instance]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Input image paths. Defaults to material/*.jpg when omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to a SAM3 image checkpoint, for example sam3.pt. Leave empty until the model finishes downloading.",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow SAM3 to download checkpoints from Hugging Face when --checkpoint is empty.",
    )
    parser.add_argument("--sam3-repo", type=Path, default=DEFAULT_SAM3_REPO)
    parser.add_argument("--bpe-path", type=Path, default=DEFAULT_BPE_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("output/sam3_masks"))
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Autocast dtype used on CUDA.",
    )
    parser.add_argument(
        "--max-inference-side",
        type=int,
        default=1600,
        help="Resize longest image side before SAM3 inference. Use 0 for original size.",
    )
    parser.add_argument(
        "--min-area-pixels",
        type=int,
        default=300,
        help="Drop restored masks smaller than this many pixels.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.00002,
        help="Drop restored masks smaller than this fraction of the original image area.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Keep at most this many instances after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--sort",
        default="x",
        choices=["x", "score"],
        help="Instance id ordering: left-to-right bbox x or descending score.",
    )
    parser.add_argument("--no-label-mask", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def resolve_images(images: list[Path]) -> list[Path]:
    if images:
        return images
    return sorted(Path("material").glob("*.jpg"))


def ensure_sam3_importable(sam3_repo: Path) -> None:
    if sam3_repo.exists():
        sys.path.insert(0, str(sam3_repo))


def validate_args(args: argparse.Namespace) -> None:
    if not args.allow_hf_download and not args.checkpoint:
        raise SystemExit(
            "--checkpoint is empty. Pass --checkpoint /path/to/sam3.pt after the "
            "model finishes downloading, or pass --allow-hf-download to let SAM3 "
            "download from Hugging Face."
        )
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not args.allow_hf_download and not args.bpe_path.exists():
        raise SystemExit(f"BPE vocabulary not found: {args.bpe_path}")


def resize_for_inference(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    if max_side <= 0:
        return image, 1.0
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image, 1.0
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def tensor_to_numpy_bool(mask_tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    mask = np.squeeze(mask)
    return mask.astype(bool)


def restore_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_img) > 0


def mask_bbox(mask: np.ndarray) -> tuple[list[int], list[float]]:
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    box_xyxy = [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]
    return bbox, box_xyxy


def color_for_id(instance_id: int) -> tuple[int, int, int]:
    hue = (instance_id * 0.61803398875) % 1.0
    x = 1 - abs((hue * 6) % 2 - 1)
    if hue < 1 / 6:
        rgb = (1, x, 0)
    elif hue < 2 / 6:
        rgb = (x, 1, 0)
    elif hue < 3 / 6:
        rgb = (0, 1, x)
    elif hue < 4 / 6:
        rgb = (0, x, 1)
    elif hue < 5 / 6:
        rgb = (x, 0, 1)
    else:
        rgb = (1, 0, x)
    return tuple(int(255 * c) for c in rgb)


def save_overlay(
    image: Image.Image,
    masks: list[np.ndarray],
    instances: list[Sam3Instance],
    output_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    for mask, instance in zip(masks, instances):
        color = color_for_id(instance.id)
        color_layer = Image.new("RGBA", image.size, (*color, 0))
        alpha = Image.fromarray(mask.astype(np.uint8) * 90, mode="L")
        color_layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, color_layer)
        draw = ImageDraw.Draw(overlay)

        x, y, w, h = instance.bbox
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(*color, 255), width=5)
        draw.text((x + 4, y + 4), f"{instance.id}:{instance.score:.2f}", fill=(*color, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path, quality=92)


def save_label_mask(masks: list[np.ndarray], output_path: Path) -> None:
    if not masks:
        return
    label = np.zeros(masks[0].shape, dtype=np.uint16)
    for idx, mask in enumerate(masks, start=1):
        label[mask] = idx
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label, mode="I;16").save(output_path)


def build_model(args: argparse.Namespace):
    ensure_sam3_importable(args.sam3_repo)
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")

    checkpoint_path = args.checkpoint or None
    model = build_sam3_image_model(
        bpe_path=str(args.bpe_path) if args.bpe_path else None,
        device=args.device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=args.allow_hf_download,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence,
    )
    return torch, processor


def autocast_context(torch_module, device: str, dtype_name: str):
    if device != "cuda" or dtype_name == "float32":
        return nullcontext()
    dtype = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
    }[dtype_name]
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def run_image(
    torch_module,
    processor,
    image_path: Path,
    args: argparse.Namespace,
) -> Sam3ImageResult:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    original_size = image.size
    inference_image, scale = resize_for_inference(image, args.max_inference_side)

    with autocast_context(torch_module, args.device, args.dtype):
        state = processor.set_image(inference_image)
        output = processor.set_text_prompt(prompt=args.prompt, state=state)

    masks_tensor = output["masks"]
    scores_tensor = output["scores"].detach().cpu()

    restored: list[tuple[np.ndarray, float, list[int], list[float], int]] = []
    min_area = max(args.min_area_pixels, math.ceil(image.width * image.height * args.min_area_ratio))

    for idx in range(int(masks_tensor.shape[0])):
        mask_small = tensor_to_numpy_bool(masks_tensor[idx])
        mask = restore_mask(mask_small, original_size)
        area = int(mask.sum())
        if area < min_area:
            continue
        bbox, box_xyxy = mask_bbox(mask)
        score = float(scores_tensor[idx].item())
        restored.append((mask, score, bbox, box_xyxy, area))

    if args.sort == "score":
        restored.sort(key=lambda item: item[1], reverse=True)
    else:
        restored.sort(key=lambda item: (item[2][0], item[2][1]))
    if args.max_instances > 0:
        restored = restored[: args.max_instances]

    image_dir = args.output_dir / image_path.stem
    mask_dir = image_dir / "instances"
    mask_dir.mkdir(parents=True, exist_ok=True)

    instances: list[Sam3Instance] = []
    masks: list[np.ndarray] = []
    for instance_id, (mask, score, bbox, box_xyxy, area) in enumerate(restored, start=1):
        mask_path = mask_dir / f"person_{instance_id:03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        masks.append(mask)
        instances.append(
            Sam3Instance(
                id=instance_id,
                score=round(score, 6),
                bbox=bbox,
                box_xyxy=[round(v, 2) for v in box_xyxy],
                area=area,
                mask_path=str(mask_path),
            )
        )

    if not args.no_label_mask:
        save_label_mask(masks, image_dir / f"{image_path.stem}_instances.png")
    if not args.no_overlay:
        save_overlay(image, masks, instances, image_dir / f"{image_path.stem}_overlay.jpg")

    result = Sam3ImageResult(
        image=str(image_path),
        prompt=args.prompt,
        width=image.width,
        height=image.height,
        inference_width=inference_image.width,
        inference_height=inference_image.height,
        checkpoint=args.checkpoint or None,
        instances=instances,
    )
    metadata_path = image_dir / f"{image_path.stem}.json"
    metadata_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    images = resolve_images(args.images)
    if not images:
        raise SystemExit("no input images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch_module, processor = build_model(args)

    all_results: list[dict] = []
    for image_path in images:
        result = run_image(torch_module, processor, image_path, args)
        all_results.append(asdict(result))
        print(
            f"{image_path.name}: {len(result.instances)} instance mask(s), "
            f"inference={result.inference_width}x{result.inference_height}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
