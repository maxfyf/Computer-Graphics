"""Generate per-person instance masks with Ultralytics YOLO segmentation.

This is the lightweight local alternative to the SAM3 pipeline. It writes the
same output layout as scripts/generate_sam3_masks.py, so downstream metadata
extraction can reuse scripts/extract_metadata_from_sam3.py.

Examples:
    python scripts/generate_yolo_person_masks.py material/*.jpg \
        --model yolo11m-seg.pt --output-dir yolo_masks

    python scripts/generate_yolo_person_masks.py material/g2.jpg \
        --model yolo11m-seg.pt --tile-size 1536 --tile-overlap 256
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


PERSON_CLASS_ID = 0


@dataclass
class YoloCandidate:
    score: float
    bbox: list[int]
    box_xyxy: list[float]
    area: int
    mask: np.ndarray
    source: str


@dataclass
class YoloInstance:
    id: int
    score: float
    bbox: list[int]
    box_xyxy: list[float]
    area: int
    mask_path: str


@dataclass
class YoloImageResult:
    image: str
    prompt: str
    width: int
    height: int
    inference_width: int
    inference_height: int
    checkpoint: str | None
    model: str
    strategy: str
    instances: list[YoloInstance]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Input image paths. Defaults to material/*.jpg when omitted.",
    )
    parser.add_argument(
        "--model",
        default="yolo11m-seg.pt",
        help="Ultralytics segmentation checkpoint, e.g. yolo11s-seg.pt or yolo11m-seg.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("yolo_masks"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or empty for Ultralytics default.")
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference when supported by the selected device.",
    )
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Ask YOLO for higher-resolution masks. Costs substantially more memory.",
    )
    parser.add_argument(
        "--auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically choose whole-image or tiled inference per image.",
    )
    parser.add_argument("--probe-imgsz", type=int, default=640)
    parser.add_argument("--probe-conf", type=float, default=0.2)
    parser.add_argument(
        "--auto-whole-max-count",
        type=int,
        default=18,
        help="Use whole-image inference when probe count is at most this and people are not tiny.",
    )
    parser.add_argument(
        "--auto-dense-count",
        type=int,
        default=20,
        help="Force tiled inference when the probe sees at least this many people.",
    )
    parser.add_argument("--auto-whole-min-height-ratio", type=float, default=0.22)
    parser.add_argument("--auto-large-area-ratio", type=float, default=0.018)
    parser.add_argument("--auto-tile-size", type=int, default=1280)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Force tiled inference with this crop size. In auto mode this is used for dense scenes.",
    )
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--mask-iou", type=float, default=0.6)
    parser.add_argument(
        "--containment",
        type=float,
        default=0.85,
        help="Drop smaller boxes that are mostly contained in a larger candidate.",
    )
    parser.add_argument(
        "--no-whole-image-pass",
        action="store_true",
        help="Disable the extra whole-image pass used to suppress tiled fragments.",
    )
    parser.add_argument("--min-area-pixels", type=int, default=300)
    parser.add_argument("--min-area-ratio", type=float, default=0.00002)
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument(
        "--sort",
        default="x",
        choices=["x", "score"],
        help="Instance id ordering: left-to-right bbox x or descending score.",
    )
    parser.add_argument("--no-label-mask", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--clear-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear Python/CUDA cache after each tile.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def resolve_images(images: list[Path]) -> list[Path]:
    if images:
        return images
    return sorted(Path("material").glob("*.jpg"))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_img) > 0


def bbox_from_mask(mask: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> tuple[list[int], list[float]]:
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()) + offset_x, int(xs.max()) + offset_x
    y0, y1 = int(ys.min()) + offset_y, int(ys.max()) + offset_y
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
    instances: list[YoloInstance],
    output_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for mask, instance in zip(masks, instances):
        color = color_for_id(instance.id)
        layer = Image.new("RGBA", image.size, (*color, 0))
        alpha = Image.fromarray(mask.astype(np.uint8) * 90, mode="L")
        layer.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, layer)
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


def iter_tiles(width: int, height: int, tile_size: int, overlap: int):
    if tile_size <= 0:
        yield 0, 0, width, height
        return
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    for y in ys:
        for x in xs:
            yield x, y, min(width, x + tile_size), min(height, y + tile_size)


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_area(box: list[int]) -> int:
    return max(0, box[2]) * max(0, box[3])


def bbox_intersection(a: list[int], b: list[int]) -> int:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def smaller_box_containment(a: list[int], b: list[int]) -> float:
    smaller = min(bbox_area(a), bbox_area(b))
    if smaller <= 0:
        return 0.0
    return bbox_intersection(a, b) / smaller


def mask_iou(a: YoloCandidate, b: YoloCandidate) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    a_crop = a.mask[iy0 - ay : iy1 - ay, ix0 - ax : ix1 - ax]
    b_crop = b.mask[iy0 - by : iy1 - by, ix0 - bx : ix1 - bx]
    inter = int(np.logical_and(a_crop, b_crop).sum())
    if inter == 0:
        return 0.0
    union = int(a.mask.sum()) + int(b.mask.sum()) - inter
    return inter / union if union > 0 else 0.0


def dedupe_candidates(
    candidates: list[YoloCandidate],
    nms_iou: float,
    duplicate_mask_iou: float,
    containment: float,
    max_instances: int,
) -> list[YoloCandidate]:
    kept: list[YoloCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = False
        replacements: list[int] = []
        for existing_index, existing in enumerate(kept):
            if bbox_iou(candidate.bbox, existing.bbox) >= nms_iou:
                duplicate = True
                break
            if mask_iou(candidate, existing) >= duplicate_mask_iou:
                duplicate = True
                break
            if smaller_box_containment(candidate.bbox, existing.bbox) >= containment:
                if bbox_area(candidate.bbox) <= bbox_area(existing.bbox):
                    duplicate = True
                    break
                if candidate.score >= existing.score * 0.6:
                    replacements.append(existing_index)
        if not duplicate:
            for index in sorted(replacements, reverse=True):
                kept.pop(index)
            kept.append(candidate)
            if max_instances > 0 and len(kept) >= max_instances:
                break
    return kept


def center_in_region(bbox: list[int], region: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    rx0, ry0, rx1, ry1 = region
    return rx0 <= cx < rx1 and ry0 <= cy < ry1


def result_candidates(
    result,
    offset_x: int,
    offset_y: int,
    crop_size: tuple[int, int],
    min_area: int,
    source: str,
    owner_region: tuple[int, int, int, int] | None = None,
) -> list[YoloCandidate]:
    if result.masks is None or result.boxes is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    candidates: list[YoloCandidate] = []
    for idx in range(len(masks)):
        if classes[idx] != PERSON_CLASS_ID:
            continue
        mask = resize_mask(masks[idx] > 0.5, crop_size)
        area = int(mask.sum())
        if area < min_area:
            continue
        bbox, box_xyxy = bbox_from_mask(mask, offset_x=offset_x, offset_y=offset_y)
        if owner_region is not None and not center_in_region(bbox, owner_region):
            continue
        local_x = bbox[0] - offset_x
        local_y = bbox[1] - offset_y
        local_w = bbox[2]
        local_h = bbox[3]
        instance_mask = mask[local_y : local_y + local_h, local_x : local_x + local_w]
        candidates.append(
            YoloCandidate(
                score=float(scores[idx]),
                bbox=bbox,
                box_xyxy=box_xyxy,
                area=area,
                mask=instance_mask,
                source=source,
            )
        )
    return candidates


def predict_crop(model, crop: Image.Image, args: argparse.Namespace):
    predict_kwargs = dict(
        source=crop,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[PERSON_CLASS_ID],
        retina_masks=args.retina_masks,
        verbose=False,
    )
    if args.device:
        predict_kwargs["device"] = args.device
    if args.half and args.device != "cpu":
        predict_kwargs["half"] = True
    return model.predict(**predict_kwargs)


def clear_memory(args: argparse.Namespace) -> None:
    if not args.clear_cache:
        return
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def probe_image(model, image: Image.Image, args: argparse.Namespace) -> list[YoloCandidate]:
    probe_args = copy.copy(args)
    probe_args.imgsz = args.probe_imgsz
    probe_args.conf = args.probe_conf
    probe_args.retina_masks = False
    probe_args.tile_size = 0
    probe_args.no_whole_image_pass = False

    width, height = image.size
    min_area = max(100, math.ceil(width * height * args.min_area_ratio * 0.25))
    results = predict_crop(model, image, probe_args)
    try:
        if not results:
            return []
        return result_candidates(
            results[0],
            offset_x=0,
            offset_y=0,
            crop_size=image.size,
            min_area=min_area,
            source="probe",
        )
    finally:
        del results
        clear_memory(args)


def auto_plan(model, image: Image.Image, args: argparse.Namespace) -> tuple[argparse.Namespace, str]:
    planned = copy.copy(args)
    if not args.auto:
        planned.tile_size = args.tile_size or 0
        return planned, "manual-tiled" if planned.tile_size > 0 else "manual-whole"

    probe = probe_image(model, image, args)
    if not probe:
        planned.tile_size = args.tile_size or 0
        return planned, "auto-fallback-whole"

    image_area = image.width * image.height
    count = len(probe)
    height_ratios = [candidate.bbox[3] / image.height for candidate in probe]
    area_ratios = [candidate.area / image_area for candidate in probe]
    median_height_ratio = float(np.median(height_ratios))
    max_area_ratio = max(area_ratios)

    force_tiled = count >= args.auto_dense_count and max_area_ratio < args.auto_large_area_ratio
    use_whole = (
        not force_tiled
        and (
            count <= 5
            or max_area_ratio >= args.auto_large_area_ratio
            or (
                count <= args.auto_whole_max_count
                and median_height_ratio >= args.auto_whole_min_height_ratio
            )
        )
    )

    if use_whole:
        planned.tile_size = 0
        planned.no_whole_image_pass = False
        planned.conf = max(args.conf, 0.3)
        return planned, (
            f"auto-whole(count={count},median_h={median_height_ratio:.2f},"
            f"max_area={max_area_ratio:.3f})"
        )

    planned.tile_size = args.tile_size or args.auto_tile_size
    planned.no_whole_image_pass = True
    return planned, (
        f"auto-tiled(count={count},median_h={median_height_ratio:.2f},"
        f"max_area={max_area_ratio:.3f},force_dense={force_tiled})"
    )


def tile_owner_region(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    image_width: int,
    image_height: int,
    overlap: int,
) -> tuple[int, int, int, int]:
    margin = max(0, overlap // 2)
    rx0 = x0 if x0 == 0 else x0 + margin
    ry0 = y0 if y0 == 0 else y0 + margin
    rx1 = x1 if x1 == image_width else x1 - margin
    ry1 = y1 if y1 == image_height else y1 - margin
    return rx0, ry0, rx1, ry1


def collect_candidates(model, image: Image.Image, args: argparse.Namespace) -> list[YoloCandidate]:
    width, height = image.size
    min_area = max(args.min_area_pixels, math.ceil(width * height * args.min_area_ratio))
    candidates: list[YoloCandidate] = []

    if args.tile_size <= 0 or not args.no_whole_image_pass:
        results = predict_crop(model, image, args)
        if results:
            candidates.extend(
                result_candidates(
                    results[0],
                    offset_x=0,
                    offset_y=0,
                    crop_size=image.size,
                    min_area=min_area,
                    source="whole",
                )
            )
        del results
        clear_memory(args)

    tiles = list(iter_tiles(width, height, args.tile_size or 0, args.tile_overlap))
    if args.tile_size <= 0:
        return candidates

    tile_iter = tqdm(
        tiles,
        desc="tiles",
        leave=False,
        disable=args.quiet or len(tiles) <= 1,
    )
    for x0, y0, x1, y1 in tile_iter:
        crop = image.crop((x0, y0, x1, y1))
        results = predict_crop(model, crop, args)
        if not results:
            clear_memory(args)
            continue
        candidates.extend(
            result_candidates(
                results[0],
                offset_x=x0,
                offset_y=y0,
                crop_size=crop.size,
                min_area=min_area,
                source=f"tile:{x0},{y0},{x1},{y1}",
                owner_region=tile_owner_region(
                    x0,
                    y0,
                    x1,
                    y1,
                    image_width=width,
                    image_height=height,
                    overlap=args.tile_overlap,
                ),
            )
        )
        del results
        del crop
        clear_memory(args)
    return candidates


def paste_local_mask(candidate: YoloCandidate, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    full = np.zeros((height, width), dtype=bool)
    x, y, w, h = candidate.bbox
    full[y : y + h, x : x + w] = candidate.mask[:h, :w]
    return full


def run_image(model, image_path: Path, args: argparse.Namespace) -> YoloImageResult:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    run_args, strategy = auto_plan(model, image, args)
    candidates = collect_candidates(model, image, run_args)
    candidates = dedupe_candidates(
        candidates,
        nms_iou=run_args.nms_iou,
        duplicate_mask_iou=run_args.mask_iou,
        containment=run_args.containment,
        max_instances=run_args.max_instances,
    )

    if run_args.sort == "score":
        candidates.sort(key=lambda c: c.score, reverse=True)
    else:
        candidates.sort(key=lambda c: (c.bbox[0], c.bbox[1]))

    image_dir = run_args.output_dir / image_path.stem
    mask_dir = image_dir / "instances"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in mask_dir.glob("person_*.png"):
        stale_mask.unlink()
    for stale_file in [
        image_dir / f"{image_path.stem}.json",
        image_dir / f"{image_path.stem}_instances.png",
        image_dir / f"{image_path.stem}_overlay.jpg",
    ]:
        if stale_file.exists():
            stale_file.unlink()

    masks: list[np.ndarray] = []
    instances: list[YoloInstance] = []
    for instance_id, candidate in enumerate(candidates, start=1):
        full_mask = paste_local_mask(candidate, image.size)
        bbox, box_xyxy = bbox_from_mask(full_mask)
        mask_path = mask_dir / f"person_{instance_id:03d}.png"
        Image.fromarray(full_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        masks.append(full_mask)
        instances.append(
            YoloInstance(
                id=instance_id,
                score=round(candidate.score, 6),
                bbox=bbox,
                box_xyxy=[round(v, 2) for v in box_xyxy],
                area=int(full_mask.sum()),
                mask_path=str(mask_path),
            )
        )

    if not run_args.no_label_mask:
        save_label_mask(masks, image_dir / f"{image_path.stem}_instances.png")
    if not run_args.no_overlay:
        save_overlay(image, masks, instances, image_dir / f"{image_path.stem}_overlay.jpg")

    result = YoloImageResult(
        image=str(image_path),
        prompt="person",
        width=width,
        height=height,
        inference_width=run_args.tile_size or width,
        inference_height=run_args.tile_size or height,
        checkpoint=run_args.model,
        model=run_args.model,
        strategy=strategy,
        instances=instances,
    )
    (image_dir / f"{image_path.stem}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    images = resolve_images(args.images)
    if not images:
        raise SystemExit("no input images found")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run `uv sync` after the pyproject.toml "
            "update, or install `ultralytics>=8.3` in this environment."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    all_results: list[dict] = []
    image_iter = tqdm(images, desc="images", disable=args.quiet)
    for image_path in image_iter:
        image_iter.set_postfix_str(image_path.name)
        result = run_image(model, image_path, args)
        all_results.append(asdict(result))
        print(f"{image_path.name}: {len(result.instances)} person mask(s), {result.strategy}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
