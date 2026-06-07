#!/usr/bin/env python3
"""Align a foreground person's tone to masked people in a group photo in HSV space.

The tool adjusts only pixels inside the source mask. It estimates the target tone
from one or more target person masks, so background colors do not pull the match.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-mask", type=Path, action="append", default=[])
    parser.add_argument(
        "--target-mask-glob",
        action="append",
        default=[],
        help="Glob pattern for target person masks. Can be passed multiple times.",
    )
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-saturation", type=float, default=0.08)
    parser.add_argument("--min-value", type=float, default=0.08)
    parser.add_argument("--max-hue-shift-deg", type=float, default=18.0)
    parser.add_argument("--min-saturation-ratio", type=float, default=0.75)
    parser.add_argument("--max-saturation-ratio", type=float, default=1.35)
    parser.add_argument("--min-value-ratio", type=float, default=0.90)
    parser.add_argument("--max-value-ratio", type=float, default=1.10)
    parser.add_argument(
        "--value-strength",
        type=float,
        default=0.0,
        help="0 disables V adjustment; 1 applies the full clipped value ratio.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Blend strength for hue and saturation correction.",
    )
    parser.add_argument("--min-valid-pixels", type=int, default=128)
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path))
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(img)
    if arr.ndim == 3:
        return np.any(arr[:, :, :3] > 0, axis=2)
    return arr > 0


def resolve_target_masks(paths: list[Path], patterns: list[str], size: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    mask_paths = [Path(p) for p in paths]
    for pattern in patterns:
        mask_paths.extend(Path(p) for p in glob.glob(pattern))
    unique = sorted({str(p): p for p in mask_paths}.values())
    if not unique:
        raise FileNotFoundError("no target masks matched --target-mask/--target-mask-glob")

    union = np.zeros((size[1], size[0]), dtype=bool)
    used: list[str] = []
    for path in unique:
        if not path.exists():
            continue
        mask = load_mask(path, size)
        if mask.any():
            union |= mask
            used.append(str(path))
    if not used:
        raise FileNotFoundError("target masks exist but all are empty or unreadable")
    return union, used


def rgb_to_hsv_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0


def hsv_to_rgb_arr(hsv: np.ndarray) -> np.ndarray:
    u8 = (np.clip(hsv, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(u8, mode="HSV").convert("RGB"), dtype=np.uint8)


def circular_mean_deg(hue_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    if hue_deg.size == 0:
        return 0.0
    radians = np.deg2rad(hue_deg)
    if weights is None:
        weights = np.ones_like(hue_deg, dtype=np.float32)
    sin_sum = float(np.sum(np.sin(radians) * weights))
    cos_sum = float(np.sum(np.cos(radians) * weights))
    if abs(sin_sum) < EPS and abs(cos_sum) < EPS:
        return float(np.median(hue_deg))
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def signed_hue_delta_deg(source_deg: float, target_deg: float) -> float:
    return ((target_deg - source_deg + 180.0) % 360.0) - 180.0


def clipped(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def collect_stats(hsv: np.ndarray, mask: np.ndarray, min_saturation: float, min_value: float) -> dict[str, Any]:
    valid_sv = mask & (hsv[:, :, 2] >= min_value)
    valid_hue = valid_sv & (hsv[:, :, 1] >= min_saturation)
    sv_pixels = hsv[valid_sv]
    hue_pixels = hsv[valid_hue]

    stats: dict[str, Any] = {
        "mask_pixels": int(mask.sum()),
        "valid_sv_pixels": int(valid_sv.sum()),
        "valid_hue_pixels": int(valid_hue.sum()),
    }
    if sv_pixels.size == 0:
        stats.update({
            "hue_mean_deg": 0.0,
            "saturation_median": 0.0,
            "value_median": 0.0,
        })
        return stats

    if hue_pixels.size > 0:
        hue_deg = hue_pixels[:, 0] * 360.0
        weights = np.clip(hue_pixels[:, 1], 0.0, 1.0)
        hue_mean = circular_mean_deg(hue_deg, weights)
    else:
        hue_mean = 0.0

    stats.update({
        "hue_mean_deg": round(float(hue_mean), 3),
        "saturation_median": round(float(np.median(sv_pixels[:, 1])), 6),
        "value_median": round(float(np.median(sv_pixels[:, 2])), 6),
    })
    return stats


def align_foreground_hsv(
    source_rgb: Image.Image,
    source_mask: np.ndarray,
    target_rgb: Image.Image,
    target_mask: np.ndarray,
    *,
    min_saturation: float = 0.08,
    min_value: float = 0.08,
    max_hue_shift_deg: float = 18.0,
    min_saturation_ratio: float = 0.75,
    max_saturation_ratio: float = 1.35,
    min_value_ratio: float = 0.90,
    max_value_ratio: float = 1.10,
    value_strength: float = 0.0,
    strength: float = 1.0,
    min_valid_pixels: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_hsv = rgb_to_hsv_arr(source_rgb)
    target_hsv = rgb_to_hsv_arr(target_rgb)

    source_stats = collect_stats(source_hsv, source_mask, min_saturation, min_value)
    target_stats = collect_stats(target_hsv, target_mask, min_saturation, min_value)
    warnings: list[str] = []

    if source_stats["valid_sv_pixels"] < min_valid_pixels:
        warnings.append("too few valid source pixels for stable HSV statistics")
    if target_stats["valid_sv_pixels"] < min_valid_pixels:
        warnings.append("too few valid target pixels for stable HSV statistics")
    if source_stats["valid_hue_pixels"] < min_valid_pixels:
        warnings.append("too few saturated source pixels for stable hue correction")
    if target_stats["valid_hue_pixels"] < min_valid_pixels:
        warnings.append("too few saturated target pixels for stable hue correction")

    raw_hue_shift = signed_hue_delta_deg(
        float(source_stats["hue_mean_deg"]),
        float(target_stats["hue_mean_deg"]),
    )
    hue_shift = clipped(raw_hue_shift, -max_hue_shift_deg, max_hue_shift_deg) * clipped(strength, 0.0, 1.0)

    source_sat = max(float(source_stats["saturation_median"]), EPS)
    target_sat = float(target_stats["saturation_median"])
    raw_sat_ratio = target_sat / source_sat
    sat_ratio = clipped(raw_sat_ratio, min_saturation_ratio, max_saturation_ratio)
    sat_ratio = sat_ratio ** clipped(strength, 0.0, 1.0)

    source_val = max(float(source_stats["value_median"]), EPS)
    target_val = float(target_stats["value_median"])
    raw_val_ratio = target_val / source_val
    val_ratio = clipped(raw_val_ratio, min_value_ratio, max_value_ratio)
    val_ratio = val_ratio ** clipped(value_strength, 0.0, 1.0)

    adjusted = source_hsv.copy()
    fg = source_mask.astype(bool)
    adjusted[:, :, 0][fg] = (adjusted[:, :, 0][fg] + hue_shift / 360.0) % 1.0
    adjusted[:, :, 1][fg] = np.clip(adjusted[:, :, 1][fg] * sat_ratio, 0.0, 1.0)
    adjusted[:, :, 2][fg] = np.clip(adjusted[:, :, 2][fg] * val_ratio, 0.0, 1.0)

    output = hsv_to_rgb_arr(adjusted)
    report = {
        "status": "success" if not warnings else "success_with_warnings",
        "source_stats": source_stats,
        "target_stats": target_stats,
        "correction": {
            "raw_hue_shift_deg": round(float(raw_hue_shift), 3),
            "applied_hue_shift_deg": round(float(hue_shift), 3),
            "raw_saturation_ratio": round(float(raw_sat_ratio), 6),
            "applied_saturation_ratio": round(float(sat_ratio), 6),
            "raw_value_ratio": round(float(raw_val_ratio), 6),
            "applied_value_ratio": round(float(val_ratio), 6),
            "strength": round(float(clipped(strength, 0.0, 1.0)), 3),
            "value_strength": round(float(clipped(value_strength, 0.0, 1.0)), 3),
        },
        "limits": {
            "max_hue_shift_deg": max_hue_shift_deg,
            "saturation_ratio": [min_saturation_ratio, max_saturation_ratio],
            "value_ratio": [min_value_ratio, max_value_ratio],
            "min_saturation": min_saturation,
            "min_value": min_value,
            "min_valid_pixels": min_valid_pixels,
        },
        "warnings": warnings,
    }
    return output, report


def main() -> None:
    args = parse_args()
    source_rgb = load_rgb(args.source_image)
    target_rgb = load_rgb(args.target_image)
    source_mask = load_mask(args.source_mask, source_rgb.size)
    target_mask, target_mask_paths = resolve_target_masks(args.target_mask, args.target_mask_glob, target_rgb.size)

    output, report = align_foreground_hsv(
        source_rgb,
        source_mask,
        target_rgb,
        target_mask,
        min_saturation=args.min_saturation,
        min_value=args.min_value,
        max_hue_shift_deg=args.max_hue_shift_deg,
        min_saturation_ratio=args.min_saturation_ratio,
        max_saturation_ratio=args.max_saturation_ratio,
        min_value_ratio=args.min_value_ratio,
        max_value_ratio=args.max_value_ratio,
        value_strength=args.value_strength,
        strength=args.strength,
        min_valid_pixels=args.min_valid_pixels,
    )
    report.update({
        "inputs": {
            "source_image": str(args.source_image),
            "source_mask": str(args.source_mask),
            "target_image": str(args.target_image),
            "target_masks": target_mask_paths,
        },
        "outputs": {
            "output_image": str(args.output_image),
            "report": str(args.report),
        },
    })

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output_image)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output_image": str(args.output_image),
        "report": str(args.report),
        "applied_hue_shift_deg": report["correction"]["applied_hue_shift_deg"],
        "applied_saturation_ratio": report["correction"]["applied_saturation_ratio"],
        "warnings": report["warnings"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
