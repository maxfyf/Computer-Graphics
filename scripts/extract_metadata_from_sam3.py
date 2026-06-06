"""Build person metadata from SAM3 instance-mask outputs.

Input layout expected from scripts/generate_sam3_masks.py:

    sam3_masks/g1/g1.json
    sam3_masks/g1/instances/person_001.png

Output:

    output/person_metadata/g1.json
    output/person_metadata/summary.json

The face box is a geometry estimate from the person mask/bbox. It is useful as
a stable placeholder for compositing, but should be replaced by a real face
detector if exact face boxes are required.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


@dataclass
class FaceMetadata:
    center: list[float]
    size: list[int]
    bbox: list[int]
    confidence: float | None
    method: str


@dataclass
class PersonMetadata:
    id: int
    source_id: int
    score: float
    bbox: list[int]
    area: int
    mask_path: str
    contour_pixels: list[list[int]]
    face: FaceMetadata


@dataclass
class ImageMetadata:
    image: str
    width: int
    height: int
    source_metadata: str
    persons: list[PersonMetadata]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-dir", type=Path, default=Path("sam3_masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/person_metadata"))
    parser.add_argument(
        "--contour-stride",
        type=int,
        default=1,
        help="Keep every Nth contour pixel. Use 1 to keep all boundary pixels.",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude a source instance id, e.g. --exclude g1:11. Can be repeated.",
    )
    parser.add_argument("--no-debug-images", action="store_true")
    return parser.parse_args()


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask_img = ImageOps.exif_transpose(Image.open(path))
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask_img) > 0


def contour_pixels(mask: np.ndarray, stride: int) -> list[list[int]]:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = mask & ~eroded
    ys, xs = np.where(boundary)
    points = np.column_stack([xs, ys])
    if stride > 1:
        points = points[::stride]
    return points.astype(int).tolist()


def estimate_face(mask: np.ndarray, bbox: list[int]) -> FaceMetadata:
    x, y, w, h = bbox
    face_h = int(round(min(0.16 * h, 0.38 * w)))
    face_h = max(12, face_h)
    face_w = max(10, int(round(face_h * 0.8)))

    band_y0 = y
    band_y1 = min(mask.shape[0], int(round(y + max(face_h * 1.7, h * 0.24))))
    band = mask[band_y0:band_y1, max(0, x) : min(mask.shape[1], x + w)]

    if band.any():
        ys, xs = np.where(band)
        center_x = float(x + np.median(xs))
    else:
        center_x = x + w / 2

    center_y = y + face_h / 2 + h * 0.02
    center_x = min(max(center_x, x + face_w / 2), x + w - face_w / 2)
    center_y = min(max(center_y, y + face_h / 2), y + h - face_h / 2)

    face_x = int(round(center_x - face_w / 2))
    face_y = int(round(center_y - face_h / 2))
    return FaceMetadata(
        center=[round(center_x, 2), round(center_y, 2)],
        size=[face_w, face_h],
        bbox=[face_x, face_y, face_w, face_h],
        confidence=None,
        method="mask_bbox_heuristic",
    )


def parse_exclusions(values: list[str]) -> set[tuple[str, int]]:
    exclusions: set[tuple[str, int]] = set()
    for value in values:
        try:
            image_id, instance_id = value.split(":", 1)
            exclusions.add((image_id, int(instance_id)))
        except ValueError as exc:
            raise SystemExit(f"invalid --exclude value: {value}") from exc
    return exclusions


def draw_debug_overlay(
    source_image: Path,
    metadata: ImageMetadata,
    output_path: Path,
) -> None:
    image = ImageOps.exif_transpose(Image.open(source_image)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for person in metadata.persons:
        x, y, w, h = person.bbox
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 255, 0), width=5)
        fx, fy, fw, fh = person.face.bbox
        draw.rectangle([fx, fy, fx + fw - 1, fy + fh - 1], outline=(255, 0, 0), width=4)
        cx, cy = person.face.center
        r = max(3, int(max(fw, fh) * 0.08))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=3)
        draw.text((x + 4, y + 4), str(person.id), fill=(255, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def convert_image(
    metadata_path: Path,
    output_dir: Path,
    exclusions: set[tuple[str, int]],
    min_score: float,
    min_area: int,
    contour_stride: int,
    debug_images: bool,
) -> ImageMetadata:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_id = metadata_path.parent.name
    image_path = Path(data["image"])
    width, height = int(data["width"]), int(data["height"])
    warnings: list[str] = []
    persons: list[PersonMetadata] = []

    for source in data.get("instances", []):
        source_id = int(source["id"])
        score = float(source["score"])
        area = int(source["area"])
        if (image_id, source_id) in exclusions:
            warnings.append(f"excluded source instance {source_id}")
            continue
        if score < min_score:
            warnings.append(f"filtered source instance {source_id}: score {score:.3f} < {min_score:.3f}")
            continue
        if area < min_area:
            warnings.append(f"filtered source instance {source_id}: area {area} < {min_area}")
            continue

        mask_path = Path(source["mask_path"])
        if not mask_path.exists():
            fallback = metadata_path.parent / "instances" / mask_path.name
            if fallback.exists():
                mask_path = fallback
            else:
                raise FileNotFoundError(f"mask not found: {source['mask_path']}")
        mask = load_mask(mask_path, (width, height))
        bbox = [int(v) for v in source["bbox"]]
        persons.append(
            PersonMetadata(
                id=len(persons) + 1,
                source_id=source_id,
                score=round(score, 6),
                bbox=bbox,
                area=int(mask.sum()),
                mask_path=str(mask_path),
                contour_pixels=contour_pixels(mask, max(1, contour_stride)),
                face=estimate_face(mask, bbox),
            )
        )

    result = ImageMetadata(
        image=str(image_path),
        width=width,
        height=height,
        source_metadata=str(metadata_path),
        persons=persons,
        warnings=warnings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{image_id}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if debug_images:
        draw_debug_overlay(image_path, result, output_dir / f"{image_id}_debug.jpg")
    return result


def main() -> None:
    args = parse_args()
    exclusions = parse_exclusions(args.exclude)
    metadata_files = sorted(p for p in args.sam3_dir.glob("*/*.json") if p.name != "summary.json")
    if not metadata_files:
        raise SystemExit(f"no SAM3 metadata files found under {args.sam3_dir}")

    results: list[dict] = []
    for metadata_path in metadata_files:
        result = convert_image(
            metadata_path=metadata_path,
            output_dir=args.output_dir,
            exclusions=exclusions,
            min_score=args.min_score,
            min_area=args.min_area,
            contour_stride=args.contour_stride,
            debug_images=not args.no_debug_images,
        )
        results.append(asdict(result))
        print(
            f"{metadata_path.parent.name}: {len(result.persons)} person(s), "
            f"{len(result.warnings)} warning(s)"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
