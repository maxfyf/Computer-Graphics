"""Extract person contours and face metadata from images and optional masks.

This script intentionally separates model inference from geometry extraction:

* If a person mask is available, connected components are treated as persons
  and boundary pixels are exported as contours.
* If no mask is available, face metadata can still be estimated with the
  scikit-image LBP face cascade, but person contours cannot be recovered
  reliably without a segmentation model.

Example:
    python PersonMetadataExtractor.py material/*.jpg --output-dir output/person_metadata --detect-faces
    python PersonMetadataExtractor.py material/*.jpg --mask-dir masks --output-dir output/person_metadata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


@dataclass
class FaceMetadata:
    center: list[float]
    size: list[int]
    bbox: list[int]
    confidence: float | None = None


@dataclass
class PersonMetadata:
    id: int
    bbox: list[int] | None
    area: int
    contour_pixels: list[list[int]]
    face: FaceMetadata | None


@dataclass
class ImageMetadata:
    image: str
    width: int
    height: int
    mask: str | None
    persons: list[PersonMetadata]
    warnings: list[str]


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask_img = ImageOps.exif_transpose(Image.open(path))
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.Resampling.NEAREST)

    mask_arr = np.asarray(mask_img)
    if mask_arr.ndim == 3:
        mask = np.any(mask_arr[:, :, :3] > 0, axis=2)
    else:
        mask = mask_arr > 0

    return mask


def find_matching_mask(mask_dir: Path | None, image_path: Path) -> Path | None:
    if mask_dir is None:
        return None

    candidates = [
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}.jpg",
        mask_dir / f"{image_path.stem}.jpeg",
        mask_dir / f"{image_path.name}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def component_bbox(component: np.ndarray) -> list[int] | None:
    ys, xs = np.where(component)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def contour_pixels(component: np.ndarray, stride: int = 1) -> list[list[int]]:
    eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3)), border_value=0)
    boundary = component & ~eroded
    ys, xs = np.where(boundary)
    points = np.column_stack([xs, ys])
    if stride > 1:
        points = points[::stride]
    return points.astype(int).tolist()


def split_person_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    labels, count = ndimage.label(mask)
    components = []
    for label_id in range(1, count + 1):
        component = labels == label_id
        if int(component.sum()) >= min_area:
            components.append(component)
    components.sort(key=lambda c: component_bbox(c)[0] if component_bbox(c) else 0)
    return components


def detect_faces(image: Image.Image, max_dim: int) -> list[FaceMetadata]:
    try:
        from skimage import color, data, feature, transform
    except Exception:
        return []

    arr = np.asarray(image)
    height, width = arr.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        small = transform.resize(
            arr,
            (max(1, int(height * scale)), max(1, int(width * scale))),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
    else:
        small = arr

    gray = color.rgb2gray(small)
    cascade = feature.Cascade(data.lbp_frontal_face_cascade_filename())
    detections = cascade.detect_multi_scale(
        gray,
        scale_factor=1.15,
        step_ratio=0.08,
        min_size=(18, 18),
        max_size=(180, 180),
    )

    faces: list[FaceMetadata] = []
    for item in detections:
        x = int(round(item["c"] / scale))
        y = int(round(item["r"] / scale))
        w = int(round(item["width"] / scale))
        h = int(round(item["height"] / scale))
        faces.append(
            FaceMetadata(
                center=[round(x + w / 2, 2), round(y + h / 2, 2)],
                size=[w, h],
                bbox=[x, y, w, h],
                confidence=None,
            )
        )
    return faces


def face_center_in_bbox(face: FaceMetadata, bbox: list[int] | None) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    cx, cy = face.center
    return x <= cx <= x + w and y <= cy <= y + h


def assign_faces_to_persons(
    persons: list[PersonMetadata], faces: list[FaceMetadata]
) -> list[FaceMetadata]:
    remaining: list[FaceMetadata] = []
    for face in faces:
        matched = False
        for person in persons:
            if person.face is None and face_center_in_bbox(face, person.bbox):
                person.face = face
                matched = True
                break
        if not matched:
            remaining.append(face)
    return remaining


def extract_image_metadata(
    image_path: Path,
    mask_path: Path | None,
    min_area: int,
    contour_stride: int,
    face_max_dim: int,
    detect_faces_enabled: bool,
) -> ImageMetadata:
    image = load_image(image_path)
    width, height = image.size
    warnings: list[str] = []
    faces = detect_faces(image, face_max_dim) if detect_faces_enabled else []
    persons: list[PersonMetadata] = []

    if mask_path is None:
        warnings.append("no mask supplied; contours were not extracted")
        if not detect_faces_enabled:
            warnings.append("face detection disabled; pass --detect-faces to estimate face boxes")
        for idx, face in enumerate(faces, start=1):
            persons.append(
                PersonMetadata(
                    id=idx,
                    bbox=None,
                    area=0,
                    contour_pixels=[],
                    face=face,
                )
            )
    else:
        mask = load_mask(mask_path, image.size)
        components = split_person_components(mask, min_area=min_area)
        if not components:
            warnings.append("mask has no component above min_area")
        for idx, component in enumerate(components, start=1):
            persons.append(
                PersonMetadata(
                    id=idx,
                    bbox=component_bbox(component),
                    area=int(component.sum()),
                    contour_pixels=contour_pixels(component, stride=contour_stride),
                    face=None,
                )
            )
        unmatched_faces = assign_faces_to_persons(persons, faces)
        if unmatched_faces:
            warnings.append(f"{len(unmatched_faces)} detected face(s) were not inside any mask bbox")

    return ImageMetadata(
        image=str(image_path),
        width=width,
        height=height,
        mask=str(mask_path) if mask_path else None,
        persons=persons,
        warnings=warnings,
    )


def draw_debug_overlay(image_path: Path, metadata: ImageMetadata, output_path: Path) -> None:
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)
    for person in metadata.persons:
        if person.bbox is not None:
            x, y, w, h = person.bbox
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 255, 0), width=6)
        if person.contour_pixels:
            step = max(1, len(person.contour_pixels) // 3000)
            for x, y in person.contour_pixels[::step]:
                draw.point((x, y), fill=(0, 255, 255))
        if person.face is not None:
            x, y, w, h = person.face.bbox
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 0, 0), width=5)
            cx, cy = person.face.center
            r = max(4, int(max(w, h) * 0.06))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Input image paths")
    parser.add_argument("--mask-dir", type=Path, help="Directory with masks named after images")
    parser.add_argument("--output-dir", type=Path, default=Path("output/person_metadata"))
    parser.add_argument("--min-area", type=int, default=500, help="Minimum mask component area")
    parser.add_argument(
        "--contour-stride",
        type=int,
        default=1,
        help="Keep every Nth contour point. Use 1 to keep all boundary pixels.",
    )
    parser.add_argument(
        "--face-max-dim",
        type=int,
        default=720,
        help="Downscale longest side to this value for face detection.",
    )
    parser.add_argument(
        "--detect-faces",
        action="store_true",
        help="Enable scikit-image LBP face detection. This is slow and less accurate than a model.",
    )
    parser.add_argument("--no-debug-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata: list[dict] = []

    for image_path in args.images:
        mask_path = find_matching_mask(args.mask_dir, image_path)
        metadata = extract_image_metadata(
            image_path=image_path,
            mask_path=mask_path,
            min_area=args.min_area,
            contour_stride=max(1, args.contour_stride),
            face_max_dim=args.face_max_dim,
            detect_faces_enabled=args.detect_faces,
        )
        all_metadata.append(asdict(metadata))

        json_path = args.output_dir / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not args.no_debug_images:
            draw_debug_overlay(image_path, metadata, args.output_dir / f"{image_path.stem}_debug.jpg")

        print(
            f"{image_path.name}: {len(metadata.persons)} person record(s), "
            f"{sum(1 for p in metadata.persons if p.face is not None)} face(s)"
        )
        for warning in metadata.warnings:
            print(f"  warning: {warning}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
