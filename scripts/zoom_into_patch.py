"""裁出候选位 + 合成结果做近距离对比。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PersonInserter import (  # noqa: E402
    CandidatePatch,
    compose_and_paste,
    find_insertion_patches,
)


def main() -> None:
    group = "g1"
    person = "p1"
    group_image = ROOT / "material" / f"{group}.jpg"
    person_image = ROOT / "material" / f"{person}.jpg"
    group_meta = ROOT / "person_metadata" / f"{group}.json"
    person_meta = ROOT / "person_metadata" / f"{person}.json"

    patches = find_insertion_patches(
        group_meta_path=group_meta,
        group_image_path=group_image,
        individual_meta_path=person_meta,
        individual_image_path=person_image,
        top_k=5,
    )
    if not patches:
        print("no patches")
        return

    out_dir = ROOT / "output" / "insertion_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(group_image)).convert("RGB")
    group_rgb = np.asarray(img, dtype=np.float32) / 255.0

    from ImageCompositor import MRFImageCompositor
    compositor = MRFImageCompositor(max_iter=300, tolerance=1e-4)

    # Top 3 candidates
    for i, patch in enumerate(patches[:3]):
        zoom = 0.5
        x0, y0, x1, y1 = patch.gap_bbox
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        w = max(800, int((x1 - x0) * 2.0))
        h = max(800, int((y1 - y0) * 2.0))
        bx0 = max(0, cx - w // 2)
        by0 = max(0, cy - h // 2)
        bx1 = min(img.width, cx + w // 2)
        by1 = min(img.height, cy + h // 2)

        before = img.crop((bx0, by0, bx1, by1))
        before.save(out_dir / f"{group}_before_{person}_cand{i+1}.jpg", quality=90)

        composed = compose_and_paste(group_rgb, patch, compositor=compositor)
        comp_img = Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8))
        after = comp_img.crop((bx0, by0, bx1, by1))
        after.save(out_dir / f"{group}_after_{person}_cand{i+1}.jpg", quality=90)
        print(f"  cand #{i+1}: bbox gap={patch.gap_bbox} contour_pts={len(patch.contour)} scale={patch.scale:.3f}")


if __name__ == "__main__":
    main()
