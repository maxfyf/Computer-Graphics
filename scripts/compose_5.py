"""用 p1 + g1 跑 5 个候选位，每个出一张完整合成图。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PersonInserter import (
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

    t0 = time.time()
    patches = find_insertion_patches(
        group_meta_path=group_meta,
        group_image_path=group_image,
        individual_meta_path=person_meta,
        individual_image_path=person_image,
        top_k=5,
    )
    print(f"Found {len(patches)} candidate(s) in {time.time() - t0:.2f}s")
    if not patches:
        return

    for i, p in enumerate(patches):
        print(f"  #{i+1}: score={p.score:.3f} scale={p.scale:.3f} "
              f"offset={p.offset} nbrs={p.neighbors} "
              f"gap_bbox={p.gap_bbox} contour_pts={len(p.contour)} "
              f"refined_area={int(p.refined_mask.sum())}")

    out_dir = ROOT / "output" / "compose_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    group_rgb = np.asarray(ImageOps.exif_transpose(
        Image.open(group_image)).convert("RGB"), dtype=np.float32) / 255.0

    from ImageCompositor import MRFImageCompositor
    compositor = MRFImageCompositor(max_iter=200, tolerance=1e-4)

    n = min(5, len(patches))
    for i in range(n):
        patch = patches[i]
        t0 = time.time()
        composed = compose_and_paste(
            group_rgb, patch, compositor=compositor, max_crop_size=200,
        )
        out_path = out_dir / f"{group}_{person}_compose_cand{i+1}.jpg"
        Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8)).save(
            out_path, quality=85
        )
        print(f"  cand #{i+1}: wrote {out_path.name} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
