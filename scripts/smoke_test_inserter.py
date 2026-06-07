"""冒烟测试：用真实 g1 + p1 数据跑 find_insertion_patches，可视化 + 一次 compose。"""
from __future__ import annotations

import sys
import time
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


def draw_candidates_overlay(
    group_image_path: Path,
    patches: list[CandidatePatch],
    output_path: Path,
    top_n: int = 5,
) -> None:
    img = ImageOps.exif_transpose(Image.open(group_image_path)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    for i, p in enumerate(patches[:top_n]):
        color = palette[i % len(palette)] + (180,)
        x0, y0, x1, y1 = p.gap_bbox
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=8)
        fx, fy, fw, fh = p.target_face_bbox
        draw.rectangle([fx, fy, fx + fw - 1, fy + fh - 1], outline=color, width=6)
        # contour
        for y, x in p.contour[::20]:
            draw.point((int(x), int(y)), fill=color)
        draw.text((x0 + 6, max(0, y0 - 30)),
                  f"#{i+1} score={p.score:.2f} scale={p.scale:.2f} nbr={p.neighbors}",
                  fill=color)
    out = Image.alpha_composite(img, overlay)
    out.convert("RGB").save(output_path, quality=85)
    print(f"  wrote {output_path}")


def main() -> None:
    group = "g1"
    person = "p1"
    group_image = ROOT / "material" / f"{group}.jpg"
    person_image = ROOT / "material" / f"{person}.jpg"
    group_meta = ROOT / "person_metadata" / f"{group}.json"
    person_meta = ROOT / "person_metadata" / f"{person}.json"

    print(f"Group: {group} ({group_image})")
    print(f"Person: {person} ({person_image})")

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
        print("No candidates. Aborting.")
        return

    for i, p in enumerate(patches):
        print(f"  #{i+1}: score={p.score:.3f} scale={p.scale:.3f} "
              f"offset={p.offset} nbrs={p.neighbors} "
              f"gap_bbox={p.gap_bbox} contour_pts={len(p.contour)} "
              f"refined_area={int(p.refined_mask.sum())}")

    out_dir = ROOT / "output" / "insertion_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving candidates overlay...")
    draw_candidates_overlay(
        group_image, patches,
        out_dir / f"{group}_candidates_{person}.jpg",
    )

    print("\nRunning compose_and_paste on top candidate (this may take a while)...")
    t0 = time.time()
    group_rgb = np.asarray(ImageOps.exif_transpose(
        Image.open(group_image)).convert("RGB"), dtype=np.float32) / 255.0
    try:
        from ImageCompositor import MRFImageCompositor
        compositor = MRFImageCompositor(max_iter=200, tolerance=1e-4)
        composed = compose_and_paste(group_rgb, patches[0], compositor=compositor)
        out_img = Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8))
        out_img.save(out_dir / f"{group}_composed_{person}.jpg", quality=85)
        print(f"  compose done in {time.time() - t0:.2f}s")
        print(f"  wrote {out_dir / (group + '_composed_' + person + '.jpg')}")
    except Exception as exc:
        print(f"  compose failed: {exc!r}")


if __name__ == "__main__":
    main()
