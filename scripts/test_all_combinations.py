"""对 3 组 (group, person) 组合跑 find_insertion_patches，验证系统通用性。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PersonInserter import find_insertion_patches  # noqa: E402


def main() -> None:
    pairs = [("g1", "p1"), ("g2", "p2"), ("g3", "p3")]
    for g, p in pairs:
        group_image = ROOT / "material" / f"{g}.jpg"
        person_image = ROOT / "material" / f"{p}.jpg"
        group_meta = ROOT / "person_metadata" / f"{g}.json"
        person_meta = ROOT / "person_metadata" / f"{p}.json"
        t0 = time.time()
        try:
            patches = find_insertion_patches(
                group_meta_path=group_meta,
                group_image_path=group_image,
                individual_meta_path=person_meta,
                individual_image_path=person_image,
                top_k=5,
            )
            dt = time.time() - t0
            if not patches:
                print(f"[{g}+{p}] no candidates in {dt:.2f}s")
                continue
            top = patches[0]
            print(f"[{g}+{p}] {len(patches)} patches in {dt:.2f}s | top: "
                  f"score={top.score:.3f} scale={top.scale:.3f} "
                  f"nbrs={top.neighbors} refined_area={int(top.refined_mask.sum())}")
        except Exception as exc:
            print(f"[{g}+{p}] ERROR: {exc!r}")


if __name__ == "__main__":
    main()
