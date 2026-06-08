"""为集体照补人：寻找单人照适合插入到合照的几个候选位置，输出调 compose 所需数据。

输入:
  - 合照元数据 (person_metadata/<group>.json) + 合照图片 (material/<group>.jpg)
  - 单人照元数据 (person_metadata/<person>.json) + 单人照图片 (material/<person>.jpg)
  - 单人照 SAM3 mask (sam3_masks/<person>/instances/person_001.png)，可省略自动找

输出:
  - list[CandidatePatch]，按 score 降序。每个含:
    - target_rgb: 合照 RGB float [0,1]
    - source_rgb: 缩放后单人 RGB float [0,1]
    - source_mask: 缩放后单人 mask bool（同 source_rgb 形状）
    - refined_mask: 精炼后插入人物 mask（在 target_rgb 坐标下）
    - offset: source_rgb 左上角在 target_rgb 的 (y, x)
    - contour: 精炼后边界像素 (y, x) 数组，在 target_rgb 坐标
    - scale / target_face_bbox / gap_bbox / score / neighbors / warnings

下一步用 compose_and_paste(group_rgb, patch) 直接拿到合成图。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


# ==================== 数据类 ====================

@dataclass
class GapInfo:
    """一行中相邻两人之间或行端的空位。"""
    row_index: int
    left_id: int | None        # 左侧邻接人 id；None 表示行首空位
    right_id: int | None       # 右侧邻接人 id；None 表示行尾空位
    face_x: float              # 空位中心 face x
    face_y: float              # 目标 face y
    face_h: float              # 目标 face h
    face_w: float              # 目标 face w
    bbox: list[int]            # [x0, y0, x1, y1] 空位 bbox
    width: float               # 空位水平宽度


@dataclass
class CandidatePatch:
    """一个候选插入位。"""
    target_rgb: np.ndarray         # (H, W, 3) float [0,1]
    source_rgb: np.ndarray         # (h, w, 3) float [0,1]
    source_mask: np.ndarray        # (h, w) bool
    refined_mask: np.ndarray       # (H, W) bool，target 坐标系
    offset: tuple[int, int]        # source_rgb 左上角在 target 坐标的 (y, x)
    contour: np.ndarray            # (N, 2) int32 边界 (y, x)，target 坐标
    scale: float
    target_face_bbox: list[int]    # [x, y, w, h] target 坐标
    gap_bbox: list[int]            # [x0, y0, x1, y1] target 坐标
    score: float
    neighbors: list[int]
    warnings: list[str] = field(default_factory=list)


# ==================== I/O helpers ====================

def _load_rgb(path: Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    m = ImageOps.exif_transpose(Image.open(path))
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(m)
    if arr.ndim == 3:
        return np.any(arr[:, :, :3] > 0, axis=2)
    return arr > 0


def _resolve_individual_mask(
    individual_meta_path: Path,
    provided: Path | None,
) -> Path:
    if provided is not None:
        if not provided.exists():
            raise FileNotFoundError(f"individual mask not found: {provided}")
        return provided

    data = json.loads(individual_meta_path.read_text(encoding="utf-8"))
    src = data.get("source_metadata")
    if src:
        src_path = Path(src)
        if not src_path.is_absolute():
            # person_metadata/<stem>.json -> ../sam3_masks/<stem>/<stem>.json
            candidate = individual_meta_path.parent.parent / src_path
            if candidate.exists():
                instances_dir = candidate.parent / "instances"
                pngs = sorted(instances_dir.glob("person_*.png"))
                if pngs:
                    return pngs[0]
        else:
            instances_dir = src_path.parent / "instances"
            pngs = sorted(instances_dir.glob("person_*.png"))
            if pngs:
                return pngs[0]

    stem = individual_meta_path.stem
    for base in [Path("sam3_masks"), Path("output/sam3_masks")]:
        p = base / stem / "instances" / "person_001.png"
        if p.exists():
            return p

    raise FileNotFoundError(
        f"cannot locate individual mask for {individual_meta_path}; "
        "pass individual_mask_path explicitly"
    )


# ==================== 行聚类 & 空位检测 ====================

def _cluster_rows(
    persons: list[dict[str, Any]],
    image_height: int,
) -> list[list[dict[str, Any]]]:
    """按 face_h 1-D KMeans (K=2) 聚类成行。face_h 是深度代理（脸大 = 靠前 = 同一排）。

    试每个候选分割点，选 within-cluster sum of squares 最小的那一个。
    比固定 ratio 阈值更鲁棒 — 适应 g2 那种 face_h 连续分布（30-88，几乎没 gap）。
    """
    if not persons:
        return []
    n = len(persons)
    if n < 2:
        return [persons]
    sorted_hs = sorted(p["face"]["bbox"][3] for p in persons)
    if sorted_hs[0] <= 0:
        return [persons]

    # K=2 KMeans：找最小 within-cluster variance 的分割
    best_score = float("inf")
    best_k = 1
    for k in range(1, n):
        left = sorted_hs[:k]
        right = sorted_hs[k:]
        lm = sum(left) / len(left)
        rm = sum(right) / len(right)
        score = sum((x - lm) ** 2 for x in left) + sum((x - rm) ** 2 for x in right)
        if score < best_score:
            best_score = score
            best_k = k
    threshold = (sorted_hs[best_k - 1] + sorted_hs[best_k]) / 2.0

    rows: list[list[dict[str, Any]]] = [[], []]
    for p in persons:
        if p["face"]["bbox"][3] < threshold:
            rows[0].append(p)
        else:
            rows[1].append(p)
    for row in rows:
        row.sort(key=lambda p: p["face"]["center"][0])
    return [r for r in rows if r]


def _find_row_gaps(row: list[dict[str, Any]], image_width: int) -> list[GapInfo]:
    """行内找空位：行首、邻接中点、行尾。"""
    if not row:
        return []
    row_index = id(row)  # 由调用方覆盖
    y_values = [p["bbox"][1] for p in row]
    y1_values = [p["bbox"][1] + p["bbox"][3] for p in row]
    y0 = int(min(y_values))
    y1 = int(max(y1_values))

    gaps: list[GapInfo] = []

    # 行首空位
    left = row[0]
    half_w = left["bbox"][2] / 2
    left_face_x = left["face"]["center"][0]
    gaps.append(GapInfo(
        row_index=row_index,
        left_id=None,
        right_id=left["id"],
        face_x=max(0.0, left_face_x - half_w),
        face_y=left["face"]["center"][1],
        face_h=float(left["face"]["bbox"][3]),
        face_w=float(left["face"]["bbox"][2]),
        bbox=[max(0, int(left_face_x - half_w * 2)), y0, int(left_face_x), y1],
        width=half_w,
    ))

    # 行内邻接空位
    # 跨排过滤：邻接两人 face y 差或 face h 比例过大视为不同排
    # （行聚类边界漏网的情况，比如 face y 接近但脸大小差很多 → 不同排）
    cross_row_y_diff = max(40.0, 0.02 * (y1 - y0))
    cross_row_h_ratio = 1.5
    for i in range(len(row) - 1):
        a = row[i]
        b = row[i + 1]
        a_y = a["face"]["center"][1]
        b_y = b["face"]["center"][1]
        a_h = float(a["face"]["bbox"][3])
        b_h = float(b["face"]["bbox"][3])
        if abs(a_y - b_y) > cross_row_y_diff:
            continue
        if a_h > 0 and b_h > 0:
            ratio = max(a_h, b_h) / min(a_h, b_h)
            if ratio > cross_row_h_ratio:
                # 一个脸大一个脸小，肯定不同排
                continue
        a_x = a["face"]["center"][0]
        b_x = b["face"]["center"][0]
        a_w = float(a["face"]["bbox"][2])
        b_w = float(b["face"]["bbox"][2])
        mid_x = (a_x + b_x) / 2
        gaps.append(GapInfo(
            row_index=row_index,
            left_id=a["id"],
            right_id=b["id"],
            face_x=mid_x,
            face_y=(a_y + b_y) / 2,
            face_h=(a_h + b_h) / 2,
            face_w=(a_w + b_w) / 2,
            bbox=[int(a_x), y0, int(b_x), y1],
            width=float(b_x - a_x),
        ))

    # 行尾空位
    right = row[-1]
    half_w = right["bbox"][2] / 2
    right_face_x = right["face"]["center"][0]
    gaps.append(GapInfo(
        row_index=row_index,
        left_id=right["id"],
        right_id=None,
        face_x=min(float(image_width), right_face_x + half_w),
        face_y=right["face"]["center"][1],
        face_h=float(right["face"]["bbox"][3]),
        face_w=float(right["face"]["bbox"][2]),
        bbox=[int(right_face_x), y0,
              min(image_width, int(right_face_x + half_w * 2)), y1],
        width=half_w,
    ))

    return gaps


# ==================== 缩放 & 放置 & 精炼 ====================

def _scale_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(scale - 1.0) < 1e-6:
        return image, mask
    h, w = image.shape[:2]
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    img_u8 = np.clip(image, 0.0, 1.0)
    img_u8 = (img_u8 * 255.0 + 0.5).astype(np.uint8)
    img_pil = Image.fromarray(img_u8).resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (new_w, new_h), Image.Resampling.NEAREST
    )
    return (
        np.asarray(img_pil, dtype=np.float32) / 255.0,
        np.asarray(mask_pil) > 0,
    )


def _existing_bbox_mask(persons: list[dict[str, Any]], size: tuple[int, int]) -> np.ndarray:
    """合照中所有人 bbox 矩形的并集 mask（fallback：SAM3 mask 拿不到时用）。"""
    H, W = size
    out = np.zeros((H, W), dtype=bool)
    for p in persons:
        x, y, w, h = p["bbox"]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x + w))
        y1 = min(H, int(y + h))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = True
    return out


def _load_existing_depth(
    group_meta_path: Path,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """从合照的 source_metadata 找到 SAM3 instance mask 目录，加载每个人真实 mask。

    返回 (union, depth_map)：
      - union[y, x] = True 当有任何人在该像素
      - depth_map[y, x] = 该像素所有人中最大的 face_h（深度代理，脸大 = 更靠前 = 更靠近相机）
    若 SAM3 instance mask 拿不到则返回 (zeros, None)，调用方应回落到 bbox。
    """
    H, W = size
    union = np.zeros((H, W), dtype=bool)
    depth = np.zeros((H, W), dtype=np.float32)

    if not group_meta_path.exists():
        return union, None
    group_data = json.loads(group_meta_path.read_text(encoding="utf-8"))
    persons: list[dict[str, Any]] = group_data.get("persons", [])
    if not persons:
        return union, None

    src = group_data.get("source_metadata")
    if not src:
        return union, None
    src_path = Path(src)
    if not src_path.is_absolute():
        # person_metadata/<stem>.json -> ../sam3_masks/<stem>/<stem>.json
        candidate = group_meta_path.parent.parent / src_path
        if candidate.exists():
            src_path = candidate
        else:
            return union, None
    instances_dir = src_path.parent / "instances"
    if not instances_dir.exists():
        return union, None

    loaded = False
    for p in persons:
        sid = p.get("source_id", p["id"])
        mask_file = instances_dir / f"person_{sid:03d}.png"
        if not mask_file.exists():
            continue
        m_img = Image.open(mask_file)
        if m_img.size != (W, H):
            m_img = m_img.resize((W, H), Image.Resampling.NEAREST)
        m = np.asarray(m_img) > 0
        if not m.any():
            del m
            continue
        loaded = True
        face_h = float(p["face"]["bbox"][3])
        # depth = max(existing, face_h) 在 m 范围内
        update = np.where(m, face_h, 0.0)
        np.maximum(depth, update, out=depth)
        union |= m
        del m, update, m_img

    if not loaded:
        return union, None
    return union, depth


def _rank_rows_by_face_y(
    rows: list[list[dict[str, Any]]],
) -> dict[int, int]:
    """给每个 row_index 排一个深度等级：行内平均 face_y 越大 = 越靠前（rank 越大）。

    不用 face_h 是因为小人物（后排前排都可能站着小孩）脸小 ≠ 靠前。
    face_y 才是稳定的深度信号：合照里前排站在地面上、脸在图下方 → face_y 大。

    rank 约定：rank 0 = 最远（后排），rank N-1 = 最近（前排）。
    """
    row_avg: list[tuple[int, float]] = []
    for r_idx, row in enumerate(rows):
        if not row:
            row_avg.append((r_idx, 0.0))
            continue
        avg = sum(p["face"]["center"][1] for p in row) / len(row)
        row_avg.append((r_idx, avg))
    # 升序：face_y 小的 = 远 = 排在前面（rank 小）
    sorted_by_avg = sorted(row_avg, key=lambda t: t[1])
    return {r_idx: rank for rank, (r_idx, _) in enumerate(sorted_by_avg)}


def _build_row_occlusion(
    group_meta_path: Path,
    persons: list[dict[str, Any]],
    rows: list[list[dict[str, Any]]],
    row_rank: dict[int, int],
    placed_row_index: int,
    size: tuple[int, int],
    dilation_px: int = 4,
) -> np.ndarray:
    """行级 z-order 遮挡：只用 placed 行**之前**的行的 SAM3 instance mask 做遮挡。

    - 同排 (rank 相等)：不遮挡（侧身挨着）
    - 后排 (rank 较小)：不遮挡（插入人物在前面）
    - 前排 (rank 较大)：用真实 mask 遮挡（他们在前面，盖住插入人物身体）
    """
    H, W = size
    out = np.zeros((H, W), dtype=bool)

    if placed_row_index not in row_rank:
        return out
    placed_rank = row_rank[placed_row_index]
    if placed_rank >= max(row_rank.values()):
        return out  # 已经是最后（最前）一排，无人挡

    person_rank: dict[int, int] = {}
    for r_idx, row in enumerate(rows):
        for p in row:
            person_rank[p["id"]] = row_rank[r_idx]

    if not group_meta_path.exists():
        return out
    group_data = json.loads(group_meta_path.read_text(encoding="utf-8"))
    src = group_data.get("source_metadata")
    if not src:
        return out
    src_path = Path(src)
    if not src_path.is_absolute():
        candidate = group_meta_path.parent.parent / src_path
        if candidate.exists():
            src_path = candidate
        else:
            return out
    instances_dir = src_path.parent / "instances"
    if not instances_dir.exists():
        return out

    # 遮挡规则：
    #   - 同排 (rank == placed_rank): 已有的人不挡目标（目标在同排里加入，侧身挨着）
    #   - 前排 (rank > placed_rank):  他们更靠前，遮挡目标
    #   - 后排 (rank < placed_rank):  目标在他们前面，不进 occlusion
    for p in persons:
        pid = p["id"]
        if pid not in person_rank or person_rank[pid] <= placed_rank:
            continue
        sid = p.get("source_id", pid)
        mask_file = instances_dir / f"person_{sid:03d}.png"
        if not mask_file.exists():
            continue
        m_img = Image.open(mask_file)
        if m_img.size != (W, H):
            m_img = m_img.resize((W, H), Image.Resampling.NEAREST)
        m = np.asarray(m_img) > 0
        if m.any():
            out |= m
        del m, m_img

    if dilation_px > 0 and out.any():
        out = ndimage.binary_dilation(out, iterations=dilation_px)
    return out


def _refine_overlap_with_depth(
    placed_mask: np.ndarray,
    place_offset: tuple[int, int],
    depth_map: np.ndarray,
    placed_face_h: float,
    depth_threshold: float = 1.3,
    dilation_px: int = 4,
) -> np.ndarray:
    """深度感知遮挡：只让"实际比插入人物更靠前"的人做遮挡。

    规则：existing person 的 face_h > placed_face_h * depth_threshold 才算"在前面"。
    其余人（同排 / 后排）不遮挡。
    再做几像素的 dilation 防止 mask 边缘漏。
    """
    H, W = depth_map.shape
    py, px = place_offset
    ph, pw = placed_mask.shape

    threshold = placed_face_h * depth_threshold
    occlude = depth_map > threshold
    if dilation_px > 0:
        occlude = ndimage.binary_dilation(occlude, iterations=dilation_px)

    out = np.zeros((H, W), dtype=bool)
    y0 = max(0, py)
    x0 = max(0, px)
    y1 = min(H, py + ph)
    x1 = min(W, px + pw)
    if y1 <= y0 or x1 <= x0:
        return out
    sy = y0 - py
    sx = x0 - px
    region = placed_mask[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    out[y0:y1, x0:x1] = region & ~occlude[y0:y1, x0:x1]
    return out


def _refine_overlap_mask(
    placed_mask: np.ndarray,
    place_offset: tuple[int, int],
    existing_mask: np.ndarray,
) -> np.ndarray:
    """把 placed_mask 放在 place_offset 后，与 existing_mask 求交精炼（无深度信息 fallback）。"""
    py, px = place_offset
    ph, pw = placed_mask.shape
    H, W = existing_mask.shape
    out = np.zeros((H, W), dtype=bool)
    y0 = max(0, py)
    x0 = max(0, px)
    y1 = min(H, py + ph)
    x1 = min(W, px + pw)
    if y1 <= y0 or x1 <= x0:
        return out
    sy = y0 - py
    sx = x0 - px
    region = placed_mask[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    out[y0:y1, x0:x1] = region & ~existing_mask[y0:y1, x0:x1]
    return out


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """从 mask 提取 (N, 2) (y, x) int32 数组。"""
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = mask & ~eroded
    ys, xs = np.where(boundary)
    return np.column_stack([ys, xs]).astype(np.int32)


def _score_gap(gap: GapInfo, individual_body_h: float, scale: float) -> float:
    required = individual_body_h * scale
    if required <= 0:
        return 0.0
    return gap.width / required


def _compute_patch(
    gap: GapInfo,
    group_rgb: np.ndarray,
    individual_rgb: np.ndarray,
    individual_mask: np.ndarray,
    individual_face: dict[str, Any],
    individual_body_h: float,
    occlusion_mask: np.ndarray,
) -> CandidatePatch | None:
    individual_face_h = float(individual_face["bbox"][3])
    individual_face_w = float(individual_face["bbox"][2])
    if individual_face_h <= 0 or individual_face_w <= 0:
        return None

    scale_h = gap.face_h / individual_face_h
    scale_w = gap.face_w / individual_face_w
    scale = (scale_h + scale_w) / 2.0
    if scale <= 0:
        return None

    scaled_rgb, scaled_mask = _scale_image_and_mask(individual_rgb, individual_mask, scale)

    indiv_face_x = float(individual_face["center"][0])
    indiv_face_y = float(individual_face["center"][1])
    target_face_x = gap.face_x
    target_face_y = gap.face_y
    ox = int(round(target_face_x - indiv_face_x * scale))
    oy = int(round(target_face_y - indiv_face_y * scale))

    # 行级 z-order 遮挡：已预计算好（只包含前排的 mask）
    refined = _refine_overlap_mask(scaled_mask, (oy, ox), occlusion_mask)
    contour = _extract_boundary(refined)
    if len(contour) == 0:
        return None

    # target face bbox
    tfx = int(round(target_face_x - gap.face_w / 2))
    tfy = int(round(target_face_y - gap.face_h / 2))
    tfw = int(round(gap.face_w))
    tfh = int(round(gap.face_h))

    score = _score_gap(gap, individual_body_h, scale)
    neighbors = [n for n in (gap.left_id, gap.right_id) if n is not None]

    return CandidatePatch(
        target_rgb=group_rgb,
        source_rgb=scaled_rgb,
        source_mask=scaled_mask,
        refined_mask=refined,
        offset=(oy, ox),
        contour=contour,
        scale=scale,
        target_face_bbox=[tfx, tfy, tfw, tfh],
        gap_bbox=gap.bbox,
        score=score,
        neighbors=neighbors,
    )


# ==================== 主入口 ====================

def find_insertion_patches(
    group_meta_path: str | Path,
    group_image_path: str | Path,
    individual_meta_path: str | Path,
    individual_image_path: str | Path,
    individual_mask_path: str | Path | None = None,
    top_k: int = 5,
) -> list[CandidatePatch]:
    """找插入候选位。返回按 score 降序的 list[CandidatePatch]。

    使用行级 z-order：插入人物在 gap 所在的行；前排（行内平均脸高更大）的 mask
    才会遮挡，后排与同排不遮挡。SAM3 instance mask 优先，拿不到时回落到 bbox。
    """
    group_meta_path = Path(group_meta_path)
    group_image_path = Path(group_image_path)
    individual_meta_path = Path(individual_meta_path)
    individual_image_path = Path(individual_image_path)
    individual_mask_path = Path(individual_mask_path) if individual_mask_path else None

    if not group_meta_path.exists():
        raise FileNotFoundError(f"group metadata not found: {group_meta_path}")
    if not group_image_path.exists():
        raise FileNotFoundError(f"group image not found: {group_image_path}")
    if not individual_meta_path.exists():
        raise FileNotFoundError(f"individual metadata not found: {individual_meta_path}")
    if not individual_image_path.exists():
        raise FileNotFoundError(f"individual image not found: {individual_image_path}")

    group_data = json.loads(group_meta_path.read_text(encoding="utf-8"))
    indiv_data = json.loads(individual_meta_path.read_text(encoding="utf-8"))

    persons: list[dict[str, Any]] = group_data["persons"]
    if not persons:
        return []
    if not indiv_data["persons"]:
        raise ValueError(f"no person in individual metadata: {individual_meta_path}")
    individual = indiv_data["persons"][0]
    if "face" not in individual or not individual["face"]:
        raise ValueError(
            f"individual {individual_meta_path} has no face metadata; "
            "use the SAM3-derived metadata (person_metadata/<stem>.json), not the LBP-cascade one"
        )

    group_rgb = _load_rgb(group_image_path)
    H, W = group_rgb.shape[:2]
    individual_rgb = _load_rgb(individual_image_path)
    iH, iW = individual_rgb.shape[:2]

    mask_path = _resolve_individual_mask(individual_meta_path, individual_mask_path)
    individual_mask = _load_mask(mask_path, (iW, iH))

    rows = _cluster_rows(persons, H)
    if not rows:
        return []

    # 行级 z-order：按行内平均 face_h 排序，rank 越大 = 越靠前
    row_rank = _rank_rows_by_face_y(rows)

    # 检查 SAM3 instance mask 是否可用；不可用则后面用 bbox
    existing_union, _ = _load_existing_depth(group_meta_path, (H, W))
    sam3_masks_available = existing_union.any()
    if not sam3_masks_available:
        # 回落到 bbox，但仍然按行做遮挡
        existing_union = _existing_bbox_mask(persons, (H, W))

    all_gaps: list[GapInfo] = []
    for r_idx, row in enumerate(rows):
        for g in _find_row_gaps(row, W):
            g.row_index = r_idx
            all_gaps.append(g)

    individual_body_h = max(
        float(individual["bbox"][3]),
        float(individual["face"]["bbox"][3]) * 7.0,
    )

    # 预计算每行的遮挡 mask（用 SAM3 真实 mask 或 bbox）
    row_occlusion: dict[int, np.ndarray] = {}
    for r_idx in range(len(rows)):
        if sam3_masks_available:
            occ = _build_row_occlusion(
                group_meta_path, persons, rows, row_rank, r_idx, (H, W),
                dilation_px=4,
            )
        else:
            # bbox fallback: 前排的 bbox 并集
            placed_rank = row_rank[r_idx]
            occ = np.zeros((H, W), dtype=bool)
            for other_r_idx, other_row in enumerate(rows):
                if row_rank[other_r_idx] <= placed_rank:
                    continue
                for p in other_row:
                    x, y, w, h = p["bbox"]
                    x0 = max(0, int(x)); y0 = max(0, int(y))
                    x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
                    if x1 > x0 and y1 > y0:
                        occ[y0:y1, x0:x1] = True
            if occ.any():
                occ = ndimage.binary_dilation(occ, iterations=4)
        row_occlusion[r_idx] = occ

    patches: list[CandidatePatch] = []
    for gap in all_gaps:
        occ = row_occlusion[gap.row_index]
        patch = _compute_patch(
            gap, group_rgb, individual_rgb, individual_mask,
            individual["face"], individual_body_h,
            occ,
        )
        if patch is not None:
            patches.append(patch)

    patches.sort(key=lambda p: p.score, reverse=True)
    return patches[:top_k]


# ==================== compose-and-paste helper ====================

def _crop_for_compose(
    group_rgb: np.ndarray,
    patch: CandidatePatch,
    margin: int = 8,
    max_crop_size: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    """把 patch 周围裁出 (target_crop, source_crop, fg_mask, boundary_mask, paste_box, downscale)。

    paste_box = (oy0, ox0, oy1, ox1) — 贴回原 group 时的方框。
    downscale = 实际下采样倍率 (1.0 表示未缩放)。当裁片超过 max_crop_size 时，
    按 max_crop_size 限制最长边并下采样，compose 完后再上采样贴回。
    """
    H, W = group_rgb.shape[:2]
    sh, sw = patch.source_rgb.shape[:2]
    py, px = patch.offset

    if patch.refined_mask.any():
        ys, xs = np.where(patch.refined_mask)
        cy0, cy1 = int(ys.min()), int(ys.max())
        cx0, cx1 = int(xs.min()), int(xs.max())
    else:
        cy0, cy1 = max(0, py), min(H - 1, py + sh - 1)
        cx0, cx1 = max(0, px), min(W - 1, px + sw - 1)

    oy0 = max(0, cy0 - margin)
    ox0 = max(0, cx0 - margin)
    oy1 = min(H, cy1 + margin + 1)
    ox1 = min(W, cx1 + margin + 1)
    if oy1 <= oy0 or ox1 <= ox0:
        raise ValueError("empty crop box")

    # 1) 先在原分辨率下做裁切 + 放置
    target_full = group_rgb[oy0:oy1, ox0:ox1].copy()
    source_full = np.zeros_like(target_full)
    source_mask_full = np.zeros(target_full.shape[:2], dtype=bool)

    sy0 = oy0 - py
    sx0 = ox0 - px
    sy1 = oy1 - py
    sx1 = ox1 - px
    sy0c, sx0c = max(0, sy0), max(0, sx0)
    sy1c, sx1c = min(sh, sy1), min(sw, sx1)
    if sy1c > sy0c and sx1c > sx0c:
        ty_off = sy0c - sy0
        tx_off = sx0c - sx0
        source_full[ty_off:ty_off + (sy1c - sy0c),
                    tx_off:tx_off + (sx1c - sx0c)] = patch.source_rgb[sy0c:sy1c, sx0c:sx1c]
        source_mask_full[ty_off:ty_off + (sy1c - sy0c),
                         tx_off:tx_off + (sx1c - sx0c)] = patch.source_mask[sy0c:sy1c, sx0c:sx1c]

    refined_full = patch.refined_mask[oy0:oy1, ox0:ox1].copy()

    # 2) 计算下采样倍率（让最长边 ≤ max_crop_size）
    crop_h, crop_w = target_full.shape[:2]
    longest = max(crop_h, crop_w)
    downscale = 1.0
    if longest > max_crop_size:
        downscale = max_crop_size / longest
        new_h = max(1, int(round(crop_h * downscale)))
        new_w = max(1, int(round(crop_w * downscale)))
    else:
        new_h, new_w = crop_h, crop_w

    # 3) 一致地 resize
    target_crop = _resize_f32(target_full, new_w, new_h)
    source_crop = _resize_f32(source_full, new_w, new_h)
    source_mask_crop = _resize_bool(source_mask_full, new_w, new_h)
    refined_crop = _resize_bool(refined_full, new_w, new_h)

    foreground_mask = source_mask_crop & refined_crop

    boundary_mask = np.zeros(target_crop.shape[:2], dtype=bool)
    if len(patch.contour) > 0:
        rel_y = (patch.contour[:, 0] - oy0) * downscale
        rel_x = (patch.contour[:, 1] - ox0) * downscale
        in_crop = (
            (rel_y >= 0) & (rel_y < target_crop.shape[0]) &
            (rel_x >= 0) & (rel_x < target_crop.shape[1])
        )
        boundary_mask[rel_y[in_crop].astype(int), rel_x[in_crop].astype(int)] = True

    return target_crop, source_crop, foreground_mask, boundary_mask, (oy0, ox0, oy1, ox1), downscale


def _resize_f32(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if img.shape[1] == new_w and img.shape[0] == new_h:
        return img.copy()
    u8 = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(u8).resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _resize_bool(mask: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if mask.shape[1] == new_w and mask.shape[0] == new_h:
        return mask.copy()
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (new_w, new_h), Image.Resampling.NEAREST
    )
    return np.asarray(pil) > 0


def compose_and_paste(
    group_rgb: np.ndarray,
    patch: CandidatePatch,
    margin: int = 8,
    compositor: Any | None = None,
    max_crop_size: int = 200,
) -> np.ndarray:
    """裁出 patch 的 compose 区域，调 MRFImageCompositor.compose，贴回 group_rgb。

    max_crop_size: 裁片最长边超过此值时先下采样到 ≤ max_crop_size，compose 完上采样回原大小。
    """
    from ImageCompositor import MRFImageCompositor as _MC

    target_crop, source_crop, fg, bd, paste_box, downscale = _crop_for_compose(
        group_rgb, patch, margin=margin, max_crop_size=max_crop_size,
    )
    if compositor is None:
        compositor = _MC()
    composited = compositor.compose(
        source_rgb=source_crop,
        target_rgb=target_crop,
        foreground_mask=fg,
        boundary_mask=bd,
    )

    oy0, ox0, oy1, ox1 = paste_box
    out_h, out_w = oy1 - oy0, ox1 - ox0
    if downscale < 1.0 and composited.shape[:2] != (out_h, out_w):
        composited = _resize_f32(composited, out_w, out_h)
    if downscale < 1.0 and fg.shape != (out_h, out_w):
        fg_full = _resize_bool(fg, out_w, out_h)
    else:
        fg_full = fg

    # 关键：只把 foreground_mask 内的像素贴回去；外侧保留原合照
    # （MRF 在前景外的源是 0/black，会被算进 L_composite 变成怪异颜色）
    out = group_rgb.copy()
    region = out[oy0:oy1, ox0:ox1]
    mask_3d = fg_full[..., None]
    out[oy0:oy1, ox0:ox1] = np.where(mask_3d, composited, region)
    return out
