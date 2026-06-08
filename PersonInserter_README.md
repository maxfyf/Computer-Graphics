# 集体照补人工具

把单人照里的人物**插入**到集体照里，插入位置是自动找出的最合适空位；插入过程基于 MRF（马尔可夫随机场）做光照一致融合，看起来像原本就在合照里。

## 背景

合照里经常有人缺席。本工具做的事：
1. 在合照里按**行**自动找几个候选空位（人之间的间隙）
2. 选一个，把单人照里的人物缩放、对齐后放进去
3. 用 `ImageCompositor.py` 里的 MRF compositor 做边缘融合（颜色 / 亮度自适应）

## 数据准备

工具依赖以下目录结构（仓库里已经准备好示例数据）：

```
material/                    # 原图
  g1.jpg g2.jpg g3.jpg       # 集体照
  p1.jpg p2.jpg p3.jpg       # 单人照（要插进去的人）

person_metadata/             # 每张图的元数据（bbox / face / contour / mask_path）
  g1.json g2.json g3.json
  p1.json p2.json p3.json

sam3_masks/                  # SAM3 instance mask（每人一帧 PNG）
  g1/
    g1.json                  # metadata: 每人的 score / bbox / mask_path
    instances/
      person_001.png         # 0/255 二值 mask
      person_002.png
      ...
```

如果你已经有 `person_metadata/`，可以直接跳到后续步骤；否则按下面顺序处理：

```bash
# 1) SAM3 给合照里每个人出 mask（需要 GPU）（依赖可选）
uv run python scripts/generate_sam3_masks.py material/g*.jpg \
    --checkpoint /path/to/sam3.pt \
    --output-dir sam3_masks

# 2) 从 SAM3 输出派生 person_metadata（bbox / face / contour 等）
uv run python scripts/extract_metadata_from_sam3.py \
    --sam3-dir sam3_masks \
    --output-dir person_metadata

# 3) 单人照同样处理
uv run python scripts/generate_sam3_masks.py material/p*.jpg \
    --checkpoint /path/to/sam3.pt \
    --output-dir sam3_masks
uv run python scripts/extract_metadata_from_sam3.py \
    --sam3-dir sam3_masks \
    --output-dir person_metadata
```

## 安装

Python 3.12，`uv` 管理依赖。

```bash
uv sync
```

`sam3` 是可选依赖：只有你需要从头生成 SAM3 masks 时才需要安装它。
`torch` 仍然走官方 CUDA 12.8 源。

如果你本地有同级 `../sam3` 仓库，可在 `pyproject.toml` 中恢复对应的 `tool.uv.sources` 配置。

## HSV 色调对齐

`ImageCompositor.py` 主要处理 Lab 亮度一致性。若单人照和合照人物存在冷暖、色相或饱和度差异，可先运行 HSV 色调对齐工具，只用 mask 内的人物像素做统计，避免背景颜色污染参考值。

```bash
uv run python scripts/align_tone_hsv.py \
  --source-image material/p1.jpg \
  --source-mask sam3_masks/p1/instances/person_001.png \
  --target-image material/g1.jpg \
  --target-mask-glob 'sam3_masks/g1/instances/person_*.png' \
  --output-image output/tone_align/g1_p1_hsv.png \
  --report output/tone_align/g1_p1_hsv_report.json
```

该工具也已注册为 orchestrator 工具：

```bash
cd Computer-Graphics
conda run -n check-numpy python orchestrator/scripts/run_tool.py compositing.align_tone_hsv
```

完整 ReAct 流程会在候选插入后调用 `compositing.align_tone_hsv`，并由 `tone_verifier.py` 检查像素数、hue shift 和 saturation ratio 是否在安全范围内。

## 快速开始

```python
from PersonInserter import find_insertion_patches, compose_and_paste
from PIL import Image
import numpy as np

# 1) 找候选位
patches = find_insertion_patches(
    group_meta_path="person_metadata/g1.json",
    group_image_path="material/g1.jpg",
    individual_meta_path="person_metadata/p1.json",
    individual_image_path="material/p1.jpg",
    top_k=5,
)

for i, p in enumerate(patches):
    print(f"#{i+1}  score={p.score:.3f}  scale={p.scale:.3f}  neighbors={p.neighbors}")

# 2) 选一个候选，调 MRF 合成 + 贴回
group_rgb = np.asarray(Image.open("material/g1.jpg").convert("RGB"), dtype=np.float32) / 255.0
final = compose_and_paste(group_rgb, patches[0], margin=8, max_crop_size=200)

Image.fromarray((final * 255).astype(np.uint8)).save("output/g1_with_p1.jpg")
```

跑仓库里的脚本能直接看到效果：

```bash
# 5 个候选各出一张完整合成图（~3 分钟）
uv run python scripts/compose_5.py

# top-3 候选的裁切前后对比
uv run python scripts/zoom_into_patch.py

# 单候选 + 可视化候选位置
uv run python scripts/smoke_test_inserter.py

# 多组数据 sanity check
uv run python scripts/test_all_combinations.py
```

输出在 `output/compose_5/`、`output/insertion_smoke/`。

## API

### `find_insertion_patches(...)`

```python
def find_insertion_patches(
    group_meta_path,            # str | Path
    group_image_path,           # str | Path
    individual_meta_path,       # str | Path
    individual_image_path,      # str | Path
    individual_mask_path=None,  # 可选；不传则从 individual_meta 的 source_metadata 找
    top_k=5,                    # 返回候选数
) -> list[CandidatePatch]
```

返回按 score 降序的 `CandidatePatch` 列表。

### `compose_and_paste(...)`

```python
def compose_and_paste(
    group_rgb,           # (H, W, 3) float32 [0, 1]
    patch,               # CandidatePatch
    margin=8,            # 裁切时在 contour 外留的像素
    compositor=None,     # 复用已有的 MRFImageCompositor
    max_crop_size=200,   # 裁片最长边超过此值先下采样；防 MRF 爆内存
    preserve_detail=True,# 下采样只算修正场，最终保留原分辨率人像细节
    feather_px=2.0,      # 贴回时 mask 边缘羽化半径；减少锯齿和硬边
) -> np.ndarray         # 合成后的 (H, W, 3) float32 [0, 1]
```

### `CandidatePatch` 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `target_rgb` | `np.ndarray` (H, W, 3) | 合照 RGB |
| `source_rgb` | `np.ndarray` (h, w, 3) | 缩放后的单人 RGB |
| `source_mask` | `np.ndarray` (h, w) bool | 缩放后的单人 mask |
| `refined_mask` | `np.ndarray` (H, W) bool | 精炼后插入人物 mask（合照坐标系） |
| `offset` | `(y, x)` tuple | source_rgb 左上角在 target 坐标系 |
| `contour` | `(N, 2) int32` | 精炼后边界 (y, x) 数组（合照坐标系） |
| `scale` | float | 缩放因子 |
| `target_face_bbox` | `[x, y, w, h]` | 估计的目标脸 bbox |
| `gap_bbox` | `[x0, y0, x1, y1]` | 这个候选空位的 bbox |
| `score` | float | 排序分数 |
| `neighbors` | `list[int]` | 左右邻接人 id |
| `warnings` | `list[str]` | 提取过程中的警告 |

## 算法

### 行聚类

合照里按 face y 聚类（gap > `max(20, 0.04 × 图片高)` 即分新行），行内按 face x 升序排。

### 候选空位

每行产生 3 类空位：
- 行首（在最左人的左侧）
- 邻接两人中间
- 行尾（在最右人的右侧）

每个空位的"目标脸"是邻接两人脸中心 / 脸高的均值。

### 缩放 & 放置

- `scale = mean(目标 face_h / 单人 face_h, 目标 face_w / 单人 face_w)`
- 缩放后单人图的 face center 对齐到空位的 face center
- 单人 mask 同步缩放

### 行级 z-order 精炼

按行的**平均 face_h** 排深度（越大越靠前）。插入行 X 时：
- 同排 / 后排的人 → **不遮挡**
- 前排的人 → 用他们的 **SAM3 instance mask** 遮挡（dilation 4 px）
- 如果 SAM3 mask 拿不到，回落到 bbox 矩形

这一步的关键是：**用真实 mask，不用 bbox**——bbox 比实际身体大很多，会把插入人物"过度遮挡"。

### MRF 合成

按 `refined_mask` 的 bbox + margin 裁出矩形，调 `MRFImageCompositor.compose`：
- `source_rgb` = 缩放后单人图放在合照对应区域（空的地方填 0）
- `target_rgb` = 合照裁片
- `foreground_mask` = source_mask ∩ refined_mask
- `boundary_mask` = contour 在裁后坐标系

**关键细节**：MRF 假定源图覆盖整个区域。源图里 0/black 的部分也会被算进 `L_composite` 变成怪异颜色。`compose_and_paste` 贴回时用 `np.where(foreground_mask, composited, 原合照)`，**只把前景内的像素写回**，前景外保留原合照。

### max_crop_size

`ImageCompositor.compose` 用稠密 N×N 仿射矩阵（N = H×W），大裁片会爆内存。`max_crop_size=200`（默认）会把裁片下采样到最长边 ≤ 200 px 来计算平滑光照/颜色修正场。

默认 `preserve_detail=True` 时，低分辨率 MRF 结果不会被直接放大贴回，而是先转成相对 source 的修正场，再应用到原分辨率人像裁片。这样保留衣服、头发和脸部细节，同时仍然避免大矩阵爆内存。如果为了复现旧行为，可以传 `preserve_detail=False`。

### feather_px

SAM3 mask 和缩放后的二值 mask 边缘都是硬边，直接 `np.where(mask, person, group)` 会产生锯齿。默认 `feather_px=2.0` 会基于原分辨率贴回 mask 生成 soft alpha，只在边缘 2px 左右过渡，主体区域仍保持完全不透明。若需要硬边调试，可设为 `0`。

## 已知限制

- **后排插入**：身体下沿会被前排真实 mask 正确遮挡，只能看清头和肩膀。要看清全身请选**前排候选**。
- **MRF 慢**：单次合成 25–35 秒（crop ≤ 200 px）。`max_iter=200` 收敛阈值 `1e-4`，要更快可改小，要更精细可改大。
- **face 估计是几何启发式**：基于 bbox 顶部 16% 推算，不是真的人脸检测。后续如果要换真检测器，只改 `PersonInserter._scale_image_and_mask` 之外的地方都不用动。
- **行聚类阈值**对极端稀疏 / 密集场景可能要调：`_cluster_rows` 里的 `max(20, 0.04 × H)`。
- **深度信息用 face_h 估算**：竖排非常规的合照（如只有 1 行或所有人在一个深度）会判定不准。

## 模块结构

```
PersonInserter.py             # 主模块
├─ @dataclass GapInfo         # 行内空位（内部）
├─ @dataclass CandidatePatch  # 输出单元
├─ _load_rgb / _load_mask     # 读图
├─ _resolve_individual_mask   # 自动找 SAM3 mask
├─ _cluster_rows              # 按 face y 分行
├─ _find_row_gaps             # 行内空位
├─ _scale_image_and_mask      # 双线性缩放图 + 最近邻缩放 mask
├─ _rank_rows_by_face_h       # 行内平均 face_h 排深度
├─ _build_row_occlusion       # 前排 SAM3 mask 并集
├─ _refine_overlap_mask       # placed & ~occlusion
├─ _compute_patch             # 组装一个 CandidatePatch
├─ find_insertion_patches     # 主入口
├─ _crop_for_compose          # 裁片 + 可选下采样
├─ _resize_f32 / _resize_bool # resize helper
└─ compose_and_paste          # 一站式合成 + 贴回
```

## 相关文件

- [`ImageCompositor.py`](ImageCompositor.py) — 底层 MRF 合成算法（包含人脸处采样优化，并提供K-Means聚类分析皮肤光强选项）
- [`PersonMetadataExtractor.py`](PersonMetadataExtractor.py) — LBP face cascade 提取人脸元数据（备选 pipeline）
- [`scripts/generate_sam3_masks.py`](scripts/generate_sam3_masks.py) — SAM3 instance mask 提取
- [`scripts/extract_metadata_from_sam3.py`](scripts/extract_metadata_from_sam3.py) — 从 SAM3 mask 派生元数据
