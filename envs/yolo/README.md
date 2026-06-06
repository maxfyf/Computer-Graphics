# YOLO Person-Mask Environment

This uv project is separate from the root SAM3 environment. Use it for the
lightweight local pipeline that generates per-person instance masks with
Ultralytics YOLO.

## 1. Create The Environment

Run from the repository root:

```bash
uv sync --project envs/yolo
```

Activate it if you prefer an interactive shell:

```bash
source envs/yolo/.venv/bin/activate
```

Or run commands without activation:

```bash
uv run --project envs/yolo python --version
```

Check the install:

```bash
uv run --project envs/yolo python -c "import ultralytics; print(ultralytics.__version__)"
```

## 2. Download YOLO Segmentation Models

Ultralytics can auto-download model weights, but GitHub downloads are often
slow. Download weights manually and pass local paths to the script.

Create a model directory:

```bash
mkdir -p models/yolo
```

Recommended 8GB local model:

```bash
aria2c -x 16 -s 16 -k 1M -c \
  -d models/yolo \
  -o yolo11s-seg.pt \
  "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11s-seg.pt"
```

Higher quality if memory allows:

```bash
aria2c -x 16 -s 16 -k 1M -c \
  -d models/yolo \
  -o yolo11m-seg.pt \
  "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11m-seg.pt"

aria2c -x 16 -s 16 -k 1M -c \
  -d models/yolo \
  -o yolo11l-seg.pt \
  "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11l-seg.pt"
```

If `aria2c` is unavailable, use `wget`:

```bash
wget -c "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11l-seg.pt" \
  -O models/yolo/yolo11l-seg.pt
```

Official Hugging Face URLs also work by replacing `hf-mirror.com` with
`huggingface.co`.

## 3. Generate Person Masks

The script writes the same directory structure as the SAM3 pipeline:

```text
output/yolo_l_auto_conf035_v2/g1/
  g1.json
  g1_instances.png
  g1_overlay.jpg
  instances/
    person_001.png
```

Default behavior is automatic. The script probes each image and chooses:

- `auto-whole(...)` for single-person or larger-person images.
- `auto-tiled(...)` for dense group photos.

Recommended command for this project:

```bash
uv run --project envs/yolo python scripts/generate_yolo_person_masks.py material/*.jpg \
  --model models/yolo/yolo11l-seg.pt \
  --output-dir output/yolo_l_auto_conf035_v2 \
  --imgsz 1024 \
  --conf 0.35 \
  --auto-tile-size 1280 \
  --tile-overlap 256 \
  --half
```

Lower-memory fallback:

```bash
uv run --project envs/yolo python scripts/generate_yolo_person_masks.py material/*.jpg \
  --model models/yolo/yolo11s-seg.pt \
  --output-dir output/yolo_s_auto_lowmem \
  --imgsz 768 \
  --conf 0.35 \
  --auto-tile-size 960 \
  --tile-overlap 192 \
  --half
```

Do not enable `--retina-masks` on an 8GB GPU unless running a very small image;
it improves mask resolution but costs much more memory.

## 4. Inspect Results

Open the overlay images:

```text
output/yolo_l_auto_conf035_v2/g1/g1_overlay.jpg
output/yolo_l_auto_conf035_v2/g2/g2_overlay.jpg
output/yolo_l_auto_conf035_v2/g3/g3_overlay.jpg
```

The JSON files include the chosen strategy:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("output/yolo_l_auto_conf035_v2")
for p in sorted(root.glob("*/*.json")):
    d = json.loads(p.read_text())
    print(p.parent.name, len(d["instances"]), d.get("strategy"))
PY
```

Expected for the current tuned output:

```text
g1 22 auto-whole(...)
g2 80 auto-tiled(...)
g3 72 auto-tiled(...)
p1 1 auto-whole(...)
p2 1 auto-whole(...)
p3 1 auto-whole(...)
```

## 5. Extract Contours And Face Metadata

After masks are generated, run the shared metadata extractor:

```bash
python scripts/extract_metadata_from_sam3.py \
  --sam3-dir output/yolo_l_auto_conf035_v2 \
  --output-dir output/yolo_l_auto_conf035_v2_person_metadata \
  --contour-stride 1
```

Output:

```text
output/yolo_l_auto_conf035_v2_person_metadata/
  g1.json
  g1_debug.jpg
  g2.json
  g2_debug.jpg
  g3.json
  g3_debug.jpg
  p1.json
  p1_debug.jpg
  summary.json
```

Each person entry contains:

- `bbox`
- `score`
- `mask_path`
- `area`
- `contour_pixels`
- `face.center`
- `face.size`
- `face.bbox`

The face box is currently a geometry estimate from the person mask and bbox,
not a dedicated face detector.

## 6. Useful Tuning

Reduce false positives and fragments:

```bash
--conf 0.4
```

Recover missed small people:

```bash
--conf 0.25 --auto-tile-size 1280 --tile-overlap 384
```

Force automatic strategy to choose tiled mode more easily:

```bash
--auto-dense-count 15
```

Force manual whole-image mode:

```bash
--no-auto --tile-size 0
```

Force manual tiled mode:

```bash
--no-auto --tile-size 1280 --tile-overlap 256 --no-whole-image-pass
```
