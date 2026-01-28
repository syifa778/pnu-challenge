import random
import shutil
from pathlib import Path
from collections import defaultdict

# =====================
# CONFIG
# =====================
DATA_ROOT = Path(".")
IMG_DIR = DATA_ROOT / "images"
LBL_DIR = DATA_ROOT / "labels"

OUT_ROOT = Path("indonesia-yolo")
SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}

random.seed(42)

# =====================
# PREPARE OUTPUT DIRS
# =====================
for split in SPLITS:
    (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# =====================
# GROUP BY CCTV
# =====================
cctv_groups = defaultdict(list)

for img_path in IMG_DIR.glob("*"):
    cctv_name = img_path.name.split("_")[0]
    cctv_groups[cctv_name].append(img_path)

# =====================
# SPLIT PER CCTV
# =====================
for cctv, imgs in cctv_groups.items():
    random.shuffle(imgs)
    n = len(imgs)

    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    split_map = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:]
    }

    print(f"{cctv}: {len(imgs)} → "
          f"train={len(split_map['train'])}, "
          f"val={len(split_map['val'])}, "
          f"test={len(split_map['test'])}")

    for split, files in split_map.items():
        for img in files:
            lbl = LBL_DIR / img.with_suffix(".txt").name

            shutil.copy(img, OUT_ROOT / "images" / split / img.name)
            shutil.copy(lbl, OUT_ROOT / "labels" / split / lbl.name)

print("Dataset split completed")
