from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2

# =====================
# CONFIG
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "models/car_detection_stage_2_yolo/weights/best.pt"

INPUT_ROOT = SCRIPT_DIR / "indonesia-dataset/cctv-images"
OUTPUT_ROOT = SCRIPT_DIR / "indonesia-dataset"

OUT_IMG_DIR = OUTPUT_ROOT / "images"
OUT_LBL_DIR = OUTPUT_ROOT / "labels"
OUT_BB_DIR  = OUTPUT_ROOT / "images-with-bb"

CONF_THRES = 0.35
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# =====================
# SETUP
# =====================
for d in (OUT_IMG_DIR, OUT_LBL_DIR, OUT_BB_DIR):
    d.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# =====================
# COLLECT IMAGES
# =====================
image_paths = [
    p for p in INPUT_ROOT.rglob("*")
    if p.suffix.lower() in IMG_EXTS
]

print(f"Found {len(image_paths)} images")

# =====================
# AUTO LABELING
# =====================
for img_path in tqdm(image_paths, desc="Auto-labeling images", unit="img"):
    # CCTV name = first folder under cctv-images
    cctv_name = img_path.relative_to(INPUT_ROOT).parts[0]

    # New filename
    new_stem = f"{cctv_name}_{img_path.stem}"
    new_img_name = new_stem + img_path.suffix

    results = model(
        source=str(img_path),
        conf=CONF_THRES,
        iou=0.5,
        device=0,
        verbose=False
    )

    r = results[0]
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        continue

    # ---------------------
    # Save raw image
    # ---------------------
    shutil.copy(img_path, OUT_IMG_DIR / new_img_name)

    # ---------------------
    # Save YOLO label
    # ---------------------
    label_path = OUT_LBL_DIR / f"{new_stem}.txt"
    with open(label_path, "w") as f:
        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id != 0:  # car only
                continue

            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # ---------------------
    # Save image with bounding boxes
    # ---------------------
    bb_path = OUT_BB_DIR / new_img_name
    cv2.imwrite(str(bb_path), r.plot())

print("Auto-labeling completed")
