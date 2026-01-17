from ultralytics import YOLO
from pathlib import Path

# =====================
# CONFIG
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_BASE_DIR = SCRIPT_DIR / "models"
DATA_YAML = SCRIPT_DIR / "indonesia-dataset/indonesia-yolo/indonesia_car.yaml"

EPOCHS = 50
IMGSZ = 640
BATCH = 16

# =====================
# LOAD MODEL
# =====================
model = YOLO(f"{MODELS_BASE_DIR}/car_detection_stage_2_yolo/weights/best.pt")

# =====================
# TRAIN
# =====================
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,

    # Best practice for domain adaptation
    lr0=1e-4,
    patience=10,
    cos_lr=True,

    # Augmentation (safe for CCTV)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.2,
    mixup=0.0,

    project=MODELS_BASE_DIR,
    name="car_detection_stage_3_yolo"
)
print("Training on Indonesia Traffic CCTV dataset completed.")