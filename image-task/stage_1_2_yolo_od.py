import os
from ultralytics import YOLO

# ============================
# CONFIG
# ============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(SCRIPT_DIR, "coco-dataset/coco_car_only/data.yaml")
MODELS_BASE_DIR = os.path.join(SCRIPT_DIR, "models")
IMG_SIZE = 640
BATCH = 16
DEVICE = 0   # set to 'cpu' if no GPU

# # ============================
# # STAGE 1: Freeze Backbone
# # ============================
print("Stage 1: Training detection head (backbone frozen)")

model = YOLO(f"{MODELS_BASE_DIR}/yolov10m.pt")

model.train(
    data=DATA_YAML,
    epochs=10,
    imgsz=IMG_SIZE,
    batch=BATCH,
    freeze=10,          # Freeze backbone layers
    device=DEVICE,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    patience=5,
    project=MODELS_BASE_DIR,
    name="car_detection_stage_1_yolo"
)

# ============================
# STAGE 2: Full Fine-Tuning
# ============================
print("Stage 2: Full model fine-tuning")

model = YOLO(f"{MODELS_BASE_DIR}/car_detection_stage_1_yolo/weights/best.pt")

model.train(
    data=DATA_YAML,
    epochs=40,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    optimizer="AdamW",
    lr0=5e-4,            # lower LR for stability
    cos_lr=True,
    patience=10,
    project=MODELS_BASE_DIR,
    name="car_detection_stage_2_yolo",
)

print("Training complete. Car-specialized YOLOv8 model ready.")
