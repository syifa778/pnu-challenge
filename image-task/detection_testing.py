from ultralytics import YOLO
from pathlib import Path

# =====================
# CONFIG
# =====================
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_BASE_DIR = SCRIPT_DIR / "models"
DATA_YAML = SCRIPT_DIR / "indonesia-dataset/indonesia-yolo/indonesia_car.yaml"

model = YOLO(f"{MODELS_BASE_DIR}/car_detection_stage_3_yolo/weights/best.pt")

model.predict(
    source=f"{SCRIPT_DIR}/traffic_test.mp4",
    conf=0.9,
    save=True,
    project=MODELS_BASE_DIR,
    name="infosec_traffic_test_stage_3_yolo_predictions"
)