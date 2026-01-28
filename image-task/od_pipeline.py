import os
from ultralytics import YOLO, RTDETR
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2
import random
from collections import defaultdict

# ============================
# CONFIG
# ============================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "coco-dataset/coco_car_only/data.yaml"
MODELS_BASE_DIR = SCRIPT_DIR / "models"
IMG_SIZE = 640
BATCH = 16
DEVICE = 0   # set to 'cpu' if no GPU

# Models to train
MODELS_TO_TRAIN = [
    {"name": "yolov10m", "file": "yolov10m.pt", "model_class": YOLO},
    {"name": "rtdetr-l", "file": "rtdetr-l.pt", "model_class": RTDETR}
]

def train_model_stages(model_config):
    """Train model in two stages: frozen backbone, then full fine-tuning"""
    model_name = model_config["name"]
    model_file = model_config["file"]
    ModelClass = model_config["model_class"]
    
    # ============================
    # STAGE 1: Freeze Backbone
    # ============================
    print(f"\n[{model_name}] Stage 1: Training detection head (backbone frozen)")
    
    model = ModelClass(str(MODELS_BASE_DIR / model_file))
    
    model.train(
        data=str(DATA_YAML),
        epochs=10,
        imgsz=IMG_SIZE,
        batch=BATCH,
        freeze=10,          # Freeze backbone layers
        device=DEVICE,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        patience=5,
        project=str(MODELS_BASE_DIR),
        name=f"car_detection_stage_1_{model_name}"
    )
    
    # ============================
    # STAGE 2: Full Fine-Tuning
    # ============================
    print(f"\n[{model_name}] Stage 2: Full model fine-tuning")
    
    model = ModelClass(str(MODELS_BASE_DIR / f"car_detection_stage_1_{model_name}/weights/best.pt"))
    
    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        optimizer="AdamW",
        lr0=5e-4,            # lower LR for stability
        cos_lr=True,
        patience=10,
        project=str(MODELS_BASE_DIR),
        name=f"car_detection_stage_2_{model_name}",
    )
    
    print(f"[{model_name}] Training complete. Car-specialized model ready.")

# Train all models
for model_config in MODELS_TO_TRAIN:
    train_model_stages(model_config)


# =====================
# AUTO LABELING FOR ALL MODELS
# =====================
def auto_label_images(model_config):
    """Auto-label images using trained model"""
    model_name = model_config["name"]
    ModelClass = model_config["model_class"]
    
    print(f"\n[{model_name}] Starting auto-labeling")
    
    MODEL_PATH = MODELS_BASE_DIR / f"car_detection_stage_2_{model_name}/weights/best.pt"
    
    INPUT_ROOT = SCRIPT_DIR / "indonesia-dataset/cctv-images"
    OUTPUT_ROOT = SCRIPT_DIR / "indonesia-dataset"
    
    OUT_IMG_DIR = OUTPUT_ROOT / f"images-{model_name}"
    OUT_LBL_DIR = OUTPUT_ROOT / f"labels-{model_name}"
    OUT_BB_DIR  = OUTPUT_ROOT / f"images-with-bb-{model_name}"
    
    CONF_THRES = 0.35
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    
    # Setup directories
    for d in (OUT_IMG_DIR, OUT_LBL_DIR, OUT_BB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    
    model = ModelClass(str(MODEL_PATH))
    
    # Collect images
    image_paths = [
        p for p in INPUT_ROOT.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ]
    
    print(f"[{model_name}] Found {len(image_paths)} images")
    
    # Auto-labeling
    for img_path in tqdm(image_paths, desc=f"[{model_name}] Auto-labeling", unit="img"):
        # CCTV name = first folder under cctv-images
        cctv_name = img_path.relative_to(INPUT_ROOT).parts[0]
        
        # New filename
        new_stem = f"{cctv_name}_{img_path.stem}"
        new_img_name = new_stem + img_path.suffix
        
        results = model(
            source=str(img_path),
            conf=CONF_THRES,
            iou=0.5,
            device=DEVICE,
            verbose=False
        )
        
        r = results[0]
        boxes = r.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        # Save raw image
        shutil.copy(img_path, OUT_IMG_DIR / new_img_name)
        
        # Save label
        label_path = OUT_LBL_DIR / f"{new_stem}.txt"
        with open(label_path, "w") as f:
            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id != 0:  # car only
                    continue
                
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        # Save image with bounding boxes
        bb_path = OUT_BB_DIR / new_img_name
        cv2.imwrite(str(bb_path), r.plot())
    
    print(f"[{model_name}] Auto-labeling completed")

for model_config in MODELS_TO_TRAIN:
    auto_label_images(model_config)


# =====================
# DATASET SPLITTING FOR ALL MODELS
# =====================
def split_dataset(model_config):
    """Split auto-labeled dataset into train/val/test"""
    model_name = model_config["name"]
    
    print(f"\n[{model_name}] Starting dataset split")
    
    DATA_ROOT = SCRIPT_DIR
    IMG_DIR = DATA_ROOT / f"indonesia-dataset/images-{model_name}"
    LBL_DIR = DATA_ROOT / f"indonesia-dataset/labels-{model_name}"
    
    OUT_ROOT = SCRIPT_DIR / f"indonesia-dataset/indonesia-{model_name}"
    SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}
    
    random.seed(42)
    
    # Prepare output dirs
    for split in SPLITS:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Group by CCTV
    cctv_groups = defaultdict(list)
    
    for img_path in IMG_DIR.glob("*"):
        cctv_name = img_path.name.split("_")[0]
        cctv_groups[cctv_name].append(img_path)
    
    # Split per CCTV
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
        
        print(f"[{model_name}] {cctv}: {len(imgs)} → "
              f"train={len(split_map['train'])}, "
              f"val={len(split_map['val'])}, "
              f"test={len(split_map['test'])}")
        
        for split, files in split_map.items():
            for img in files:
                lbl = LBL_DIR / img.with_suffix(".txt").name
                
                shutil.copy(img, OUT_ROOT / "images" / split / img.name)
                shutil.copy(lbl, OUT_ROOT / "labels" / split / lbl.name)
    
    print(f"[{model_name}] Dataset split completed")

for model_config in MODELS_TO_TRAIN:
    split_dataset(model_config)


# =====================
# STAGE 3: FINE-TUNE ON INDONESIA DATASET FOR ALL MODELS
# =====================
def train_on_indonesia_dataset(model_config):
    """Fine-tune model on Indonesia CCTV dataset"""
    model_name = model_config["name"]
    ModelClass = model_config["model_class"]
    
    print(f"\n[{model_name}] Stage 3: Training on Indonesia Traffic CCTV dataset")
    
    DATA_YAML = SCRIPT_DIR / f"indonesia-dataset/indonesia-{model_name}/indonesia_car.yaml"
    
    EPOCHS = 50
    IMGSZ = 640
    BATCH = 16
    
    # Load Stage 2 trained model
    model = ModelClass(str(MODELS_BASE_DIR / f"car_detection_stage_2_{model_name}/weights/best.pt"))
    
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        
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
        
        project=str(MODELS_BASE_DIR),
        name=f"car_detection_stage_3_{model_name}"
    )
    print(f"[{model_name}] Training on Indonesia dataset completed.")

for model_config in MODELS_TO_TRAIN:
    train_on_indonesia_dataset(model_config)

# =====================
# PREDICTION / INFERENCE FOR ALL MODELS
# =====================
def run_predictions(model_config):
    """Run predictions on test video"""
    model_name = model_config["name"]
    ModelClass = model_config["model_class"]
    
    print(f"\n[{model_name}] Running inference")
    
    model = ModelClass(str(MODELS_BASE_DIR / f"car_detection_stage_3_{model_name}/weights/best.pt"))
    
    test_video = SCRIPT_DIR / "traffic_test.mp4"
    
    if test_video.exists():
        model.predict(
            source=str(test_video),
            conf=0.9,
            save=True,
            project=str(MODELS_BASE_DIR),
            name=f"infosec_traffic_test_stage_3_{model_name}_predictions"
        )
        print(f"[{model_name}] Predictions completed")
    else:
        print(f"[{model_name}] Test video not found at {test_video}")

for model_config in MODELS_TO_TRAIN:
    run_predictions(model_config)

print("\nAll pipelines completed successfully!")
