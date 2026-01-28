from ultralytics import YOLO
from pathlib import Path
import cv2

# =====================
# CONFIG
# =====================
VIDEOS = ["pasteur-bandung", "yogya", "aa-bandung", "paskal-bandung", "pvj-bandung"]   
SCRIPT_DIR = Path(__file__).resolve().parent

for video_name in VIDEOS:
    print(f"[INFO] Processing video: {video_name}")
    MODEL_PATH = SCRIPT_DIR / "models/car_detection_stage_3_yolov10m/weights/best.pt"
    VIDEO_PATH = SCRIPT_DIR / f"indonesia-dataset/cctv-videos/{video_name}.mp4"
    # pasteur-bandung , yogya
    OUT_DIR = SCRIPT_DIR / "label_studio_data/images"
    OUT_DIR.mkdir(exist_ok=True)

    CONF_THRES = 0.9
    INTERVAL_SEC = 1.0   # ← sampling every N seconds

    # =====================
    # LOAD MODEL
    # =====================
    model = YOLO(MODEL_PATH)

    # =====================
    # VIDEO SETUP
    # =====================
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps * INTERVAL_SEC))

    print(f"[INFO] FPS: {fps:.2f}")
    print(f"[INFO] Sampling every {INTERVAL_SEC}s → every {step} frames")

    frame_idx = 0
    crop_idx = 0

    # =====================
    # PROCESS VIDEO
    # =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- FRAME SAMPLER ----
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # ---- YOLO INFERENCE ----
        results = model(frame, conf=CONF_THRES, verbose=False)

        for r in results:
            for box in r.boxes:
                # class 0 = car (adjust if needed)
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                fname = f"{video_name}_car_f{frame_idx}_c{crop_idx}.jpg"
                cv2.imwrite(str(OUT_DIR / fname), crop)
                crop_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[SUCCESS] Saved {video_name} {crop_idx} car crops")
