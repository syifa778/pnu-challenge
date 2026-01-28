import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
from models.vit_model_classification import ViTCarClassifier
from datasets_classification import CLASS_NAMES

# =====================
# CONFIGURATION
# =====================
SCRIPT_DIR = Path(__file__).parent
MODELS_BASE_DIR = SCRIPT_DIR / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detection models configuration
DETECTION_MODELS = [
    {"name": "yolov10m", "file": "yolov10m.pt", "model_class": YOLO}
]

# Classification model configuration
CLASSIFICATION_CONFIG = {
    "model_name": "vit_b_16",  # Change to "resnet50" if using CNN
    "model_type": "vit",  # "vit" or "cnn"
    "checkpoint_path": "models/vit_b_16/best.pth",
    "num_classes": len(CLASS_NAMES),
    "class_names": CLASS_NAMES
}

# Inference configuration
INFERENCE_CONFIG = {
    "detection_conf": 0.9,  # Detection confidence threshold
    "classification_enabled": True,  # Enable/disable classification
    "save_video": True,  # Save annotated video
    "save_results_json": True,  # Save detection+classification results to JSON
    "show_realtime": False,  # Show video in real-time (not recommended for long videos)
}


# =====================
# CLASSIFICATION MODEL LOADER
# =====================
def load_classification_model(config):
    """Load the trained classification model"""
    print(f"\n[CLASSIFICATION] Loading model: {config['model_name']}")
    
    # Initialize model based on type
    if config["model_type"] == "vit":
        from models.vit_model_classification import ViTCarClassifier
        model = ViTCarClassifier(num_classes=config["num_classes"])
    else:  # cnn
        from models.cnn_model_classification import CNNCarClassifier
        model = CNNCarClassifier(backbone=config["model_name"], num_classes=config["num_classes"])
    
    # Load checkpoint
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classification model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"[CLASSIFICATION] Model loaded successfully")
    print(f"[CLASSIFICATION] Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    print(f"[CLASSIFICATION] Macro F1: {checkpoint.get('macro_f1', 'N/A'):.4f}")
    
    return model


# =====================
# IMAGE PREPROCESSING FOR CLASSIFICATION
# =====================
def get_classification_transform():
    """Get transform for classification input"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def classify_crop(crop_img, classification_model, transform):
    """Classify a cropped car image"""
    # Convert BGR (OpenCV) to RGB (PIL)
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    
    # Transform and add batch dimension
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = classification_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, dim=1)
    
    return pred_class.item(), conf.item()


# =====================
# COLOR CODING FOR CAR TYPES
# =====================
def get_car_type_color(car_type):
    """Return BGR color for each car type for visualization"""
    color_map = {
        "sedan": (255, 0, 0),      # Blue
        "suv": (0, 255, 0),         # Green
        "mpv": (255, 0, 255),       # Magenta
        "truck": (0, 255, 255),     # Yellow
        "bus": (255, 255, 0),       # Cyan
        "pickup": (128, 0, 128),    # Purple
    }
    return color_map.get(car_type, (255, 255, 255))  # White as default


# =====================
# COMBINED DETECTION + CLASSIFICATION INFERENCE
# =====================
def run_detection_classification(detection_model_config, classification_model, test_video_path):
    """Run detection and classification on test video"""
    
    model_name = detection_model_config["name"]
    ModelClass = detection_model_config["model_class"]
    
    print(f"\n{'='*70}")
    print(f"RUNNING DETECTION + CLASSIFICATION: {model_name}")
    print(f"{'='*70}\n")
    
    # Load detection model
    detection_model_path = MODELS_BASE_DIR / f"car_detection_stage_3_{model_name}/weights/best.pt"
    print(f"[DETECTION] Loading model from: {detection_model_path}")
    detection_model = ModelClass(str(detection_model_path))
    
    # Setup video capture
    cap = cv2.VideoCapture(str(test_video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {test_video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] Resolution: {width}x{height}")
    print(f"[VIDEO] FPS: {fps}")
    print(f"[VIDEO] Total frames: {total_frames}")
    
    # Setup video writer
    output_dir = MODELS_BASE_DIR / f"infosec_traffic_test_stage_3_{model_name}_with_classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_video_path = output_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Prepare classification transform
    transform = get_classification_transform()
    
    # Store results
    results_data = {
        "video_info": {
            "source": str(test_video_path),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "detection_model": model_name,
        "classification_model": CLASSIFICATION_CONFIG["model_name"],
        "frames": []
    }
    
    # Statistics
    car_type_counts = {car_type: 0 for car_type in CLASS_NAMES}
    total_detections = 0
    
    frame_idx = 0
    print(f"\n[PROCESSING] Starting frame-by-frame inference...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        detection_results = detection_model.predict(
            source=frame,
            conf=INFERENCE_CONFIG["detection_conf"],
            verbose=False
        )[0]
        
        frame_results = {
            "frame_number": frame_idx,
            "detections": []
        }
        
        # Process each detection
        boxes = detection_results.boxes
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf_det = float(box.conf[0])
            
            # Crop the detected car
            crop = frame[y1:y2, x1:x2]
            
            # Classify if enabled and crop is valid
            car_type = "Unknown"
            conf_class = 0.0
            
            if INFERENCE_CONFIG["classification_enabled"] and crop.size > 0:
                try:
                    class_idx, conf_class = classify_crop(crop, classification_model, transform)
                    car_type = CLASS_NAMES[class_idx]
                    car_type_counts[car_type] += 1
                    total_detections += 1
                except Exception as e:
                    print(f"[WARNING] Classification failed for detection {i} in frame {frame_idx}: {e}")
                    car_type = "Error"
            
            # Store detection data
            frame_results["detections"].append({
                "bbox": [x1, y1, x2, y2],
                "detection_conf": conf_det,
                "car_type": car_type,
                "classification_conf": conf_class
            })
            
            # Draw bounding box with car type
            color = get_car_type_color(car_type)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{car_type} {conf_class:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # print(f"[DEBUG] Label: {label} at ({x1}, {y1})")
            
            # Safe y position (avoid negative coordinates)
            label_y = max(y1 - label_size[1] - 10, 0)

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, label_y),
                (x1 + label_size[0], label_y + label_size[1] + 10),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, label_y + label_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        
        # Add frame statistics overlay
        stats_text = f"Frame: {frame_idx}/{total_frames} | Detections: {len(boxes)}"
        cv2.putText(
            frame,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Write frame
        if INFERENCE_CONFIG["save_video"]:
            out.write(frame)
        
        # Show frame (optional)
        if INFERENCE_CONFIG["show_realtime"]:
            cv2.imshow('Detection + Classification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Store frame results
        results_data["frames"].append(frame_results)
        
        # Progress indicator
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"[PROGRESS] {progress:.1f}% - Frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Add summary statistics
    results_data["summary"] = {
        "total_frames_processed": frame_idx,
        "total_detections": total_detections,
        "car_type_distribution": car_type_counts
    }
    
    # Save results to JSON
    if INFERENCE_CONFIG["save_results_json"]:
        json_path = output_dir / "detection_classification_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\n[RESULTS] Saved to: {json_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Output video: {output_video_path}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total car detections: {total_detections}")
    print(f"\nCar type distribution:")
    for car_type, count in sorted(car_type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"  {car_type:15s}: {count:5d} ({percentage:5.2f}%)")
    print(f"{'='*70}\n")


# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("CAR DETECTION + CLASSIFICATION INFERENCE")
    print(f"{'='*70}\n")
    print(f"Device: {DEVICE}")
    print(f"Detection confidence threshold: {INFERENCE_CONFIG['detection_conf']}")
    print(f"Classification enabled: {INFERENCE_CONFIG['classification_enabled']}")
    
    # Load classification model
    classification_model = None
    if INFERENCE_CONFIG["classification_enabled"]:
        try:
            classification_model = load_classification_model(CLASSIFICATION_CONFIG)
        except Exception as e:
            print(f"[ERROR] Failed to load classification model: {e}")
            print("[INFO] Continuing with detection only...")
            INFERENCE_CONFIG["classification_enabled"] = False
    
    # Test video path
    test_video = SCRIPT_DIR / "traffic_test.mp4"
    
    if not test_video.exists():
        print(f"[ERROR] Test video not found at {test_video}")
        exit(1)
    
    # Run inference for each detection model
    for model_config in DETECTION_MODELS:
        try:
            run_detection_classification(
                detection_model_config=model_config,
                classification_model=classification_model,
                test_video_path=test_video
            )
        except Exception as e:
            print(f"[ERROR] Failed to run inference for {model_config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n[INFO] All inference tasks completed!")