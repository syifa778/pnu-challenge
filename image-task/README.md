# PNU CHALLENGE : IMAGE TASK

## Dataset Preparation

### DETECTION : COCO Car Only Dataset

```bash
cd image-task/coco-dataset

# Download Dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip Dataset
unzip train2017.zip && rm train2017.zip
unzip val2017.zip && rm val2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip

# Extract Car Only 
uv run coco_car_only_extractor.py

# Expected output
# Processing train split
# 12251 images kept in train

# Processing val split
# 535 images kept in val
```

### DETECTION : Indonesia Car CCTV Dataset

```bash
# Frame Sampling
cd indonesia-dataset &&
uv run cctv_frame_sampler.py --video ./cctv-videos/pvj-bandung.mp4 --output ./cctv-images/pvj-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/aa-bandung.mp4 --output ./cctv-images/aa-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/paskal-bandung.mp4 --output ./cctv-images/paskal-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/pasteur-bandung.mp4 --output ./cctv-images/pasteur-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/yogya.mp4 --output ./cctv-images/yogya --interval 0.05

# Pseudolabelling
cd .. && 
uv run pseudolabelling_with_stage_2_yolo.py

# Split Data
cd indonesia-dataset &&
uv run split_dataset.py
# Expected output :
# aa-bandung: 3807 → train=2284, val=761, test=762
# paskal-bandung: 4430 → train=2658, val=886, test=886
# pasteur-bandung: 4073 → train=2443, val=814, test=816
# yogya: 3843 → train=2305, val=768, test=770
# pvj-bandung: 3657 → train=2194, val=731, test=732
# Dataset split completed
```

## Car Detection Stages YOLOv10 Pretrained on COCO Based
YOLOv10 chosen because the architecture has scientific paper and based on ultralytic docs and specifically yolov10m beside their medium version for general-purpose use, the model has reasonable MAP and FLOPs compared to last and new version of yolo.
### Stage 1 & 2 : Frozen backbone & Full fine-tuning
Data : COCO car only dataset 
Stage 1 : Train on frozen pretrained yolov10 with COCO model backbone
Stage 2 : Train on full fine-tuning trained model from stage 1

```bash
uv run stage_1_2_yolo_od.py
```

### Stage 3 : Full fine-tuning
Data : Indonesia CCTV Pseudolabelling

```bash
cd .. &&
uv run stage_3_yolo_od.py
```

## Car Detection Stages Pretrained on COCO Based

### Stage 1 & 2 : Frozen backbone & Full fine-tuning
Data : COCO car only dataset 
Stage 1 : Train on frozen pretrained yolov10 with COCO model backbone
Stage 2 : Train on full fine-tuning trained model from stage 1

```bash
uv run stage_1_2_yolo_od.py
```

### Stage 3 : Full fine-tuning
Data : Indonesia CCTV Pseudolabelling

```bash
cd .. &&
uv run stage_3_yolo_od.py
```

## Car Detection Test
Data : Infosec test video
```bash
uv run detection_testing.py
```

## Car Detection Overall Pipeline for both YOLO and RT-DETR
```bash
uv run od_pipeline.py
```