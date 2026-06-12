# Car Retrieval System Using Object Detection and Car Type Classification
This work presents the development of an end-to-end Car Retrieval System that integrates object detection and car type classification. The system is designed to detect multiple car instances in real-world scenes and classify each detected vehicle into one of several car types commonly found in Indonesia, such as Sedan, MPV, SUV, Pickup, Truck, and Bus. Two different paradigms are explored for each task: convolutional neural network (CNN)-based models and attention-based models. The proposed system is trained and evaluated on the COCO 2017 dataset for object detection task and Indonesia Traffic CCTV for both tasks to demonstrate its effectiveness and practical applicability.

## Quick Result Overview
[Infosec Video Testing](https://drive.google.com/file/d/1ZdygRvZ0-n5xtmNrsQ3P2nUCjmxw7FY1/view?usp=sharing)

## Dataset Preparation

### COCO Car Only Dataset

```bash
cd image-task/coco-dataset

# Download Dataset
wget http://images.cocodataset.org/zips/train2017.zip && wget http://images.cocodataset.org/zips/val2017.zip && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip Dataset
unzip train2017.zip && rm train2017.zip && unzip val2017.zip && rm val2017.zip && unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip

# Extract Car Only (car, bus, truck) 
uv run coco_car_only_extractor.py

# Expected output
# Processing train split
# 16270 images kept in train

# Processing val split
# 707 images kept in val
```

### Indonesia Car CCTV Dataset

[CCTV Indonesia Videos](https://drive.google.com/drive/folders/1pK8SQ-aluYJSzkJV1dlgpulXsVDK7oSq?usp=sharing)

```bash
# Frame Sampling
cd indonesia-dataset &&
uv run cctv_frame_sampler.py --video ./cctv-videos/pvj-bandung.mp4 --output ./cctv-images/pvj-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/aa-bandung.mp4 --output ./cctv-images/aa-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/paskal-bandung.mp4 --output ./cctv-images/paskal-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/pasteur-bandung.mp4 --output ./cctv-images/pasteur-bandung --interval 0.05 &&
uv run cctv_frame_sampler.py --video ./cctv-videos/yogya.mp4 --output ./cctv-images/yogya --interval 0.05

# aa-bandung: 3807 → train=2284, val=761, test=762
# paskal-bandung: 4430 → train=2658, val=886, test=886
# pasteur-bandung: 4073 → train=2443, val=814, test=816
# yogya: 3843 → train=2305, val=768, test=770
# pvj-bandung: 3657 → train=2194, val=731, test=732
```

### Classification Dataset
[Label Studio Result](https://drive.google.com/drive/folders/1pK8SQ-aluYJSzkJV1dlgpulXsVDK7oSq?usp=sharing)

``` bash
  car_type  total_annotations  
0      bus                 45      
1      mpv                542      
2   pickup                 79       
3    sedan                193       
4      suv                395       
5    truck                 61      
```

### Car Detection Overall Pipeline for both YOLO and RT-DETR Stage 1, 2, 3 : Frozen backbone, Full fine-tuning, and Full fine-tuning
```bash
uv run od_pipeline.py
```

### Car Classification Train
```bash
uv run train_eval_classification.py
```

### Testing Detection and Classification
Final model: yolov10m-based detection and vitb16-based classification
```bash
uv run testing.py
```
