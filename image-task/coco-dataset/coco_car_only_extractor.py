import json
from pathlib import Path
from shutil import copyfile
from collections import defaultdict

# =====================
# CONFIG
# =====================
COCO_ROOT = Path(".")
OUTPUT_ROOT = Path("coco_car_only")

CAR_CATEGORY_ID = 3  # COCO car category (1-based)
BUS_CATEGORY_ID = 6  # COCO bus category (1-based)
TRUCK_CATEGORY_ID = 8  # COCO truck category (1-based)


SPLITS = {
    "train": {
        "ann": COCO_ROOT / "annotations/instances_train2017.json",
        "img": COCO_ROOT / "train2017",
    },
    "val": {
        "ann": COCO_ROOT / "annotations/instances_val2017.json",
        "img": COCO_ROOT / "val2017",
    },
}

# =====================
# UTILS
# =====================
def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    return x_c, y_c, w / img_w, h / img_h

# =====================
# MAIN
# =====================
for split, paths in SPLITS.items():
    print(f"\nProcessing {split} split")

    out_img_dir = OUTPUT_ROOT / "images" / split
    out_lbl_dir = OUTPUT_ROOT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(paths["ann"]) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    car_anns = [
        ann for ann in coco["annotations"]
        if ann["category_id"] == CAR_CATEGORY_ID or ann["category_id"] == BUS_CATEGORY_ID or ann["category_id"] == TRUCK_CATEGORY_ID
    ]

    anns_per_image = defaultdict(list)
    for ann in car_anns:
        anns_per_image[ann["image_id"]].append(ann)

    kept_images = 0

    for img_id, anns in anns_per_image.items():
        img_info = images[img_id]
        src_img = paths["img"] / img_info["file_name"]

        if not src_img.exists():
            continue

        # Copy image
        dst_img = out_img_dir / img_info["file_name"]
        copyfile(src_img, dst_img)

        # Write label
        label_path = out_lbl_dir / img_info["file_name"].replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                bbox = coco_to_yolo_bbox(
                    ann["bbox"],
                    img_info["width"],
                    img_info["height"]
                )
                f.write(f"0 {' '.join(f'{v:.6f}' for v in bbox)}\n")

        kept_images += 1

    print(f"{kept_images} images kept in {split}")
