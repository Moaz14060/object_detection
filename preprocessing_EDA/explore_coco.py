# scripts/explore_yolo_balanced.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import pandas as pd
from tqdm import tqdm
import yaml

# --- USER CONFIG ---
BASE_DIR = r"E:\all e drive\AIU COMPUTER SCIENCE BACLERIOS\AIU COMPUTER SCIENCE SIXTH SEMESTER SPRING (24-25)\DEPI\Final Project\CocoDataSet\coco_preprocess\Milestone_1_Data_Preprocessing"
DATASET_DIR = os.path.join(BASE_DIR, "outputs", "balanced_dataset")
# Allow YAML to live either under the base dir or under outputs/ (some runs place it there)
possible_yaml_paths = [
    os.path.join(BASE_DIR, "data_balanced.yaml"),
    os.path.join(BASE_DIR, "outputs", "data_balanced.yaml"),
]
YAML_FILE = next((p for p in possible_yaml_paths if os.path.exists(p)), possible_yaml_paths[0])
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "balanced_exploration")
SAMPLE_COUNT = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD CLASSES ---
with open(YAML_FILE, "r") as f:
    data_yaml = yaml.safe_load(f)
CLASS_NAMES = data_yaml["names"]
NC = len(CLASS_NAMES)

# --- FUNCTION: EXPLORE YOLO DATASET ---
def explore_yolo_split(split):
    print(f"\n=== Exploring {split.upper()} Split ===")
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    class_counts = Counter()
    objs_per_image = []
    widths = []
    heights = []

    for img_file in tqdm(img_files, desc=f"Processing {split} images"):
        lbl_file = os.path.splitext(img_file)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_file)
        img_path = os.path.join(img_dir, img_file)
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        objs_per_image.append(len(lines))
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            class_counts[cls_id] += 1

    # --- Statistics summary ---
    summary = {
        "split": split,
        "num_images": len(img_files),
        "num_annotations_total": sum(class_counts.values()),
        "avg_objects_per_image": np.mean(objs_per_image) if objs_per_image else 0,
        "image_width_mean": np.mean(widths) if widths else 0,
        "image_height_mean": np.mean(heights) if heights else 0
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, f"summary_{split}.csv"), index=False)
    print("Summary:", summary)

    # --- Convert class_counts to readable form ---
    # Convert class names to strings (yaml may contain numeric labels)
    class_count_named = {str(CLASS_NAMES[k]): v for k, v in class_counts.items()}

    # --- Plots ---
    sns.set_style("whitegrid")

    # 1. Class distribution (with class names on Y-axis)
    plt.figure(figsize=(10, 0.4 * NC))  # height scales with number of classes

    # Ensure all classes are present even if count = 0
    counts = [class_counts.get(i, 0) for i in range(NC)]
    classes = [str(CLASS_NAMES[i]) for i in range(NC)]

    # Sort by count descending (most common at top)
    sorted_pairs = sorted(zip(counts, classes), reverse=True)
    sorted_counts, sorted_classes = zip(*sorted_pairs)

    sns.barplot(x=sorted_counts, y=sorted_classes, palette="crest")
    plt.title(f"Class Distribution - {split}")
    plt.xlabel("Object Count")
    plt.ylabel("Class Name")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{split}.png"))
    plt.close()



    # 2. Objects per image histogram
    plt.figure(figsize=(7, 5))
    plt.hist(objs_per_image, bins=30, color="royalblue")
    plt.xlabel("Objects per image")
    plt.ylabel("Number of images")
    plt.title(f"Objects per Image - {split}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"objects_per_image_{split}.png"))
    plt.close()

    # 3. Image resolution scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, s=10, alpha=0.4)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(f"Image Resolution Scatter - {split}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"image_resolution_{split}.png"))
    plt.close()

    # --- Sample image visualization ---
    print("Generating example annotated images...")
    samples = random.sample(img_files, min(SAMPLE_COUNT, len(img_files)))
    for i, fname in enumerate(samples):
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(lbl_path):
            continue
        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cls_name = str(CLASS_NAMES[int(cls)])
            cv2.putText(img, cls_name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_path = os.path.join(OUTPUT_DIR, f"sample_{split}_{i}.jpg")
        cv2.imwrite(out_path, img)

    # --- Insights ---
    print("\n📊 Key Insights:")
    top5 = class_counts.most_common(5)
    rare5 = class_counts.most_common()[-5:]
    print(f"Top 5 most common classes: {[CLASS_NAMES[c] for c, _ in top5]}")
    print(f"5 rarest classes: {[CLASS_NAMES[c] for c, _ in rare5]}")
    print(f"Average objects per image: {summary['avg_objects_per_image']:.2f}")
    print(f"Average image size: {summary['image_width_mean']:.0f}×{summary['image_height_mean']:.0f}")

    print("\n💡 Analysis:")
    print(f"- The dataset contains {len(img_files)} images.")
    print(f"- {sum(class_counts.values())} total labeled objects across {NC} classes.")
    print(f"- {top5[0][1]} '{CLASS_NAMES[top5[0][0]]}' instances dominate the dataset.")
    print(f"- The rarest objects ({[CLASS_NAMES[c] for c, _ in rare5]}) may still need attention.\n")

    return summary, class_count_named


if __name__ == "__main__":
    print("Starting YOLO Balanced Dataset Exploration ...")
    explore_yolo_split("train")
    explore_yolo_split("val")
    print(f"\n✅ Exploration finished. Visual reports saved in:\n{OUTPUT_DIR}")
####################################################################################

########################### Version 1 ######################################

""" # scripts/explore_coco.py
import os, json, random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from collections import Counter
import pandas as pd

# ---- USER CONFIG ----
BASE_DIR = r"E:\all e drive\AIU COMPUTER SCIENCE BACLERIOS\AIU COMPUTER SCIENCE SIXTH SEMESTER SPRING (24-25)\DEPI\Final Project\CocoDataSet\coco_preprocess"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANN_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_ANN = os.path.join(ANN_DIR, "instances_train2017.json")
VAL_ANN = os.path.join(ANN_DIR, "instances_val2017.json")
IMG_DIR_TRAIN = os.path.join(DATA_DIR, "train2017")
IMG_DIR_VAL = os.path.join(DATA_DIR, "val2017")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "exploration")

SAMPLE_COUNT = 8

# ✅ Selected 24 classes for autonomous-driving dataset
SELECTED_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "fire hydrant", "bench", "parking meter",
    "umbrella", "backpack", "handbag", "tie", "cell phone", "dog", "cat",
    "horse", "bird", "skateboard", "boat", "suitcase"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
def explore(ann_file, img_dir, split_name="train"):
    print(f"\n=== Exploring {split_name.upper()} split ===")
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c['id']: c['name'] for c in cats}
    selected_cat_ids = coco.getCatIds(catNms=SELECTED_CLASSES)

    # --- Basic statistics ---
    img_ids = coco.getImgIds()
    images = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    widths = [img['width'] for img in images]
    heights = [img['height'] for img in images]
    class_counts = Counter()
    for a in anns:
        if a['category_id'] in selected_cat_ids:
            class_counts[cat_id_to_name[a['category_id']]] += 1

    objs_per_image = [len(coco.getAnnIds(imgIds=img['id'], catIds=selected_cat_ids)) for img in images]

    summary = {
        "split": split_name,
        "num_images": len(images),
        "num_annotations_total": len(anns),
        "num_annotations_selected": sum(class_counts.values()),
        "image_width_mean": np.mean(widths),
        "image_height_mean": np.mean(heights),
        "avg_objects_selected_per_image": np.mean(objs_per_image)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, f"summary_{split_name}.csv"), index=False)
    print("Summary:", summary)

    # --- Visualization section ---
    sns.set_style("whitegrid")

    # 1. Class distribution
    plt.figure(figsize=(10,6))
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    sns.barplot(x=counts, y=classes, palette="crest")
    plt.title(f"Class Distribution - {split_name}")
    plt.xlabel("Object count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{split_name}.png"))
    plt.close()

    # 2. Objects per image histogram
    plt.figure(figsize=(7,5))
    plt.hist(objs_per_image, bins=30, color="royalblue")
    plt.xlabel("Objects per image")
    plt.ylabel("Number of images")
    plt.title(f"Objects per Image Histogram ({split_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"objects_per_image_{split_name}.png"))
    plt.close()

    # 3. Image dimension distribution
    plt.figure(figsize=(6,6))
    plt.scatter(widths, heights, s=10, alpha=0.4)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(f"Image Resolution Scatter ({split_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"image_resolution_{split_name}.png"))
    plt.close()

    # --- Sample image visualization ---
    print("Generating example annotated images...")
    sample_img_ids = random.sample(img_ids, min(SAMPLE_COUNT, len(img_ids)))
    for i, img_id in enumerate(sample_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        anns_ids = coco.getAnnIds(imgIds=img_id, catIds=selected_cat_ids)
        anns_sel = coco.loadAnns(anns_ids)
        for ann in anns_sel:
            x, y, w, h = map(int, ann['bbox'])
            cat_name = cat_id_to_name[ann['category_id']]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, cat_name, (x, max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = os.path.join(OUTPUT_DIR, f"sample_{split_name}_{i}.jpg")
        cv2.imwrite(out_path, img)

    # --- Automatic insights ---
    print("\n📊 Key Insights:")
    top5 = class_counts.most_common(5)
    print(f"Top 5 most common objects: {top5}")
    rare5 = class_counts.most_common()[-5:]
    print(f"Least common objects: {rare5}")
    print(f"Average objects per image: {summary['avg_objects_selected_per_image']:.2f}")
    print(f"Average image size: {summary['image_width_mean']:.0f}×{summary['image_height_mean']:.0f}")

    print("\n💡 Analysis Q&A:")
    print("Q1: Which objects dominate the dataset?")
    print(f"A1: {top5[0][0]} appears most frequently, suggesting focus on urban scenes.")
    print("Q2: Are there underrepresented classes?")
    print(f"A2: The rarest are {', '.join([c for c,_ in rare5])}, consider augmenting them.")
    print("Q3: How dense are scenes?")
    print(f"A3: On average {summary['avg_objects_selected_per_image']:.1f} relevant objects per image.")
    print("Q4: Are image sizes consistent?")
    print(f"A4: Widths range {min(widths)}–{max(widths)}, heights {min(heights)}–{max(heights)}.")
    print("Q5: What do examples show?")
    print("A5: Sample images saved with bounding boxes illustrate diversity in object placement and scale.\n")

    return summary, class_counts


if __name__ == "__main__":
    print("Starting Dataset Exploration ...")
    explore(TRAIN_ANN, IMG_DIR_TRAIN, "train")
    explore(VAL_ANN, IMG_DIR_VAL, "val")
    print(f"\n✅ Exploration finished. Check visual reports in:\n{OUTPUT_DIR}")

"""

