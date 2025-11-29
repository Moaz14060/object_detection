import os
import random
import shutil
import statistics
from collections import defaultdict

def balance_dataset(
    base_dir,
    strategy="median",
    fixed_target=500,
    seed=42
):
    """
    Balances a YOLO-style dataset by undersampling classes in the training set.

    Parameters:
        base_dir (str): Root folder containing 'train/images' and 'train/labels'.
        strategy (str): Balancing method: 'min', 'median', or 'fixed'.
        fixed_target (int): Target count per class when strategy='fixed'.
        seed (int): Random seed for reproducibility.

    Output:
        Creates a new folder 'train_balanced' containing balanced images and labels.
    """
    
    random.seed(seed)

    # ---------- PATH SETUP ----------
    train_images_dir = os.path.join(base_dir, "train", "images")
    train_labels_dir = os.path.join(base_dir, "train", "labels")
    out_dir = os.path.join(base_dir, "train_balanced")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    # ---------- COLLECT IMAGE-CLASS MAPPING ----------
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_classes = {}
    class_to_images = defaultdict(list)

    for img in image_files:
        lbl = os.path.splitext(img)[0] + ".txt"
        lbl_path = os.path.join(train_labels_dir, lbl)
        classes_in_image = set()

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    classes_in_image.add(class_id)

        image_classes[img] = classes_in_image
        for c in classes_in_image:
            class_to_images[c].append(img)

    # ---------- COMPUTE CLASS COUNTS ----------
    class_counts = {c: len(set(imgs)) for c, imgs in class_to_images.items()}
    print("Original class counts (train):")
    for c in sorted(class_counts):
        print(f"  class {c}: {class_counts[c]}")

    # ---------- DETERMINE TARGET COUNTS ----------
    if not class_counts:
        print("No labeled data found. Aborting.")
        return

    if strategy == "min":
        target = min(class_counts.values())
        target_counts = {c: target for c in class_counts}
    elif strategy == "median":
        vals = list(class_counts.values())
        target = int(statistics.median(vals))
        target_counts = {c: target for c in class_counts}
    elif strategy == "fixed":
        target_counts = {c: fixed_target for c in class_counts}
    else:
        raise ValueError("Unknown strategy. Use 'min', 'median', or 'fixed'.")

    print("\nTarget counts per class (undersampling):")
    for c in sorted(target_counts):
        print(f"  class {c}: {target_counts[c]}")

    # ---------- SELECT IMAGES ----------
    selected_images = set()

    for c, imgs in class_to_images.items():
        imgs_unique = list(set(imgs))
        available = len(imgs_unique)
        tgt = target_counts.get(c, 0)
        if available <= tgt:
            pick = imgs_unique
        else:
            pick = random.sample(imgs_unique, tgt)
        selected_images.update(pick)

    print(f"\nTotal images selected for balanced training: {len(selected_images)}")

    # ---------- COPY FILES ----------
    copied = 0
    for img in selected_images:
        src_img = os.path.join(train_images_dir, img)
        src_lbl = os.path.join(train_labels_dir, os.path.splitext(img)[0] + ".txt")
        dst_img = os.path.join(out_dir, "images", img)
        dst_lbl = os.path.join(out_dir, "labels", os.path.splitext(img)[0] + ".txt")

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied += 1
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    print(f"\nCopied {copied} images (and their labels) to: {out_dir}")
    print("Dataset balancing complete!")



# Example usage:
# balance_dataset(r"F:\data1", strategy="median", fixed_target=500, seed=42)
