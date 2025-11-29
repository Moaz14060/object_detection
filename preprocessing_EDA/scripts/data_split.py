import os
import random
import shutil

def split_dataset(base_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """
    Splits a dataset of images and labels into train, validation, and test sets.

    Parameters:
        base_dir (str): Path to the base dataset folder containing 'images' and 'labels' subfolders.
        train_ratio (float): Proportion of data for training.
        valid_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
    """

    # Ensure ratios sum to 1.0
    total = train_ratio + valid_ratio + test_ratio
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    splits = {
        "train": train_ratio,
        "valid": valid_ratio,
        "test": test_ratio
    }

    # Create split folders
    for split in splits.keys():
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "labels"), exist_ok=True)

    # Get and shuffle images
    images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    # Calculate split sizes
    n = len(images)
    train_end = int(n * splits["train"])
    valid_end = train_end + int(n * splits["valid"])

    train_files = images[:train_end]
    valid_files = images[train_end:valid_end]
    test_files  = images[valid_end:]

    def move_files(file_list, split_name):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            
            # Move image
            shutil.move(os.path.join(images_dir, img_file),
                        os.path.join(base_dir, split_name, "images", img_file))
            
            # Move label if exists
            if os.path.exists(os.path.join(labels_dir, label_file)):
                shutil.move(os.path.join(labels_dir, label_file),
                            os.path.join(base_dir, split_name, "labels", label_file))

    # Move files for each split
    move_files(train_files, "train")
    move_files(valid_files, "valid")
    move_files(test_files, "test")

    print(f"Done! Split {n} images into:")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(valid_files)}")
    print(f"Test:  {len(test_files)}")

# Example usage:
# split_dataset(r"F:\export", train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)