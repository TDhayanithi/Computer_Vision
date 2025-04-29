import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, test_dir, split_ratio=0.7):
    # List all the class names (e.g., 'benign', 'malignant', 'normal')
    classes = os.listdir(source_dir)
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue  # Skip if not a folder
        
        images = os.listdir(cls_path)
        random.shuffle(images)

        # Split dataset into train, val, and test
        train_index = int(len(images) * split_ratio)
        val_index = int(len(images) * (split_ratio + (1 - split_ratio) / 2))
        
        train_images = images[:train_index]
        val_images = images[train_index:val_index]
        test_images = images[val_index:]

        # Create class folders in train, val, and test dirs
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Copy images to train, val, and test folders
        for img in train_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_dir, cls, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.copy2(src, dst)

        for img in test_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(test_dir, cls, img)
            shutil.copy2(src, dst)

    print("✅ Dataset successfully split into 'train', 'val', and 'test' folders in 'data/'.")

if __name__ == "__main__":
    source_data_dir = "LIDC_Y-Net"   # Update here
    train_output_dir = "data/train"
    val_output_dir = "data/val"
    test_output_dir = "data/test"  # Make sure this path exists

    split_dataset(source_data_dir, train_output_dir, val_output_dir, test_output_dir)
