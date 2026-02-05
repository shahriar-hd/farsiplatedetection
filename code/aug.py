# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:34:58 2025

@author: kut
"""

import os
import cv2
import glob
import random
import numpy as np
import albumentations as A
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = "yolo_dataset"
TARGET_COUNT = 5000  # Minimum required instances per class

# The exact class list used in your training
CLASS_NAMES = [
    "plate",      
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "alef",       
    "be",         
    "pe",         
    "te",         
    "se",         
    "jim",        
    "dal",        
    "ze",         
    "sin",        
    "shin",       
    "sad",        
    "ta",         
    "za",         
    "ein",        
    "fe",         
    "ghaf",       
    "kaf",        
    "gaf",        
    "lam",        
    "mim",        
    "noon",       
    "vav",        
    "he",         
    "ye",         
    "zhe",        
    "disabled",   
    "protocol",   
    "S",          
    "D",          
]

# List of specific classes you want to augment if they are below TARGET_COUNT
# Based on your request: "پ الف ث ش ظ ک گ معلولین دیپلمات و سرویس و تشریفات"
# Map these Persian concepts to your English CLASS_NAMES:
TARGET_CLASSES_NAMES = [
    "alef", "pe", "se", "shin", "za", "kaf", "gaf", 
    "disabled", "protocol", "S", "D"
]

# Convert names to IDs
TARGET_CLASS_IDS = [CLASS_NAMES.index(name) for name in TARGET_CLASSES_NAMES if name in CLASS_NAMES]

# ==========================================
# 2. AUGMENTATION PIPELINE
# ==========================================

# Define the augmentation pipeline
# Rotations up to 25 degrees (safer than 30 for bounding boxes), noise, color shifts.
# NO FLIPS included.
aug_pipeline = A.Compose([
    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    # Ensure bounding boxes are handled correctly for YOLO format
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def read_yolo_label(txt_path):
    """Reads a YOLO label file and returns a list of boxes and class IDs."""
    boxes = []
    class_labels = []
    if not os.path.exists(txt_path):
        return boxes, class_labels
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id = int(parts[0])
            # x_center, y_center, width, height
            bbox = [float(x) for x in parts[1:]] 
            
            boxes.append(bbox)
            class_labels.append(cls_id)
            
    return boxes, class_labels

def save_yolo_label(txt_path, boxes, class_labels):
    """Saves boxes and labels to a text file in YOLO format."""
    with open(txt_path, 'w') as f:
        for bbox, cls_id in zip(boxes, class_labels):
            # Limit values to [0, 1] to prevent errors
            xc, yc, w, h = bbox
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

# ==========================================
# 4. MAIN PROCESS
# ==========================================

def balance_dataset():
    print(f"Target classes to augment: {TARGET_CLASSES_NAMES}")
    print(f"Target count per class: {TARGET_COUNT}")

    # 1. Index all files
    # We look into all splits because we want to balance the dataset globally 
    # (or you can restrict to 'train' only if you prefer).
    # Typically, we only augment TRAIN data. Let's focus on TRAIN + VAL.
    
    splits = ['train', 'val', 'test']
    all_files_map = {cid: [] for cid in TARGET_CLASS_IDS} # Stores paths to images containing the class
    
    print("Indexing dataset...")
    for split in splits:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)
        
        # Get all txt files
        txt_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        
        for txt_file in tqdm(txt_files, desc=f"Scanning {split}"):
            # Check what classes are in this file
            _, class_ids_in_file = read_yolo_label(txt_file)
            
            # Identify corresponding image
            basename = os.path.basename(txt_file).replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, basename)
            
            if not os.path.exists(img_path):
                continue
                
            # Add to map if it contains our target classes
            unique_classes = set(class_ids_in_file)
            for cid in unique_classes:
                if cid in TARGET_CLASS_IDS:
                    all_files_map[cid].append({
                        'img_path': img_path,
                        'txt_path': txt_file,
                        'split': split
                    })

    # 2. Check deficits and Generate
    print("\n--- Augmentation Status ---")
    
    for cid in TARGET_CLASS_IDS:
        current_count = len(all_files_map[cid])
        class_name = CLASS_NAMES[cid]
        
        if current_count >= TARGET_COUNT:
            print(f"Class '{class_name}' (ID {cid}): OK ({current_count} instances)")
            continue
            
        needed = TARGET_COUNT - current_count
        print(f"Class '{class_name}' (ID {cid}): Needs {needed} more (Current: {current_count})")
        
        if current_count == 0:
            print(f"  WARNING: No instances found for '{class_name}'. Cannot augment.")
            continue

        # Start generating
        generated_count = 0
        pbar = tqdm(total=needed, desc=f"Augmenting {class_name}")
        
        while generated_count < needed:
            # Randomly select a source image containing this class
            source_data = random.choice(all_files_map[cid])
            
            src_img_path = source_data['img_path']
            src_txt_path = source_data['txt_path']
            
            # Load Image
            image = cv2.imread(src_img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load Labels
            bboxes, labels = read_yolo_label(src_txt_path)
            
            # Apply Augmentation
            try:
                transformed = aug_pipeline(image=image, bboxes=bboxes, class_labels=labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                
                # If augmentation pushed boxes out of view completely, skip
                if len(aug_bboxes) == 0 and len(bboxes) > 0:
                    continue
                    
            except Exception as e:
                # Sometimes geometric transformations fail on edge cases
                continue
                
            # Save Output
            # We save in the SAME folder as the source
            img_dir = os.path.dirname(src_img_path)
            lbl_dir = os.path.dirname(src_txt_path)
            
            # Create a unique name
            base_name = os.path.splitext(os.path.basename(src_img_path))[0]
            new_name = f"{base_name}_aug_{generated_count}_{random.randint(100,999)}"
            
            target_img_path = os.path.join(img_dir, new_name + ".jpg")
            target_txt_path = os.path.join(lbl_dir, new_name + ".txt")
            
            # Write Image (Convert back to BGR for OpenCV)
            cv2.imwrite(target_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            # Write Label
            save_yolo_label(target_txt_path, aug_bboxes, aug_labels)
            
            generated_count += 1
            pbar.update(1)
            
        pbar.close()

    print("\nData balancing complete.")

if __name__ == "__main__":
    balance_dataset()