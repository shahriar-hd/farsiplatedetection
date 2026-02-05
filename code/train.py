# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 13:30:03 2025

@author: kut
"""

from ultralytics import YOLO
import torch
import os

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dev: {torch.cuda.get_device_name(0)}")
print(torch.version.cuda)

def train_model():
    model = YOLO('yolov8n.pt') 
    yaml_path = os.path.join("yolo_dataset", "data.yaml")

    custom_augs = {
        'fliplr': 0.0,
        'flipud': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }

    # 3. Train the model
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=64,
        name='iran_lpr_model',
        device=0, 
        patience=10,
        verbose=True,
        **custom_augs
    )

    print("Training finished.")
    
    # 4. Validate the model
    metrics = model.val()
    print(f"mAP@50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_model()