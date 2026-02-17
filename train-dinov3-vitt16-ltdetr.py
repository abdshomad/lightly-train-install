#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train DINOv3 ViTT16-LTDETR Object Detection Model for Chicken Detection

This script trains a lightly-train object detection model using the DINOv3 ViTT16-LTDETR
architecture pre-trained on COCO dataset, fine-tuned on the chicken detection dataset.

The model will be trained to detect two classes:
- 0: chicken
- 1: not-chicken
"""

import warnings
from pathlib import Path
import torch
import lightly_train

# Suppress AccumulateGrad stream mismatch warning
# This warning occurs during DDP training and can be safely ignored
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

# Suppress torchmetrics deprecation warning about non-tuple sequence indexing
# This is a known issue in torchmetrics and will be fixed in future versions
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
    module="torchmetrics",
    category=UserWarning,
)


if __name__ == "__main__":
    output_dir = Path("chicken-detection-models/dinov3/vitt16-ltdetr-chicken")
    
    # Check if output directory exists and is not empty
    # If it exists, resume training from the last checkpoint
    resume_interrupted = output_dir.exists() and any(output_dir.iterdir())
    
    if resume_interrupted:
        print(f"Output directory '{output_dir}' exists and is not empty.")
        print("Resuming training from last checkpoint...")
    
    lightly_train.train_object_detection(
        out=str(output_dir),
        model="dinov3/vitt16-ltdetr-coco",
        steps=200,  # Small number of steps for demonstration, default is 90_000.
        batch_size=4,  # Small batch size for demonstration, default is 16.
        resume_interrupted=resume_interrupted,  # Resume if output directory exists
        data={
            "path": "chicken-detection-labelme-format/yolo-format",
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "chicken",
                1: "not-chicken",
            },
        },
    )
