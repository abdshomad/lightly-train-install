#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train DINOv3 ViTT16-LTDETR Object Detection Model for Chicken Detection

This script trains a lightly-train object detection model using the DINOv3 ViTT16-LTDETR
architecture pre-trained on COCO dataset, fine-tuned on the chicken detection dataset.

The model will be trained to detect two classes:
- 0: chicken
- 1: not-chicken
"""

import lightly_train


if __name__ == "__main__":
    lightly_train.train_object_detection(
        out="chicken-detection-models/dinov3/vitt16-ltdetr-chicken",
        model="dinov3/vitt16-ltdetr-coco",
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
