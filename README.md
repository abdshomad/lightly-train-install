This repository contains training scripts and setup for object detection models on the chicken detection dataset using LightlyTrain.

## Overview

This project supports training object detection models using:
- **LightlyTrain** - State-of-the-art object detection with DINOv3 backbones
- **Chicken Detection Dataset** - Custom dataset with 2 classes (chicken, not-chicken)

## Quick Start

### 1. Add Submodules

```bash
git submodule add https://github.com/lightly-ai/lightly-train lightly-train
git submodule add https://github.com/abdshomad/chicken-detection-labelme-format
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
uv venv

# Sync dependencies
uv sync
```

### 4. Convert Dataset to YOLO Format

```bash
cd chicken-detection-labelme-format
uv venv
source .venv/bin/activate
uv run convert_labelme_to_yolo.py
# This will produce chicken-detection-labelme-format/yolo-format
cd ..
```

### 5. Train Model

Train the DINOv3 ViTT16-LTDETR model:

```bash
# Using uv run (recommended)
uv run python train-dinov3-vitt16-ltdetr.py

# Or activate venv first
source .venv/bin/activate
python train-dinov3-vitt16-ltdetr.py
```

## Training Scripts

### DINOv3 ViTT16-LTDETR

**Script:** `train-dinov3-vitt16-ltdetr.py`

Trains a DINOv3 ViTT16-LTDETR object detection model pre-trained on COCO, fine-tuned on the chicken detection dataset.

**Configuration:**
- Model: `dinov3/vitt16-ltdetr-coco`
- Output: `chicken-detection-models/dinov3/vitt16-ltdetr-chicken`
- Dataset: `chicken-detection-labelme-format/yolo-format`
- Classes: `{0: "chicken", 1: "not-chicken"}`

**Usage:**
```bash
uv run python train-dinov3-vitt16-ltdetr.py
```

## Dataset Structure

The dataset should be in YOLO format:

```
chicken-detection-labelme-format/yolo-format/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training labels (YOLO format)
    └── val/            # Validation labels (YOLO format)
```

## Project Structure

```
lightly-train-install/
├── train-dinov3-vitt16-ltdetr.py    # DINOv3 training script
├── free_gpu.py                      # GPU memory management utility
├── pyproject.toml                   # Python dependencies
├── lightly-train/                   # LightlyTrain submodule
├── chicken-detection-labelme-format/ # Dataset submodule
└── chicken-detection-models/        # Trained model outputs (created during training)
```

## Dependencies

Key dependencies (see `pyproject.toml` for full list):
- `lightly-train>=0.14.1` - LightlyTrain framework
- `torch` - PyTorch
- `torchvision` - Computer vision utilities
- `tensorboard` - Training visualization

## Documentation

- `INSTALL.md` - Detailed installation guide
- `AGENTS.md` - Development guidelines and rules

## Notes

- Always use `uv sync` to manage dependencies
- Never modify files within git submodules (see `AGENTS.md`)
- Training outputs are saved to `chicken-detection-models/` directory 
