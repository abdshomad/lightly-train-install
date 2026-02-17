# Installation Guide

Complete step-by-step installation guide for training object detection models on chicken detection dataset.

This guide covers:
- **LightlyTrain** - DINOv3-based object detection training

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **CUDA-capable GPU(s)** with NVIDIA drivers installed
- **Git** installed
- **`uv`** package manager installed ([Installation guide](https://github.com/astral-sh/uv))
- **nvidia-smi** available (for GPU monitoring)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.11 or higher

# Check CUDA availability
nvidia-smi  # Should show GPU information

# Check uv installation
uv --version
```

## 🚀 Installation Steps

### Step 1: Clone Repository and Initialize Submodules

```bash
# If cloning the repository for the first time
git clone <repository-url>
cd lightly-train-install

# Initialize git submodules
git submodule update --init --recursive
```

**Important:** The repository uses git submodules for:
- `lightly-train/` - The LightlyTrain framework
- `chicken-detection-labelme-format/` - The dataset and configs

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment using uv
uv venv

# Sync dependencies from pyproject.toml
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all required dependencies (PyTorch, torchvision, etc.)

### Step 3: Activate Virtual Environment (Optional)

While `uv run` can execute commands directly, you can also activate the environment:

```bash
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```

### Step 4: Convert Dataset to YOLO Format

For LightlyTrain, the dataset must be in YOLO format:

```bash
cd chicken-detection-labelme-format
uv venv
source .venv/bin/activate
uv run convert_labelme_to_yolo.py
cd ..
```

This will create the YOLO format dataset at `chicken-detection-labelme-format/yolo-format/`.

**Verify:** Check that the following directories exist:
- `chicken-detection-labelme-format/yolo-format/images/train/`
- `chicken-detection-labelme-format/yolo-format/images/val/`
- `chicken-detection-labelme-format/yolo-format/labels/train/`
- `chicken-detection-labelme-format/yolo-format/labels/val/`

## 🎯 Training with LightlyTrain

### Quick Start: DINOv3 ViTT16-LTDETR

Train a DINOv3 ViTT16-LTDETR model on the chicken detection dataset:

```bash
# Using uv run (recommended)
uv run python train-dinov3-vitt16-ltdetr.py

# Or activate venv first
source .venv/bin/activate
python train-dinov3-vitt16-ltdetr.py
```

**Training Script:** `train-dinov3-vitt16-ltdetr.py`

**Configuration:**
- Model: `dinov3/vitt16-ltdetr-coco` (pre-trained on COCO)
- Output: `chicken-detection-models/dinov3/vitt16-ltdetr-chicken`
- Dataset: `chicken-detection-labelme-format/yolo-format`
- Classes: `{0: "chicken", 1: "not-chicken"}`

**Training Output:**
After training completes, the model will be saved to:
- `chicken-detection-models/dinov3/vitt16-ltdetr-chicken/exported_models/exported_best.pt` - Best model checkpoint
- `chicken-detection-models/dinov3/vitt16-ltdetr-chicken/exported_models/exported_last.pt` - Last epoch checkpoint
- `chicken-detection-models/dinov3/vitt16-ltdetr-chicken/checkpoints/` - Training checkpoints
- `chicken-detection-models/dinov3/vitt16-ltdetr-chicken/train.log` - Training logs

### Loading Trained Model for Inference

```python
import lightly_train

# Load the trained model
model = lightly_train.load_model("chicken-detection-models/dinov3/vitt16-ltdetr-chicken/exported_models/exported_best.pt")

# Run inference
results = model.predict("path/to/image.jpg")
print(results["labels"])   # Class labels
print(results["bboxes"])   # Bounding boxes
print(results["scores"])   # Confidence scores
```

### Customizing Training Parameters

You can modify `train-dinov3-vitt16-ltdetr.py` to customize training:

```python
lightly_train.train_object_detection(
    out="chicken-detection-models/dinov3/vitt16-ltdetr-chicken",
    model="dinov3/vitt16-ltdetr-coco",
    steps=90000,        # Number of training steps (default: 90000)
    batch_size=16,      # Batch size (default: 16)
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
```

## 🧪 Testing Installation

### Test LightlyTrain Installation

```bash
# Test import of lightly_train
uv run python -c "import lightly_train; print('LightlyTrain version:', lightly_train.__version__)"
```

### Test Training Script

Verify the training script can be loaded:

```bash
# Check script syntax
uv run python -m py_compile train-dinov3-vitt16-ltdetr.py
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Git Submodules Not Initialized

**Error:** `FileNotFoundError: lightly-train/... not found`

**Solution:**
```bash
git submodule update --init --recursive
```

#### 2. Dataset Not Converted to YOLO Format

**Error:** `FileNotFoundError: chicken-detection-labelme-format/yolo-format/...`

**Solution:**
```bash
cd chicken-detection-labelme-format
uv run convert_labelme_to_yolo.py
cd ..
```

#### 3. CUDA Out of Memory (OOM)

**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Solutions:**
1. **Free GPU memory:**
   ```bash
   uv run python free_gpu.py --kill
   ```

2. **Reduce batch size** in training script:
   - Edit `batch_size` parameter in `train-dinov3-vitt16-ltdetr.py`
   - Start with smaller values (e.g., 4 or 8)

3. **Reduce number of training steps:**
   - Edit `steps` parameter in `train-dinov3-vitt16-ltdetr.py`

#### 4. Missing Python Dependencies

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Sync dependencies
uv sync

# Or add missing package
uv add <package-name>
```

#### 5. Path Resolution Errors

**Error:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:**
- Verify dataset paths in `train-dinov3-vitt16-ltdetr.py`
- Ensure `chicken-detection-labelme-format/yolo-format/` directory exists
- Check that YOLO format dataset has been created (images and labels directories)

#### 6. LightlyTrain Import Errors

**Error:** `ModuleNotFoundError: No module named 'lightly_train'`

**Solution:**
```bash
# Ensure dependencies are synced
uv sync

# Verify lightly-train is installed
uv run python -c "import lightly_train; print(lightly_train.__version__)"
```

### Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all prerequisites are met
3. Ensure all submodules and dependencies are installed
4. Check GPU memory availability with `nvidia-smi`
5. Review the `README.md` for additional information

## 📦 Dependency Management

### Adding Dependencies

```bash
# Add a new package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>
```

### Updating Dependencies

```bash
# Sync with pyproject.toml
uv sync

# Update a specific package
uv add <package-name>@latest
```

### Current Dependencies

See `pyproject.toml` for the complete list. Key dependencies include:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `faster-coco-eval` - COCO evaluation metrics
- `PyYAML` - YAML configuration parsing
- `tensorboard` - Training visualization
- `transformers` - Hugging Face transformers (for DINOv3)
- `torchmetrics` - PyTorch metrics
- `termcolor` - Colored terminal output

## 🔄 Updating the Installation

### Update Git Submodules

```bash
# Update submodules to latest commits
git submodule update --remote

# Or update to specific commits
cd lightly-train
git pull origin main
cd ..
```

**⚠️ Important:** Do not modify files within submodule directories directly. See `AGENTS.md` for more information.

### Update Python Dependencies

```bash
# Sync dependencies after pyproject.toml changes
uv sync
```

## ✅ Installation Checklist

Use this checklist to verify your installation:

### Installation Checklist
- [ ] Python 3.11+ installed
- [ ] CUDA and GPU drivers installed (`nvidia-smi` works)
- [ ] `uv` package manager installed
- [ ] Repository cloned
- [ ] Git submodules initialized (`lightly-train/` and `chicken-detection-labelme-format/`)
- [ ] Python virtual environment created (`uv venv`)
- [ ] Dependencies installed (`uv sync`)
- [ ] Dataset converted to YOLO format (`chicken-detection-labelme-format/yolo-format/`)
- [ ] Training script present (`train-dinov3-vitt16-ltdetr.py`)
- [ ] GPU memory available (checked with `nvidia-smi`)

## 📚 Next Steps

After installation:

1. **Free GPU Memory:** Run `uv run python free_gpu.py --kill` if needed
2. **Start Training:** Run `uv run python train-dinov3-vitt16-ltdetr.py`
3. **Monitor Training:** Check `chicken-detection-models/dinov3/vitt16-ltdetr-chicken/train.log`
4. **Load Trained Model:** Use `lightly_train.load_model()` to load the best checkpoint

For more information, see:
- `README.md` - Project overview and usage
- `AGENTS.md` - Development guidelines
- LightlyTrain documentation: https://docs.lightly.ai/train/stable/

## 📄 License

See the LightlyTrain repository for license information.
