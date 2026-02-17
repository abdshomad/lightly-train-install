#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test DINOv3 ViTT16-LTDETR Object Detection Model

This script tests a trained lightly-train object detection model on random
validation images from the chicken detection dataset.

It loads the trained model checkpoint and runs inference on 20 random images
from the validation set, saving the results with bounding boxes to the
labs/test-results folder.
"""

import random
import warnings
from pathlib import Path
import torch
import lightly_train
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

# Suppress AccumulateGrad stream mismatch warning
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

# Suppress torchmetrics deprecation warning
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing is deprecated",
    module="torchmetrics",
    category=UserWarning,
)


def get_random_images(image_dir: Path, num_images: int = 20):
    """Get random images from a directory.
    
    Args:
        image_dir: Path to directory containing images
        num_images: Number of random images to select
        
    Returns:
        List of Path objects for selected images
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(image_dir.glob(f'*{ext}')))
    
    if len(all_images) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Select random images
    num_images = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_images)
    
    return selected_images


def main():
    # Paths
    model_path = Path("chicken-detection-models/dinov3/vitt16-ltdetr-chicken/exported_models/exported_best.pt")
    val_images_dir = Path("chicken-detection-labelme-format/yolo-format/images/val")
    output_dir = Path("labs/test-results")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using train-dinov3-vitt16-ltdetr.py")
        return
    
    # Check if validation images directory exists
    if not val_images_dir.exists():
        print(f"Error: Validation images directory not found at {val_images_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = lightly_train.load_model(str(model_path))
    print(f"Model loaded successfully!")
    print(f"Model classes: {model.classes}")
    print()
    
    # Get random images
    print(f"Selecting 20 random images from: {val_images_dir}")
    selected_images = get_random_images(val_images_dir, num_images=20)
    print(f"Selected {len(selected_images)} images:")
    for img in selected_images:
        print(f"  - {img.name}")
    print()
    
    # Run inference on each image
    print("Running inference...")
    print("=" * 60)
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}] Processing: {img_path.name}")
        
        try:
            # Run prediction
            results = model.predict(str(img_path))
            
            # Print detection summary
            num_detections = len(results["labels"]) if results["labels"] is not None else 0
            print(f"  Detections: {num_detections}")
            
            if num_detections > 0:
                # Print class and confidence for each detection
                for j in range(num_detections):
                    label = results["labels"][j].item()
                    score = results["scores"][j].item()
                    class_name = model.classes[label]
                    bbox = results["bboxes"][j]
                    print(f"    - {class_name}: {score:.3f} at {bbox.tolist()}")
            
            # Visualize and save results
            image = read_image(str(img_path))
            
            if num_detections > 0:
                # Draw bounding boxes
                labels = [f"{model.classes[label.item()]} {score.item():.2f}" 
                         for label, score in zip(results["labels"], results["scores"])]
                image_with_boxes = draw_bounding_boxes(
                    image,
                    boxes=results["bboxes"],
                    labels=labels,
                    width=2,
                )
            else:
                image_with_boxes = image
            
            # Save visualization
            output_path = output_dir / f"{img_path.stem}_result.jpg"
            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_boxes.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"{img_path.name} - {num_detections} detections", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved result to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Testing complete! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
