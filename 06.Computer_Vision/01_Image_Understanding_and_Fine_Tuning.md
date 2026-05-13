# Computer Vision with Hugging Face

## Overview

Hugging Face makes it easy to apply state-of-the-art computer vision models — from image classification and object detection to background removal and custom fine-tuning. This document covers inference with pre-trained models and the complete fine-tuning workflow for custom image classification.

---

## Topics Covered

1. [Image Classification](#1-image-classification)
2. [Object Detection](#2-object-detection-with-bounding-boxes)
3. [Image Segmentation (Background Removal)](#3-image-segmentation-background-removal)
4. [Fine-Tuning for Custom Classification](#4-fine-tuning-a-model-for-custom-classification)
5. [Complete Fine-Tuning Workflow](#complete-fine-tuning-workflow)

---

## Setup

```bash
pip install transformers datasets torch torchvision matplotlib
```

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import load_dataset
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from torchvision.transforms import Compose, Normalize, ToTensor
import torch
```

---

## Part 1: Inference with Pre-Trained Models

### 1. Image Classification

Classify what's in an image using a pre-trained MobileNet model:

```python
from datasets import load_dataset
from transformers import pipeline

# Load a sample image
dataset = load_dataset("nlphuji/flickr30k")
image = dataset['test'][134]['image']   # PIL Image object

# Load classification pipeline
classifier = pipeline(
    "image-classification",
    model="google/mobilenet_v2_1.0_224"
)

# Classify the image
results = classifier(image)

print(f"Top prediction: {results[0]['label']}")
print(f"Confidence:     {results[0]['score']:.2%}")

# All predictions
for pred in results[:5]:
    print(f"  {pred['label']:40s} {pred['score']:.2%}")
```

**Output format:**
```
[
  {'label': 'LABEL_283', 'score': 0.9812},
  {'label': 'LABEL_281', 'score': 0.0091},
  ...
]
```

---

### 2. Object Detection with Bounding Boxes

Detect multiple objects in an image and visualize them:

```python
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load object detection pipeline (DETR: DEtection TRansformer)
detector = pipeline(
    "object-detection",
    model="facebook/detr-resnet-50",
    revision="no_timm"
)

# Run detection with a confidence threshold
detection_outputs = detector(image, threshold=0.95)
print(f"Detected {len(detection_outputs)} objects")

# Visualize results with bounding boxes
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple']
ax.imshow(image)

for n, obj in enumerate(detection_outputs):
    box = obj['box']  # {xmin, ymin, xmax, ymax}
    color = colors[n % len(colors)]
    
    # Draw rectangle
    rect = patches.Rectangle(
        (box['xmin'], box['ymin']),
        box['xmax'] - box['xmin'],
        box['ymax'] - box['ymin'],
        linewidth=2,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        box['xmin'], box['ymin'] - 8,
        f"{obj['label']}: {obj['score']:.2f}",
        color='white', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
    )

ax.set_title(f"Object Detection ({len(detection_outputs)} objects)")
ax.axis('off')
plt.tight_layout()
plt.show()
```

**Detection output format:**
```python
[
    {
        'label': 'cat',
        'score': 0.9987,
        'box': {'xmin': 52, 'ymin': 45, 'xmax': 345, 'ymax': 290}
    },
    ...
]
```

---

### 3. Image Segmentation (Background Removal)

```python
from transformers import pipeline

# Load background removal model
bg_remover = pipeline(
    "image-segmentation",
    model="briai/RMBG-1.4",
    trust_remote_code=True
)

# Remove background
bg_output = bg_remover(image)

# Display before/after
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')
axes[1].imshow(bg_output)
axes[1].set_title("Background Removed")
axes[1].axis('off')
plt.tight_layout()
plt.show()
```

---

## Part 2: Fine-Tuning for Custom Classification

Fine-tuning adapts a pre-trained model to classify your own custom categories.

### When to Fine-Tune

| Use Case | Approach |
|----------|----------|
| Standard categories (cats, dogs, cars) | Use pre-trained model directly |
| Custom categories (your product types) | Fine-tune a pre-trained model |
| No training data | Zero-shot with CLIP |
| Small dataset (100-1000 images) | Fine-tune with pre-trained backbone |

---

## Complete Fine-Tuning Workflow

### Step 1: Load and Split Dataset

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("ideepankarsharma2003/Midjourney_v6_Classification_small_shuffled")
dataset = dataset['train']

# Split into train and test
data_splits = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Train samples: {len(data_splits['train'])}")
print(f"Test samples:  {len(data_splits['test'])}")
```

### Step 2: Create Label Mappings

```python
# Get list of class names
labels = data_splits["train"].features["label"].names

# Create bidirectional mappings
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

print(f"Labels: {labels}")
print(f"label2id: {label2id}")
```

### Step 3: Load Pre-Trained Model with Custom Head

```python
from transformers import AutoModelForImageClassification

checkpoint = "google/mobilenet_v2_1.0_224"

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),      # Number of your custom classes
    id2label=id2label,           # Map IDs → class names
    label2id=label2id,           # Map class names → IDs
    ignore_mismatched_sizes=True # Required when changing the classification head
)

print(f"Model: {checkpoint}")
print(f"Custom classes: {len(labels)}")
```

### Step 4: Prepare Dataset with Transformations

```python
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Normalize, ToTensor

# Load image processor (handles resize and normalization config)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Build transformation pipeline using model's own normalization stats
normalize = Normalize(
    mean=image_processor.image_mean,
    std=image_processor.image_std
)
transform = Compose([ToTensor(), normalize])

def transforms_fn(examples):
    """Convert images to normalized tensors for training."""
    examples["pixel_values"] = [
        transform(img.convert("RGB"))
        for img in examples["image"]
    ]
    del examples["image"]   # Remove original images to save memory
    return examples

# Apply to both splits
data_splits = data_splits.with_transform(transforms_fn)
print("Transforms applied successfully")
```

### Step 5: Configure Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="dataset_finetune",           # Where to save checkpoints
    learning_rate=6e-5,                      # Low LR for fine-tuning
    gradient_accumulation_steps=4,           # Simulate larger batch
    num_train_epochs=3,                      # Training epochs
    per_device_train_batch_size=16,          # Batch size per GPU
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",             # Evaluate after each epoch
    save_strategy="epoch",                   # Save checkpoint each epoch
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    remove_unused_columns=False,             # Important: keep pixel_values
    load_best_model_at_end=True,             # Load best checkpoint after training
    metric_for_best_model="accuracy"
)
```

### TrainingArguments Explained

| Argument | Description | Typical Value |
|----------|-------------|---------------|
| `learning_rate` | Step size for optimizer | `1e-5` to `6e-5` |
| `num_train_epochs` | How many passes through training data | 3–10 |
| `per_device_train_batch_size` | Samples per GPU per step | 8–32 |
| `gradient_accumulation_steps` | Steps before optimizer update | 4–16 |
| `evaluation_strategy` | When to run eval (`"epoch"` or `"steps"`) | `"epoch"` |
| `load_best_model_at_end` | Restore best checkpoint at end | `True` |

### Step 6: Create Trainer and Train

```python
from transformers import Trainer, DefaultDataCollator

# Data collator handles batching
data_collator = DefaultDataCollator()

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_splits["train"],
    eval_dataset=data_splits["test"],
    processing_class=image_processor,  # Used for padding/resizing
    data_collator=data_collator,
)

# Train
trainer.train()
print("Training complete!")
```

### Step 7: Evaluate the Fine-Tuned Model

```python
import numpy as np

# Get predictions on test set
predictions = trainer.predict(data_splits["test"])

# Calculate accuracy
pred_labels = predictions.predictions.argmax(axis=1)
true_labels = predictions.label_ids
accuracy = (pred_labels == true_labels).mean()

print(f"Test Accuracy: {accuracy:.2%}")
```

---

## Fine-Tuning Architecture Diagram

```
Pre-trained MobileNetV2 (ImageNet)
│
├── Feature Extractor Layers    ← FROZEN or fine-tuned with low LR
│   (convolutional layers)
│
└── Classification Head         ← REPLACED with new head
    (original 1000 classes)     → (your N custom classes)
    
Training updates:
  - Head: Updated with full learning rate
  - Backbone: Updated with small learning rate (or frozen)
```

---

## Common Computer Vision Models

| Model | Task | Strengths |
|-------|------|-----------|
| `google/mobilenet_v2_1.0_224` | Classification | Fast, lightweight |
| `google/vit-base-patch16-224` | Classification | High accuracy |
| `facebook/detr-resnet-50` | Object detection | Fast detection |
| `facebook/detr-resnet-101` | Object detection | Higher accuracy |
| `briai/RMBG-1.4` | Segmentation / BG removal | High quality |
| `openai/clip-vit-base-patch32` | Zero-shot classification | No training needed |

---

## Practical Tips

| Tip | Details |
|-----|---------|
| Use `ignore_mismatched_sizes=True` | Required when replacing the classification head |
| Set `remove_unused_columns=False` | Prevents training from losing `pixel_values` |
| Use `gradient_accumulation_steps` | Simulate larger batches when GPU memory is limited |
| Use `load_best_model_at_end=True` | Automatically restores the best checkpoint |
| Use CPU for small experiments | Works fine, just slower |
| Use GPU (`cuda`) for real training | Dramatically faster (10-100x) |
