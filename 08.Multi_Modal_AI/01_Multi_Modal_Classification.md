# Multi-Modal Models for Classification

> A comprehensive guide to multi-modal classification using CLIP, CLAP, Vision Language Models, and fusion techniques for production-grade AI systems.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Core Concepts](#2-architecture--core-concepts)
3. [CLIP for Product Categorization](#3-clip-for-product-categorization)
4. [CLIP Score for Quality Measurement](#4-clip-score-for-quality-measurement)
5. [Vision Language Models (VLMs)](#5-vision-language-models-vlms)
6. [CLAP for Audio Analysis](#6-clap-for-audio-analysis)
7. [Multi-Modal Fusion (Video + Audio)](#7-multi-modal-fusion-video--audio)
8. [Real-World Applications](#8-real-world-applications)
9. [Best Practices & Optimization](#9-best-practices--optimization)
10. [Quick Reference](#10-quick-reference)

---

## 1. Overview

Multi-modal models process and reason across multiple data types — images, text, audio, and video — within a unified framework. By learning a **shared embedding space**, these models enable cross-modal understanding and zero-shot classification without task-specific training.

### Why Multi-Modal Classification?

| Challenge | Single-Modal Limitation | Multi-Modal Advantage |
|---|---|---|
| Product categorization | Relies on text OR image only | Combines visual + textual cues |
| Content moderation | Misses context from other modalities | Fuses signals from all modalities |
| Video understanding | Ignores audio or visual track | Analyzes both frames and audio |
| Audio classification | No visual context | Cross-references with visual data |

### Models Covered

| Model | Modality | Parameters | Key Capability |
|---|---|---|---|
| **CLIP** | Image + Text | ~150M | Zero-shot image classification |
| **CLAP** | Audio + Text | ~150M | Zero-shot audio classification |
| **Qwen2-VL** | Image + Text | 2B | Visual reasoning, VQA, entailment |

### When to Use Multi-Modal Models

- **Zero-shot classification** — no labeled training data available
- **Cross-modal search** — find images by text description (or vice versa)
- **Content quality scoring** — measure alignment between modalities
- **Visual reasoning** — answer questions about images
- **Multi-modal fusion** — combine visual + audio signals for robust predictions

---

## 2. Architecture & Core Concepts

### 2.1 CLIP Architecture

CLIP (Contrastive Language-Image Pre-training) uses a **dual-encoder** architecture trained on 400M image-text pairs from the internet. The model learns to align visual and textual representations in a shared 512-dimensional embedding space.

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIP Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐              ┌──────────────┐            │
│   │    Image      │              │     Text      │            │
│   │   (Input)     │              │   (Input)     │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          ▼                              ▼                     │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   Vision      │              │    Text       │            │
│   │  Encoder       │              │   Encoder     │            │
│   │ (ViT/CNN)      │              │ (Transformer) │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          │    d = 512                    │    d = 512         │
│          ▼                              ▼                     │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   Image       │              │    Text       │            │
│   │ Embedding     │              │  Embedding    │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          └──────────┬───────────────────┘                    │
│                     ▼                                        │
│          ┌─────────────────────┐                             │
│          │  Cosine Similarity  │                             │
│          │  (Aligned Space)    │                             │
│          └─────────────────────┘                             │
│                                                              │
│   Training: Contrastive Loss (match pairs, separate others)  │
│   Result:   Similar concepts cluster together                │
└─────────────────────────────────────────────────────────────┘
```

**Key Properties:**

- **Embedding dimension:** 512
- **Text encoder:** Transformer (max 77 tokens)
- **Vision encoder:** ViT-Base/32 (patch size 32×32)
- **Training data:** 400M image-text pairs (internet-scale)
- **Loss function:** InfoNCE contrastive loss

### 2.2 CLAP Architecture

CLAP (Contrastive Language-Audio Pre-training) applies the same dual-encoder concept to the **audio modality**, trained on 633K matched audio-description pairs.

```
┌─────────────────────────────────────────────────────────────┐
│                    CLAP Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐              ┌──────────────┐            │
│   │    Audio      │              │     Text      │            │
│   │  Waveform     │              │   (Input)     │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          ▼                              ▼                     │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   Audio       │              │    Text       │            │
│   │  Encoder      │              │   Encoder     │            │
│   │ (HTSAT)       │              │ (Transformer) │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          ▼                              ▼                     │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   Audio       │              │    Text       │            │
│   │ Embedding     │              │  Embedding    │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          └──────────┬───────────────────┘                    │
│                     ▼                                        │
│          ┌─────────────────────┐                             │
│          │  Cosine Similarity  │                             │
│          │  (Aligned Space)    │                             │
│          └─────────────────────┘                             │
│                                                              │
│   Training: 633K audio-description pairs                     │
│   Model:    laion/clap-htsat-unfused                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Vision Language Model (VLM) Architecture

VLMs go beyond embedding alignment — they perform **generative reasoning** over visual and textual inputs using a decoder.

```
┌─────────────────────────────────────────────────────────────┐
│                    VLM Architecture                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐              ┌──────────────┐            │
│   │    Image      │              │   Text/Prompt │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          ▼                              ▼                     │
│   ┌──────────────┐              ┌──────────────┐            │
│   │   Vision      │              │    Text       │            │
│   │  Encoder       │              │   Encoder     │            │
│   │   (ViT)        │              │ (Transformer) │            │
│   └──────┬───────┘              └──────┬───────┘            │
│          │                              │                     │
│          └──────────┬───────────────────┘                    │
│                     ▼                                        │
│          ┌─────────────────────┐                             │
│          │   Feature Fusion    │                             │
│          │  (Cross-Attention)  │                             │
│          └──────────┬──────────┘                             │
│                     ▼                                        │
│          ┌─────────────────────┐                             │
│          │   Language Model    │                             │
│          │     (Decoder)       │                             │
│          └──────────┬──────────┘                             │
│                     ▼                                        │
│          ┌─────────────────────┐                             │
│          │  Generated Output   │                             │
│          │  (Answer/Reasoning) │                             │
│          └─────────────────────┘                             │
│                                                              │
│   Capabilities: VQA, Image-Text Matching, Entailment,        │
│                 Multimodal Reasoning                         │
│   Example:     Qwen/Qwen2-VL-2B-Instruct (2B parameters)    │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Entailment Types in VLMs

| Type | Description | Example |
|---|---|---|
| **Entailment** | Text logically follows from image | Image shows dog → "A pet is present" |
| **Contradiction** | Text contradicts the image | Image shows dog → "A cat is sitting" |
| **Neutral** | Text neither follows nor contradicts | Image shows dog → "It might be hungry" |

---

## 3. CLIP for Product Categorization

### 3.1 Setup & Installation

```bash
pip install transformers torch pillow torchvision
```

```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
```

### 3.2 Model Loading

```python
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()
```

### 3.3 Zero-Shot Image Classification

Zero-shot classification allows categorization into **any class** without retraining — simply provide the candidate labels as text.

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Input Image  │────▶│  CLIP Dual       │────▶│  Similarity   │
│               │     │  Encoder         │     │  Scores       │
├──────────────┤     ├──────────────────┤     ├───────────────┤
│  Categories:  │────▶│  Text Encoder    │────▶│  Softmax      │
│  ["cat","dog"]│     │                  │     │  → [0.9, 0.1] │
└──────────────┘     └──────────────────┘     └───────────────┘
```

```python
image = Image.open("product.jpg")

categories = [
    "electronics",
    "clothing",
    "home and garden",
    "toys and games",
    "sports equipment",
]

inputs = processor(
    text=categories,
    images=image,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

predicted_idx = probs.argmax(dim=1).item()
predicted_category = categories[predicted_idx]
confidence = probs[0][predicted_idx].item()

print(f"Category: {predicted_category} (confidence: {confidence:.4f})")
```

### 3.4 Batch Processing Multiple Images

```python
images = [Image.open(f"img_{i}.jpg") for i in range(10)]

inputs = processor(
    text=categories,
    images=images,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

for i, image_probs in enumerate(probs):
    idx = image_probs.argmax().item()
    print(f"Image {i}: {categories[idx]} ({image_probs[idx]:.4f})")
```

### 3.5 Prompt Engineering for Better Results

```python
prompt_templates = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a bright photo of a {}",
    "a cropped photo of a {}",
]

def build_prompts(categories, templates):
    return [t.format(c) for c in categories for t in templates]

expanded_prompts = build_prompts(categories, prompt_templates)
```

> **Tip:** Using multiple prompt templates and averaging scores over them improves accuracy by 2–5% on most benchmarks.

---

## 4. CLIP Score for Quality Measurement

### 4.1 What is CLIP Score?

CLIP Score measures the **alignment** between an image and a text description. It quantifies how well a caption matches an image on a scale of **0–100**.

```
┌──────────────┐                     ┌──────────────┐
│    Image      │───▶ Embedding ────▶ │              │
│               │     (512-d)         │   Cosine     │───▶ CLIP Score
├──────────────┤                     │  Similarity  │     (0–100)
│    Text       │───▶ Embedding ────▶ │              │
│  (caption)    │     (512-d)         │              │
└──────────────┘                     └──────────────┘
```

### 4.2 Use Cases

| Use Case | Description |
|---|---|
| **Caption quality evaluation** | Score how well generated captions describe images |
| **Data filtering** | Remove mislabeled or low-quality image-text pairs |
| **Ranking** | Rank image-text pairs by relevance |
| **Dataset curation** | Build high-quality training datasets |

### 4.3 Implementation

```bash
pip install torchmetrics
```

```python
import torch
from torchmetrics.functional.multimodal import clip_score
from PIL import Image
from torchvision.transforms.functional import to_tensor

image = Image.open("product.jpg")
image_tensor = to_tensor(image).to(torch.uint8)

description = "A red sports car parked on a city street"

score = clip_score(
    image_tensor.unsqueeze(0),
    [description],
    model_name="openai/clip-vit-base-patch32"
)

print(f"CLIP Score: {score.item():.2f}")
```

### 4.4 Batch Scoring

```python
image_tensors = []
descriptions = []

for img_path, desc in dataset_pairs:
    img = Image.open(img_path)
    image_tensors.append(to_tensor(img).to(torch.uint8))
    descriptions.append(desc)

image_batch = torch.stack(image_tensors)

scores = clip_score(
    image_batch,
    descriptions,
    model_name="openai/clip-vit-base-patch32"
)

threshold = 25.0
low_quality = [(i, s.item()) for i, s in enumerate(scores) if s < threshold]
print(f"Low quality pairs: {len(low_quality)} / {len(descriptions)}")
```

### 4.5 Score Interpretation

| Score Range | Quality | Action |
|---|---|---|
| **0–20** | Poor alignment | Discard or re-label |
| **20–35** | Weak alignment | Review manually |
| **35–50** | Moderate alignment | Acceptable for training |
| **50–70** | Good alignment | High-quality pair |
| **70–100** | Excellent alignment | Gold standard |

---

## 5. Vision Language Models (VLMs)

### 5.1 Overview

Vision Language Models extend beyond embedding alignment to **generative reasoning** — they can answer questions, perform entailment, and reason about visual content.

### 5.2 Capabilities

| Capability | Description | Example |
|---|---|---|
| **VQA** | Visual Question Answering | "How many products are shown?" |
| **Image-Text Matching** | Determine if text describes image | "Does this image show a kitchen?" |
| **Entailment** | Logical relationship between image and text | Entailment / Contradiction / Neutral |
| **Multimodal Reasoning** | Complex reasoning across modalities | "Is the product safely packaged?" |

### 5.3 Setup with Qwen2-VL

```bash
pip install transformers torch pillow accelerate
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
```

### 5.4 Visual Question Answering (VQA)

```python
image = Image.open("product.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What product is shown in this image? Describe its key features."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256)

response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
```

### 5.5 Visual Entailment Classification

```python
image = Image.open("scene.jpg")

entailment_prompt = """Given the image, classify the relationship between the image and the following statement.

Statement: "A person is cooking in a modern kitchen."

Choose one:
- Entailment: The statement is supported by the image
- Contradiction: The statement contradicts the image
- Neutral: The statement cannot be confirmed or refuted by the image

Provide your answer with reasoning."""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": entailment_prompt}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(result)
```

### 5.6 VLM vs CLIP Comparison

| Feature | CLIP | VLM (Qwen2-VL) |
|---|---|---|
| **Architecture** | Dual encoder | Encoder + Decoder |
| **Output** | Similarity scores | Generated text |
| **Classification** | Zero-shot via text prompts | Reasoning + classification |
| **Reasoning** | None | Chain-of-thought |
| **Speed** | Fast (embedding only) | Slower (autoregressive) |
| **Use case** | Retrieval, categorization | VQA, reasoning, entailment |

---

## 6. CLAP for Audio Analysis

### 6.1 Overview

CLAP (Contrastive Language-Audio Pre-training) applies the contrastive learning paradigm to **audio-text alignment**, enabling zero-shot audio classification using natural language descriptions.

### 6.2 Setup

```bash
pip install transformers torch librosa datasets
```

```python
import torch
import librosa
from transformers import ClapModel, ClapProcessor

model_name = "laion/clap-htsat-unfused"

model = ClapModel.from_pretrained(model_name)
processor = ClapProcessor.from_pretrained(model_name)

model.eval()
```

### 6.3 Zero-Shot Audio Emotion Recognition

```python
audio_path = "speech_sample.wav"
audio, sr = librosa.load(audio_path, sr=48000)

emotion_categories = [
    "joyful and happy",
    "fearful and scared",
    "angry and aggressive",
    "sad and melancholic",
    "disgusted",
    "surprised",
    "calm and neutral"
]

inputs = processor(
    text=emotion_categories,
    audios=audio,
    sampling_rate=sr,
    return_tensors="pt",
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits_per_audio.softmax(dim=-1)

predicted_idx = probs.argmax(dim=-1).item()
predicted_emotion = emotion_categories[predicted_idx]
confidence = probs[0][predicted_idx].item()

print(f"Emotion: {predicted_emotion} (confidence: {confidence:.4f})")
```

### 6.4 Audio-Text Similarity Flow

```
┌──────────────┐                      ┌──────────────┐
│  Audio File   │───▶ Audio Encoder ──▶│              │
│  (.wav)       │     (HTSAT)         │   Cosine     │───▶ Similarity
├──────────────┤                      │  Similarity  │     Scores
│  Text Labels  │───▶ Text Encoder ──▶│              │     (per label)
│  (emotions)   │     (Transformer)   │              │
└──────────────┘                      └──────────────┘
```

### 6.5 Audio Classification Categories

| Domain | Example Categories |
|---|---|
| **Emotion** | joy, fear, anger, sadness, disgust, surprise, neutral |
| **Music Genre** | rock, jazz, classical, electronic, hip-hop |
| **Environment** | indoor, outdoor, urban, nature, transportation |
| **Speech Events** | speech, music, silence, noise, applause |

---

## 7. Multi-Modal Fusion (Video + Audio)

### 7.1 Business Case: Competitor Ad Sentiment Analysis

Combine **visual cues** from video frames and **audio cues** from the soundtrack to classify competitor advertisements by sentiment.

### 7.2 Multi-Modal Fusion Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│               Multi-Modal Fusion Pipeline                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌────────────┐                                              │
│   │  Video File │                                              │
│   └─────┬──────┘                                              │
│         │                                                     │
│         ├──────────────┬────────────────┐                     │
│         ▼              ▼                │                     │
│   ┌──────────┐  ┌────────────┐          │                     │
│   │ Extract   │  │ Extract    │          │                     │
│   │  Frames   │  │  Audio     │          │                     │
│   │ (1 fps)   │  │ Track      │          │                     │
│   └─────┬────┘  └─────┬──────┘          │                     │
│         │              │                  │                     │
│         ▼              ▼                  │                     │
│   ┌──────────┐  ┌────────────┐          │                     │
│   │   CLIP   │  │    CLAP    │          │                     │
│   │  Visual   │  │   Audio    │          │                     │
│   │ Classifier│  │ Classifier │          │                     │
│   └─────┬────┘  └─────┬──────┘          │                     │
│         │              │                  │                     │
│         │    Scores     │    Scores       │                     │
│         │   (visual)    │   (audio)       │                     │
│         │              │                  │                     │
│         └──────┬───────┘                  │                     │
│                ▼                           │                     │
│   ┌──────────────────────┐                │                     │
│   │   Multi-Modal Fusion │                │                     │
│   │                      │                │                     │
│   │  final = α*visual +  │                │                     │
│   │         β*audio      │                │                     │
│   │  (α + β = 1)         │                │                     │
│   └──────────┬───────────┘                │                     │
│              ▼                              │                     │
│   ┌──────────────────────┐                │                     │
│   │  Final Classification│                │                     │
│   │  (Sentiment/Emotion) │                │                     │
│   └──────────────────────┘                │                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.3 Implementation

```python
import torch
import librosa
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor

emotion_categories = [
    "joy", "fear", "anger",
    "sadness", "disgust", "surprise", "neutral"
]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

clip_model.eval()
clap_model.eval()


def extract_frames(video_path, fps=1):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
    cap.release()
    return frames


def classify_visual(frames, categories):
    inputs = clip_processor(
        text=categories, images=frames,
        return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    return probs.mean(dim=0)


def classify_audio(audio_path, categories):
    audio, sr = librosa.load(audio_path, sr=48000)
    inputs = clap_processor(
        text=categories, audios=audio,
        sampling_rate=sr, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = clap_model(**inputs)
        probs = outputs.logits_per_audio.softmax(dim=-1)
    return probs.squeeze(0)


def multimodal_fusion(video_path, audio_path, categories, alpha=0.6, beta=0.4):
    frames = extract_frames(video_path, fps=1)
    visual_scores = classify_visual(frames, categories)
    audio_scores = classify_audio(audio_path, categories)

    fused_scores = alpha * visual_scores + beta * audio_scores

    predicted_idx = fused_scores.argmax().item()
    return categories[predicted_idx], fused_scores[predicted_idx].item()


emotion, confidence = multimodal_fusion(
    "competitor_ad.mp4",
    "competitor_ad.wav",
    emotion_categories,
    alpha=0.6,
    beta=0.4
)

print(f"Predicted Emotion: {emotion} (confidence: {confidence:.4f})")
```

### 7.4 Fusion Strategy Comparison

| Strategy | Formula | Best For |
|---|---|---|
| **Weighted Average** | `α*visual + β*audio` | General-purpose, tunable |
| **Max Confidence** | `max(visual_conf, audio_conf)` | When one modality dominates |
| **Product Rule** | `visual * audio` (normalized) | When both modalities must agree |
| **Learned Fusion** | Small classifier on concatenated scores | When you have training data |

> **Tip:** Start with a 60/40 (visual/audio) split and tune based on validation performance. Visual cues tend to carry more signal for ad sentiment.

---

## 8. Real-World Applications

| Company | Application | Model | Impact |
|---|---|---|---|
| **Amazon** | E-commerce Product Classification | CLIP | 80% reduction in manual categorization |
| **Meta** | Social Media Content Moderation | CLIP + VLM | 50% faster harmful content detection |
| **Google** | Ad Performance Analysis | Multi-modal Fusion | 30% improvement in ROI prediction |
| **Siemens** | Medical Image Analysis | VLM (Qwen2-VL) | 40% reduction in report turnaround |
| **Netflix** | Automated Video Subtitling | CLIP + CLAP | 25% improvement in subtitle quality |

### Application Decision Matrix

| Your Need | Recommended Model | Why |
|---|---|---|
| Categorize products with no training data | CLIP | Zero-shot, fast, accurate |
| Evaluate caption quality at scale | CLIP Score | Automated, no human review |
| Answer questions about images | VLM (Qwen2-VL) | Generative reasoning |
| Classify audio by emotion/genre | CLAP | Zero-shot audio classification |
| Analyze video content holistically | CLIP + CLAP Fusion | Multi-modal coverage |

---

## 9. Best Practices & Optimization

### 9.1 Performance Optimization

| Technique | Speedup | Trade-off |
|---|---|---|
| **FP16 inference** | ~2× faster | Minor precision loss |
| **INT8 quantization** | ~4× faster (CPU) | Slight accuracy drop |
| **Batch processing** | ~3× throughput | Higher memory usage |
| **Embedding caching** | ~10× for repeated queries | Storage overhead |
| **ONNX Runtime** | ~2× faster inference | Export complexity |

```python
# FP16 inference
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16)
model = model.half().cuda()

# INT8 quantization (for CPU deployment)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", quantization_config=quantization_config)

# Cache embeddings for frequent queries
import hashlib, pickle

embedding_cache = {}

def get_cached_embedding(text, model, processor):
    key = hashlib.md5(text.encode()).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    
    embedding_cache[key] = embedding
    return embedding
```

### 9.2 Key Constraints

| Constraint | Limit | Workaround |
|---|---|---|
| CLIP max text tokens | 77 tokens | Truncate or summarize text |
| CLIP image resolution | 224×224 (default) | Processor handles resizing |
| Domain shift | Lower accuracy on niche domains | Fine-tune on domain data |
| Batch memory | Scales with batch size | Use gradient checkpointing |

### 9.3 Production Checklist

- [ ] Use FP16 or INT8 for inference
- [ ] Batch requests for throughput
- [ ] Cache embeddings for repeated queries
- [ ] Set appropriate confidence thresholds
- [ ] Handle out-of-distribution inputs gracefully
- [ ] Monitor embedding drift over time
- [ ] Fine-tune on domain-specific data if accuracy is insufficient

---

## 10. Quick Reference

### Model Summary

| Model | Hugging Face ID | Modality | Key Use Case |
|---|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | Image + Text | Zero-shot image classification |
| CLAP | `laion/clap-htsat-unfused` | Audio + Text | Zero-shot audio classification |
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | Image + Text | Visual reasoning, VQA |

### One-Liner Commands

```python
# CLIP zero-shot classification
probs = model(**processor(text=labels, images=img, return_tensors="pt", padding=True)).logits_per_image.softmax(dim=1)

# CLIP Score
score = clip_score(image_tensor.unsqueeze(0), [caption], model_name="openai/clip-vit-base-patch32")

# CLAP zero-shot audio classification
probs = model(**processor(text=labels, audios=audio, sampling_rate=48000, return_tensors="pt", padding=True)).logits_per_audio.softmax(dim=-1)
```

### Common Pitfalls

| Pitfall | Solution |
|---|---|
| Text exceeds 77 tokens | Truncate or use shorter descriptions |
| Poor zero-shot accuracy | Use descriptive, specific category names |
| Slow inference on CPU | Use INT8 quantization |
| Domain mismatch | Fine-tune on domain-specific data |
| Imbalanced fusion weights | Tune α/β on a validation set |

### Workflow Decision Tree

```
START
  │
  ├── Need to classify IMAGES?
  │     ├── Have training data? ──▶ Fine-tune CLIP or use CNN
  │     └── No training data? ────▶ Zero-shot CLIP
  │
  ├── Need to classify AUDIO?
  │     ├── Have training data? ──▶ Fine-tune audio model
  │     └── No training data? ────▶ Zero-shot CLAP
  │
  ├── Need VISUAL REASONING?
  │     └── ──────────────────────▶ VLM (Qwen2-VL)
  │
  ├── Need IMAGE-TEXT QUALITY?
  │     └── ──────────────────────▶ CLIP Score
  │
  └── Need VIDEO UNDERSTANDING?
        └── ──────────────────────▶ CLIP + CLAP Fusion
```

---

> **Last Updated:** May 2026 | **Models:** CLIP, CLAP, Qwen2-VL | **Framework:** Hugging Face Transformers
