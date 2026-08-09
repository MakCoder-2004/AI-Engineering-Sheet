# Multi-Modal Models for Generation

> A comprehensive guide to multi-modal generation systems covering visual QA, document understanding, image/video synthesis, and quality evaluation — all powered by Hugging Face `transformers` and `diffusers`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Visual Question Answering with ViLT](#2-visual-question-answering-with-vilt)
3. [Document VQA with LayoutLM](#3-document-vqa-with-layoutlm)
4. [Image Generation with Stable Diffusion](#4-image-generation-with-stable-diffusion)
5. [Image Editing with ControlNet](#5-image-editing-with-controlnet)
6. [Image Inpainting](#6-image-inpainting)
7. [Video Generation with CogVideoX](#7-video-generation-with-cogvideox)
8. [Quality Evaluation with CLIP Score](#8-quality-evaluation-with-clip-score)
9. [Real-World Applications](#9-real-world-applications)
10. [Optimization & Best Practices](#10-optimization--best-practices)
11. [Quick Reference](#11-quick-reference)

---

## 1. Overview

Multi-modal generation models bridge multiple data types — text, images, video, and document layouts — to perform tasks that require understanding and producing content across modalities.

### Capability Matrix

| Model | Input | Output | Domain | Params | VRAM |
|---|---|---|---|---|---|
| **ViLT** (VQA) | Image + Text | Text (answer) | Natural images | ~110M | ~4 GB |
| **LayoutLM** (Doc QA) | Document + Text | Text (answer) | Scanned documents | ~340M | ~6 GB |
| **Stable Diffusion** | Text prompt | Image (512×512) | General generation | ~860M UNet + 123M CLIP | ~8 GB (fp16) |
| **ControlNet** | Text + Conditioning image | Image | Structured editing | +360M per conditioner | ~10 GB |
| **CogVideoX** | Text prompt | Video (480×720) | Video generation | 2B | ~10–12 GB |

### Architecture Family Tree

```
Multi-Modal Generation
├── Understanding (Vision + Language → Text)
│   ├── ViLT ─────── Natural Image VQA
│   └── LayoutLM ─── Document VQA (layout-aware)
│
├── Image Synthesis (Text → Image)
│   ├── Stable Diffusion ─── Unconditional / Text-conditioned
│   ├── ControlNet ───────── Structure-conditioned editing
│   └── Inpainting ───────── Mask-conditioned regeneration
│
├── Video Synthesis (Text → Video)
│   └── CogVideoX ── Cascade diffusion (3-stage)
│
└── Evaluation
    └── CLIP Score ── Frame-text alignment metric
```

### Multi-Modal Pipeline Flow

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  User Input   │────▶│  Multi-Modal     │────▶│   Output     │
│  (Text/Image) │     │  Model Pipeline  │     │  (Text/Img/) │
└──────────────┘     └──────────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     │  Pre-trained │
                     │  Weights     │
                     │  (HuggingFace│
                     │   Hub)       │
                     └─────────────┘
```

---

## 2. Visual Question Answering with ViLT

### Architecture

**ViLT** (Vision-and-Language Transformer) unifies vision and language processing into a single transformer — eliminating the need for a separate object detector, making it significantly faster than earlier VQA models.

```
┌────────┐    ┌──────────┐
│ Image  │───▶│ Patch    │──┐
│        │    │ Embedding │  │    ┌───────────────────┐    ┌──────────┐
└────────┘    └──────────┘  ├──▶│   Unified          │───▶│ Answer   │
                            │   │   Transformer       │    │ Head     │
┌────────┐    ┌──────────┐  │   │   (12 layers)      │    │ (3,129   │
│Question │──▶│  Text    │──┘   │                     │    │  classes)│
│ (Text) │    │ Embedding│      └───────────────────┘    └──────────┘
└────────┘    └──────────┘               │
                                   ┌─────┴─────┐
                                   │  Joint    │
                                   │  Attention│
                                   │  (image ↔ │
                                   │   text)   │
                                   └───────────┘
```

**Key design decisions:**

| Feature | Detail |
|---|---|
| Vision encoder | Linear projection of image patches (no Region Proposal Network) |
| Text encoder | WordPiece tokenization + learned embeddings |
| Fusion | Early fusion — both modalities processed together from layer 1 |
| Training data | VQA v2.0 — 265k images, 1.1M questions |
| Output vocabulary | 3,129 possible answers |
| Speed advantage | No object detector → ~4× faster than detector-based models |

### Code Example

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests

image = Image.open(requests.get("https://example.com/photo.jpg", stream=True).raw)
question = "What animal is in the picture?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

encoding = processor(image, question, return_tensors="pt")
outputs = model(**encoding)

logits = outputs.logits
predicted_answer = model.config.id2label[logits.argmax(-1).item()]
print(f"Answer: {predicted_answer}")
```

### Getting Top-K Answers

```python
import torch

top_k = 5
probs = torch.softmax(logits, dim=-1)
top_probs, top_indices = probs.topk(top_k, dim=-1)

for i in range(top_k):
    answer = model.config.id2label[top_indices[0][i].item()]
    confidence = top_probs[0][i].item()
    print(f"  {answer}: {confidence:.2%}")
```

### Applications

| Use Case | Description |
|---|---|
| Image accessibility | Auto-describe image content for visually impaired users |
| E-commerce | Answer customer questions about product images |
| Education | Interactive visual learning and quiz generation |
| Medical imaging | Preliminary image-based diagnostic Q&A |

---

## 3. Document VQA with LayoutLM

### Architecture

**LayoutLM** extends the standard transformer by incorporating **layout embeddings** (bounding box coordinates), enabling the model to understand the spatial structure of documents.

```
┌─────────────────────┐
│   Scanned Document   │
│  ┌─────────────────┐ │
│  │ Invoice #12345  │ │     ┌────────────────────────────────────┐
│  │ Date: 2025-01   │ │     │          LayoutLM Encoder          │
│  │ Amount: $4,200  │ │     │                                    │
│  │   ┌───────┐     │ │     │  Text Embedding ─┐                │
│  │   │ Table │     │ │────▶│  Layout Embedding ┼─▶ Transformer │
│  │   │ Data  │     │ │     │  Image Embedding ─┘    (12 layers) │
│  │   └───────┘     │ │     │                                    │
│  └─────────────────┘ │     └──────────────┬─────────────────────┘
└─────────────────────┘                     │
                                            ▼
                                   ┌──────────────┐
                                   │  Answer Span  │
                                   │  Extraction   │
                                   └──────────────┘

Embedding Breakdown:
┌────────────────┬─────────────────────────────────────────┐
│ Token Embed    │ WordPiece tokens from OCR text           │
│ Layout Embed   │ (x0, y0, x1, y1) bounding boxes per token│
│ Segment Embed  │ Document region type (title, body, etc.) │
└────────────────┴─────────────────────────────────────────┘
```

### Code Example

```python
from transformers import pipeline

doc_qa = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa"
)

result = doc_qa(
    document_image,
    "What was the gross income?"
)

print(result)
# [{'answer': '$42,000', 'score': 0.97, 'start': 156, 'end': 163}]
```

### Supported Document Types

| Document Type | Example Use Cases |
|---|---|
| Invoices | Extract vendor, amount, due date, line items |
| Financial reports | Revenue, profit margins, quarterly figures |
| Tax forms | Extract filing status, deductions, totals |
| Contracts | Key dates, parties, clauses |
| Medical records | Patient info, diagnoses, prescriptions |
| IDs / Passports | Name, DOB, expiry, document number |

### Comparison: ViLT vs LayoutLM

| Feature | ViLT | LayoutLM |
|---|---|---|
| Input domain | Natural images (photos) | Scanned documents, forms |
| Spatial awareness | Patch-based (grid) | Bounding box coordinates |
| OCR dependency | None | External OCR required |
| Best for | Scene understanding | Document information extraction |
| Model | `dandelin/vilt-b32-finetuned-vqa` | `impira/layoutlm-document-qa` |

---

## 4. Image Generation with Stable Diffusion

### Diffusion Process

Stable Diffusion is a **latent diffusion model** that generates images by learning to reverse a noise-adding process, conditioned on text prompts via CLIP.

```
Forward Process (Training)                Reverse Process (Generation)
========================                  ============================

  Clean Image                              Random Noise
  ┌───────┐                                ┌───────┐
  │       │  t=0                            │ ░░░░░ │  t=T
  │ ████  │                                │░░░░░░░│
  │       │  ──Add Noise──▶  ...           │░░░░░░░│
  └───────┘                                └───────┘
                                               │
  Noisy Image t=1         ...                   │  ──Denoise (UNet)──▶
  ┌───────┐                                    │
  │ █ ░ ░ │                                    ▼
  │░ █ ░░ │  ──Add Noise──▶  ...           Denoised t=T-1
  └───────┘                                ┌───────┐
                                           │ ▓▓░░▓ │
  Pure Noise t=T                           │░▓▓▓░░ │
  ┌───────┐                                └───────┘
  │░░░░░░░│                                     │
  │░░░░░░░│                                ...continues...
  └───────┘                                     │
                                                ▼
                                           Generated Image
                                           ┌───────┐
                                           │ █████ │
                                           │ █████ │
                                           └───────┘
```

### Stable Diffusion Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Stable Diffusion Pipeline               │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │  Text Prompt │───▶│ CLIP Text    │───▶│  UNet      │  │
│  │  "A lion..." │    │ Encoder      │    │  (860M)    │  │
│  └─────────────┘    │ (123M)       │    │            │  │
│                     └──────────────┘    │  Denoise   │  │
│  ┌─────────────┐                       │  x N steps │  │
│  │ Random      │──────────────────────▶│            │  │
│  │ Latent      │                       └─────┬──────┘  │
│  └─────────────┘                             │         │
│                                              ▼         │
│                                        ┌────────────┐  │
│                                        │  VAE       │  │
│                                        │  Decoder   │  │
│                                        │            │  │
│                                        │ Latent →   │  │
│                                        │ Pixel      │  │
│                                        └─────┬──────┘  │
│                                              │         │
└──────────────────────────────────────────────┼─────────┘
                                               ▼
                                        ┌────────────┐
                                        │  Output    │
                                        │  Image     │
                                        │  512×512   │
                                        └────────────┘
```

### Key Parameters

| Parameter | Typical Value | Effect |
|---|---|---|
| `num_inference_steps` | 20–50 | More steps → higher quality, slower |
| `guidance_scale` | 7.5 | Higher → more prompt adherence, less diversity |
| `torch_dtype` | `torch.float16` | Halves VRAM usage with minimal quality loss |
| Scheduler | PNDM / Euler A | Controls the denoising schedule |
| Seed | Integer | Reproducibility — same seed + prompt = same image |

### Code Example

```python
import torch
from diffusers import StableDiffusionPipeline

sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = sd_pipe(
    "A majestic lion in a sunlit African savanna",
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

image.save("lion_savanna.png")
```

### Generating with Seed (Reproducibility)

```python
from diffusers import EulerAncestralDiscreteScheduler
import torch

generator = torch.Generator("cuda").manual_seed(42)

sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    sd_pipe.scheduler.config
)

image = sd_pipe(
    "A futuristic city at sunset, cyberpunk style",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator
).images[0]
```

### Model Specifications

| Spec | Value |
|---|---|
| UNet parameters | 860M |
| CLIP text encoder | 123M |
| Training data | LAION-5B subset |
| Output resolution | 512 × 512 |
| VRAM (fp16) | ~8 GB |
| Inference time (20 steps) | ~3 sec on RTX 3090 |

---

## 5. Image Editing with ControlNet

### Architecture

**ControlNet** adds spatial conditioning to a pre-trained diffusion model by cloning the UNet encoder weights and injecting conditioning signals through **zero convolutions**, preserving the base model's capabilities.

```
                        ControlNet Architecture
 ┌───────────────────────────────────────────────────────────────┐
 │                                                               │
 │  ┌──────────┐                                                 │
 │  │ Conditioning│    ┌─────────────────┐                       │
 │  │ Image      │───▶│ ControlNet      │                       │
 │  │ (Canny/    │    │ (trainable copy  │                       │
 │  │  Depth/    │    │  of UNet encoder)│                       │
 │  │  Pose)     │    └────────┬────────┘                       │
 │  └──────────┘              │                                  │
 │                    Zero Convolutions                           │
 │                    (initialized to 0)                          │
 │                             │                                  │
 │                             ▼                                  │
 │  ┌──────────┐     ┌─────────────────┐     ┌────────────┐     │
 │  │  Text     │────▶│  Original UNet  │────▶│  VAE       │     │
 │  │  Prompt   │     │  (frozen)       │     │  Decoder   │     │
 │  └──────────┘     └─────────────────┘     └────────────┘     │
 │                                                    │          │
 └────────────────────────────────────────────────────┼──────────┘
                                                      ▼
                                               ┌────────────┐
                                               │  Edited    │
                                               │  Image     │
                                               └────────────┘
```

### Conditioning Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Source Image │───▶│ Canny Edge  │───▶│  ControlNet +    │───▶│  New Image  │
│             │    │ Detector    │    │  Stable Diffusion │    │  (preserves │
└─────────────┘    └─────────────┘    │  + Text Prompt    │    │  structure) │
                                      └──────────────────┘    └─────────────┘
  ┌─────────────┐          ┌──────────────┐
  │ Original    │          │ Edges /      │
  │ Photo       │          │ Structure    │     Text: "Albert Einstein"
  │             │          │ Map          │         │
  └─────────────┘          └──────────────┘         ▼
                                              ┌─────────────┐
                                              │ Controlled  │
                                              │ Generation  │
                                              └─────────────┘
```

### Supported Conditioning Types

| Conditioner | Input | Preserves |
|---|---|---|
| Canny Edge | Binary edge map | Object outlines and contours |
| Depth Map | Monocular depth estimation | 3D spatial structure |
| Pose (OpenPose) | Keypoint skeleton | Human body pose |
| Segmentation | Semantic class labels | Region boundaries |
| Normal Map | Surface normals | Surface geometry |
| Lineart | Extracted line drawing | Sketch structure |

### Code Example (Canny Edge)

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

image = np.array(source_image)
canny_image = cv2.Canny(image, 100, 200)
canny_image = Image.fromarray(canny_image)

output = pipe(
    prompt="Albert Einstein",
    image=canny_image,
    num_inference_steps=20,
    guidance_scale=7.5
)

output.images[0].save("einstein_controlled.png")
```

---

## 6. Image Inpainting

### Concept

Inpainting regenerates **masked regions** of an image while preserving the rest. White pixels in the mask indicate areas to regenerate; black pixels indicate areas to keep.

```
┌───────────────────────┐     ┌───────────────────────┐
│      Original Image    │     │       Mask             │
│                       │     │  ┌──────────┐          │
│     ┌──────────┐      │     │  │ ████████ │  White =  │
│     │  Object  │      │     │  │ ████████ │  Regenerate│
│     │  to      │      │  +  │  │ ████████ │          │
│     │  Remove  │      │     │  └──────────┘          │
│     └──────────┘      │     │  Black = Keep Original │
│                       │     │                       │
└───────────────────────┘     └───────────────────────┘
              │                          │
              └──────────┬───────────────┘
                         ▼
              ┌───────────────────────┐
              │  Inpainting Pipeline   │
              │  (mask = -1.0 for     │
              │   regeneration zones) │
              └──────────┬────────────┘
                         ▼
              ┌───────────────────────┐
              │      Result Image     │
              │                       │
              │     ┌──────────┐      │
              │     │ Filled / │      │
              │     │ Replaced │      │
              │     │ Content  │      │
              │     └──────────┘      │
              │                       │
              └───────────────────────┘
```

### Use Cases

| Use Case | Description | Example |
|---|---|---|
| Object removal | Remove unwanted elements from photos | Erase power lines, photobombers |
| Object replacement | Replace one object with another | Swap a dog for a cat in a scene |
| Defect correction | Fix damaged or corrupted regions | Restore old photographs |
| Creative editing | Add new elements to existing images | Insert a sunset into a daytime photo |

### Code Example

```python
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("original.png").resize((512, 512))
mask_image = Image.open("mask.png").resize((512, 512))

result = pipe(
    prompt="A serene mountain landscape",
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=20
).images[0]

result.save("inpainted.png")
```

### Mask Preparation

```python
import numpy as np
from PIL import Image, ImageDraw

mask = Image.new("L", (512, 512), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([150, 100, 350, 300], fill=255)

init_array = np.array(init_image).astype(np.float32) / 255.0
mask_array = np.array(mask).astype(np.float32) / 255.0
init_array[mask_array > 0.5] = -1.0
```

---

## 7. Video Generation with CogVideoX

### Cascade Diffusion Architecture

CogVideoX uses a **three-stage cascade** to generate high-quality video from text prompts.

```
┌─────────────────────────────────────────────────────────────────┐
│                   CogVideoX Cascade Pipeline                     │
│                                                                 │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │  Text     │────▶│  Stage 1:    │────▶│  Low-res frames  │    │
│  │  Prompt   │     │  Base UNet   │     │  (coarse)        │    │
│  └──────────┘     └──────────────┘     └────────┬─────────┘    │
│                                                   │              │
│                                                   ▼              │
│                   ┌──────────────┐     ┌──────────────────┐    │
│                   │  Stage 2:    │────▶│  Smoothed frames │    │
│                   │  Interpolation│    │  (temporal fill) │    │
│                   │  UNet        │     └────────┬─────────┘    │
│                   └──────────────┘              │              │
│                                                 ▼              │
│                   ┌──────────────┐     ┌──────────────────┐    │
│                   │  Stage 3:    │────▶│  Final frames    │    │
│                   │  Super-Res   │     │  480×720, up to  │    │
│                   │  UNet        │     │  49 frames       │    │
│                   └──────────────┘     └──────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Temporal Processing:
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│ F 1  │─▶│ F 2  │─▶│ F 3  │─▶│ ...  │─▶│ F 49 │
└──────┘  └──────┘  └──────┘  └──────┘  └──────┘
   │         │         │                    │
   └─────3D Convolutions (temporal dimension)┘
```

### Model Specifications

| Spec | Value |
|---|---|
| Resolution | 480 × 720 pixels |
| Max frames | 49 frames |
| Duration | ~2–3 seconds at 8 fps |
| Parameters | 2B |
| VRAM requirement | ~10–12 GB |
| Temporal processing | 3D convolutions across frame dimension |
| Model ID | `THUDM/CogVideoX-2b` |

### Memory Optimization Techniques

```
┌───────────────────────────────────────────────────────┐
│              Memory Optimization Strategy              │
│                                                       │
│  ┌─────────────────┐     ┌─────────────────────────┐ │
│  │ CPU Offloading   │     │ Move inactive model      │ │
│  │ enable_model_   │     │ components to CPU RAM    │ │
│  │ cpu_offload()   │     │ sequentially             │ │
│  └─────────────────┘     └─────────────────────────┘ │
│                                                       │
│  ┌─────────────────┐     ┌─────────────────────────┐ │
│  │ VAE Slicing     │     │ Process latent slices    │ │
│  │ vae.enable_     │     │ one at a time instead    │ │
│  │ slicing()       │     │ of full batch            │ │
│  └─────────────────┘     └─────────────────────────┘ │
│                                                       │
│  ┌─────────────────┐     ┌─────────────────────────┐ │
│  │ VAE Tiling      │     │ Decode latent space in   │ │
│  │ vae.enable_     │     │ spatial tiles for very   │ │
│  │ tiling()        │     │ high resolutions         │ │
│  └─────────────────┘     └─────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### Code Example

```python
import torch
from diffusers import CogVideoXPipeline
from export_utils import export_to_video

video_pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
)

video_pipe.enable_model_cpu_offload()
video_pipe.vae.enable_slicing()

video_frames = video_pipe(
    prompt="A golden retriever playing fetch in a park on a sunny afternoon",
    num_inference_steps=20,
    num_frames=20,
    guidance_scale=6
).frames[0]

export_to_video(video_frames, "output_video.mp4", fps=8)
```

### Adjusting Generation Parameters

```python
video_frames = video_pipe(
    prompt="Ocean waves crashing on a rocky shore at sunset",

    num_inference_steps=30,
    num_frames=20,
    guidance_scale=6,

    generator=torch.Generator("cpu").manual_seed(42)
).frames[0]
```

| Parameter | Range | Effect |
|---|---|---|
| `num_inference_steps` | 10–50 | Higher → better quality, slower |
| `num_frames` | 1–49 | More frames → longer video, more VRAM |
| `guidance_scale` | 1–15 | Higher → more prompt adherence |

---

## 8. Quality Evaluation with CLIP Score

### Concept

CLIP Score measures the alignment between generated visual content (image/video frames) and the text prompt used to create it, leveraging OpenAI's CLIP model.

```
┌──────────────────────────────────────────────────────┐
│                  CLIP Score Pipeline                   │
│                                                       │
│  ┌──────────┐     ┌──────────┐                       │
│  │  Text     │────▶│  CLIP    │───▶ Text Embedding   │
│  │  Prompt   │     │  Text    │     (512-d)    │     │
│  └──────────┘     │  Encoder │                │     │
│                    └──────────┘                │     │
│                                                 ▼     │
│                                          ┌─────────┐ │
│                                          │ Cosine  │ │
│                                          │ Similar │ │
│                                          │ = Score │ │
│                                          └────┬────┘ │
│                                                │      │
│  ┌──────────┐     ┌──────────┐                │      │
│  │ Generated │────▶│  CLIP    │───▶ Image  ───┘      │
│  │  Frame    │     │  Visual  │     Embedding        │
│  └──────────┘     │  Encoder │     (512-d)          │
│                    └──────────┘                      │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Score Interpretation

| Score Range | Interpretation |
|---|---|
| < 15 | Poor alignment — content does not match prompt |
| 15–22 | Moderate alignment — some elements match |
| 22–28 | Good alignment — most elements are captured |
| 28–35 | Strong alignment — high fidelity to prompt |
| > 35 | Excellent alignment (rare in practice) |

### Temporal Consistency Tracking

For video generation, CLIP Score can be computed per-frame to detect consistency issues:

```
CLIP Score
   30 ┤  ●─────●─────●─────●─────●─────●─────●
   25 ┤
   20 ┤          ○                              ○ ← Drops indicate
   15 ┤                                              scene breaks
   10 ┤
      └──────────────────────────────────────────
       F1     F5     F10    F15    F20    F25   F30
                        Frames
```

### Code Example

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(image, prompt):
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    score = (image_embeds @ text_embeds.T).item() * 100
    return score

score = compute_clip_score(generated_image, "A majestic lion in a sunlit African savanna")
print(f"CLIP Score: {score:.2f}")
```

### Limitations

| Limitation | Description |
|---|---|
| No motion assessment | Only evaluates static frame quality, not motion coherence |
| Temporal blindness | May not catch flicker, jitter, or morphing artifacts |
| Prompt bias | Long/complex prompts may score lower even with good generation |
| Not a human proxy | High CLIP score does not guarantee perceptual quality |

---

## 9. Real-World Applications

### Industry Deployments

| Company | Application | Model / Technique | Impact |
|---|---|---|---|
| **UiPath** | Automated document processing | LayoutLM + OCR | 80% reduction in manual data entry |
| **Meta** | Interactive advertising | ControlNet + SD | 300% increase in click-through rates |
| **TikTok** | Video content creation | CogVideoX + SD | 2B+ views on generated content |
| **Siemens** | Medical image analysis | ViLT (fine-tuned) | 40% faster diagnosis |

### Use Case Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Decision Tree                  │
│                                                              │
│  What do you need?                                           │
│       │                                                      │
│       ├──▶ Understand existing content                       │
│       │       │                                              │
│       │       ├── Natural images? ──▶ ViLT (VQA)            │
│       │       └── Documents/forms? ──▶ LayoutLM (Doc QA)    │
│       │                                                      │
│       ├──▶ Generate new content                              │
│       │       │                                              │
│       │       ├── Images? ──▶ Stable Diffusion               │
│       │       ├── Controlled edits? ──▶ ControlNet           │
│       │       └── Partial edits? ──▶ Inpainting              │
│       │                                                      │
│       ├──▶ Generate video ──▶ CogVideoX                     │
│       │                                                      │
│       └──▶ Evaluate quality ──▶ CLIP Score                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Optimization & Best Practices

### Memory Optimization

| Technique | Method | VRAM Savings |
|---|---|---|
| **CPU Offloading** | `pipe.enable_model_cpu_offload()` | ~40–60% |
| **VAE Slicing** | `pipe.vae.enable_slicing()` | ~30% on VAE step |
| **VAE Tiling** | `pipe.vae.enable_tiling()` | Enables high-res decode |
| **fp16 Precision** | `torch_dtype=torch.float16` | ~50% baseline reduction |
| **Attention Slicing** | `pipe.enable_attention_slicing()` | ~20–30% |

### Speed Optimization

| Technique | Impact | Trade-off |
|---|---|---|
| `torch.float16` | ~2× faster inference | Minor precision loss |
| Reduce `num_inference_steps` | Linear speed-up | Lower quality |
| `torch.compile()` | ~10–30% faster | One-time compile overhead |
| Attention slicing | Enables lower VRAM | Slightly slower |

### Quality Optimization

| Technique | When to Use |
|---|---|
| **Negative prompts** | Suppress unwanted artifacts (`"blurry, low quality, deformed"`) |
| **Higher guidance scale** | When prompt adherence is critical (7.5 → 12–15) |
| **Face restoration (GFPGAN)** | When generating faces that look distorted |
| **Upscaling (Real-ESRGAN)** | When higher resolution is needed beyond 512×512 |
| **More inference steps** | When quality matters more than speed |

### Negative Prompts Example

```python
image = sd_pipe(
    "A portrait of a woman in a garden",
    negative_prompt="blurry, low quality, deformed, disfigured, bad anatomy",
    num_inference_steps=30,
    guidance_scale=12
).images[0]
```

### Complete Optimization Template

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

image = pipe(
    prompt="A professional product photo of a smartphone",
    negative_prompt="blurry, low quality, watermark, text",
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
```

### Optimization Decision Flowchart

```
                           ┌─────────────┐
                           │ Out of VRAM?│
                           └──────┬──────┘
                                  │
                        ┌─────────┴────────┐
                       Yes                  No
                        │                   │
                        ▼                   ▼
                  ┌──────────┐        ┌──────────┐
                  │ fp16?    │        │ Need     │
                  │ Already? │        │ Speed?   │
                  └─────┬────┘        └────┬─────┘
                    No  │  Yes            │
                   ┌────┘    │     ┌──────┴──────┐
                   ▼         ▼    Yes             No
              ┌────────┐ ┌──────┐ │               │
              │Enable  │ │CPU   │ ▼               ▼
              │fp16    │ │Off-  │ ┌─────────┐  ┌──────┐
              └────────┘ │load  │ │torch    │  │Higher│
                         └──────┘ │compile  │  │Steps │
                            │     └─────────┘  └──────┘
                            ▼
                      ┌──────────┐
                      │VAE Slice │
                      └──────────┘
                            │
                            ▼
                      Still OOM?
                      ┌────┴────┐
                     Yes         No
                      │          │
                      ▼          ▼
               ┌──────────┐   Done
               │Attention │
               │Slicing   │
               └──────────┘
```

---

## 11. Quick Reference

### One-Liner Pipelines

```python
# --- Visual QA (ViLT) ---
from transformers import pipeline
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
answer = vqa(image="photo.jpg", question="What color is the car?")

# --- Document QA (LayoutLM) ---
doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
result = doc_qa(image="invoice.png", question="What is the total amount?")

# --- Image Generation (Stable Diffusion) ---
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
image = pipe("a cozy cabin in the snow", num_inference_steps=20).images[0]

# --- Controlled Editing (ControlNet) ---
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16).to("cuda")

# --- Inpainting ---
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
result = pipe(prompt="clear blue sky", image=original, mask_image=mask).images[0]

# --- Video Generation (CogVideoX) ---
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
frames = pipe("a cat jumping over a fence", num_frames=20).frames[0]
```

### Model Hub Quick Lookup

| Task | Model ID | Library |
|---|---|---|
| Visual QA | `dandelin/vilt-b32-finetuned-vqa` | `transformers` |
| Document QA | `impira/layoutlm-document-qa` | `transformers` |
| Image generation | `runwayml/stable-diffusion-v1-5` | `diffusers` |
| ControlNet (Canny) | `lllyasviel/sd-controlnet-canny` | `diffusers` |
| Inpainting | `runwayml/stable-diffusion-inpainting` | `diffusers` |
| Video generation | `THUDM/CogVideoX-2b` | `diffusers` |
| CLIP (evaluation) | `openai/clip-vit-base-patch32` | `transformers` |

### Key Import Summary

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import CogVideoXPipeline
import torch
```

### VRAM Requirements Summary

```
VRAM Usage (approximate)
  4 GB ████████████░░░░░░░░░░  ViLT (VQA)
  6 GB ████████████████░░░░░░  LayoutLM (Doc QA)
  8 GB ████████████████████░░  Stable Diffusion (fp16)
 10 GB ██████████████████████  ControlNet
 12 GB ████████████████████████ CogVideoX (with offloading)
     └──────────────────────────
       0    4    8   12   16 GB
```
