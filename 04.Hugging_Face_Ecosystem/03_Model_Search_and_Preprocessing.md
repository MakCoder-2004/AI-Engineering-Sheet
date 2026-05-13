# Searching & Preprocessing for Hugging Face Models

## Overview

Before using a model, you need to (1) **find the right model** for your task and (2) **preprocess your data** in the format the model expects. This document covers the full workflow from model discovery to data preprocessing for text, images, and audio.

---

## Topics Covered

1. [Searching Models on Hugging Face Hub](#1-searching-models-on-hugging-face-hub)
2. [Text Preprocessing (Tokenization)](#2-text-preprocessing-tokenization)
3. [Image Preprocessing](#3-image-preprocessing)
4. [Audio Preprocessing](#4-audio-preprocessing)
5. [Pipelines vs. Direct Model Usage](#5-pipelines-vs-direct-model-usage)
6. [Finding Models for Pipelines](#6-finding-models-for-pipelines)
7. [Generation Parameters](#7-generation-parameters)
8. [Model Evaluation Metrics](#8-model-evaluation-metrics)
9. [AutoClasses: The Flexible Approach](#9-autoclasses-the-flexible-approach)

---

## 1. Searching Models on Hugging Face Hub

Use `HfApi` to programmatically find models for any task:

```python
from huggingface_hub import HfApi

api = HfApi()

# Search for text-to-image models by CompVis, sorted by downloads
models = api.list_models(
    task="text-to-image",
    author="CompVis",               # Filter by author
    tags="diffusers:StableDiffusionPipeline",  # Filter by tags
    sort="downloads"                # Sort by most downloaded
)

# Get the top model
top_model = models[0]
print(top_model.modelId)
print(top_model.downloads)
```

### Available Search Filters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `task` | Filter by ML task | `"text-classification"`, `"image-classification"` |
| `author` | Filter by creator | `"facebook"`, `"google"`, `"microsoft"` |
| `tags` | Filter by tags | `"pytorch"`, `"transformers"` |
| `sort` | Sort results | `"downloads"`, `"likes"`, `"lastModified"` |
| `limit` | Max results returned | `10` |
| `search` | Keyword search | `"bert"`, `"gpt"` |

---

## 2. Text Preprocessing (Tokenization)

Tokenization converts raw text into numbers that a model can process. The process has four steps:

```
Raw Text → Normalization → Tokenization → Token IDs → Padding
"Hello!"     lowercase      ["hello", "!"]   [7592, 999]   [7592, 999, 0, 0]
```

### Tokenization Steps

| Step | What Happens | Example |
|------|-------------|---------|
| 1. Normalization | Lowercase, remove special chars | `"Hello!"` → `"hello"` |
| 2. Tokenization | Split into subwords | `"playing"` → `["play", "##ing"]` |
| 3. Token → ID | Map each token to a number | `"play"` → `7784` |
| 4. Padding | Make all sequences same length | Add `[PAD]` tokens |

### Code Example

```python
from transformers import AutoTokenizer

# Load tokenizer for a specific model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-squad")

text = "This is an example text."

# Tokenize and get tensors
encoded_input = tokenizer(
    text,
    return_tensors="pt",    # Return PyTorch tensors
    padding=True,            # Pad to the longest sequence in batch
    truncation=True,         # Truncate if longer than model's max length
    max_length=512           # Maximum token length
)

print(encoded_input)
# {
#   'input_ids': tensor([[101, 2023, 2003, ...]]),
#   'attention_mask': tensor([[1, 1, 1, ...]])
# }
print(encoded_input['input_ids'].shape)   # torch.Size([1, N_tokens])
```

### Tokenizer Output Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `input_ids` | `(batch, seq_len)` | Token integer IDs |
| `attention_mask` | `(batch, seq_len)` | 1=real token, 0=padding |
| `token_type_ids` | `(batch, seq_len)` | Used for BERT-style models (sentence A vs B) |

---

## 3. Image Preprocessing

Image preprocessing prepares visual data for vision models:

```
Raw PIL Image → Resize → Normalize pixel values → Convert to tensor
(H, W, 3)         (224, 224, 3)    (0-1 range)        (1, 3, H, W)
```

### Code Example: Image Captioning with BLIP

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

checkpoint = "Salesforce/blip-image-captioning-base"

# Load model and processor together
model = BlipForConditionalGeneration.from_pretrained(checkpoint)
processor = BlipProcessor.from_pretrained(checkpoint)

# Load a sample image from a dataset
image = load_dataset("nlphuji/flickr30k")["test"][11]["image"]
# image is a PIL Image object

# Preprocess: resize, normalize, convert to tensor
inputs = processor(
    images=image,       # PIL Image or numpy array
    return_tensors="pt" # Return PyTorch tensors
)
# → inputs['pixel_values'] shape: (1, 3, 384, 384)

# Generate caption
outputs = model.generate(**inputs)

# Decode output tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)  # → "a man is standing on a rock by the ocean"
```

### Image Preprocessing Steps

1. **Resize**: Models expect fixed-size inputs (e.g., 224×224 for ViT-based models)
2. **Normalize**: Pixel values scaled from [0, 255] to [0, 1] using mean/std
3. **Tensor conversion**: NumPy arrays → PyTorch tensors
4. **Channel reordering**: PIL (H, W, C) → PyTorch (C, H, W), plus batch dim

---

## 4. Audio Preprocessing

Audio preprocessing converts raw waveforms into model-compatible features:

```
Raw Audio → Resample → Feature Extraction → Tensor
(array, sr)  (16kHz)   (mel spectrogram)  (1, 80, T)
```

### Code Example: Preprocessing for Whisper

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor

# Load dataset
dataset = load_dataset("CSTR-Edinburgh/vctk")["train"]

# Step 1: Resample to 16kHz (Whisper expects 16kHz)
dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate=16_000)
)

# Step 2: Load the model's processor
processor = AutoProcessor.from_pretrained("openai/whisper-small")

# Step 3: Extract features from audio
audio_sample = dataset[0]["audio"]

audio_inputs = processor(
    audio_sample["array"],      # NumPy array of waveform
    sampling_rate=16_000,        # Tell processor about the sample rate
    return_tensors="pt"          # Return PyTorch tensors
)

print(audio_inputs.keys())
# → dict_keys(['input_features'])
# input_features shape: (1, 80, 3000) = (batch, mel_bins, time_frames)
```

### Audio Preprocessing Steps

| Step | Purpose | How |
|------|---------|-----|
| Resampling | Match model's expected sample rate | `cast_column(..., Audio(sampling_rate=16000))` |
| Padding/Truncation | Make all samples same length | Done by processor automatically |
| Feature extraction | Convert waveform to spectrogram | `AutoProcessor.from_pretrained(...)` |
| Normalization | Scale features to stable range | Automatic in processor |

---

## 5. Pipelines vs. Direct Model Usage

| Approach | Pros | Cons |
|----------|------|------|
| `pipeline()` | Fast to implement, minimal code | Less control, harder to customize |
| Direct model | Full control, fine-tuning support, custom preprocessing | More code, manual tensor handling |

### Side-by-Side Comparison

**Pipeline approach (simple):**
```python
from transformers import pipeline

pipe = pipeline(task="image-to-text", model="Salesforce/blip-image-captioning-base")
result = pipe(image)
print(result)  # → [{'generated_text': 'a man standing on a rock'}]
```

**Direct model approach (flexible):**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)
print(caption)
```

---

## 6. Finding Models for Pipelines

```python
from transformers import pipeline
from huggingface_hub import HfApi

api = HfApi()

# Find the top text-to-image models
available_models = list(
    api.list_models(
        task="text-to-image",
        limit=5
    )
)

# Use the top model automatically
pipe = pipeline(
    "text-to-image",
    model=available_models[0].id
)

print(f"Using: {available_models[0].id}")
output = pipe("A beautiful sunset over the ocean")
```

---

## 7. Generation Parameters

Control how models generate output:

| Parameter | Effect | When to Use |
|-----------|--------|-------------|
| `temperature` | Higher = more creative/random | Creative text generation |
| `max_new_tokens` | Limits output length | Control response size |
| `top_p` | Nucleus sampling (keeps top P% probability mass) | Alternative to temperature |
| `top_k` | Keep only top K token candidates | Focused generation |
| `num_beams` | Beam search width (higher = better quality, slower) | Translation, summarization |
| `do_sample` | Enable sampling (vs greedy) | Must be True for temperature |

```python
from transformers import pipeline

audio_pipe = pipeline(
    task="text-to-audio",
    model="facebook/musicgen-small",
    framework="pt"
)

generate_kwargs = {
    "temperature": 0.8,     # Some creativity
    "max_new_tokens": 256   # About 5 seconds of audio
}

outputs = audio_pipe(
    "Classic rock riff with electric guitar",
    generate_kwargs=generate_kwargs
)

print(outputs)  # Audio data in numpy array
```

---

## 8. Model Evaluation Metrics

```python
from evaluate import evaluator

# Create evaluator for image classification
task_evaluator = evaluator("image-classification")
```

### Common Metrics by Task

| Task | Metric | Description |
|------|--------|-------------|
| Classification | **Accuracy** | % of correctly classified examples |
| Classification | **F1 Score** | Harmonic mean of precision and recall |
| Classification | **Precision** | TP / (TP + FP) — of all predicted positives, how many are correct? |
| Classification | **Recall** | TP / (TP + FN) — of all actual positives, how many did we find? |
| Generation | **BLEU** | Overlap with reference translations |
| Summarization | **ROUGE** | Overlap with reference summaries |
| QA | **Exact Match** | Answer exactly matches reference |

---

## 9. AutoClasses: The Flexible Approach

AutoClasses automatically detect and load the right model class:

```python
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering
)

# Load model and tokenizer separately
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Combine into a custom pipeline
from transformers import pipeline
custom_pipeline = pipeline(
    task="sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

output = custom_pipeline("This course is excellent!")
print(output)  # → [{'label': 'POSITIVE', 'score': 0.999}]
```

### Common AutoClasses Reference

| AutoClass | Task |
|-----------|------|
| `AutoTokenizer` | Text tokenization for any model |
| `AutoProcessor` | Multi-modal preprocessing (images, audio) |
| `AutoModel` | Base model (no task head) |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForCausalLM` | Text generation (GPT-style) |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForSeq2SeqLM` | Translation, summarization (T5-style) |
| `AutoModelForImageClassification` | Image classification |
| `AutoImageProcessor` | Image preprocessing |

---

## Summary: Model Usage Decision Tree

```
Do I need to fine-tune the model?
  YES → Use AutoClasses + Trainer
  NO  ↓

Do I need custom preprocessing or post-processing?
  YES → Use AutoClasses directly
  NO  ↓

Do I need a quick result with minimal code?
  YES → Use pipeline()
  NO  ↓

Do I need to search for the best model first?
  YES → Use HfApi().list_models() then pipeline()
```
