# Speech Recognition and Audio Generation with Hugging Face

## Overview

This document covers the complete pipeline for speech processing with Hugging Face — from automatic speech recognition (ASR) using OpenAI's Whisper to audio generation with Microsoft's SpeechT5, speaker embeddings for voice cloning, and fine-tuning text-to-speech models for new languages and speakers.

---

## Topics Covered

1. [Automatic Speech Recognition with Whisper](#part-1-automatic-speech-recognition-with-whisper)
2. [Audio Generation with SpeechT5](#part-2-audio-generation-with-speecht5)
3. [Speaker Embeddings for Voice Cloning](#part-3-speaker-embeddings-for-voice-cloning)
4. [Visualization](#part-4-visualization)
5. [Fine-Tuning Text-to-Speech Models](#part-5-fine-tuning-text-to-speech-models)
6. [Model Comparison Tables](#model-comparison-tables)
7. [Best Practices & Tips](#best-practices--tips)
8. [Quick Reference](#quick-reference)

---

## Architecture & Concepts

### End-to-End Speech Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SPEECH PROCESSING OVERVIEW                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │  Audio    │────▶│   Feature    │────▶│    Model     │            │
│  │  Input    │     │  Extraction  │     │  (Encoder-   │            │
│  │ (Waveform)│     │(Mel-Spectro) │     │   Decoder)   │            │
│  └──────────┘     └──────────────┘     └──────┬───────┘            │
│                                               │                     │
│                          ┌────────────────────┼────────────┐       │
│                          ▼                    ▼            ▼       │
│                   ┌────────────┐     ┌────────────┐ ┌──────────┐  │
│                   │   Text     │     │  Mel-Spec   │ │ Speaker  │  │
│                   │ (ASR/Trans)│     │ (Gen/TTS)   │ │ Embed    │  │
│                   └────────────┘     └─────┬──────┘ └──────────┘  │
│                                           │                        │
│                                     ┌─────▼──────┐                │
│                                     │  Vocoder   │                │
│                                     │ (HiFi-GAN) │                │
│                                     └─────┬──────┘                │
│                                           │                        │
│                                     ┌─────▼──────┐                │
│                                     │  Waveform  │                │
│                                     │  (Audio)   │                │
│                                     └────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### Whisper Architecture (ASR)

```
┌──────────────────────────────────────────────────┐
│              WHISPER ENCODER-DECODER              │
├──────────────────────────────────────────────────┤
│                                                  │
│  Raw Audio (16kHz)                               │
│       │                                          │
│       ▼                                          │
│  ┌─────────────┐                                 │
│  │ Log-Mel     │  80 mel bins                    │
│  │ Spectrogram │  (audio → visual repr.)         │
│  └──────┬──────┘                                 │
│         │                                        │
│  ┌──────▼──────┐                                 │
│  │  ENCODER    │  Processes audio features       │
│  │  (Transformer│ across time frames             │
│  │   Blocks)   │                                 │
│  └──────┬──────┘                                 │
│         │  Encoded representations               │
│         ▼                                        │
│  ┌─────────────┐                                 │
│  │  DECODER    │  Autoregressive text generation │
│  │ (Transformer│  (token by token)               │
│  │   + LM Head)│                                 │
│  └──────┬──────┘                                 │
│         │                                        │
│         ▼                                        │
│  ┌─────────────┐                                 │
│  │ Text Output │  "Hello, how are you?"          │
│  └─────────────┘                                 │
└──────────────────────────────────────────────────┘
```

### SpeechT5 Architecture (Audio Generation)

```
┌──────────────────────────────────────────────────────────────┐
│                  SPEECHT5 THREE-COMPONENT SYSTEM              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐                                          │
│  │  PREPROCESSOR  │  Handles resampling, feature extraction  │
│  │  (Processor)   │  Converts raw audio → mel-spectrograms   │
│  └───────┬────────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐     ┌───────────────────┐               │
│  │     MODEL      │────▶│ Speaker Embedding │               │
│  │  (SpeechT5     │     │ (512-dim vector)  │               │
│  │  Transformer)  │     │ Conditions output  │               │
│  └───────┬────────┘     │ on target voice    │               │
│          │              └───────────────────┘               │
│          ▼  Mel-spectrogram output                          │
│  ┌────────────────┐                                          │
│  │    VOCODER     │  HiFi-GAN (GAN-based waveform gen)      │
│  │ (SpeechT5      │  Reconstructs phase info lost in        │
│  │  HiFi-GAN)     │  spectrogram computation                │
│  └───────┬────────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                          │
│  │   Waveform     │  Audio output (16kHz)                   │
│  │   (Audio)      │                                          │
│  └────────────────┘                                          │
└──────────────────────────────────────────────────────────────┘
```

### Speaker Embedding Pipeline (Voice Cloning)

```
┌────────────────────────────────────────────────────────────────┐
│              VOICE CLONING / CONVERSION PIPELINE               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Reference Audio          Target Text                          │
│  (target voice)           (what to say)                        │
│       │                        │                               │
│       ▼                        ▼                               │
│  ┌─────────────┐      ┌──────────────┐                        │
│  │   Speaker   │      │   Text       │                        │
│  │   Encoder   │      │  Tokenizer   │                        │
│  │  (X-Vector) │      │              │                        │
│  └──────┬──────┘      └──────┬───────┘                        │
│         │                    │                                 │
│         ▼                    ▼                                 │
│  ┌─────────────┐      ┌──────────────┐                        │
│  │   Speaker   │      │  input_ids   │                        │
│  │ Embedding   │      │  (tokens)    │                        │
│  │ (512-dim)   │      └──────┬───────┘                        │
│  └──────┬──────┘             │                                 │
│         │                    │                                 │
│         └────────┬───────────┘                                │
│                  ▼                                             │
│         ┌──────────────┐                                      │
│         │   TTS Model  │  SpeechT5 (conditioned on speaker)   │
│         │  (Decoder)   │                                      │
│         └──────┬───────┘                                      │
│                │  Mel-spectrogram                              │
│                ▼                                               │
│         ┌──────────────┐                                      │
│         │   Vocoder    │  HiFi-GAN                            │
│         │ (HiFi-GAN)   │                                      │
│         └──────┬───────┘                                      │
│                │                                               │
│                ▼                                               │
│         ┌──────────────┐                                      │
│         │ Generated    │  Speech in target voice              │
│         │ Speech       │                                      │
│         └──────────────┘                                      │
└────────────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
pip install transformers datasets torch speechbrain librosa matplotlib
```

```python
import torch
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    SpeechT5Processor,
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset, Audio, DatasetDict
from speechbrain.inference.speaker import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
import librosa.display
import matplotlib.pyplot as plt
```

---

## Part 1: Automatic Speech Recognition with Whisper

Whisper is a transformer-based encoder-decoder model trained on **680,000 hours** of labeled audio. It supports transcription (speech-to-text), translation (speech-to-text in another language), and language identification.

### Whisper Model Sizes

| Model     | Parameters | Disk Size | Relative Speed | Accuracy     | Use Case                    |
|-----------|-----------|-----------|----------------|--------------|-----------------------------|
| `tiny`    | 39M       | ~150 MB   | Fastest        | Lowest       | Prototyping, embedded       |
| `base`    | 74M       | ~290 MB   | Very Fast      | Low          | Quick transcription         |
| `small`   | 244M      | ~940 MB   | Fast           | Good         | Production (balanced)       |
| `medium`  | 769M      | ~2.9 GB   | Moderate       | Very Good    | High-accuracy tasks         |
| `large`   | 1.55B     | ~5.9 GB   | Slowest        | Best         | State-of-the-art            |

### Implementation

#### Step 1: Load Model and Processor

The processor handles both audio feature extraction (log-mel spectrograms) and text tokenization:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

model.generation_config.language = "english"
model.generation_config.task = "transcribe"
```

#### Step 2: Load and Prepare Audio Dataset

```python
from datasets import load_dataset, Audio

dataset = load_dataset("CSTR-Edinburgh/vctk")["train"]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

print(f"Dataset: {len(dataset)} audio samples")
print(f"Features: {dataset.column_names}")
```

> **Note:** Whisper expects **16,000 Hz** sampling rate. Always cast the audio column to the correct rate.

#### Step 3: Process Audio and Generate Transcription

```python
import torch

sample = dataset[0]["audio"]

input_preprocessed = processor(
    sample["array"],
    sampling_rate=sample["sampling_rate"],
    return_tensors="pt"
)

with torch.no_grad():
    predicted_ids = model.generate(input_preprocessed.input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(f'Transcription: "{transcription[0]}"')
```

**What happens during processing:**
1. Audio is resampled to 16kHz (if needed)
2. Log-mel spectrogram is computed (80 bins)
3. Features are normalized and padded to model's expected length
4. Encoder processes the spectrogram into hidden representations
5. Decoder generates text tokens autoregressively
6. Special tokens (`<|startoftranscript|>`, `<|notimestamps|>`) are stripped

**Preprocessed input shape:** `(1, 80, time_frames)` where 80 = mel bins.

---

## Part 2: Audio Generation with SpeechT5

SpeechT5 is a unified-modal encoder-decoder model for speech processing. It uses three critical components:

| Component       | Role                                                        |
|----------------|-------------------------------------------------------------|
| **Preprocessor** | Resampling, feature extraction (raw audio → mel-spectrograms) |
| **Model**       | SpeechT5 transformer that transforms features               |
| **Vocoder**     | HiFi-GAN converts spectrograms → audio waveforms            |

### Why the Vocoder is Essential

Spectrograms represent **magnitude** but lack **phase information** needed to reconstruct waveforms. HiFi-GAN (High-Fidelity GAN) is a generative adversarial network specifically designed to infer the missing phase and produce high-quality audio waveforms from mel-spectrograms.

### Implementation

```python
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

processor_st5 = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model_st5 = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

- **Processor**: Converts raw audio to mel-spectrogram features
- **Model**: Takes mel-spectrograms as input, outputs transformed mel-spectrograms
- **Vocoder**: Generates waveforms at native sampling rate (typically 16kHz)

---

## Part 3: Speaker Embeddings for Voice Cloning

Speaker embeddings capture unique vocal characteristics in a fixed-dimensional vector:

| Characteristic | Captured by Embedding |
|---------------|----------------------|
| Pitch         | Yes                  |
| Timbre        | Yes                  |
| Speaking rate | Yes                  |
| Accent        | Yes                  |
| Prosody       | Yes                  |

### Speaker Encoder Details

| Property       | Value                                         |
|---------------|-----------------------------------------------|
| Architecture  | X-Vector (Time-Delay Neural Network)          |
| Model         | `speechbrain/spkrec-xvect-voxceleb`           |
| Training Data | VoxCeleb (1,251 speakers, 150k+ utterances)   |
| Output        | 512-dimensional vector                        |

### Implementation

#### Step 1: Load Speaker Encoder

```python
from speechbrain.inference.speaker import EncoderClassifier

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="tmpdir_spkrec"
)
```

#### Step 2: Extract Speaker Embedding

```python
audio_array = dataset[0]["audio"]["array"]
audio_tensor = torch.tensor(audio_array).float()

speaker_embeddings = speaker_model.encode_batch(audio_tensor)

speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
speaker_embeddings = speaker_embeddings.unsqueeze(0)

print(f"Shape: {speaker_embeddings.shape}")
print(f"L2 norm: {torch.norm(speaker_embeddings, dim=2).item():.4f}")
```

**Processing pipeline inside `encode_batch`:**
1. Computes MFCCs (Mel-Frequency Cepstral Coefficients) from audio
2. Extracts statistical features across time windows
3. Passes through TDNN layers to produce embedding
4. Output shape: `(batch, time, 512)`

**L2 normalization** creates a unit vector, improving cosine similarity matching. The L2 norm after normalization should be ~1.0.

---

## Part 4: Visualization

### Waveform Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio_waveform(audio_array, sampling_rate, title="Audio Waveform"):
    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio_array)) / sampling_rate
    plt.plot(time, audio_array)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Mel Spectrogram Visualization

```python
import librosa
import librosa.display

def visualize_mel_spectrogram(audio_array, sampling_rate, title="Mel Spectrogram"):
    plt.figure(figsize=(12, 6))

    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sampling_rate,
        n_mels=80,
        fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    librosa.display.specshow(
        mel_spec_db,
        sr=sampling_rate,
        x_axis='time',
        y_axis='mel',
        fmax=8000
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
```

### Speaker Embedding Visualization

```python
def visualize_speaker_embedding(embedding_vector):
    plt.figure(figsize=(16, 4))
    emb_1d = embedding_vector.squeeze().cpu().numpy()

    plt.imshow(emb_1d.reshape(1, -1), aspect='auto', cmap='viridis')
    plt.colorbar(label='Embedding Value')
    plt.xlabel("Embedding Dimension (0-511)")
    plt.ylabel("Speaker")
    plt.title("Speaker Embedding Visualization (512 dimensions)")
    plt.tight_layout()
    plt.show()
```

### Comprehensive Generated Speech Visualization

```python
def visualize_generated_speech(speech_array, sample_rate=16000):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time = np.arange(len(speech_array)) / sample_rate
    axes[0, 0].plot(time, speech_array)
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Generated Speech Waveform")
    axes[0, 0].grid(True, alpha=0.3)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(speech_array)), ref=np.max)
    img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[0, 1])
    axes[0, 1].set_title("Spectrogram (Frequency vs Time)")
    fig.colorbar(img, ax=axes[0, 1], format="%+2.0f dB")

    mel_spec = librosa.feature.melspectrogram(y=speech_array, sr=sample_rate, n_mels=80)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    img2 = librosa.display.specshow(mel_db, sr=sample_rate, x_axis='time', y_axis='mel', ax=axes[1, 0])
    axes[1, 0].set_title("Mel Spectrogram (Model Output)")
    fig.colorbar(img2, ax=axes[1, 0], format="%+2.0f dB")

    axes[1, 1].hist(speech_array, bins=50, alpha=0.7)
    axes[1, 1].set_xlabel("Amplitude")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Amplitude Distribution\nStd: {speech_array.std():.3f}")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

---

## Part 5: Fine-Tuning Text-to-Speech Models

### When to Fine-Tune TTS

| Use Case                              | Approach                    |
|---------------------------------------|-----------------------------|
| Standard English speech               | Use pre-trained model       |
| New language or dialect               | Fine-tune on target language |
| Domain-specific vocabulary            | Fine-tune with domain data  |
| Custom speaker voice                  | Fine-tune with speaker data |
| Improve naturalness in specific domain| Fine-tune on domain data    |

### Fine-Tuning Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    TTS FINE-TUNING ARCHITECTURE                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Text Input ──▶ ┌──────────┐    ┌─────────────────┐               │
│                 │ Encoder  │───▶│  Combined       │───┐           │
│                 │ (Tuned)  │    │  Features       │   │           │
│                 └──────────┘    └────────┬────────┘   │           │
│                                         │            │           │
│  Reference Audio ──▶ ┌──────────┐       │            │           │
│                       │ Speaker  │──────┘            │           │
│                       │ Encoder  │                   ▼           │
│                       │ (FROZEN) │           ┌──────────────┐    │
│                       └──────────┘           │   Decoder    │    │
│                                              │  (Tuned)     │    │
│                                              └──────┬───────┘    │
│                                                     │            │
│                                                     ▼            │
│                                              ┌──────────────┐    │
│                                              │ Mel-Spec     │    │
│                                              │ (predicted)  │    │
│                                              └──────┬───────┘    │
│                                                     │            │
│                                                     ▼            │
│                                              ┌──────────────┐    │
│                                              │   Vocoder    │    │
│                                              │ (HiFi-GAN,   │    │
│                                              │   FROZEN)    │    │
│                                              └──────┬───────┘    │
│                                                     │            │
│                                                     ▼            │
│                                              ┌──────────────┐    │
│                                              │  Waveform    │    │
│                                              │  (Audio)     │    │
│                                              └──────────────┘    │
└────────────────────────────────────────────────────────────────────┘

What changes during fine-tuning:
  - Encoder: Learns pronunciation for new phonemes
  - Decoder: Generates mel-spectrograms for target language/speaker
  - Speaker Encoder: FROZEN (preserves voice identity)
  - Vocoder: FROZEN (already generates high-quality waveforms)
```

### Complete Fine-Tuning Workflow

#### Step 1: Load and Prepare Dataset

VoxPopuli is a large-scale multilingual speech corpus from EU parliament recordings (2009-2020), covering 18 languages.

```python
from datasets import load_dataset, DatasetDict

dataset = load_dataset(
    "facebook/voxpopuli",
    "it",
    split="train",
    trust_remote_code=True
)

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

print(f"Train: {len(dataset['train']):,}  |  Test: {len(dataset['test']):,}")
```

**Dataset features:** `audio`, `raw_text`, `normalized_text`, `speaker_id`, `gender`

#### Step 2: Load All Components

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.inference.speaker import EncoderClassifier

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)
```

#### Step 3: Create Preprocessing Pipeline

```python
def prepare_dataset(example):
    audio = example["audio"]

    processed = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False
    )

    processed["labels"] = processed["labels"][0]

    with torch.no_grad():
        audio_tensor = torch.tensor(audio["array"]).float()
        speaker_emb = speaker_model.encode_batch(audio_tensor)
        speaker_emb = torch.nn.functional.normalize(speaker_emb, dim=2)
        processed["speaker_embeddings"] = speaker_emb.squeeze().cpu().numpy()

    return processed


dataset["train"] = dataset["train"].map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    num_proc=4
)

dataset["test"] = dataset["test"].map(
    prepare_dataset,
    remove_columns=dataset["test"].column_names,
    num_proc=4
)
```

**Preprocessing steps:**
1. Text tokenization (SentencePiece tokenizer)
2. Target audio → log-mel spectrogram (labels)
3. Speaker embedding generation (512-dim from reference audio)
4. L2 normalization of embeddings

#### Step 4: Create Custom Data Collator

```python
@dataclass
class TTSDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        batch = self.processor.pad(input_ids, padding=True, return_tensors="pt")

        labels = [feature["labels"] for feature in features]
        max_label_len = max(l.shape[0] for l in labels)

        padded_labels = []
        for label in labels:
            pad_len = max_label_len - label.shape[0]
            padded = np.pad(
                label,
                ((0, pad_len), (0, 0)),
                mode='constant',
                constant_values=0
            )
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels)
        speaker_embeddings = [feature["speaker_embeddings"] for feature in features]
        batch["speaker_embeddings"] = torch.tensor(speaker_embeddings)

        return batch


data_collator = TTSDataCollator(processor=processor)
```

**Why dynamic padding:** Mel-spectrograms vary in length (different utterance durations). Padding to the longest in each batch (not a fixed global max) minimizes memory usage.

#### Step 5: Configure Training Arguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./speecht5_italian_tts",
    overwrite_output_dir=True,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,

    learning_rate=1e-5,
    warmup_steps=500,
    lr_scheduler_type="linear",

    num_train_epochs=3,

    evaluation_strategy="steps",
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,

    predict_with_generate=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    report_to=["none"],

    label_names=["labels"],
    generation_max_length=512,
    generation_num_beams=4,
)
```

#### Training Arguments Explained

| Argument                      | Value  | Why                                                |
|------------------------------|--------|-----------------------------------------------------|
| `per_device_train_batch_size`| 4      | Small due to mel-spectrogram memory usage           |
| `gradient_accumulation_steps`| 8      | Effective batch = 4 × 8 = 32                        |
| `learning_rate`              | 1e-5   | Lower than ASR — TTS is more sensitive to LR        |
| `warmup_steps`               | 500    | Gradually ramp up LR to avoid early instability     |
| `num_train_epochs`           | 3      | Sufficient for fine-tuning from pre-trained English  |
| `fp16`                       | Auto   | Mixed precision for ~2x speedup on GPU              |
| `label_names`                | ["labels"] | Tells trainer which tensor is the target         |

#### Step 6: Initialize Trainer and Train

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()
```

**What happens during each training step:**
1. Text tokens → Encoder → Hidden states
2. Hidden states + speaker embedding → Decoder
3. Decoder predicts mel-spectrogram frames autoregressively
4. Loss = MSE between predicted and target mel frames
5. Gradients backpropagate through encoder and decoder
6. Speaker encoder remains **frozen**

#### Step 7: Generate Speech with Fine-Tuned Model

```python
text = "se sono italiano posso cantare l'opera lirica"

test_speaker_emb = dataset["test"][5]["speaker_embeddings"]
speaker_embedding = torch.tensor(test_speaker_emb).unsqueeze(0)

inputs = processor(text=text, return_tensors="pt")

with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embedding,
        vocoder=vocoder
    )

print(f"Duration: {len(speech) / 16000:.2f}s  |  Sample rate: 16,000 Hz")
```

### Evaluation Metrics

| Metric                        | What it Measures          | Direction |
|------------------------------|---------------------------|-----------|
| Mel Cepstral Distortion (MCD)| Spectral similarity       | Lower     |
| Mean Opinion Score (MOS)     | Human-rated naturalness   | Higher    |
| Speaker Similarity           | Cosine similarity of embeddings | Higher |
| Word Error Rate (WER)        | ASR-transcribed accuracy  | Lower     |

---

## Model Comparison Tables

### Speech Recognition Models

| Model                          | Parameters | Task          | Training Data         | Strengths                    |
|-------------------------------|-----------|---------------|-----------------------|------------------------------|
| `openai/whisper-tiny`         | 39M       | ASR           | 680k hours            | Fast, lightweight            |
| `openai/whisper-small`        | 244M      | ASR           | 680k hours            | Good balance                 |
| `openai/whisper-large`        | 1.55B     | ASR           | 680k hours            | State-of-the-art accuracy    |
| `facebook/wav2vec2-base`      | 95M       | ASR           | 960 hours (LibriSpeech)| Strong self-supervised pretraining |

### Audio Generation Models

| Model                                  | Parameters | Task              | Strengths                         |
|----------------------------------------|-----------|-------------------|-----------------------------------|
| `microsoft/speecht5_tts`               | ~100M     | Text-to-Speech    | Balanced, supports voice cloning  |
| `microsoft/speecht5_vc`                | ~100M     | Voice Conversion  | Change speaker identity           |
| `microsoft/speecht5_hifigan`           | —         | Vocoder           | High-fidelity waveform generation |
| `suno/bark`                            | —         | TTS + Music       | Expressive, multilingual          |

### Speaker Encoder Models

| Model                                      | Architecture | Embedding Dim | Training Data        | Use Case                  |
|--------------------------------------------|-------------|---------------|----------------------|---------------------------|
| `speechbrain/spkrec-xvect-voxceleb`       | X-Vector    | 512           | VoxCeleb (1,251 spk) | Speaker verification, TTS |
| `speechbrain/spkrec-ecapa-voxceleb`       | ECAPA-TDNN  | 192           | VoxCeleb             | Higher accuracy           |

---

## Best Practices & Tips

### Data Preparation

| Tip                                          | Details                                        |
|----------------------------------------------|------------------------------------------------|
| Resample to model's expected rate            | Usually 16kHz — mismatched rates degrade quality |
| Use clean recordings                         | Background noise harms TTS training            |
| Normalize text                               | Expand numbers, abbreviations before training  |
| Precompute speaker embeddings                | Saves significant training time                |
| Dataset size: 10+ hours                      | Reasonable quality                             |
| Dataset size: 50+ hours                      | Good quality                                   |

### Training

| Tip                                          | Details                                        |
|----------------------------------------------|------------------------------------------------|
| GPU Memory: 16GB+ recommended               | Mel-spectrograms are memory-intensive           |
| Use mixed precision (FP16)                   | ~2x speedup on compatible GPUs                 |
| Small batch size (2-4)                       | Mel-spectrograms consume significant GPU memory |
| Gradient accumulation (4-8)                  | Simulates larger effective batch size           |
| Lower learning rate (1e-5)                   | TTS models are sensitive — too high causes artifacts |
| Freeze speaker encoder                       | Preserves speaker identity                      |
| Monitor for overfitting                      | TTS models can memorize individual speakers     |

### Inference

| Tip                                          | Details                                        |
|----------------------------------------------|------------------------------------------------|
| Use `torch.no_grad()`                        | Disable gradients for faster inference          |
| Cache models locally                         | Avoid re-downloading on each run                |
| Normalize speaker embeddings (L2)            | Consistent cosine similarity matching           |
| Test with different speaker embeddings       | Verify voice diversity and quality              |

### Common Pitfalls to Avoid

| Pitfall                         | Consequence                              | Fix                               |
|--------------------------------|------------------------------------------|-----------------------------------|
| Mismatched sample rates        | Garbled or incorrect output              | Always cast to 16kHz              |
| Training with background noise | Model learns noise artifacts             | Use clean recordings only         |
| Overfitting to single speaker  | Model fails to generalize                | Use multiple speakers             |
| Skipping L2 normalization      | Inconsistent similarity measures         | Always normalize embeddings       |
| Too high learning rate         | Speech artifacts, robotic output         | Use 1e-5 for TTS fine-tuning      |

---

## Quick Reference

### ASR Pipeline (Whisper)

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.generation_config.language = "english"
model.generation_config.task = "transcribe"

dataset = load_dataset("CSTR-Edinburgh/vctk")["train"]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

sample = dataset[0]["audio"]
inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")

with torch.no_grad():
    ids = model.generate(inputs.input_features)

transcription = processor.batch_decode(ids, skip_special_tokens=True)
```

### Audio Generation Pipeline (SpeechT5)

```python
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

### Speaker Embedding Pipeline

```python
from speechbrain.inference.speaker import EncoderClassifier
import torch

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="tmpdir_spkrec"
)

audio_tensor = torch.tensor(audio_array).float()
embeddings = speaker_model.encode_batch(audio_tensor)
embeddings = torch.nn.functional.normalize(embeddings, dim=2)
```

### TTS Fine-Tuning Pipeline

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# ... preprocess dataset, create data collator ...

training_args = Seq2SeqTrainingArguments(
    output_dir="./speecht5_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()

# Generate speech
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
```

### Real-World Applications

| Application                          | Models Used                        |
|--------------------------------------|-------------------------------------|
| Real-time transcription services     | Whisper (medium or large)           |
| Voice assistants with custom voices  | SpeechT5 + speaker embeddings       |
| Dubbing and localization             | Whisper (translation) + SpeechT5    |
| Speaker diarization (who spoke when) | X-Vector speaker encoder            |
| Emotion recognition from speech      | Whisper + classification head        |
| Accessibility tools                  | Whisper ASR + TTS for screen readers |
| Podcast transcription                | Whisper (large for accuracy)         |
