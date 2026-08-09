# Fine-Tuning Large Language Models

> A complete guide to fine-tuning transformer-based LLMs for downstream tasks, demonstrated with BERT sentiment classification on the IMDB dataset.

---

## 1. Overview: Why Fine-Tune?

Pre-trained LLMs capture broad linguistic patterns from massive corpora, but they lack task-specific precision. Fine-tuning adapts a general-purpose model to a narrow domain by continuing training on a labeled dataset — unlocking high accuracy with far fewer resources than training from scratch.

**Key benefits:**

| Benefit | Description |
|---------|-------------|
| **Domain Adaptation** | Specialize a general model for medical, legal, financial, or other domains |
| **Task Performance** | Achieve state-of-the-art results on classification, QA, NER, summarization |
| **Resource Efficiency** | Train with a fraction of the compute vs. pre-training from scratch |
| **Rapid Iteration** | Experiment with different architectures and hyperparameters quickly |

---

## 2. Pipelines vs. AutoClasses

Hugging Face offers two abstraction levels for working with models. Pipelines are ideal for quick inference, while AutoClasses unlock full customization and fine-tuning support.

| Aspect | Pipelines | AutoClasses |
|--------|-----------|-------------|
| **Purpose** | Streamline common tasks | Full model customization |
| **Model Selection** | Automatic (heuristic-based) | Manual and explicit |
| **Control Level** | Minimal — black box | Fine-grained control over every component |
| **Tokenizer** | Auto-selected | Manually loaded via `AutoTokenizer` |
| **Fine-Tuning** | Not supported | Fully supported with `Trainer` API |
| **Use Case** | Prototyping / quick demos | Production systems / research |

**When to use which:**

```
Use Pipelines when:
  ✓ Rapid prototyping
  ✓ No training required
  ✓ Standard NLP tasks (sentiment, translation, etc.)

Use AutoClasses when:
  ✓ Fine-tuning on custom data
  ✓ Custom model architectures
  ✓ Production deployment with control
```

---

## 3. The LLM Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM LIFECYCLE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────┐         ┌──────────────────────┐            │
│   │    PRE-TRAINING      │         │     FINE-TUNING      │            │
│   │                      │         │                      │            │
│   │  • Broad data        │ ──────► │  • Domain-specific   │            │
│   │  • General patterns  │         │  • Task-specific     │            │
│   │  • Self-supervised   │         │  • Supervised        │            │
│   │  • Massive compute   │         │  • Moderate compute  │            │
│   │  • Weeks / months    │         │  • Hours / days      │            │
│   │                      │         │                      │            │
│   └──────────────────────┘         └──────────────────────┘            │
│           │                                │                            │
│           ▼                                ▼                            │
│   ┌──────────────────────┐         ┌──────────────────────┐            │
│   │  Examples:           │         │  Examples:           │            │
│   │  • Masked LM (BERT)  │         │  • Sentiment class.  │            │
│   │  • Causal LM (GPT)   │         │  • Named Entity Rec. │            │
│   │  • Seq2Seq (T5)      │         │  • Question Answer.  │            │
│   └──────────────────────┘         └──────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration & Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `"bert-base-uncased"` | Pre-trained BERT model (110M params, lowercase) |
| `DATASET_NAME` | `"imdb"` | Binary sentiment classification dataset (pos/neg) |
| `MAX_LENGTH` | `64` | Maximum token length per input sequence |
| `NUM_EPOCHS` | `3` | Number of full passes over the training data |
| `BATCH_SIZE` | `8` | Samples processed per gradient update |
| `LEARNING_RATE` | `2e-5` | Step size for weight updates |
| `SHARD_SIZE` | `4` | Dataset split factor (use 1/4 of data for speed) |
| `WEIGHT_DECAY` | `0.01` | L2 regularization to prevent overfitting |

---

## 5. End-to-End Fine-Tuning Pipeline

```
┌───────────────────────────────────────────────────────────────────┐
│                 FINE-TUNING PIPELINE OVERVIEW                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐   ┌────────────┐   ┌───────────┐   ┌────────────┐  │
│  │  LOAD   │──►│ TOKENIZE   │──►│   TRAIN   │──►│ EVALUATE   │  │
│  │  DATA   │   │ & PREPROCESS│   │   MODEL   │   │ & METRICS  │  │
│  └─────────┘   └────────────┘   └───────────┘   └────────────┘  │
│       │              │               │                │           │
│       ▼              ▼               ▼                ▼           │
│  IMDB Dataset   BERT Tokenizer  Trainer API    Accuracy, F1     │
│  train / test   padding, trunc  3 epochs       Precision, Recall│
│                                                                   │
│                              │                                    │
│                              ▼                                    │
│                    ┌─────────────────┐                            │
│                    │  SAVE & DEPLOY  │                            │
│                    │  Inference      │                            │
│                    │  Pipeline       │                            │
│                    └─────────────────┘                            │
└───────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Loading the Dataset

Load the IMDB movie review dataset and take a subset (1/4 shard) for faster training iteration.

```python
from datasets import load_dataset

train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test")

# Use a subset (1/4) for faster training
train_data = train_data.shard(num_shards=4, index=0)
test_data = test_data.shard(num_shards=4, index=0)

print(f"Training samples:   {len(train_data)}")
print(f"Evaluation samples: {len(test_data)}")
```

---

### Step 2: Loading Model & Tokenizer

Load the pre-trained BERT model with a classification head and its matching tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # Binary classification: Positive / Negative
)
```

> **Note:** `AutoModelForSequenceClassification` adds a classification head on top of BERT's pooled output. The head is randomly initialized, while BERT's weights are pre-trained.

---

### Step 3: Tokenization Pipeline

Convert raw text into token IDs, attention masks, and apply padding/truncation.

```python
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# Remove raw text — no longer needed after tokenization
train_data = train_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])
```

**Tokenization flow:**

```
┌──────────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Raw Text         │    │  BERT         │    │  Model Input     │
│                  │───►│  Tokenizer    │───►│                  │
│  "Great movie!"  │    │               │    │  input_ids       │
│                  │    │  • Lowercase  │    │  attention_mask  │
│                  │    │  • Subword    │    │  labels          │
│                  │    │  • Pad/Trunc  │    │                  │
└──────────────────┘    └───────────────┘    └──────────────────┘
```

---

### Step 4: Training Configuration

Define hyperparameters and training behavior using `TrainingArguments`.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./finetuned_model",
    evaluation_strategy="epoch",      # Evaluate at end of each epoch
    num_train_epochs=3,               # Total training epochs
    learning_rate=2e-5,               # AdamW learning rate
    per_device_train_batch_size=8,    # Batch size per GPU
    per_device_eval_batch_size=8,     # Eval batch size per GPU
    weight_decay=0.01,                # L2 regularization
    save_strategy="epoch",            # Save checkpoint each epoch
    logging_steps=50,                 # Log metrics every 50 steps
    load_best_model_at_end=True       # Keep best checkpoint
)
```

**Key argument breakdown:**

| Argument | Role |
|----------|------|
| `output_dir` | Directory for checkpoints and final model |
| `evaluation_strategy` | When to run evaluation (`"epoch"`, `"steps"`, `"no"`) |
| `learning_rate` | Controls step size — too high diverges, too slow converges |
| `weight_decay` | Regularization strength to reduce overfitting |
| `save_strategy` | Checkpoint frequency — matches `evaluation_strategy` for best results |
| `logging_steps` | Frequency of training loss logging |

---

### Step 5: Building the Trainer

The `Trainer` class orchestrates the training loop, evaluation, checkpointing, and logging.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)
```

**Trainer internal flow:**

```
┌─────────────────────────────────────────────────────┐
│                   TRAINER LOOP                      │
│                                                     │
│  for each epoch:                                    │
│    ┌──────────┐   ┌───────────┐   ┌──────────────┐ │
│    │  FORWARD │──►│  COMPUTE  │──►│  BACKWARD    │ │
│    │  PASS    │   │  LOSS     │   │  PASS (grad) │ │
│    └──────────┘   └───────────┘   └──────────────┘ │
│         │                               │           │
│         │    ┌───────────┐              │           │
│         └───►│ EVALUATE  │◄─────────────┘           │
│              │ on test   │  (optimizer step)         │
│              └───────────┘                          │
│                    │                                │
│                    ▼                                │
│              ┌───────────┐                          │
│              │  SAVE     │                          │
│              │  CHECKPOINT│                         │
│              └───────────┘                          │
└─────────────────────────────────────────────────────┘
```

---

### Step 6: Training & Evaluation

#### Training

```python
trainer.train()
```

#### Evaluation with Multiple Metrics

```python
import numpy as np
import evaluate

outputs = trainer.predict(test_data)
predicted_labels = np.argmax(outputs.predictions, axis=1)
true_labels = outputs.label_ids

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

accuracy = accuracy_metric.compute(predictions=predicted_labels, references=true_labels)
f1 = f1_metric.compute(predictions=predicted_labels, references=true_labels)
precision = precision_metric.compute(predictions=predicted_labels, references=true_labels)
recall = recall_metric.compute(predictions=predicted_labels, references=true_labels)

print(f"Accuracy:  {accuracy['accuracy']:.4f}")
print(f"F1 Score:  {f1['f1']:.4f}")
print(f"Precision: {precision['precision']:.4f}")
print(f"Recall:    {recall['recall']:.4f}")
```

**Metric definitions:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were found |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

---

### Step 7: Production Inference Pipeline

A self-contained class for deploying the fine-tuned model in production.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentPipeline:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.label_map = {0: "Negative", 1: "Positive"}

    def predict(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()
        return [
            {"text": text, "sentiment": self.label_map[pred]}
            for text, pred in zip(texts, predictions)
        ]


# Usage
pipeline = SentimentPipeline("./finetuned_model")
results = pipeline.predict([
    "This movie was absolutely fantastic!",
    "Terrible plot and awful acting.",
    "A masterpiece of cinema."
])
for result in results:
    print(f"  {result['sentiment']:>8s}  │  {result['text']}")
```

**Production pipeline flow:**

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw Text    │───►│  Tokenize     │───►│  Model       │───►│  Argmax    │
│  (list)      │    │  (no grad)    │    │  Inference   │    │  + Label   │
└──────────────┘    └───────────────┘    └──────────────┘    └────────────┘
```

---

## 8. Hyperparameter Guide

| Hyperparameter | Typical Range | Recommended Start | Notes |
|----------------|---------------|-------------------|-------|
| `learning_rate` | `1e-5` – `5e-5` | `2e-5` | Lower for larger models; use warmup |
| `batch_size` | `4` – `32` | `8` – `16` | Limited by GPU memory |
| `num_epochs` | `2` – `5` | `3` | More epochs risk overfitting on small data |
| `max_length` | `32` – `512` | `128` | Depends on task; shorter = faster |
| `weight_decay` | `0.0` – `0.1` | `0.01` | Higher for small datasets |
| `warmup_steps` | `0` – `1000` | `10%` of total | Prevents early instability |
| `adam_epsilon` | `1e-8` – `1e-6` | `1e-8` | Rarely needs tuning |

---

## 9. Best Practices

### Data Preparation

- **Shuffle your data** before splitting to avoid label-order bias
- **Use stratified splits** for imbalanced datasets
- **Start with a subset** to validate the pipeline before full training
- **Inspect tokenized outputs** to verify padding and truncation behavior

### Training

- **Use `2e-5` as a default learning rate** — it works well for most BERT-class models
- **Match `save_strategy` with `evaluation_strategy`** to enable `load_best_model_at_end`
- **Enable mixed precision** (`fp16=True`) to halve GPU memory usage with minimal accuracy loss
- **Monitor train vs. eval loss** to detect overfitting early
- **Use `weight_decay`** (0.01–0.1) for regularization on small datasets

### Evaluation

- **Don't rely on accuracy alone** — use F1 for imbalanced classes
- **Compute a confusion matrix** to understand per-class errors
- **Test on truly unseen data** — hold out a separate test set

### Production

- **Wrap inference in a class** with `model.eval()` and `torch.no_grad()`
- **Reuse the tokenizer** from training — mismatched tokenization breaks predictions
- **Batch predictions** for throughput gains over single-sample inference
- **Version your models** and track which dataset/code produced each checkpoint

---

## 10. Quick Reference

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FINE-TUNING QUICK REFERENCE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. LOAD DATA       load_dataset("imdb", split="train")            │
│  2. LOAD MODEL      AutoModelForSequenceClassification             │
│  3. TOKENIZE        tokenizer(text, padding, truncation)            │
│  4. CONFIGURE       TrainingArguments(lr=2e-5, epochs=3)           │
│  5. TRAIN           Trainer(model, args, data).train()              │
│  6. EVALUATE        trainer.predict() + metrics                     │
│  7. DEPLOY          Custom inference pipeline class                 │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Libraries: transformers · datasets · evaluate · torch · numpy     │
│  Model:     bert-base-uncased (110M params)                        │
│  Task:      Binary Sentiment Classification (IMDB)                 │
└─────────────────────────────────────────────────────────────────────┘
```
