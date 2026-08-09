# Analyzing Car Reviews with LLMs

> An end-to-end NLP project that applies **4 production-style LLM pipelines** to real car review data — covering sentiment classification, translation, extractive question answering, and summarization.

---

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Project Overview](#1-project-overview) | What it demonstrates, skills covered |
| 2 | [Dataset Description](#2-dataset-description) | Data format and structure |
| 3 | [Project Architecture](#3-project-architecture) | Pipeline diagram and flow |
| 4 | [Task 1: Sentiment Classification](#4-task-1-sentiment-classification) | Model, code, evaluation |
| 5 | [Task 2: Translation + BLEU](#5-task-2-translation--bleu-evaluation) | Translation and scoring |
| 6 | [Task 3: Extractive QA](#6-task-3-extractive-question-answering) | Logit extraction, diagram |
| 7 | [Task 4: Summarization](#7-task-4-summarization) | Abstractive summarization |
| 8 | [Models Used](#8-models-used) | Reference table |
| 9 | [Results Interpretation](#9-results-interpretation-guide) | How to read the metrics |
| 10 | [Extensions & Improvements](#10-extensions--improvements) | Next steps |
| 11 | [Quick Reference](#11-quick-reference) | Cheat sheet |

---

## 1. Project Overview

This project is a **hands-on demonstration** of how to apply pre-trained Transformer models to real-world text data using the Hugging Face ecosystem.

### Skills Covered

| Category | Topics |
|----------|--------|
| **Transformers** | `pipeline` API, `AutoTokenizer`, `AutoModelForQuestionAnswering` |
| **Inference** | Batch classification, single-pass translation, logit-based span extraction |
| **Evaluation** | Accuracy, F1 Score, BLEU Score via the `evaluate` library |
| **Data Handling** | CSV parsing (semicolon-delimited), text file I/O |
| **Deep Learning** | `torch.no_grad()` context, `argmax` over logits, token ID decoding |

### Key Dependencies

```
transformers
torch
pandas
evaluate
```

---

## 2. Dataset Description

### Source Files

| File | Format | Purpose |
|------|--------|---------|
| `data/car_reviews.csv` | Semicolon-delimited CSV | Car reviews with sentiment labels |
| `data/reference_translations.txt` | Plain text (one per line) | Reference Spanish translations for BLEU |

### CSV Schema

```
+----------------------------------------------+-----------+
|  Review                                      |  Class    |
+----------------------------------------------+-----------+
|  "I love this car, it drives smoothly..."    |  POSITIVE |
|  "The engine failed after two weeks..."      |  NEGATIVE |
|  ...                                         |  ...      |
+----------------------------------------------+-----------+
```

| Column | Type | Description |
|--------|------|-------------|
| `Review` | `str` | Free-text car review |
| `Class` | `str` | Sentiment label — `POSITIVE` or `NEGATIVE` |

### Loading

```python
import pandas as pd

df = pd.read_csv("data/car_reviews.csv", delimiter=";")
reviews     = df['Review'].tolist()
real_labels = df['Class'].tolist()
```

---

## 3. Project Architecture

```
                        ┌─────────────────────────┐
                        │   data/car_reviews.csv   │
                        │   (semicolon-delimited)  │
                        └────────────┬────────────┘
                                     │
                          df['Review'], df['Class']
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  TASK 1           │  │  TASK 2           │  │  TASK 3 & 4      │
   │  Sentiment        │  │  Translation      │  │  QA + Summary    │
   │  Classification   │  │  (EN → ES)        │  │                  │
   ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
   │ distilbert-base   │  │ Helsinki-NLP      │  │ deepset/minilm   │
   │ -uncased-finetuned│  │ /opus-mt-en-es   │  │ -uncased-squad2  │
   │ -sst-2-english    │  │                   │  │ cnicu/t5-small   │
   ├──────────────────┤  ├──────────────────┤  │ -booksum         │
   │ Pipeline:         │  │ Pipeline:         │  ├──────────────────┤
   │ sentiment-analysis│  │ translation       │  │ AutoTokenizer +  │
   ├──────────────────┤  ├──────────────────┤  │ AutoModelForQA   │
   │ Metrics:          │  │ Metric:           │  │ Pipeline:        │
   │ Accuracy, F1      │  │ BLEU              │  │ summarization    │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ Predicted labels  │  │ Translated text   │  │ Extracted answer │
   │ + Accuracy / F1   │  │ + BLEU score      │  │ + Summary text   │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Execution Pipeline Flow

```
 Step 1          Step 2           Step 3           Step 4
 ───────        ────────         ────────         ────────
 Load CSV ──► Classify ──────► Translate ──────► QA Extract ──► Summarize
   │             │                  │                  │             │
   │             ▼                  ▼                  ▼             ▼
   │        Accuracy / F1      BLEU Score      Answer Span    Summary Text
   │             │                  │                  │             │
   └─────────────┴──────────────────┴──────────────────┴─────────────┘
                              Console Output
```

---

## 4. Task 1: Sentiment Classification

### Goal

Classify each car review as **POSITIVE** or **NEGATIVE** using a fine-tuned DistilBERT model, then evaluate against ground-truth labels.

### Model

**`distilbert-base-uncased-finetuned-sst-2-english`** — a distilled (lightweight) version of BERT fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset.

### Implementation

```python
from transformers import pipeline
import evaluate

classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english'
)

predicted_labels = classifier(reviews)

for review, prediction, label in zip(reviews, predicted_labels, real_labels):
    print(f"Review: {review}\n"
          f"Actual Sentiment: {label}\n"
          f"Predicted Sentiment: {prediction['label']} "
          f"(Confidence: {prediction['score']:.4f})\n")
```

### Label Mapping & Evaluation

The pipeline returns string labels (`"POSITIVE"` / `"NEGATIVE"`). These must be mapped to binary integers for metric computation:

```
"POSITIVE"  ──►  1
"NEGATIVE"  ──►  0
```

```python
accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

references  = [1 if label == "POSITIVE" else 0 for label in real_labels]
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]

accuracy_result = accuracy.compute(references=references, predictions=predictions)
f1_result       = f1.compute(references=references, predictions=predictions)

print(f"Accuracy: {accuracy_result['accuracy']}")
print(f"F1 Score: {f1_result['f1']}")
```

### Metrics Explained

| Metric | Formula | When It Matters |
|--------|---------|-----------------|
| **Accuracy** | `(TP + TN) / Total` | Balanced datasets — overall correctness |
| **F1 Score** | `2 · (Precision · Recall) / (Precision + Recall)` | Imbalanced datasets — harmonic mean of precision and recall |

> **Why both?** Accuracy alone can be misleading with imbalanced classes. F1 accounts for false positives and false negatives, giving a more robust measure.

---

## 5. Task 2: Translation + BLEU Evaluation

### Goal

Translate the **first car review** from English to Spanish and measure translation quality against human references using the BLEU metric.

### Model

**`Helsinki-NLP/opus-mt-en-es`** — a Marian-based neural machine translation model from the Opus-MT project, trained on parallel EN→ES corpora.

### Implementation

```python
from transformers import pipeline
import evaluate

translator        = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translator(reviews[0], max_length=27)[0]['translation_text']

with open("data/reference_translations.txt", 'r') as file:
    references = [line.strip() for line in file.readlines()]

bleu       = evaluate.load("bleu")
bleu_score = bleu.compute(
    predictions=[translated_review],
    references=[references]
)
print(f"BLEU Score: {bleu_score['bleu']}")
```

### Translation Evaluation Flow

```
┌─────────────────────┐
│  Original Review     │
│  (English)           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│  opus-mt-en-es      │     │  data/reference_translations  │
│  (EN → ES Model)    │     │  (Human Spanish Translations) │
└─────────┬───────────┘     └──────────────┬───────────────┘
          │                                │
          ▼                                │
┌─────────────────────┐                    │
│  Model Translation   │                    │
│  (Spanish)           │                    │
└─────────┬───────────┘                    │
          │                                │
          └────────────┬───────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   BLEU Score     │
              │   (0.0 – 1.0)   │
              └─────────────────┘
```

### BLEU Score Explained

| Component | Description |
|-----------|-------------|
| **N-gram precision** | Counts matching n-grams (1-gram through 4-gram) between prediction and references |
| **Brevity penalty** | Penalizes translations shorter than the reference to discourage partial translations |
| **Range** | `0.0` (no match) to `1.0` (perfect match) |

| BLEU Range | Interpretation |
|------------|---------------|
| `< 0.1` | Almost no overlap |
| `0.1 – 0.3` | Low quality |
| `0.3 – 0.5` | Understandable |
| `0.5 – 0.7` | Good quality |
| `> 0.7` | Excellent (near-human) |

> **Note:** BLEU measures *n-gram overlap*, not semantic equivalence. A semantically correct translation using different words may score lower than expected.

---

## 6. Task 3: Extractive Question Answering

### Goal

Given a review as context and a question, extract the **exact answer span** from the text using start/end position logits.

### Model

**`deepset/minilm-uncased-squad2`** — a compact MiniLM model fine-tuned on SQuAD 2.0, capable of handling unanswerable questions.

### Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_ckp = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckp)
model     = AutoModelForQuestionAnswering.from_pretrained(model_ckp)

context  = reviews[1]
question = "What did he like about the brand?"

inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start_idx   = torch.argmax(outputs.start_logits)
end_idx     = torch.argmax(outputs.end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer      = tokenizer.decode(answer_span)

print("Answer:", answer)
```

### How Start/End Logits Work

```
Input Tokens:  [CLS] What did he like about the brand? [SEP] The ride quality ... 
                                                                       is excellent ...

                ┌──────────────────────────────────────────────────────────────────┐
Start Logits:  │ -3.2  -1.1  -0.4  0.2  -0.8  -1.5  -2.0  -4.1  │  ...  5.1  ... │
                └──────────────────────────────────────────────────────────────────┘
                                                                 ▲
                                                          argmax picks this

                ┌──────────────────────────────────────────────────────────────────┐
End   Logits:  │ -4.0  -2.3  -1.8  -0.5  -0.1  -0.3  -1.2  -3.7  │ 4.8  ...  2.1 │
                └──────────────────────────────────────────────────────────────────┘
                                                                         ▲
                                                                  argmax picks this

Answer Span = tokens[start_idx : end_idx + 1]  ──►  decode  ──►  "excellent"
```

### Extraction Pipeline

```
┌────────────────┐     ┌───────────────┐     ┌────────────────┐
│  Question +    │────►│  Tokenizer     │────►│  Model Forward │
│  Context       │     │  (subwords)    │     │  Pass          │
└────────────────┘     └───────────────┘     └───────┬────────┘
                                                      │
                                              ┌───────┴────────┐
                                              │                │
                                              ▼                ▼
                                     start_logits       end_logits
                                              │                │
                                              ▼                ▼
                                      argmax(start)      argmax(end)
                                              │                │
                                              └───────┬────────┘
                                                      │
                                                      ▼
                                            answer_span [s:e]
                                                      │
                                                      ▼
                                              tokenizer.decode()
                                                      │
                                                      ▼
                                              Final Answer String
```

### Why `torch.no_grad()`?

Disables gradient computation during inference, which:

- **Reduces memory usage** — no need to store activations for backpropagation
- **Speeds up computation** — skips gradient-related operations
- **Is standard practice** for inference-only passes

---

## 7. Task 4: Summarization

### Goal

Generate a concise **abstractive summary** of the last car review using a T5-based model.

### Model

**`cnicu/t5-small-booksum`** — T5-small fine-tuned on the BookSum dataset, capable of generating coherent abstractive summaries.

### Implementation

```python
from transformers import pipeline

text_to_summarize = reviews[-1]

summarizer      = pipeline("summarization", model="cnicu/t5-small-booksum")
outputs         = summarizer(text_to_summarize, max_length=53)
summarized_text = outputs[0]['summary_text']

print(f"Original:\n{text_to_summarize}")
print(f"Summary:\n{summarized_text}")
```

### Model Choice Rationale

| Factor | Detail |
|--------|--------|
| **Architecture** | T5 (Text-to-Text Transfer Transformer) — encoder-decoder |
| **Size** | `small` variant (~60M params) — fast inference |
| **Training** | Fine-tuned on BookSum — strong abstractive capability |
| `max_length` | Set to 53 tokens to produce concise summaries |

### Extractive vs. Abstractive Summarization

```
Extractive:  Copies key sentences verbatim from the source.
             ──► Output is a subset of the input.

Abstractive: Generates new text that captures the meaning.
             ──► Output may contain words not in the input.

             This project uses ABSTRACTIVE (T5-based).
```

---

## 8. Models Used

| Model | Task | Architecture | ~Params | Source |
|-------|------|-------------|---------|--------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment Classification | DistilBERT (encoder-only) | ~66M | Hugging Face |
| `Helsinki-NLP/opus-mt-en-es` | EN→ES Translation | MarianMT (encoder-decoder) | ~75M | University of Helsinki |
| `deepset/minilm-uncased-squad2` | Extractive QA | MiniLM (encoder-only) | ~22M | deepset AI |
| `cnicu/t5-small-booksum` | Summarization | T5-Small (encoder-decoder) | ~60M | Community |

### Pipeline vs. Manual Inference

| Approach | Used In | Pros | Cons |
|----------|---------|------|------|
| `pipeline()` | Tasks 1, 2, 4 | Minimal code, handles preprocessing/postprocessing | Less control over internals |
| `AutoTokenizer` + `AutoModel` | Task 3 | Full control over tokenization and logits | More boilerplate |

---

## 9. Results Interpretation Guide

### Task 1 — Sentiment Classification

```
Accuracy ──── Overall proportion of correctly classified reviews.
              • 1.0  = all correct
              • 0.5  = random (binary)

F1 Score ──── Harmonic mean of precision and recall.
              • > 0.9  = excellent
              • 0.7–0.9 = good
              • < 0.7  = needs improvement
```

### Task 2 — Translation (BLEU)

```
BLEU Score ── N-gram overlap between model output and references.
              • > 0.5   = high-quality translation
              • 0.3–0.5 = adequate
              • < 0.3   = significant deviation
```

### Task 3 — Extractive QA

```
Answer ────── The extracted span is the model's best guess for the answer.
              Check:
              • Does it directly answer the question?
              • Is it a valid substring of the context?
              • Empty/CLS token may indicate "unanswerable" (SQuAD 2.0 feature).
```

### Task 4 — Summarization

```
Summary ───── Evaluate qualitatively:
              • Does it capture the main points?
              • Is it fluent and grammatically correct?
              • Is it significantly shorter than the original?
              • Does it avoid hallucinated information?
```

---

## 10. Extensions & Improvements

### Evaluation Enhancements

| Current | Improvement | Library / Method |
|---------|------------|-----------------|
| Accuracy + F1 | Per-class precision, recall, confusion matrix | `sklearn.metrics` |
| BLEU | chrF, COMET, BERTScore | `evaluate`, `bert_score` |
| Manual inspection | ROUGE score for summaries | `evaluate.load("rouge")` |
| Single reference | Multiple reference translations | Extend `reference_translations.txt` |

### Model Upgrades

| Task | Current Model | Upgrade Option | Benefit |
|------|--------------|----------------|---------|
| Sentiment | DistilBERT | `roberta-base` or `deberta-v3` | Higher accuracy |
| Translation | opus-mt-en-es | `facebook/nllb-200-distilled-600M` | Multilingual |
| QA | MiniLM | `deepset/roberta-base-squad2` | Better span extraction |
| Summarization | t5-small-booksum | `facebook/bart-large-cnn` | More fluent summaries |

### Infrastructure

- **Batch processing** — Add chunking for very long reviews exceeding model context windows
- **GPU acceleration** — Use `device=0` in pipeline calls for CUDA-enabled machines
- **Caching** — Cache model downloads and inference results with `datasets` or local storage
- **API deployment** — Wrap each pipeline in a FastAPI endpoint for production use
- **Testing** — Add unit tests for each task with known inputs/expected outputs

---

## 11. Quick Reference

### One-Line Pipeline Calls

```python
from transformers import pipeline

# Sentiment
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

# Summarization
summarizer = pipeline("summarization", model="cnicu/t5-small-booksum")
```

### Manual Inference Pattern (QA)

```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
model     = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer = tokenizer.decode(
    inputs["input_ids"][0][
        torch.argmax(outputs.start_logits) : torch.argmax(outputs.end_logits) + 1
    ]
)
```

### Metric Quick Reference

```python
import evaluate

# Accuracy
accuracy = evaluate.load("accuracy")
result   = accuracy.compute(references=[0, 1, 1], predictions=[0, 1, 0])

# F1 Score (binary)
f1     = evaluate.load("f1")
result = f1.compute(references=[0, 1, 1], predictions=[0, 1, 0])

# BLEU Score
bleu   = evaluate.load("bleu")
result = bleu.compute(
    predictions=["translation text"],
    references=[["reference one", "reference two"]]
)
```

### File Structure

```
project/
├── data/
│   ├── car_reviews.csv              # Semicolon-delimited reviews
│   └── reference_translations.txt   # Spanish reference translations
└── Project-Analyzing_car_reviews_with_llms.py   # Main script
```
