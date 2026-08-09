# Hugging Face Pipelines

## Overview

The `pipeline()` function from Hugging Face `transformers` is the **fastest way** to use state-of-the-art AI models with just a few lines of code. It handles model loading, tokenization, preprocessing, inference, and post-processing automatically — letting you focus on results rather than implementation details.

---

## How Pipelines Work

```
                    ┌─────────────────────────────────────────────┐
                    │              pipeline("task", model=...)     │
                    │                                             │
Input (text/image)  │  Tokenizer/Processor → Model → Decoder     │  Output
──────────────────► │     (preprocessing)  (inference) (post)    │ ──────────►
                    └─────────────────────────────────────────────┘
```

A pipeline handles:
1. **Loading** the model and tokenizer from Hugging Face Hub
2. **Preprocessing** the input (tokenization, feature extraction)
3. **Inference** (forward pass through the model)
4. **Post-processing** (decoding, formatting the result)

---

## Installation

```bash
pip install transformers datasets torch
pip install pypdf  # For document Q&A
```

---

## Available Pipeline Tasks

| Task | `pipeline()` task name | Example Use |
|------|------------------------|-------------|
| Text generation | `"text-generation"` | Write code, stories, completions |
| Translation | `"translation_XX_to_YY"` | Translate between languages |
| Sentiment analysis | `"sentiment-analysis"` | Classify text as positive/negative |
| Text classification | `"text-classification"` | Custom categories |
| Zero-shot classification | `"zero-shot-classification"` | No training data needed |
| Summarization | `"summarization"` | Condense long documents |
| Question answering | `"question-answering"` | Extract answers from documents |
| Image classification | `"image-classification"` | Label images |
| Image-to-text | `"image-to-text"` | Caption images |
| ASR | `"automatic-speech-recognition"` | Transcribe audio |
| Text-to-audio | `"text-to-audio"` | Generate music/sounds |

---

## 1. Text Generation

```python
from transformers import pipeline

# Load GPT-2 for text generation
generation_pipeline = pipeline(
    "text-generation",
    model="openai-community/gpt2"
)

# Generate text
results = generation_pipeline(
    "What if AI",
    max_length=100,
    pad_token_id=generation_pipeline.tokenizer.eos_token_id
)

for result in results:
    print(result['generated_text'])
```

**Key Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `max_length` | Maximum total token length of output | `100` |
| `max_new_tokens` | Max tokens to generate (doesn't include input) | `50` |
| `num_return_sequences` | How many different completions to generate | `3` |
| `temperature` | Creativity (0=deterministic, 1=normal, 2=chaotic) | `0.8` |

---

## 2. Translation

```python
from transformers import pipeline

# Spanish to English translation
translator = pipeline(
    task="translation_es_to_en",
    model="Helsinki-NLP/opus-mt-es-en"
)

spanish_text = "Este curso sobre LLMs se está poniendo muy interesante"
translations = translator(spanish_text, clean_up_tokenization_spaces=True)

print(translations[0]["translation_text"])
# → "This course on LLMs is getting very interesting"
```

**Available language pairs:**
- `Helsinki-NLP/opus-mt-en-es` — English → Spanish
- `Helsinki-NLP/opus-mt-es-en` — Spanish → English
- `Helsinki-NLP/opus-mt-en-fr` — English → French
- `Helsinki-NLP/opus-mt-de-en` — German → English

---

## 3. Sentiment Analysis (Text Classification)

```python
from transformers import pipeline

classification_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Classify single text
result = classification_pipeline("Wi-Fi is slower than a snail.")
print(result)
# → [{'label': 'NEGATIVE', 'score': 0.9998}]

# Classify multiple texts
texts = [
    "This product is amazing!",
    "Terrible customer service.",
    "It was okay, nothing special."
]
results = classification_pipeline(texts)
for text, res in zip(texts, results):
    print(f"{res['label']} ({res['score']:.2%}): {text}")
```

---

## 4. Zero-Shot Classification

> **No training data required.** Provide candidate labels and the model figures it out.

```python
from transformers import pipeline

category_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text = "Hey Datacamp, we would like to feature you in our next newsletter. Interested?"
categories = ["marketing", "support", "sales"]

output = category_pipeline(text, categories)

print(f"Top label: {output['labels'][0]}")
print(f"Score: {output['scores'][0]:.2%}")

# Full breakdown:
for label, score in zip(output['labels'], output['scores']):
    print(f"  {label}: {score:.2%}")
```

**Output:**
```
Top label: marketing
Score: 95.2%
  marketing: 95.2%
  sales: 3.1%
  support: 1.7%
```

---

## 5. QNLI (Question-Natural Language Inference)

Check if a given text **answers** a question:

```python
from transformers import pipeline

qnli_pipeline = pipeline(
    "text-classification",
    model="cross-encoder/qnli-electra-base"
)

result = qnli_pipeline({
    "text": "Where is Seattle located?",
    "text_pair": "Seattle is located in Washington state."
})

print(result)
# → [{'label': 'entailment', 'score': 0.997}]
# "entailment" means the text_pair answers the question
```

---

## 6. Text Summarization

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = """
Data Science is an interdisciplinary field that uses scientific methods, 
algorithms, and systems to extract knowledge from structured and unstructured data. 
It combines statistics, computer science, and domain expertise to solve complex problems.
This field has grown tremendously over the past decade, with applications in healthcare,
finance, marketing, and technology sectors worldwide.
"""

summary = summarizer(
    text,
    min_new_tokens=10,
    max_new_tokens=50,
    do_sample=False
)

print(summary[0]['summary_text'])
```

---

## 7. Document Question Answering

Extract answers from documents (PDFs, text files):

```python
from pypdf import PdfReader
from transformers import pipeline

# Step 1: Extract text from PDF
reader = PdfReader("US-Employee_Policy.pdf")
document_text = ""
for page in reader.pages:
    document_text += page.extract_text()

# Step 2: Load QA pipeline
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Step 3: Ask questions
question = "How many volunteer days are offered annually?"
result = qa_pipeline(question=question, context=document_text)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2%}")
print(f"Found at character: {result['start']}–{result['end']}")
```

---

## 8. Using AutoClasses for More Control

`AutoClasses` give you **direct access** to the model and tokenizer for custom workflows:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load model and tokenizer separately
my_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
my_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Create a pipeline using your manually loaded components
my_pipeline = pipeline(
    task="sentiment-analysis",
    model=my_model,
    tokenizer=my_tokenizer
)

output = my_pipeline("This course is pretty good, I guess.")
print(f"Sentiment: {output[0]['label']} ({output[0]['score']:.2%})")
```

### Pipelines vs. AutoClasses

| Feature | `pipeline()` | `AutoClasses` |
|---------|-------------|---------------|
| Speed to implement | ⚡ Very fast | Slower |
| Customization | Limited | Full control |
| Preprocessing control | Automatic | Manual |
| Model swapping | Easy | Requires code changes |
| Fine-tuning support | No | Yes |
| Best for | Quick experiments | Production, custom tasks |

---

## Pipeline with Custom Model (from Hub)

```python
from transformers import pipeline
from huggingface_hub import HfApi

api = HfApi()

# Discover top text-to-image models
available_models = list(api.list_models(task="text-to-image", limit=5))

# Use the top model in a pipeline
pipe = pipeline(
    "text-to-image",
    model=available_models[0].id
)

print(f"Using model: {available_models[0].id}")
result = pipe("A sunset over mountains")
```

---

## Best Practices

### ✅ Do's
- **Cache models locally** — they download automatically on first use and are cached in `~/.cache/huggingface/`
- **Use `torch.no_grad()`** for inference (pipelines handle this automatically)
- **Specify the model** when reproducibility matters — don't rely on the default
- **Batch inputs** for better throughput: `pipeline(["text1", "text2", ...])`

### ❌ Don'ts
- Don't use CPU for large models if you have a GPU available
- Don't load the same model multiple times in one script
- Don't use summarization pipeline on very short texts (< 50 words)

---

## Quick Reference Card

```python
# Text generation
gen = pipeline("text-generation", model="gpt2")
gen("Start of story")

# Translation
tr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
tr("Hello world")

# Sentiment
clf = pipeline("sentiment-analysis")  # Uses default model
clf("I love this!")

# Zero-shot
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
zsc("text here", candidate_labels=["sports", "politics", "tech"])

# Summarization
summ = pipeline("summarization", model="facebook/bart-large-cnn")
summ("Long text here...", max_new_tokens=50)

# QA
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
qa(question="What is X?", context="X is ...")
```
