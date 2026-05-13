# Hugging Face Datasets

## Overview

The `datasets` library by Hugging Face provides easy access to thousands of publicly available datasets for NLP, computer vision, audio, and more — all with a unified, efficient API. Datasets are stored in memory-mapped files (Arrow format), making them fast and memory-efficient even for large collections.

---

## Installation

```bash
pip install datasets
```

---

## Loading a Dataset

```python
from datasets import load_dataset

# Load a specific split of a dataset from the Hub
# Each dataset has splits — check the dataset card for available splits
my_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

# Display dataset details
print(my_dataset)
```

**Output:**
```
Dataset({
    features: ['question', 'options', 'answer', 'category', ...],
    num_rows: 12032
})
```

---

## Dataset Structure

A `Dataset` object behaves like a smart dictionary of lists:

```python
# Access columns (features)
print(my_dataset.column_names)   # ['question', 'options', 'answer', ...]
print(my_dataset.features)       # Schema with types

# Access rows like a list
first_row = my_dataset[0]
print(first_row['question'])

# Access columns like a dict
all_questions = my_dataset['question']  # Returns a list of all questions
```

---

## Common Splits

| Split | Description | Typical Use |
|-------|-------------|-------------|
| `"train"` | Training data | Fine-tuning models |
| `"validation"` / `"val"` | Validation data | Hyperparameter tuning |
| `"test"` | Test data | Final evaluation |

Not all datasets have all splits — always check the dataset card on [huggingface.co/datasets](https://huggingface.co/datasets).

---

## Loading Multiple Splits

```python
from datasets import load_dataset

# Load all splits at once (returns a DatasetDict)
dataset_dict = load_dataset("imdb")
print(dataset_dict)
# DatasetDict({
#     train: Dataset({num_rows: 25000}),
#     test: Dataset({num_rows: 25000}),
#     unsupervised: Dataset({num_rows: 50000})
# })

train_data = dataset_dict["train"]
test_data  = dataset_dict["test"]

# Or load one split at a time
train_data = load_dataset("imdb", split="train")
test_data  = load_dataset("imdb", split="test")
```

---

## Working with Datasets

### Inspect
```python
# View a sample
print(my_dataset[0])

# View column types
print(my_dataset.features)

# Number of rows
print(len(my_dataset))

# Describe (if numeric)
print(my_dataset.info)
```

### Filter
```python
# Keep only positive examples
positive_data = my_dataset.filter(lambda example: example["label"] == 1)
```

### Map (Transform)
```python
# Apply a function to every example
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized = my_dataset.map(tokenize, batched=True)
```

### Select
```python
# Take first 100 samples
small_dataset = my_dataset.select(range(100))
```

### Shuffle and Split
```python
# Create train/test split
split = my_dataset.train_test_split(test_size=0.2, seed=42)
train = split["train"]
test  = split["test"]
```

### Sharding (for large datasets)
```python
# Split into N shards and use just one (useful for quick experiments)
small_train = train_data.shard(num_shards=10, index=0)
# This gives you 1/10th of the data
```

---

## Working with Audio Datasets

```python
from datasets import load_dataset, Audio

# Load dataset
dataset = load_dataset("CSTR-Edinburgh/vctk")["train"]

# Resample audio to 16kHz (standard for speech models)
dataset = dataset.cast_column(
    "audio",
    Audio(sampling_rate=16_000)
)

# Access audio data
sample = dataset[0]["audio"]
print(sample["array"])         # NumPy array of audio waveform
print(sample["sampling_rate"]) # 16000
print(sample["path"])          # File path (if available)
```

---

## Working with Image Datasets

```python
from datasets import load_dataset

# Load an image dataset
dataset = load_dataset("nlphuji/flickr30k")["test"]

# Access images as PIL objects
image = dataset[0]["image"]
print(image.size)  # (width, height) in pixels
image.show()
```

---

## Notable Public Datasets

| Dataset | Task | Load String |
|---------|------|-------------|
| IMDB | Sentiment Analysis | `"imdb"` |
| SQuAD | QA | `"squad"` |
| COCO | Image Captioning | `"HuggingFaceM4/COCO"` |
| MMLU | Knowledge Eval | `"TIGER-Lab/MMLU-Pro"` |
| VoxPopuli | Speech (multilingual) | `"facebook/voxpopuli"` |
| VCTK | Speech (TTS) | `"CSTR-Edinburgh/vctk"` |
| Flickr30k | Images + Captions | `"nlphuji/flickr30k"` |
| Common Voice | Speech (multilingual) | `"mozilla-foundation/common_voice_13_0"` |

---

## DatasetDict vs. Dataset

```python
# DatasetDict: multiple splits
from datasets import DatasetDict

dataset_dict = load_dataset("imdb")
print(type(dataset_dict))   # <class 'datasets.dataset_dict.DatasetDict'>

# Accessing a split
train = dataset_dict["train"]
print(type(train))          # <class 'datasets.arrow_dataset.Dataset'>
```

---

## Best Practices

- ✅ Always check the dataset card (on HuggingFace Hub) for split names and features
- ✅ Use `.shard()` for quick experiments on large datasets
- ✅ Use `batched=True` in `.map()` for faster processing
- ✅ Use `cast_column("audio", Audio(sampling_rate=16000))` before feeding to speech models
- ✅ Store preprocessed datasets with `.save_to_disk()` to avoid reprocessing

```python
# Cache preprocessed data
tokenized_dataset.save_to_disk("./preprocessed_data")

# Load it back later
from datasets import load_from_disk
dataset = load_from_disk("./preprocessed_data")
```
