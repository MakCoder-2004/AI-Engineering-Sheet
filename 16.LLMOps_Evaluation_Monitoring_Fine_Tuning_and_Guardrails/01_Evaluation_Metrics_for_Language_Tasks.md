# Metrics for Language Tasks

> A comprehensive guide to evaluating LLM outputs across text classification, generation, summarization, translation, question answering, and bias detection.

---

## 1. Overview: Why Evaluation Matters

Evaluating language model outputs is fundamental to building reliable NLP systems. Without rigorous metrics, we cannot:

- **Compare models** objectively against one another
- **Detect regressions** when updating prompts, fine-tuning, or switching models
- **Ensure safety** by measuring toxicity, bias, and hallucination rates
- **Justify deployment** decisions with quantifiable evidence

Every NLP task demands a tailored evaluation strategy — a single metric does not fit all.

```
┌─────────────────────────────────────────────────────┐
│               Evaluation Pipeline                    │
│                                                     │
│   Model Output ──► Metric Selection ──► Scoring     │
│                        │                             │
│                        ▼                             │
│              ┌─────────────────┐                     │
│              │ Task-Specific   │                     │
│              │ Metric Suite    │                     │
│              └────────┬────────┘                     │
│                       ▼                              │
│            Quantitative Report                       │
└─────────────────────────────────────────────────────┘
```

---

## 2. Task → Metric Mapping

| Task | Primary Metrics | Notes |
|---|---|---|
| **Text Classification** | Accuracy, F1 Score | Handles class imbalance via F1 |
| **Text Generation** | Perplexity, BLEU | Perplexity measures fluency; BLEU measures overlap |
| **Summarization** | ROUGE | Captures n-gram recall against reference |
| **Translation** | BLEU, METEOR | METEOR handles synonyms and morphology |
| **QA (Extractive)** | Exact Match, F1 | Span-level token overlap |
| **QA (Generative)** | BLEU, ROUGE | Evaluates free-form answer quality |
| **Bias / Safety Audit** | Toxicity, Regard | Measures harmful or skewed outputs |

```
                    ┌──────────────┐
                    │  Your Task?  │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Classification    Generation        Summarization
   ┌─────────┐     ┌──────────┐     ┌──────────────┐
   │Accuracy │     │Perplexity│     │   ROUGE-1    │
   │F1 Score │     │  BLEU    │     │   ROUGE-2    │
   └─────────┘     └──────────┘     │   ROUGE-L    │
                                     └──────────────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Translation        QA (Ext.)      QA (Gen.)
    ┌──────────┐    ┌─────────────┐  ┌──────────┐
    │  BLEU    │    │ Exact Match │  │  BLEU    │
    │ METEOR   │    │   F1 Score  │  │  ROUGE   │
    └──────────┘    └─────────────┘  └──────────┘
```

---

## 3. Classification Metrics

Classification tasks (sentiment analysis, topic labeling, spam detection) require metrics that account for class imbalance.

### 3.1 Confusion Matrix

```
                    Predicted
                 ┌──────┬──────┐
                 │  Pos │  Neg │
              ┌──┼──────┼──────┤
    Actual    │P │  TP  │  FN  │
              │N │  FP  │  TN  │
              └──┴──────┴──────┘
```

| Term | Meaning |
|---|---|
| **TP** | True Positive — correctly predicted positive |
| **TN** | True Negative — correctly predicted negative |
| **FP** | False Positive — incorrectly predicted positive |
| **FN** | False Negative — incorrectly predicted negative |

### 3.2 Accuracy

**What it measures:** Overall proportion of correct predictions.

```
              TP + TN
Accuracy = ─────────────
            TP + TN + FP + FN
```

- **Strength:** Simple, interpretable
- **Weakness:** Misleading on imbalanced datasets (e.g., 95% class A → 95% accuracy by always predicting A)

### 3.3 Precision

**What it measures:** Of all samples predicted positive, how many were actually positive.

```
               TP
Precision = ───────
             TP + FP
```

- **Use when:** The cost of false positives is high (e.g., spam filtering)

### 3.4 Recall

**What it measures:** Of all actual positive samples, how many were correctly predicted.

```
              TP
Recall = ─────────
           TP + FN
```

- **Use when:** The cost of false negatives is high (e.g., disease detection)

### 3.5 F1 Score

**What it measures:** Harmonic mean of precision and recall — a single balanced metric.

```
                    2 × Precision × Recall
F1 Score = ──────────────────────────────────
            Precision + Recall
```

- **Use when:** You need a balance between precision and recall, especially with imbalanced classes

| Metric | Best When... | Sensitive To |
|---|---|---|
| Accuracy | Classes are balanced | Class imbalance |
| Precision | False positives are costly | Over-prediction |
| Recall | False negatives are costly | Under-prediction |
| F1 Score | Need balanced view | Both FP and FN |

### 3.6 Code Example

```python
import evaluate

accuracy  = evaluate.load("accuracy")
f1        = evaluate.load("f1")
precision = evaluate.load("precision")
recall    = evaluate.load("recall")

preds = [1, 0, 1, 1, 0, 1]
refs  = [1, 0, 0, 1, 0, 1]

print(accuracy.compute(predictions=preds, references=refs))
print(f1.compute(predictions=preds, references=refs))
print(precision.compute(predictions=preds, references=refs))
print(recall.compute(predictions=preds, references=refs))
```

---

## 4. Perplexity

### What It Measures

Perplexity quantifies how well a language model predicts a sequence of tokens. It is the **exponential of the average negative log-likelihood** of a tokenized sequence.

```
                     1   N
Perplexity = exp( ─── × Σ  -log P(wi | w1...wi-1) )
                     N  i=1
```

### How to Interpret

| Perplexity Range | Interpretation |
|---|---|
| **Low (1–20)** | Model is very confident; predictions closely match the text |
| **Medium (20–100)** | Reasonable model; acceptable for most tasks |
| **High (100+)** | Model is uncertain; poor fit to the data |

- **Lower is better.** A perplexity of 1 means perfect prediction.
- Perplexity is **task-agnostic** — useful for evaluating language model quality independently of a downstream task.

### Code Example

```python
import evaluate

perplexity = evaluate.load("perplexity", module_type="metric")

text = [
    "The cat sat on the mat.",
    "Artificial intelligence is transforming industries worldwide."
]

result = perplexity.compute(model_id="gpt2", predictions=text)
print(f"Mean Perplexity: {result['mean_perplexity']:.2f}")
```

> **Note:** Perplexity depends on the tokenizer. Comparing perplexity across models with different vocabularies can be misleading.

---

## 5. BLEU Score

### What It Measures

BLEU (**B**ilingual **E**valuation **U**nderstudy) measures the overlap of **n-grams** between a candidate (model output) and one or more reference texts. It was originally designed for machine translation.

### Formula

```
                       BP  W
BLEU = BP × exp( Σ   ── × log(pn) )
                      n=1

where:
  pn = modified n-gram precision for n-grams of length n
  W  = uniform weights (typically 1/4 for n = 1,2,3,4)
  BP = brevity penalty

         ⎧ 1                     if c > r
BP =     ⎨
         ⎩ exp(1 - r/c)         if c ≤ r

  c = length of candidate
  r = length of reference
```

- **n-gram precision:** The proportion of n-grams in the candidate that appear in the reference.
- **Brevity Penalty (BP):** Penalizes short translations that would otherwise achieve high precision by omitting words.

### When to Use

| Scenario | Recommended |
|---|---|
| Machine Translation evaluation | Yes |
| Text Generation evaluation | Yes (with caution) |
| Summarization | ROUGE is generally preferred |
| Open-ended dialogue | Not ideal (no single reference) |

### Code Example

```python
import evaluate

bleu = evaluate.load("bleu")

predictions  = ["The cat is on the mat"]
references   = [["The cat is sitting on the mat"]]

result = bleu.compute(predictions=predictions, references=references)
print(result)
# Keys: 'bleu', 'precisions', 'brevity_penalty', 'length_ratio', ...
```

### Limitations

- Does **not** consider semantics — synonyms are penalized
- Sensitive to **word order**
- Poor at evaluating **creative or diverse** outputs

---

## 6. ROUGE Score

### What It Measures

ROUGE (**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation) measures how much of the reference text is captured (recalled) by the model output. Primarily used for **summarization**.

### Variants

| Variant | What It Compares | Formula |
|---|---|---|
| **ROUGE-1** | Unigram (single word) overlap | `Overlap / Reference Unigrams` |
| **ROUGE-2** | Bigram (two-word) overlap | `Overlap / Reference Bigrams` |
| **ROUGE-L** | Longest Common Subsequence (LCS) | `LCS Length / Reference Length` |

```
ROUGE-1 Recall =    Matching Unigrams
                 ─────────────────────
                   Reference Unigrams

ROUGE-L Recall =       LCS Length
                 ─────────────────────
                   Reference Length
```

- **ROUGE-1:** Captures keyword overlap.
- **ROUGE-2:** Captures phrase-level quality.
- **ROUGE-L:** Captures structural similarity via the longest common subsequence — tolerant of insertions.

### Code Example

```python
import evaluate

rouge = evaluate.load("rouge")

predictions = ["The cat sat on the mat."]
references  = ["The cat is sitting on the mat."]

result = rouge.compute(predictions=predictions, references=references)
print(result)
# Keys: 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
```

### When to Use

| Scenario | Recommended |
|---|---|
| Summarization | Yes — primary metric |
| Generative QA | Yes — as a secondary metric |
| Translation | BLEU or METEOR preferred |
| Open-ended generation | Not ideal |

---

## 7. METEOR Score

### What It Measures

METEOR (**M**etric for **E**valuation of **T**ranslation with **E**xplicit **OR**dering) computes a **harmonic mean of precision and recall**, with **recall weighted higher** than precision. It also accounts for:

- **Synonymy** — matches via WordNet synonyms
- **Stemming** — matches words with the same root (e.g., "run" ≈ "running")

### Formula

```
                      P × R
F_mean = α × ────────────────
              α × P + (1 - α) × R      (typically α = 0.9, weighting recall)

Penalty = γ × (chunks / unigrams)β     (γ = 0.5, β = 3)

METEOR = F_mean × (1 - Penalty)
```

### Advantages Over BLEU

| Feature | BLEU | METEOR |
|---|---|---|
| Precision / Recall | Precision only | Both (recall weighted higher) |
| Synonym support | No | Yes (WordNet) |
| Stemming support | No | Yes |
| Correlation with humans | Moderate | Higher |
| Brevity penalty | Yes | Fragmentation penalty |

### Code Example

```python
import evaluate

meteor = evaluate.load("meteor")

predictions = ["He is happy."]
references  = [["He is very happy."]]

result = meteor.compute(predictions=predictions, references=references)
print(result)
# {'meteor': 0.94}
```

---

## 8. Exact Match

### What It Measures

Exact Match (EM) checks whether the predicted answer is **character-for-character identical** to the reference. It is the strictest evaluation metric.

### Formula

```
                1  if prediction == reference
EM =           ─
                0  otherwise
```

### When to Use

| Scenario | Recommended |
|---|---|
| Extractive QA (SQuAD-style) | Yes — primary metric alongside F1 |
| Slot filling / NER | Yes |
| Generative / open-ended tasks | No — too strict |

### Limitations

- A single extra space or synonym drops the score to **0**
- Does **not** capture partial correctness
- Typically paired with **F1 Score** for a balanced evaluation

### Code Example

```python
import evaluate

em = evaluate.load("exact_match")

predictions = ["Paris", "Mars", "42"]
references  = ["Paris", "Venus", "42"]

result = em.compute(predictions=predictions, references=references)
print(result)
# {'exact_match': 0.6667}
```

---

## 9. Bias & Safety Metrics

Evaluating LLM outputs is not limited to task performance. **Fairness and safety** are equally critical.

### 9.1 Toxicity

Measures the likelihood that generated text contains rude, disrespectful, or harmful language.

```python
import evaluate

toxicity_metric = evaluate.load("toxicity")

texts = [
    "The scientist published her findings.",
    "He is a terrible person and should be fired."
]

result = toxicity_metric.compute(predictions=texts, aggregation="maximum")
print(result)
# {'max_toxicity': 0.87, 'toxicity': [0.03, 0.87]}
```

| Aggregation Mode | Output |
|---|---|
| `None` | Per-sample toxicity scores |
| `"maximum"` | Highest toxicity across samples |
| `"ratio"` | Ratio of samples above a toxicity threshold |

### 9.2 Regard

Measures the **sentiment polarity** (positive / neutral / negative) of generated text when comparing how the model describes different demographic groups.

```python
import evaluate

regard = evaluate.load("regard")

group_a = ["She is a dedicated and brilliant engineer."]
group_b = ["He works in the kitchen."]

results = regard.compute(data=group_a)
print(results)
# {'positive': 0.72, 'neutral': 0.20, 'negative': 0.08}
```

| Score | Interpretation |
|---|---|
| **Positive Regard** | Language is favorable or complimentary |
| **Neutral Regard** | Language is factual, no sentiment |
| **Negative Regard** | Language is unfavorable or derogatory |

> **Best practice:** Compute regard scores across multiple demographic groups and compare for disparities.

---

## 10. LLM Challenges

| # | Challenge | Description | Impact |
|---|---|---|---|
| 1 | **Multi-language support** | Quality varies drastically across languages; most benchmarks are English-centric | Unreliable outputs for low-resource languages |
| 2 | **Open vs. closed models** | Open models offer transparency; closed models often have higher performance but are opaque | Trade-off between auditability and capability |
| 3 | **Scalability & compute cost** | Large models require significant GPU/TPU resources for both inference and fine-tuning | Barrier to entry for smaller organizations |
| 4 | **Bias in training data** | Models inherit and amplify societal biases present in web-scale corpora | Discriminatory or stereotypical outputs |
| 5 | **Hallucinations** | Models generate false information with high confidence | Critical risk in high-stakes domains (medical, legal, finance) |

---

## 11. Hallucination Mitigation Strategies

| Strategy | Description | When to Use |
|---|---|---|
| **Diverse training data** | Curate balanced, representative datasets | During model pre-training or fine-tuning |
| **Bias audits** | Systematically evaluate outputs across demographics | Before deployment and periodically after |
| **Fine-tuning for domain-specific tasks** | Specialize the model on curated domain data | When deploying in specialized fields (medical, legal) |
| **Prompt engineering** | Craft precise, constrained prompts to guide output | Quick, low-cost approach; no model changes needed |
| **Retrieval-Augmented Generation (RAG)** | Augment generation with retrieved factual context | When factual accuracy is critical |
| **Chain-of-Thought (CoT) reasoning** | Prompt the model to reason step-by-step | Complex reasoning and multi-step tasks |
| **Self-consistency checks** | Sample multiple outputs and select the majority answer | When inference budget allows multiple passes |
| **Temperature control** | Lower temperature = more deterministic, safer outputs | Production systems where stability is preferred |
| **RLHF** | Train models via human feedback to say "I don't know" | Long-term investment; requires feedback pipeline |

```
    Hallucination Mitigation Layer Cake
    ┌─────────────────────────────────────┐
    │          RLHF / Alignment            │  ← Long-term
    ├─────────────────────────────────────┤
    │     Self-Consistency / CoT           │  ← Inference-time
    ├─────────────────────────────────────┤
    │   RAG / Context Grounding            │  ← Architecture-level
    ├─────────────────────────────────────┤
    │  Fine-tuning / Domain Adaptation     │  ← Training-time
    ├─────────────────────────────────────┤
    │  Diverse Data / Bias Audits          │  ← Data-level
    └─────────────────────────────────────┘
```

---

## 12. Metric Selection Guide

### Decision Table

| If your task is... | And you need to measure... | Use this metric |
|---|---|---|
| Classification | Overall correctness | Accuracy |
| Classification | Balance of FP/FN on imbalanced data | F1 Score |
| Generation | How fluently the model predicts text | Perplexity |
| Translation | N-gram overlap with reference | BLEU |
| Translation | Semantic similarity (synonyms, stemming) | METEOR |
| Summarization | How much of the reference is recalled | ROUGE |
| Extractive QA | Strict correctness | Exact Match |
| Extractive QA | Partial overlap correctness | F1 Score |
| Generative QA | Overlap with reference answers | BLEU / ROUGE |
| Safety audit | Harmful or toxic language | Toxicity |
| Fairness audit | Sentiment disparity across groups | Regard |

### Decision Flowchart

```
                         ┌─────────────┐
                         │  What is    │
                         │ your task?  │
                         └──────┬──────┘
                                │
            ┌───────────┬───────┼────────┬───────────┐
            ▼           ▼       ▼        ▼           ▼
        Classification  Gen  Summary  Translation   QA
            │           │       │        │           │
            ▼           ▼       ▼        ▼           ▼
        Accuracy    Perplexity ROUGE   BLEU     Extractive?
        F1 Score    BLEU              METEOR      │
                                              ┌───┴───┐
                                              Yes     No
                                              │       │
                                              ▼       ▼
                                            EM + F1  BLEU
                                                    ROUGE
```

---

## 13. Quick Reference

### All Metrics at a Glance

| Metric | Library Call | Measures | Lower / Higher |
|---|---|---|---|
| Accuracy | `evaluate.load("accuracy")` | Correct predictions / Total | **Higher** |
| Precision | `evaluate.load("precision")` | TP / (TP + FP) | **Higher** |
| Recall | `evaluate.load("recall")` | TP / (TP + FN) | **Higher** |
| F1 Score | `evaluate.load("f1")` | Harmonic mean of P & R | **Higher** |
| Perplexity | `evaluate.load("perplexity")` | Model confidence on text | **Lower** |
| BLEU | `evaluate.load("bleu")` | N-gram precision | **Higher** |
| ROUGE | `evaluate.load("rouge")` | N-gram recall | **Higher** |
| METEOR | `evaluate.load("meteor")` | Harmonic P & R + synonyms | **Higher** |
| Exact Match | `evaluate.load("exact_match")` | String equality | **Higher** |
| Toxicity | `evaluate.load("toxicity")` | Harmful language probability | **Lower** |
| Regard | `evaluate.load("regard")` | Sentiment by demographic | **Compare groups** |

### Common Imports

```python
import evaluate

metrics = {
    "accuracy":   evaluate.load("accuracy"),
    "f1":         evaluate.load("f1"),
    "precision":  evaluate.load("precision"),
    "recall":     evaluate.load("recall"),
    "perplexity": evaluate.load("perplexity", module_type="metric"),
    "bleu":       evaluate.load("bleu"),
    "rouge":      evaluate.load("rouge"),
    "meteor":     evaluate.load("meteor"),
    "exact_match": evaluate.load("exact_match"),
    "toxicity":   evaluate.load("toxicity"),
    "regard":     evaluate.load("regard"),
}
```

---

> **Key takeaway:** No single metric tells the full story. Always combine multiple metrics tailored to your task, and include bias/safety evaluations before deploying LLM-powered systems.
