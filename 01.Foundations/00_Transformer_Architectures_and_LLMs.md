# Introduction to Large Language Models (LLMs) & Transformer Architectures

> A comprehensive foundational reference covering transformer architectures, attention mechanisms, the LLM lifecycle, and practical selection guidance.

---

## Table of Contents

1. [Overview: What is a Transformer?](#1-overview-what-is-a-transformer)
2. [The Attention Mechanism](#2-the-attention-mechanism)
3. [Transformer Architecture Components](#3-transformer-architecture-components)
4. [Architecture 1: Encoder-Only (BERT Family)](#4-architecture-1-encoder-only-bert-family)
5. [Architecture 2: Decoder-Only (GPT Family)](#5-architecture-2-decoder-only-gpt-family)
6. [Architecture 3: Encoder-Decoder (T5 / BART Family)](#6-architecture-3-encoder-decoder-t5--bart-family)
7. [Architecture Comparison Table](#7-architecture-comparison-table)
8. [The LLM Lifecycle](#8-the-llm-lifecycle)
9. [Key Concepts](#9-key-concepts)
10. [Popular LLMs Comparison Table](#10-popular-llms-comparison-table)
11. [When to Use Which Architecture](#11-when-to-use-which-architecture)
12. [Quick Reference](#12-quick-reference)

---

## 1. Overview: What is a Transformer?

A **transformer** is a deep learning architecture introduced in the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017). It was designed to process, understand, and generate sequential data — primarily natural language text — without relying on recurrence or convolution.

### Why Transformers Matter

| Aspect | Before Transformers (RNNs/LSTMs) | With Transformers |
|---|---|---|
| **Processing** | Sequential — one token at a time | Parallel — entire sequences at once |
| **Long-range dependencies** | Degrades over distance | Captured via attention mechanism |
| **Training speed** | Slow (sequential bottleneck) | Fast (highly parallelizable on GPUs) |
| **Scalability** | Limited | Scales to billions of parameters |
| **Context understanding** | Limited context window | Global context via self-attention |

### Core Properties

- **Parallel processing** — Handles entire sequences simultaneously, not token-by-token.
- **Self-attention** — Every token attends to every other token, capturing relationships regardless of distance.
- **Positional encoding** — Injects order information since the model has no inherent notion of sequence order.
- **Scalable** — Performance reliably improves with more data, more parameters, and more compute.

### The Three Transformer Architectures at a Glance

```
+---------------------------------------------------------------------+
|                    TRANSFORMER FAMILY TREE                          |
+---------------------------------------------------------------------+
|                                                                     |
|   ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐    |
|   │   ENCODER-ONLY  │  │   DECODER-ONLY  │  │ ENCODER-DECODER  │    |
|   │                 │  │                 │  │   (Seq2Seq)      │    |
|   │  Understanding  │  │   Generating    │  │  Understanding + │    |
|   │   input text    │  │     text        │  │    Generating    │    |
|   │                 │  │                 │  │                  │    |
|   │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌────────────┐  │    |
|   │  │   BERT    │  │  │  │    GPT    │  │  │  │  T5, BART  │  │    |
|   │  └───────────┘  │  │  └───────────┘  │  │  └────────────┘  │    |
|   └─────────────────┘  └─────────────────┘  └──────────────────┘    |
|                                                                     |
+---------------------------------------------------------------------+
```

---

## 2. The Attention Mechanism

The attention mechanism is the single most important innovation behind transformers. It allows the model to focus on relevant parts of the input when producing each part of the output.

### 2.1 Self-Attention

Self-attention computes a weighted representation of the entire sequence for each token, where the weights determine how much each other token "matters" to the current one.

**Intuition**: When reading the sentence *"The bank of the river was muddy"*, the word **"bank"** should attend more strongly to **"river"** than to financial contexts. Self-attention learns these relationships dynamically.

**The three learned projections for each token:**

| Component | Role | Analogy |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What do I contain?" | A database index |
| **Value (V)** | "What information do I carry?" | The actual content |

**Scaled dot-product attention formula:**

```
            Q × Kᵀ
Attention = ─────── × V
             √dₖ
```

Where `dₖ` is the dimensionality of the key vectors (scaling prevents softmax saturation).

**Step-by-step data flow:**

```
Input Tokens:  [The]  [cat]  [sat]

       ┌──────────────────────────────────────────────┐
       │         Learned Linear Projections           │
       │                                              │
       │  [The] ──→ Q₁, K₁, V₁                      │
       │  [cat] ──→ Q₂, K₂, V₂                      │
       │  [sat] ──→ Q₃, K₃, V₃                      │
       └──────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────┐
       │        Step 1: Dot Product Scores            │
       │                                              │
       │  score(i,j) = Qᵢ · Kⱼ                       │
       │                                              │
       │      The   cat   sat                         │
       │  The [0.8  0.2  0.1]                        │
       │  cat [0.3  0.9  0.4]                        │
       │  sat [0.1  0.5  0.7]                        │
       └──────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────┐
       │     Step 2: Scale by √dₖ                     │
       │                                              │
       │  scaled_score = score / √64  (if dₖ = 64)   │
       └──────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────┐
       │     Step 3: Softmax → Attention Weights      │
       │                                              │
       │      The   cat   sat                         │
       │  The [0.50  0.30  0.20]                     │
       │  cat [0.20  0.55  0.25]                     │
       │  sat [0.15  0.35  0.50]                     │
       │                                              │
       │  Each row sums to 1.0                        │
       └──────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────┐
       │     Step 4: Weighted Sum of Values           │
       │                                              │
       │  output_i = Σⱼ (weightᵢⱼ × Vⱼ)              │
       │                                              │
       │  Output: context-aware representation for    │
       │  each token, blending information from all   │
       │  tokens weighted by relevance.               │
       └──────────────────────────────────────────────┘
```

### 2.2 Multi-Head Attention

Rather than computing a single attention function, multi-head attention runs **h parallel attention heads**, each with its own learned Q, K, V projections. The outputs are concatenated and linearly projected.

```
          Input Embedding
               │
       ┌───────┼───────┐
       │       │       │
       ▼       ▼       ▼
   ┌──────┐┌──────┐┌──────┐
   │Head 1││Head 2││Head h│   (h = 8, 16, 32, ...)
   │Q K V ││Q K V ││Q K V │
   │Attn  ││Attn  ││Attn  │
   └──┬───┘└──┬───┘└──┬───┘
      │       │       │
      ▼       ▼       ▼
   head₁    head₂    headₕ
      │       │       │
      └───┬───┘───────┘
          │
          ▼
     [Concatenate]
          │
          ▼
    [Linear Projection]
          │
          ▼
      Output Embedding
```

**Why multiple heads?**

| Benefit | Explanation |
|---|---|
| **Different perspectives** | Each head can attend to different types of relationships (syntactic, semantic, positional). |
| **Richer representations** | Concatenating heads captures a diverse set of features simultaneously. |
| **Robustness** | Multiple heads reduce the risk of losing important information from a single attention pattern. |

### 2.3 Masked Attention (Decoder)

In the decoder, **masked self-attention** prevents tokens from attending to future positions. This is critical for autoregressive generation — you can't peek at the next word when predicting it.

```
Attention Mask (causal / look-ahead mask):

        Pos 1  Pos 2  Pos 3  Pos 4
Pos 1  [  ✓      ✗      ✗      ✗  ]
Pos 2  [  ✓      ✓      ✗      ✗  ]
Pos 3  [  ✓      ✓      ✓      ✗  ]
Pos 4  [  ✓      ✓      ✓      ✓  ]

✓ = allowed to attend    ✗ = masked (set to -∞ before softmax)
```

---

## 3. Transformer Architecture Components

### Full Transformer Block Diagram

```
+====================================================================+
|                     FULL TRANSFORMER (Original)                    |
+====================================================================+
|                                                                    |
|  INPUT                                                            │
│    │                                                               |
|    ▼                                                               |
|  ┌─────────────────────────────────┐                              │
|  │     Token Embedding +           │                              │
|  │     Positional Encoding         │                              │
|  └──────────────┬──────────────────┘                              │
│                  │                                                  |
|                  ▼                                                  |
|  ┌──────────────────────────────────────────────────────────────┐ │
|  │                    ENCODER (× N layers)                      │ │
|  │                                                              │ │
|  │  ┌────────────────────────────────────────────────────────┐  │ │
|  │  │  Multi-Head Self-Attention                             │  │ │
|  │  │  ┌──────────┐  ┌─────────┐  ┌──────────────────────┐  │  │ │
|  │  │  │ Attention│→ │ Add &   │→ │  Feed-Forward        │  │  │ │
|  │  │  │          │  │ Layer   │  │  Network (FFN)       │  │  │ │
|  │  │  │ Q, K, V  │  │ Norm    │  │  (2 linear layers    │  │  │ │
|  │  │  │          │  │         │  │   + ReLU/GELU)       │  │  │ │
|  │  │  └──────────┘  └─────────┘  └──────────┬───────────┘  │  │ │
|  │  │                                         │              │  │ │
|  │  │                            Add & Layer Norm             │  │ │
|  │  └─────────────────────────────────────────┼──────────────┘  │ │
|  │                                            │                 │ │
|  └────────────────────────────────────────────┼─────────────────┘ │
│                                               │                    |
|                                               ▼                    |
|  ┌──────────────────────────────────────────────────────────────┐ │
|  │                    DECODER (× N layers)                      │ │
|  │                                                              │ │
|  │  ┌────────────────────────────────────────────────────────┐  │ │
|  │  │  Masked Multi-Head                                      │  │ │
|  │  │  Self-Attention         Cross-Attention                 │  │ │
|  │  │  (over decoder input)   (attend to encoder output)      │  │ │
|  │  │         │                        │                      │  │ │
|  │  │         ▼                        ▼                      │  │ │
|  │  │    Add & Layer Norm       Add & Layer Norm              │  │ │
|  │  │                  │                                     │  │ │
|  │  │                  ▼                                     │  │ │
|  │  │         Feed-Forward Network                           │  │ │
|  │  │         Add & Layer Norm                               │  │ │
|  │  └────────────────────────────────────────────────────────┘  │ │
|  │                                                              │ │
|  └──────────────────────────────────────────────────────────────┘ │
│                                                                    |
|                  ▼                                                  |
|  ┌──────────────────────────────────────────────────────────────┐ │
|  │  Linear Projection → Softmax → Output Probabilities          │ │
|  └──────────────────────────────────────────────────────────────┘ │
│                                                                    |
|  OUTPUT (one token at a time)                                     │
+====================================================================+
```

### 3.1 Token Embedding

Every input token (word or subword) is mapped to a dense vector of dimension `d_model` (typically 512, 768, 1024, or larger).

```
Token ID: [1045, 10932, 5432, ...]     (from tokenizer)
                │
                ▼
Embedding Matrix (vocab_size × d_model)
                │
                ▼
Dense Vectors: [[0.23, -0.41, ...], ...]    (d_model dimensions each)
```

### 3.2 Positional Encoding

Since transformers process all tokens in parallel, they have no inherent sense of order. Positional encoding injects positional information using sinusoidal functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

| Property | Detail |
|---|---|
| **Deterministic** | No learned parameters; fixed by position and dimension index. |
| **Unique** | Each position gets a unique encoding vector. |
| **Generalizable** | Can extrapolate to sequence lengths not seen during training. |
| **Alternatives** | Learned positional embeddings (BERT), Rotary Position Embeddings (RoPE, used in LLaMA), ALiBi. |

```
Example positional encoding pattern (simplified):

Dim:    0     1     2     3     4     5
      ─────────────────────────────────
Pos 0: sin()  cos() sin()  cos() sin() cos()
Pos 1: sin()  cos() sin()  cos() sin() cos()
Pos 2: sin()  cos() sin()  cos() sin() cos()
Pos 3: sin()  cos() sin()  cos() sin() cos()
 ...

→ Low dimensions change slowly (long-range patterns)
→ High dimensions change rapidly  (fine-grained position)
```

### 3.3 Feed-Forward Network (FFN)

Each transformer layer contains a position-wise feed-forward network:

```
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂

Dimensions:  d_model → d_ff → d_model
             (768    → 3072  → 768)   [BERT-base example]

Activation: ReLU, GELU, or SwiGLU (modern models)
```

### 3.4 Layer Normalization & Residual Connections

Every sub-layer (attention, FFN) is wrapped with a **residual connection** followed by **layer normalization**:

```
output = LayerNorm(x + Sublayer(x))

   x ─────────────────────┐
   │                       │  (residual / skip connection)
   ▼                       │
 Sublayer(x)               │
   │                       │
   └──────── + ────────────┘
             │
             ▼
        LayerNorm
             │
             ▼
          output
```

**Why this matters:**
- **Residual connections** enable gradient flow through very deep networks (mitigates vanishing gradients).
- **Layer normalization** stabilizes training by normalizing activations across the feature dimension.

---

## 4. Architecture 1: Encoder-Only (BERT Family)

### Overview

Encoder-only models process the **entire input bidirectionally** — every token attends to every other token, including tokens both before and after it. This makes them exceptionally good at **understanding** text.

```
+============================================================+
|               ENCODER-ONLY ARCHITECTURE                    |
+============================================================+
|                                                            |
|   Input: "The cat sat on the [MASK]"                      │
|                                                            |
|   ┌──────────────────────────────────────────────────┐    │
|   │  Token + Position Embedding Layer                │    │
|   └────────────────────┬─────────────────────────────┘    │
|                        │                                  │
|          ┌─────────────▼──────────────┐                   │
|          │   ENCODER LAYER 1          │                   │
|          │                            │                   │
|          │   Multi-Head Self-Attention │  ← BIDIRECTIONAL │
|          │   (all tokens see all)      │    (full context)│
|          │   Add & Layer Norm          │                   │
|          │   Feed-Forward Network      │                   │
|          │   Add & Layer Norm          │                   │
|          └─────────────┬──────────────┘                   │
|                        │                                  │
|                        │  ... (repeat × N)                │
|                        │                                  │
|          ┌─────────────▼──────────────┐                   │
|          │   ENCODER LAYER N          │                   │
|          │   (same structure)         │                   │
|          └─────────────┬──────────────┘                   │
|                        │                                  │
|                        ▼                                  │
|   ┌──────────────────────────────────────────────────┐    │
|   │  Output: Contextual Embeddings for every token   │    │
|   │                                                  │    │
|   │  [CLS] The  cat  sat  on  the  [MASK]           │    │
|   │   │     │    │    │    │    │     │              │    │
|   │   ▼     ▼    ▼    ▼    ▼    ▼     ▼              │    │
|   │  vec  vec  vec  vec  vec  vec   vec              │    │
|   │   │                              │               │    │
|   │   ▼                              ▼               │    │
|   │ [CLS] → sentence-level     [MASK] → "mat"       │    │
|   │          classification        (fill the blank)  │    │
|   └──────────────────────────────────────────────────┘    │
|                                                            |
+============================================================+
```

### How BERT Pre-training Works

BERT is trained with two self-supervised objectives:

```
┌─────────────────────────────────────────────────────────────────┐
│                 BERT PRE-TRAINING OBJECTIVES                     │
├─────────────────────────────┬───────────────────────────────────┤
│  1. Masked Language Model   │  2. Next Sentence Prediction     │
│     (MLM)                   │     (NSP)                        │
│                             │                                   │
│  Input:                     │  Input:                          │
│  "The [MASK] sat on [MASK]"│  [CLS] Sentence A [SEP] B [SEP]  │
│                             │                                   │
│  Task: Predict masked       │  Task: Is Sentence B the actual  │
│  tokens (~15% of tokens)   │  next sentence after A?          │
│                             │                                   │
│  Example:                   │  Example:                        │
│  [MASK] → "cat"            │  A: "I love cats"                │
│  [MASK] → "mat"            │  B: "They are fluffy" → IsNext   │
│                             │  B: "Stocks fell"    → NotNext   │
└─────────────────────────────┴───────────────────────────────────┘
```

### Common Tasks

| Task | Description | Typical Setup |
|---|---|---|
| **Text Classification** | Assign categories to text | `[CLS]` token output → classifier head |
| **Named Entity Recognition** | Tag tokens as person, org, location, etc. | Token-level output → CRF / linear layer |
| **Extractive QA** | Find the answer span in a passage | Predict start/end token positions |
| **Sentiment Analysis** | Determine positive/negative/neutral | `[CLS]` token output → classifier |
| **Semantic Similarity** | Measure how similar two texts are | Embedding distance / cross-encoder |

### Popular Encoder-Only Models

| Model | Parameters | `d_model` | Layers | Heads | Training Data | Developer |
|---|---|---|---|---|---|---|
| **BERT-base** | 110M | 768 | 12 | 12 | BooksCorpus + Wikipedia (16GB) | Google |
| **BERT-large** | 340M | 1024 | 24 | 16 | BooksCorpus + Wikipedia (16GB) | Google |
| **RoBERTa** | 355M | 1024 | 24 | 16 | 160GB text | Meta |
| **DistilBERT** | 66M | 768 | 6 | 12 | Same as BERT (distilled) | Hugging Face |
| **ALBERT-base** | 12M | 768 | 12 | 12 | BooksCorpus + Wikipedia | Google |
| **DeBERTa-v3** | 184M | 768 | 12 | 12 | 78GB text | Microsoft |
| **ELECTRA-base** | 110M | 768 | 12 | 12 | Same as BERT | Google/Stanford |
| **Sentence-BERT** | 110M | 768 | 12 | 12 | SNLI + MultiNLI | UKP/DFKI |

---

## 5. Architecture 2: Decoder-Only (GPT Family)

### Overview

Decoder-only models process input **autoregressively** — predicting the next token given all previous tokens. They use **masked self-attention** so each token can only attend to itself and tokens before it. This makes them powerful **generators**.

```
+============================================================+
|               DECODER-ONLY ARCHITECTURE                    |
+============================================================+
|                                                            |
|   Input: "The cat sat on the"                              │
|   Goal: Predict next token → "mat"                        │
|                                                            |
|   ┌──────────────────────────────────────────────────┐    │
|   │  Token + Position Embedding Layer                │    │
|   └────────────────────┬─────────────────────────────┘    │
|                        │                                  │
|          ┌─────────────▼──────────────┐                   │
|          │   DECODER LAYER 1          │                   │
|          │                            │                   │
|          │   Masked Multi-Head        │                   │
|          │   Self-Attention            │  ← CAUSAL MASK   │
|          │   (tokens only see past)    │    (left-to-right│
|          │   Add & Layer Norm          │     only)        │
|          │   Feed-Forward Network      │                   │
|          │   Add & Layer Norm          │                   │
|          └─────────────┬──────────────┘                   │
|                        │                                  │
|                        │  ... (repeat × N)                │
|                        │                                  │
|          ┌─────────────▼──────────────┐                   │
|          │   DECODER LAYER N          │                   │
|          │   (same structure)         │                   │
|          └─────────────┬──────────────┘                   │
|                        │                                  │
|                        ▼                                  │
|   ┌──────────────────────────────────────────────────┐    │
|   │  Output: Next-token probability distribution     │    │
|   │                                                  │    │
|   │  P("mat" | "The cat sat on the") = 0.72         │    │
|   │  P("floor"| "The cat sat on the") = 0.15        │    │
|   │  P("couch"| "The cat sat on the") = 0.08        │    │
|   │  ...                                             │    │
|   │                                                  │    │
|   │  → Select "mat" → append to input → repeat      │    │
|   └──────────────────────────────────────────────────┘    │
|                                                            |
+============================================================+
```

### Autoregressive Generation Flow

```
Step 1:  Input:  "The cat sat on the"
         Output: "mat"  ←────────── highest probability

Step 2:  Input:  "The cat sat on the mat"
         Output: "and"  ←────────── highest probability

Step 3:  Input:  "The cat sat on the mat and"
         Output: "fell" ←────────── highest probability

Step 4:  Input:  "The cat sat on the mat and fell"
         Output: "asleep"

         ... continue until <EOS> token or max length
```

### Common Tasks

| Task | Description | Approach |
|---|---|---|
| **Text Generation** | Produce coherent, fluent text | Prompt → autoregressive decoding |
| **Language Modeling** | Predict the next word/token | Core pre-training objective |
| **Code Generation** | Generate source code from prompts | Trained on code corpora |
| **Summarization** | Condense long documents | Prompt engineering or fine-tuning |
| **Conversational AI** | Multi-turn dialogue | Chat formatting + RLHF alignment |
| **In-context Learning** | Learn from examples in the prompt | Zero-shot, few-shot prompting |
| **Reasoning** | Chain-of-thought, math, logic | Prompting strategies + training |

### Popular Decoder-Only Models

| Model | Parameters | Context Window | Training Data | Developer |
|---|---|---|---|---|
| **GPT-2** | 1.5B | 1,024 tokens | WebText (40GB) | OpenAI |
| **GPT-3** | 175B | 2,048 tokens | 570GB mixed | OpenAI |
| **GPT-3.5** (ChatGPT) | ~175B | 4,096 tokens | + RLHF | OpenAI |
| **GPT-4** | Undisclosed | 8K–128K tokens | Undisclosed | OpenAI |
| **GPT-4o** | Undisclosed | 128K tokens | Undisclosed | OpenAI |
| **LLaMA 2** | 7B–70B | 4,096 tokens | 2T tokens | Meta |
| **LLaMA 3** | 8B–70B | 8,192 tokens | 15T+ tokens | Meta |
| **LLaMA 3.1** | 8B–405B | 128K tokens | 15T+ tokens | Meta |
| **Mistral 7B** | 7B | 32K tokens | Undisclosed | Mistral AI |
| **Mixtral 8×7B** | 47B (MoE) | 32K tokens | Undisclosed | Mistral AI |
| **Gemma 2** | 2B–27B | 8K tokens | 2T+ tokens | Google |
| **Qwen 2.5** | 0.5B–72B | 128K tokens | 18T+ tokens | Alibaba |
| **DeepSeek-V3** | 671B (MoE) | 128K tokens | 14.8T tokens | DeepSeek |
| **Phi-4** | 14B | 16K tokens | Synthetic + web | Microsoft |

---

## 6. Architecture 3: Encoder-Decoder (T5 / BART Family)

### Overview

Encoder-Decoder (sequence-to-sequence) models use a **bidirectional encoder** to understand the input and a **causal decoder** to generate the output. They are designed for tasks that require both **comprehension** and **generation**.

```
+====================================================================+
|               ENCODER-DECODER (SEQ2SEQ) ARCHITECTURE               |
+====================================================================+
|                                                                    |
|   Input: "translate English to French: The cat is black"          │
|   Output: "Le chat est noir"                                      │
|                                                                    |
|   ┌──────────────────────────┐    ┌──────────────────────────┐   │
|   │        ENCODER           │    │         DECODER          │   │
|   │                          │    │                          │   │
│   │  "The cat is black"     │    │  "<BOS> Le chat est     │   │
│   │        │                 │    │          noir <EOS>"     │   │
│   │        ▼                 │    │          │               │   │
│   │  ┌──────────────────┐   │    │          ▼               │   │
│   │  │ Token + Position │   │    │  ┌──────────────────┐   │   │
│   │  │ Embedding        │   │    │  │ Token + Position │   │   │
│   │  └────────┬─────────┘   │    │  │ Embedding        │   │   │
│   │           ▼              │    │  └────────┬─────────┘   │   │
│   │  ┌──────────────────┐   │    │           ▼              │   │
│   │  │ Self-Attention   │   │    │  ┌──────────────────┐   │   │
│   │  │ (bidirectional)  │   │    │  │ Masked Self-     │   │   │
│   │  └────────┬─────────┘   │    │  │ Attention        │   │   │
│   │           ▼              │    │  └────────┬─────────┘   │   │
│   │  ┌──────────────────┐   │    │           ▼              │   │
│   │  │ Feed-Forward     │   │    │  ┌──────────────────┐   │   │
│   │  └────────┬─────────┘   │    │  │ Cross-Attention  │   │   │
│   │           │              │    │  │ (decoder attends │   │   │
│   │     ... × N layers      │    │  │  to ENCODER      │   │   │
│   │           │              │    │  │  output) ◄───────┼───┼───┤
│   │           ▼              │    │  └────────┬─────────┘   │   │
│   │  ┌──────────────────┐   │    │           ▼              │   │
│   │  │ Encoder Output   │───┼───→│  ┌──────────────────┐   │   │
│   │  │ (context vectors)│   │    │  │ Feed-Forward     │   │   │
│   │  └──────────────────┘   │    │  └────────┬─────────┘   │   │
│   │                          │    │           │              │   │
│   │                          │    │     ... × N layers      │   │
│   │                          │    │           │              │   │
│   │                          │    │           ▼              │   │
│   │                          │    │  ┌──────────────────┐   │   │
│   │                          │    │  │ Linear → Softmax │   │   │
│   │                          │    │  │ Next token probs │   │   │
│   │                          │    │  └──────────────────┘   │   │
│   └──────────────────────────┘    └──────────────────────────┘   │
|                                                                    |
+====================================================================+
```

### Cross-Attention — The Bridge Between Encoder and Decoder

```
  Encoder Output (K_encoder, V_encoder)
         │
         ▼
  ┌──────────────────────────────┐
  │     CROSS-ATTENTION          │
  │                              │
  │  Q: from decoder hidden state│
  │  K: from encoder output      │
  │  V: from encoder output      │
  │                              │
  │  "Which parts of the input   │
  │   should I focus on to       │
  │   generate the next output   │
  │   token?"                    │
  └──────────────────────────────┘
         │
         ▼
  Decoder gets informed by the most
  relevant encoder representations
```

### Common Tasks

| Task | Description | Example |
|---|---|---|
| **Machine Translation** | Translate between languages | English → French |
| **Summarization** | Generate concise summaries | Article → bullet points |
| **Conversational Agents** | Multi-turn dialogue systems | Chatbots |
| **Paraphrasing** | Rewrite text while preserving meaning | Simplification |
| **Style Transfer** | Change writing style | Formal → informal |
| **Data-to-Text** | Convert structured data to natural language | Table → description |

### Popular Encoder-Decoder Models

| Model | Parameters | `d_model` | Encoder Layers | Decoder Layers | Developer |
|---|---|---|---|---|---|
| **T5-base** | 220M | 768 | 12 | 12 | Google |
| **T5-large** | 770M | 1024 | 24 | 24 | Google |
| **T5-3B** | 3B | 1024 | 24 | 24 | Google |
| **T5-11B** | 11B | 1024 | 24 | 24 | Google |
| **Flan-T5-xl** | 3B | 2048 | 24 | 24 | Google |
| **BART-base** | 140M | 768 | 6 | 6 | Meta |
| **BART-large** | 400M | 1024 | 12 | 12 | Meta |
| **mBART-50** | 680M | 1024 | 12 | 12 | Meta |
| **Pegasus-large** | 568M | 1024 | 16 | 16 | Google |
| **UL2** | 20B | — | — | — | Google |
| **NLLB-200** | up to 54.5B | — | — | — | Meta |

### T5's Text-to-Text Framework

T5 treats **every NLP task as a text-to-text problem**, providing a unified interface:

```
┌────────────────────────────────────────────────────────────────┐
│                  T5: EVERYTHING IS TEXT-TO-TEXT                │
├────────────────────┬───────────────────────────────────────────┤
│ Task               │ Input → Output                            │
├────────────────────┼───────────────────────────────────────────┤
│ Translation        │ "translate English to French: Hello"      │
│                    │   → "Bonjour"                             │
├────────────────────┼───────────────────────────────────────────┤
│ Summarization      │ "summarize: [long article text]"          │
│                    │   → "Short summary of the article."       │
├────────────────────┼───────────────────────────────────────────┤
│ Classification     │ "cola sentence: The cat sat down"         │
│                    │   → "acceptable"                          │
├────────────────────┼───────────────────────────────────────────┤
│ QA                 │ "question: What color? context: Sky is..."│
│                    │   → "blue"                                │
├────────────────────┼───────────────────────────────────────────┤
│ Sentiment          │ "sst2 sentence: This movie was great!"    │
│                    │   → "positive"                            │
└────────────────────┴───────────────────────────────────────────┘
```

---

## 7. Architecture Comparison Table

| Dimension | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---|---|---|---|
| **Example** | BERT, RoBERTa | GPT, LLaMA | T5, BART |
| **Attention** | Bidirectional | Causal (masked) | Bidirectional + Causal + Cross |
| **Direction** | Both directions | Left-to-right only | Encoder: both; Decoder: left-to-right |
| **Strength** | Understanding text | Generating text | Understanding + generating |
| **Pre-training** | Masked LM (MLM) | Causal LM (CLM) | Span corruption / denoising |
| **Output** | Token embeddings / `[CLS]` vector | Next-token probabilities | Sequence of tokens |
| **Generation** | Not natively generative | Autoregressive generation | Autoregressive generation |
| **Parallelism** | Full sequence parallel | Training: parallel; Inference: sequential | Full encoder parallel; decoder sequential |
| **Typical Size** | 100M–1B | 1B–400B+ | 200M–11B (traditionally) |
| **Fine-tuning** | Task-specific head | Prompt engineering / LoRA | Task prefixes |
| **Best for** | Classification, extraction | Open-ended generation | Structured output tasks |
| **Speed (inference)** | Fast (single forward pass) | Slower (token by token) | Moderate |

---

## 8. The LLM Lifecycle

Building and deploying an LLM follows a well-defined lifecycle. Understanding this pipeline is critical for practitioners.

```
+====================================================================+
│                      THE LLM LIFECYCLE                             |
+====================================================================+
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │  1. PRE-     │───→│  2. FINE-    │───→│  3. ALIGNMENT &      │ │
│  │  TRAINING    │    │  TUNING      │    │     EVALUATION       │ │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘ │
│                                                      │             │
│                                                      ▼             │
│         ┌─────────────────────────────────────────────────┐       │
│         │         4. DEPLOYMENT & MONITORING              │       │
│         └─────────────────────────────────────────────────┘       │
│                                                                    │
+====================================================================+
```

### 8.1 Pre-Training

The model learns general language patterns from massive text corpora.

```
+--------------------------------------------------------+
|                  PRE-TRAINING                          |
|                                                        |
|  Objective:    Self-supervised learning                |
|  Data:         Billions to trillions of tokens         |
|  Compute:      Thousands of GPU-hours                  |
|  Cost:         $100K – $10M+                           |
|                                                        |
|  ┌───────────────────────────────────────────────┐    |
|  │           Training Pipeline                   │    |
|  │                                               │    |
|  │  Raw Text Corpus                              │    |
|  │       │                                       │    |
|  │       ▼                                       │    |
|  │  Tokenization                                 │    |
|  │       │                                       │    |
|  │       ▼                                       │    |
|  │  Batch Creation + Data Loading                │    |
|  │       │                                       │    |
|  │       ▼                                       │    |
|  │  Forward Pass → Loss Computation              │    |
|  │       │                                       │    |
|  │       ▼                                       │    |
|  │  Backward Pass → Gradient Update              │    |
|  │       │                                       │    |
|  │       ▼                                       │    |
|  │  Repeat for millions of steps                 │    |
|  └───────────────────────────────────────────────┘    |
|                                                        |
|  Result: Base model with general language abilities    |
+--------------------------------------------------------+
```

| Objective | Used By | Description |
|---|---|---|
| **Causal LM (CLM)** | GPT, LLaMA | Predict next token given previous tokens |
| **Masked LM (MLM)** | BERT, RoBERTa | Predict randomly masked tokens in input |
| **Prefix LM** | UniLM, PaLM | Prefix is bidirectional, suffix is causal |
| **Denoising** | T5, BART | Reconstruct corrupted text |
| **Span Corruption** | T5 | Replace random spans with sentinel tokens |

### 8.2 Fine-Tuning

Adapting the base model to specific tasks or domains.

```
+--------------------------------------------------------+
|                  FINE-TUNING                           |
|                                                        |
|  Base Model + Task-Specific Data → Specialized Model   |
|                                                        |
|  ┌───────────────────┐  ┌────────────────────────┐    |
|  │  Full Fine-Tuning │  │  Parameter-Efficient   │    |
|  │                   │  │  Fine-Tuning (PEFT)     │    |
|  │  Update ALL       │  │                        │    |
|  │  parameters       │  │  • LoRA (rank          │    |
|  │                   │  │    decomposition)       │    |
|  │  Expensive        │  │  • QLoRA (quantized)   │    |
|  │  Risk of          │  │  • Adapters            │    |
|  │  catastrophic     │  │  • Prefix tuning       │    |
|  │  forgetting       │  │                        │    |
|  │                   │  │  Cheap, fast, popular   │    |
|  └───────────────────┘  └────────────────────────┘    |
|                                                        |
+--------------------------------------------------------+
```

**LoRA (Low-Rank Adaptation) — most popular PEFT method:**

```
Original weight matrix: W (d × d)     ← frozen, not updated

           x
           │
     ┌─────┴─────┐
     │  W (frozen)│────→ Wx
     └─────┬─────┘         │
           │               │
     ┌─────┴─────┐         │
     │  A (d × r)│         │    r << d (e.g., r = 8 or 16)
     │  B (r × d)│         │    Only A and B are trained
     └─────┬─────┘         │
           │               │
           ▼               ▼
         BAx      +       Wx   =  (W + BA)x
           │               │          │
           └───────┬───────┘          │
                   ▼                  │
              Updated output          │
                                    trained parameters: 2 × d × r
                                    (tiny fraction of d × d)
```

### 8.3 Alignment & Evaluation

Ensuring the model is helpful, harmless, and honest, then measuring performance.

```
+--------------------------------------------------------+
|              ALIGNMENT & EVALUATION                    |
|                                                        |
|  ┌────────────────────────────────────────────────┐   |
|  │  ALIGNMENT                                      │   |
|  │                                                │   |
|  │  1. SFT (Supervised Fine-Tuning)               │   |
|  │     • Human-written demonstration data          │   |
|  │     • Teaches format and style                  │   |
|  │                                                │   |
|  │  2. RLHF (Reinforcement Learning from          │   |
|  │     Human Feedback)                             │   |
|  │     • Train reward model on preferences         │   |
|  │     • Optimize policy with PPO                  │   |
|  │                                                │   |
|  │  3. DPO (Direct Preference Optimization)       │   |
|  │     • Directly optimize from preferences        │   |
|  │     • No separate reward model needed           │   |
|  └────────────────────────────────────────────────┘   |
|                                                        |
|  ┌────────────────────────────────────────────────┐   |
|  │  EVALUATION                                     │   |
|  │                                                │   |
|  │  • Benchmarks: MMLU, HellaSwag, GSM8K,         │   |
|  │    HumanEval, MT-Bench, Arena                   │   |
|  │  • Human evaluation (chatbot arena)             │   |
|  │  • Automated metrics (BLEU, ROUGE, BERTScore)  │   |
|  └────────────────────────────────────────────────┘   |
+--------------------------------------------------------+
```

### 8.4 Deployment

Making the model available for use in production.

```
+--------------------------------------------------------+
|                  DEPLOYMENT                            |
|                                                        |
|  ┌──────────────────────────────────────────────┐     |
|  │          Optimization Techniques              │     |
|  │                                              │     |
|  │  • Quantization (INT8, INT4, GPTQ, AWQ)     │     |
|  │  • Distillation (smaller student model)      │     |
|  │  • Pruning (remove unimportant weights)      │     |
|  │  • Speculative decoding                      │     |
|  │  • Flash Attention                           │     |
|  └──────────────────────────────────────────────┘     |
|                                                        |
|  ┌──────────────────────────────────────────────┐     |
|  │          Serving Infrastructure               │     |
|  │                                              │     |
|  │  • vLLM, TGI, TensorRT-LLM, Triton          │     |
|  │  • Continuous batching                       │     |
|  │  • Paged attention (KV cache management)     │     |
|  │  • Load balancing, auto-scaling              │     |
|  └──────────────────────────────────────────────┘     |
|                                                        |
|  ┌──────────────────────────────────────────────┐     |
|  │          Monitoring                           │     |
|  │                                              │     |
|  │  • Latency, throughput                       │     |
|  │  • Output quality sampling                   │     |
|  │  • Safety / guardrails                       │     |
|  │  • Cost tracking                             │     |
|  └──────────────────────────────────────────────┘     |
+--------------------------------------------------------+
```

---

## 9. Key Concepts

### 9.1 Tokenization

Tokenization converts raw text into a sequence of integers (token IDs) that the model can process.

```
Text: "Hello, how are you?"
          │
          ▼
┌──────────────────────────────────────────────────┐
│              TOKENIZER                           │
│                                                  │
│  Subword tokenization:                           │
│  "Hello, how are you?" → [15496, 11, 703, 389,  │
│                            345, 30]              │
│                                                  │
│  Each number maps to a row in the embedding      │
│  matrix, producing a dense vector.               │
└──────────────────────────────────────────────────┘
```

**Common tokenization methods:**

| Method | Used By | Approach | Example |
|---|---|---|---|
| **BPE** | GPT-2, GPT-3 | Merge most frequent byte pairs iteratively | "lowest" → `["low", "est"]` |
| **WordPiece** | BERT | Greedy longest-match with `##` prefix for subwords | "unaffable" → `["un", "##aff", "##able"]` |
| **SentencePiece** | T5, LLaMA | Language-independent; treats text as raw bytes | Works directly on any script |
| **tiktoken** | GPT-4, GPT-4o | Fast BPE implementation in Rust | Optimized for speed |

```
Tokenization tradeoffs:

  Word-level:    "Hello world" → ["Hello", "world"]
                 + Simple, interpretable
                 - Huge vocabulary, can't handle unseen words

  Character:     "Hello world" → ["H","e","l","l","o"," ","w",...]
                 + Tiny vocabulary, no unknown tokens
                 - Very long sequences, less semantic meaning

  Subword:       "Hello world" → ["Hel", "lo", "world"]
                 + Best of both worlds
                 + Handles rare/unseen words via subword pieces
                 - Standard approach for modern LLMs
```

### 9.2 Context Window

The **context window** (or context length) is the maximum number of tokens the model can process in a single forward pass.

```
+------------------------------------------------------------------+
|                    CONTEXT WINDOW                                 |
|                                                                    |
|  Tokens:  [t₁] [t₂] [t₃] ... [tₙ] [tₙ₊₁] ... [tₘ]            |
|                                                                    |
|  └─────────────── Context Window ───────────────┘  │  Excess     │
|     ← max tokens the model can see at once →     │  tokens      │
|                                                   │  are truncated│
|                                                                    |
|  Classic models:    512 – 2,048 tokens                           │
|  Modern models:     8K – 128K tokens                             │
|  Frontier models:   128K – 1M+ tokens (Gemini, Claude)           │
|                                                                    |
|  Why it matters:                                                  |
|  • Determines how much text the model can "remember" at once      │
|  • Affects ability to process long documents                      │
|  • Directly impacts compute cost (attention is O(n²))            │
+------------------------------------------------------------------+
```

| Technique | Purpose | Models Using It |
|---|---|---|
| **Rotary Position Embeddings (RoPE)** | Better length extrapolation | LLaMA, Qwen, Mistral |
| **ALiBi** | Linear bias for length extrapolation | BLOOM, MPT |
| **Sliding Window Attention** | Fixed-size local attention | Mistral, Gemma |
| **Flash Attention** | Memory-efficient attention | Most modern models |
| **Ring Attention** | Distribute context across devices | LSDA |
| **YaRN / NTK-aware scaling** | Extend context at inference | Fine-tuned LLaMA models |

### 9.3 Parameters

Parameters are the learnable weights in the model. More parameters generally means more capability, but also more compute.

```
Where parameters live in a transformer:

┌───────────────────────────────────────────────────┐
│  Component                    │ Typical % of Total │
├───────────────────────────────┼────────────────────┤
│  Token Embedding Matrix       │ ~5–10%             │
│  Attention Q, K, V, O weights │ ~30–40%            │
│  Feed-Forward Network weights │ ~50–60%            │
│  Layer Norm parameters        │ <1%                │
│  Output projection            │ ~5–10% (tied or not)│
└───────────────────────────────┴────────────────────┘

Scaling law (Kaplan et al.):
  Performance improves predictably as a power-law function of:
  • Parameter count (N)
  • Dataset size (D)
  • Compute budget (C)

  Roughly: Loss ∝ N^(-0.076) × D^(-0.095)
```

### 9.4 Temperature & Sampling

```
Temperature controls output randomness:

  Low temperature (T = 0.1):           High temperature (T = 1.5):
  ┌────────────────────────┐          ┌────────────────────────┐
  │ Focused, deterministic │          │ Diverse, creative      │
  │                        │          │                        │
  │ Token A: 95% ████████ │          │ Token A: 25% ██        │
  │ Token B:  4% █         │          │ Token B: 30% ██        │
  │ Token C:  1% ▏         │          │ Token C: 25% ██        │
  │                        │          │ Token D: 20% ██        │
  │ → Always picks A       │          │ → Any could be chosen  │
  └────────────────────────┘          └────────────────────────┘

  T = 0:   Greedy decoding (always highest probability)
  T = 0.7: Balanced (common for chatbots)
  T = 1.0: Default distribution
  T > 1:   More random / creative
```

### 9.5 Key Innovations Summary

| Innovation | Before | After | Impact |
|---|---|---|---|
| **Self-Attention** | Sequential dependency capture | Global context in O(1) hops | Eliminates distance bottleneck |
| **Parallelization** | RNN sequential processing | Full sequence in parallel | Orders of magnitude faster training |
| **Positional Encoding** | Implicit from sequence order | Explicit encoding injection | Enables parallel processing with order |
| **Multi-Head Attention** | Single attention pattern | Multiple diverse patterns | Richer representations |
| **Residual + LayerNorm** | Unstable deep networks | Stable 100+ layer training | Enables deep architectures |
| **Scaling Laws** | Unclear scaling behavior | Predictable performance gains | Guides compute-optimal training |
| **RLHF / DPO** | Untuned generation | Aligned to human preferences | Safer, more helpful outputs |

---

## 10. Popular LLMs Comparison Table

| Model | Developer | Arch. | Params | Context | Open Source | Key Strength |
|---|---|---|---|---|---|---|
| **GPT-4o** | OpenAI | Decoder | Undisclosed | 128K | No | Best general-purpose; multimodal |
| **Claude 3.5 Sonnet** | Anthropic | Decoder | Undisclosed | 200K | No | Long context; safety; nuanced reasoning |
| **Gemini 2.0** | Google | Decoder (MoE) | Undisclosed | 1M+ | No | Massive context; multimodal |
| **LLaMA 3.1 405B** | Meta | Decoder | 405B | 128K | Yes | Best open-weight model |
| **LLaMA 3.1 70B** | Meta | Decoder | 70B | 128K | Yes | Strong open model; efficient |
| **Qwen 2.5 72B** | Alibaba | Decoder | 72B | 128K | Yes | Multilingual; code strong |
| **DeepSeek-V3** | DeepSeek | Decoder (MoE) | 671B | 128K | Yes | MoE efficiency; strong reasoning |
| **Mistral Large** | Mistral AI | Decoder | 123B | 128K | Yes | European; balanced performance |
| **Mixtral 8×7B** | Mistral AI | Decoder (MoE) | 47B | 32K | Yes | MoE efficiency at 7B cost |
| **BERT-large** | Google | Encoder | 340M | 512 | Yes | Best for classification/NLU |
| **RoBERTa-large** | Meta | Encoder | 355M | 512 | Yes | Robust BERT variant |
| **T5-11B** | Google | Enc-Dec | 11B | 512 | Yes | Unified text-to-text |
| **Flan-T5-xxl** | Google | Enc-Dec | 11B | 512 | Yes | Instruction-tuned T5 |
| **BART-large** | Meta | Enc-Dec | 400M | 1024 | Yes | Strong for summarization |
| **Phi-4** | Microsoft | Decoder | 14B | 16K | Yes | Small but highly capable |
| **Gemma 2 27B** | Google | Decoder | 27B | 8K | Yes | Efficient; research-friendly |

---

## 11. When to Use Which Architecture

### Decision Flowchart

```
                         What is your task?
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
                 ▼              ▼              ▼
          Understanding?  Generating?   Transforming
          (classify,      (write,        input→output
           extract,        chat,          (translate,
           label)          complete)       summarize)
                 │              │              │
                 ▼              ▼              ▼
          ┌──────────┐  ┌──────────┐   ┌──────────────┐
          │ ENCODER- │  │ DECODER- │   │ ENCODER-     │
          │ ONLY     │  │ ONLY     │   │ DECODER      │
          │          │  │          │   │              │
          │ BERT     │  │ GPT      │   │ T5           │
          │ RoBERTa  │  │ LLaMA    │   │ BART         │
          │ DeBERTa  │  │ Mistral  │   │ Pegasus      │
          └──────────┘  └──────────┘   └──────────────┘
```

### Detailed Selection Guide

| Scenario | Recommended Architecture | Example Models | Why |
|---|---|---|---|
| **Text classification** | Encoder-only | BERT, RoBERTa, DeBERTa | Bidirectional context captures full meaning |
| **Named entity recognition** | Encoder-only | BERT, RoBERTa | Token-level understanding with full context |
| **Sentiment analysis** | Encoder-only | DistilBERT, BERT | Fast inference, accurate classification |
| **Open-ended text generation** | Decoder-only | GPT-4, LLaMA, Qwen | Trained for coherent long-form generation |
| **Chatbots / assistants** | Decoder-only | GPT-4, Claude, LLaMA | Natural conversational flow with RLHF |
| **Code generation** | Decoder-only | GPT-4, DeepSeek-Coder | Autoregressive generation suits code |
| **Machine translation** | Encoder-Decoder | T5, mBART, NLLB | Dedicated encoder for source understanding |
| **Summarization** | Encoder-Decoder | BART, Pegasus, T5 | Encoder reads full doc, decoder generates summary |
| **Question answering (extractive)** | Encoder-only | BERT, RoBERTa | Identify answer span in context |
| **Question answering (generative)** | Decoder-only | GPT-4, LLaMA | Generate fluent natural answers |
| **Document QA with RAG** | Decoder-only | LLaMA, Mistral | Generation + retrieval augmentation |
| **Speech-to-text** | Encoder-Decoder | Whisper | Audio encoder → text decoder |
| **Image captioning** | Encoder-Decoder | ViT + GPT, BLIP | Vision encoder → text decoder |

### Practical Considerations

```
┌─────────────────────────────────────────────────────────────┐
│                   PRACTICAL TIPS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. START SIMPLE                                            │
│     • Try a decoder-only model first (most versatile)       │
│     • Use encoder-only for pure classification tasks        │
│                                                             │
│  2. CONSIDER RESOURCES                                      │
│     • <1B params:  CPU viable, edge devices                 │
│     • 1B–7B:       Single GPU (consumer)                    │
│     • 7B–30B:      Single GPU (high-end) or multi-GPU       │
│     • 30B–70B:     Multi-GPU setup                          │
│     • 70B+:        Cluster or cloud                         │
│                                                             │
│  3. FINE-TUNING APPROACH                                    │
│     • Few samples:    Few-shot prompting or LoRA            │
│     • Thousands:      Full fine-tuning or QLoRA             │
│     • Domain shift:   Continued pre-training + fine-tune    │
│                                                             │
│  4. DEPLOYMENT OPTIMIZATION                                 │
│     • Quantize to INT4/INT8 for 2–4× memory reduction       │
│     • Use speculative decoding for 2–3× speedup             │
│     • Batch requests for throughput                          │
│                                                             │
│  5. EVALUATION                                              │
│     • Use task-specific benchmarks                           │
│     • Always include human evaluation for production        │
│     • Test for safety, bias, and robustness                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Quick Reference

### Transformer Architectures — One-Liner Summary

| Architecture | One-Liner | Analogy |
|---|---|---|
| **Encoder-Only** | "Read and understand" | A careful reader who annotates text |
| **Decoder-Only** | "Predict and generate" | A storyteller who speaks one word at a time |
| **Encoder-Decoder** | "Read, then rewrite" | A translator who reads the source, then writes the target |

### Key Equations

```
Self-Attention:         Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

Multi-Head Attention:   MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ
                        where headᵢ = Attention(QWᵢᵠ, KWᵢᴋ, VWᵢⱽ)

Feed-Forward:           FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂

Residual + Norm:        output = LayerNorm(x + Sublayer(x))

Positional Encoding:    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
                        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Autoregressive:         P(x₁,...,xₙ) = Πᵢ P(xᵢ | x₁,...,xᵢ₋₁)
```

### Key Papers

| Paper | Year | Contribution |
|---|---|---|
| *Attention Is All You Need* (Vaswani et al.) | 2017 | Introduced the Transformer |
| *BERT* (Devlin et al.) | 2018 | Bidirectional encoder pre-training |
| *GPT-2* (Radford et al.) | 2019 | Demonstrated zero-shot transfer |
| *T5* (Raffel et al.) | 2019 | Text-to-text unified framework |
| *Scaling Laws* (Kaplan et al.) | 2020 | Predictable scaling behavior |
| *GPT-3* (Brown et al.) | 2020 | Few-shot learning at scale |
| *LoRA* (Hu et al.) | 2021 | Parameter-efficient fine-tuning |
| *InstructGPT / RLHF* (Ouyang et al.) | 2022 | Alignment via human feedback |
| *LLaMA* (Touvron et al.) | 2023 | Open foundation models |
| *Mixture of Experts* | 2023–24 | Sparse scaling (Mixtral, DeepSeek) |

### Common Token Counts Reference

```
1 token  ≈  4 characters (English)
1 token  ≈  ¾ of a word
100 tokens ≈  75 words  ≈  1 paragraph
500 tokens ≈  375 words ≈  1 page
1K tokens  ≈  750 words ≈  1.5 pages
4K tokens  ≈  3,000 words ≈  6 pages (≈ 1 chapter)
8K tokens  ≈  6,000 words ≈  12 pages
32K tokens ≈  24,000 words ≈  a short book
128K tokens ≈  96,000 words ≈  a full novel
```

---

> **Note**: The field of LLMs evolves rapidly. Model capabilities, sizes, and best practices change frequently. Always consult the latest benchmarks and documentation when making architectural decisions.
