# Embeddings
> A comprehensive guide to vector representations, semantic similarity, retrieval, recommendation, clustering, and production embedding systems for AI engineering.

## Table of Contents
- [1. Big Picture](#1-big-picture)
- [2. What Is an Embedding?](#2-what-is-an-embedding)
- [3. Why Embeddings Matter](#3-why-embeddings-matter)
- [4. Core Vector-Space Intuition](#4-core-vector-space-intuition)
- [5. Essential Math](#5-essential-math)
- [6. Types of Embeddings](#6-types-of-embeddings)
- [7. How Embedding Models Work](#7-how-embedding-models-work)
- [8. Word Embeddings](#8-word-embeddings)
- [9. Contextual Token Embeddings](#9-contextual-token-embeddings)
- [10. Sentence and Document Embeddings](#10-sentence-and-document-embeddings)
- [11. Multimodal Embeddings](#11-multimodal-embeddings)
- [12. Dense, Sparse, and Hybrid Representations](#12-dense-sparse-and-hybrid-representations)
- [13. Similarity Search From Scratch](#13-similarity-search-from-scratch)
- [14. Embeddings in Semantic Search](#14-embeddings-in-semantic-search)
- [15. Embeddings in RAG](#15-embeddings-in-rag)
- [16. Common Use Cases](#16-common-use-cases)
- [17. Choosing an Embedding Model](#17-choosing-an-embedding-model)
- [18. Model Families to Know](#18-model-families-to-know)
- [19. Evaluation](#19-evaluation)
- [20. Production Engineering](#20-production-engineering)
- [21. Advanced Topics](#21-advanced-topics)
- [22. Common Pitfalls](#22-common-pitfalls)
- [23. Code Examples](#23-code-examples)
- [24. Practical Project](#24-practical-project)
- [25. Quick Reference](#25-quick-reference)
- [26. Self-Check Questions](#26-self-check-questions)
- [27. Resources and References](#27-resources-and-references)

## 1. Big Picture

Embeddings are one of the most important primitives in modern AI systems. They convert messy real-world data such as text, images, audio, code, users, products, queries, or documents into numeric vectors that machine learning systems can compare, search, cluster, rank, and classify.

In traditional keyword systems, two pieces of text match only when they share words or exact terms. Embedding systems instead compare meaning. The sentences "How do I reset my password?" and "I forgot my login credentials" may share few words, but a good embedding model places them close together in vector space because they express related intent.

For AI engineers, embeddings are the foundation beneath semantic search, retrieval-augmented generation, recommendations, clustering, duplicate detection, routing, memory, personalization, and many evaluation workflows. Before using a vector database or building RAG, you should understand embeddings at the raw vector level: generate vectors, normalize them, compute similarity, rank candidates, and inspect failure cases.

```text
Raw object                 Embedding model                  Vector
-----------------          ------------------               -------------------------
"refund policy"    --->    text embedding model      --->   [0.012, -0.183, ..., 0.044]
product image      --->    image embedding model     --->   [0.221,  0.008, ..., -0.031]
audio clip         --->    audio embedding model     --->   [-0.019, 0.772, ..., 0.105]
source code        --->    code embedding model      --->   [0.331, -0.412, ..., 0.006]
```

The key idea: once objects become vectors in a shared space, the distance between vectors becomes a useful proxy for relatedness.

## 2. What Is an Embedding?

An embedding is a learned numeric representation of an object. In AI engineering, it is usually a fixed-length vector of floating-point numbers.

Example:

```text
Text: "The cat sleeps on the sofa."
Embedding: [0.0142, -0.0921, 0.3374, ..., -0.0188]
Dimension: 384, 768, 1024, 1536, 3072, or another model-specific size
```

An embedding model maps an input object `x` into a vector:

```math
f(x) = v \in \mathbb{R}^d
```

Where:

| Symbol | Meaning |
| --- | --- |
| `x` | Input object, such as text, image, audio, or code |
| `f` | Embedding model |
| `v` | Output embedding vector |
| `d` | Number of dimensions |
| `R^d` | A real-valued vector space with `d` dimensions |

### What the Numbers Mean

Individual embedding dimensions are usually not human-interpretable. Dimension 17 does not reliably mean "sentiment" and dimension 212 does not reliably mean "finance." Meaning is distributed across many dimensions. The pattern of numbers as a whole represents useful features learned during training.

This is called a distributed representation.

### Key Properties

| Property | Explanation |
| --- | --- |
| Fixed length | A short sentence and a long paragraph can both become vectors of the same dimension. |
| Learned | The model learns useful geometry from data and training objectives. |
| Comparable | Similarity functions such as cosine similarity compare vectors. |
| Composable | Embeddings can be averaged, clustered, indexed, projected, or used as ML features. |
| Lossy | An embedding compresses the input; it does not preserve every detail. |

## 3. Why Embeddings Matter

Embeddings matter because they allow software systems to operate on meaning rather than exact symbols.

### Keyword Matching vs Semantic Matching

| Query | Document | Keyword Search | Embedding Search |
| --- | --- | --- | --- |
| "reset password" | "forgot login credentials" | Weak match | Strong match |
| "cheap flights" | "low-cost airfare" | Weak match | Strong match |
| "refund policy" | "returns and reimbursements" | Weak match | Strong match |
| "SKU-XJ19" | "SKU-XJ19 warranty" | Strong match | May be weak unless exact terms are preserved |

The final row is important. Embeddings are not always better than keyword search. Semantic models may miss exact identifiers, rare names, dates, part numbers, legal citations, or code symbols. Production search systems often combine vector search with keyword search.

### Main AI Engineering Uses

| Use case | How embeddings help |
| --- | --- |
| Semantic search | Find text with similar meaning to a query. |
| RAG | Retrieve relevant chunks before calling an LLM. |
| Recommendations | Find related products, articles, songs, jobs, or users. |
| Clustering | Group similar documents or tickets without labels. |
| Classification | Compare an input to label embeddings or train a classifier on vectors. |
| Duplicate detection | Detect near-duplicate questions, documents, or records. |
| Anomaly detection | Flag items far from normal clusters. |
| Routing | Send a query to the right tool, agent, queue, or model. |
| Memory | Store and retrieve relevant past interactions. |
| Evaluation | Compare generated text to reference text semantically. |

## 4. Core Vector-Space Intuition

Imagine every sentence as a point in a high-dimensional map. In that map, points near each other have related meaning.

```text
                    password / account cluster
                         o "forgot password"
                       o
              o "reset my login"        o "change credentials"


        cooking cluster
          o "bake salmon"
             o "roast fish with herbs"


                                  finance cluster
                                      o "quarterly revenue"
                                   o "earnings guidance"
```

Real embedding spaces are not 2D. They may have hundreds or thousands of dimensions. We use 2D diagrams only as intuition.

### Semantic Directions

Classic word embeddings showed that vector differences can encode relationships:

```text
king - man + woman ~= queen
Paris - France + Japan ~= Tokyo
```

This analogy behavior is not perfect and should not be treated as symbolic reasoning. But it illustrates a useful fact: embedding spaces often contain directions that correspond to latent semantic features.

### Neighborhoods

For most application work, the most important operation is nearest-neighbor search:

```text
Given query vector q, find the documents d_i whose vectors are closest to q.
```

This is the core operation behind semantic search and RAG.

## 5. Essential Math

You do not need advanced math to use embeddings well, but you must understand vector comparison.

### Vector

A vector is an ordered list of numbers:

```math
v = [v_1, v_2, ..., v_d]
```

Example:

```text
v = [0.20, -0.10, 0.70]
```

### Dimension

The dimension is the number of values in the vector.

```text
[0.20, -0.10, 0.70] has dimension 3.
```

Modern embedding models often use dimensions such as 384, 768, 1024, 1536, or 3072. Bigger is not automatically better. Larger vectors can improve quality, but they also cost more storage, memory bandwidth, and compute.

### Dot Product

The dot product multiplies corresponding dimensions and sums the result:

```math
a \cdot b = \sum_{i=1}^{d} a_i b_i
```

For two vectors:

```text
a = [1, 2, 3]
b = [4, 5, 6]

a dot b = 1*4 + 2*5 + 3*6 = 32
```

If vectors are normalized to unit length, dot product is equivalent to cosine similarity.

### Vector Norm

The L2 norm is the vector length:

```math
||a||_2 = \sqrt{\sum_{i=1}^{d} a_i^2}
```

### Normalization

L2 normalization rescales a vector to length 1:

```math
\hat{a} = \frac{a}{||a||_2}
```

Why it matters:

- It makes vector comparison depend on direction rather than magnitude.
- It makes cosine similarity computable as a dot product.
- It often improves retrieval consistency.

### Cosine Similarity

Cosine similarity measures the angle between two vectors:

```math
cos(a, b) = \frac{a \cdot b}{||a||_2 ||b||_2}
```

Range:

| Value | Meaning |
| --- | --- |
| `1` | Same direction, very similar |
| `0` | Orthogonal, unrelated under this representation |
| `-1` | Opposite direction |

In practice, many text embedding similarities are positive, especially when vectors come from models trained to place related examples close together.

### Euclidean Distance

Euclidean distance is straight-line distance:

```math
d(a, b) = ||a - b||_2 = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}
```

For normalized vectors, cosine similarity and Euclidean distance produce equivalent rankings:

```math
||a - b||_2^2 = 2 - 2(a \cdot b)
```

This is why many systems can use dot product, cosine similarity, or L2 distance depending on index support.

### Similarity vs Distance

| Concept | Higher means | Lower means | Examples |
| --- | --- | --- | --- |
| Similarity | More related | Less related | Cosine similarity, dot product |
| Distance | Less related | More related | Euclidean distance, angular distance |

### Top-K Retrieval

For query vector `q` and document vectors `D = {d_1, d_2, ..., d_n}`, retrieval ranks documents by similarity:

```math
score_i = sim(q, d_i)
```

Then returns the top `k` documents:

```math
topK(q, D) = argsort(score_i)[:k]
```

## 6. Types of Embeddings

Embeddings can represent many kinds of objects.

| Type | Input | Output | Used for |
| --- | --- | --- | --- |
| Word embedding | Single word or subword | Word vector | Classic NLP, analogy, features |
| Token embedding | Token in context | Contextual token vector | Transformer internals, token classification |
| Sentence embedding | Sentence | One vector | Semantic similarity, intent matching |
| Document embedding | Paragraph/document/chunk | One vector | Search, RAG, clustering |
| Query embedding | Search query/question | One vector | Retrieval |
| Code embedding | Code snippet/function | One vector | Code search, duplicate code detection |
| Image embedding | Image | Image vector | Image search, visual classification |
| Audio embedding | Audio clip | Audio vector | Audio search, speaker/music similarity |
| User embedding | User profile or behavior | User vector | Recommendations, personalization |
| Product embedding | Product text/image/behavior | Product vector | Recommendations, catalog search |
| Graph embedding | Node/edge/subgraph | Graph vector | Link prediction, node classification |

### Static vs Contextual

| Type | Description | Example |
| --- | --- | --- |
| Static embedding | Same word always gets same vector. | `bank` has one vector. |
| Contextual embedding | Word vector depends on surrounding words. | `bank` differs in "river bank" vs "bank account." |

### Single-Vector vs Multi-Vector

| Type | Description | Common systems |
| --- | --- | --- |
| Single-vector | One vector per sentence, document chunk, image, or object. | SBERT, OpenAI embeddings, BGE, E5 |
| Multi-vector | Multiple vectors per document, often one per token or passage unit. | ColBERT-style late interaction |

Single-vector embeddings are easier to store and search. Multi-vector systems can improve retrieval quality by preserving token-level interactions, but they require more storage and more complex search.

## 7. How Embedding Models Work

An embedding model is trained so that useful relationships become geometric relationships.

### High-Level Pipeline

```text
Text input
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Neural network encoder
  |
  v
Token-level hidden states
  |
  v
Pooling or projection
  |
  v
Embedding vector
```

### Training Objective Matters

Different training objectives produce different embedding behavior.

| Objective | Main idea | Typical result |
| --- | --- | --- |
| Predict nearby words | Words appearing in similar contexts become close. | Word2Vec-style word vectors |
| Matrix factorization | Co-occurrence statistics define geometry. | GloVe-style word vectors |
| Masked language modeling | Predict masked tokens from context. | Contextual token representations |
| Contrastive learning | Pull positives together, push negatives apart. | Strong retrieval/similarity embeddings |
| NLI supervision | Entailment pairs close, contradictions farther. | Sentence similarity embeddings |
| Query-document ranking | Questions close to answer passages. | Retrieval embeddings |
| Image-text contrastive | Matching image/caption pairs close. | Multimodal embeddings |

### Encoder, Pooling, Projection

Modern text embedding models commonly use transformer encoders.

Steps:

1. Tokenize the input.
2. Run a transformer to produce hidden states for each token.
3. Pool token states into one vector.
4. Optionally apply a projection layer.
5. Normalize the vector.

Common pooling methods:

| Pooling method | Description | Notes |
| --- | --- | --- |
| CLS pooling | Use the hidden state of a special classification token. | Works when model was trained for it. |
| Mean pooling | Average token hidden states, ignoring padding. | Common and robust for Sentence Transformers. |
| Max pooling | Take max value per dimension over tokens. | Less common for general embeddings. |
| Last-token pooling | Use final token hidden state. | Common for decoder-only embedding models. |
| Weighted pooling | Weight tokens by learned or positional importance. | More specialized. |

Mean pooling example:

```math
e = \frac{1}{N}\sum_{i=1}^{N} h_i
```

Where `h_i` is the hidden state of token `i` and `N` is the number of non-padding tokens.

## 8. Word Embeddings

Word embeddings were an early breakthrough because they represented words as dense vectors rather than one-hot IDs.

### One-Hot Encoding Problem

Vocabulary:

```text
[cat, dog, king, queen, laptop]
```

One-hot vectors:

```text
cat    = [1, 0, 0, 0, 0]
dog    = [0, 1, 0, 0, 0]
king   = [0, 0, 1, 0, 0]
queen  = [0, 0, 0, 1, 0]
laptop = [0, 0, 0, 0, 1]
```

Problems:

- Every word is equally distant from every other word.
- There is no built-in notion that `king` and `queen` are related.
- Vectors become huge for large vocabularies.
- The representation is sparse and not semantically meaningful.

### Dense Word Embeddings

Dense word embeddings solve this by representing each word with a compact learned vector.

```text
king  = [0.25, -0.13, 0.88, ...]
queen = [0.21, -0.10, 0.91, ...]
```

Similar words appear in similar contexts, so they learn similar vectors. This follows the distributional hypothesis:

```text
You shall know a word by the company it keeps.
```

### Word2Vec

Word2Vec popularized efficient neural word embeddings.

Main architectures:

| Architecture | Goal |
| --- | --- |
| CBOW | Predict a target word from surrounding context words. |
| Skip-gram | Predict surrounding context words from a target word. |

Skip-gram intuition:

```text
Sentence: "the cat sat on the mat"
Target:   "cat"
Context:  "the", "sat"

Train the model so vector(cat) helps predict vector(the) and vector(sat).
```

Training with the full vocabulary is expensive, so Word2Vec uses techniques such as negative sampling or hierarchical softmax.

Negative sampling teaches the model to distinguish true context pairs from random noise pairs:

```text
Positive pair: (cat, sat)
Negative pair: (cat, refrigerator)
Negative pair: (cat, inflation)
```

### GloVe

GloVe learns word vectors from global word-word co-occurrence statistics. It trains vectors so their dot products reflect the log of co-occurrence counts.

Simplified objective:

```math
J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
```

Where:

| Symbol | Meaning |
| --- | --- |
| `X_ij` | Number of times word `j` appears in context of word `i` |
| `w_i` | Main vector for word `i` |
| `\tilde{w}_j` | Context vector for word `j` |
| `b_i`, `\tilde{b}_j` | Bias terms |
| `f(X_ij)` | Weighting function that limits the effect of very frequent pairs |

### FastText

FastText extends word embeddings with subword information. Instead of learning only a vector for each full word, it also learns vectors for character n-grams.

Why this helps:

- Better handling of rare words.
- Better handling of misspellings.
- Better handling of morphologically rich languages.
- Ability to produce vectors for words not seen during training.

Example:

```text
"embedding" might be represented using subwords like:
emb, mbe, bed, edd, ddi, din, ing
```

### Limitations of Static Word Embeddings

Static word embeddings cannot handle polysemy well.

Example:

```text
"I deposited cash at the bank."
"The boat stopped near the river bank."
```

Classic Word2Vec or GloVe gives `bank` one vector. Contextual models solve this by producing token embeddings that depend on surrounding words.

## 9. Contextual Token Embeddings

Contextual embeddings are generated by models such as BERT and other transformers. The same token can receive different vectors in different contexts.

### Static vs Contextual Example

| Sentence | Token | Representation |
| --- | --- | --- |
| "The bank approved the loan." | bank | Financial institution meaning |
| "The river bank collapsed." | bank | Land beside river meaning |

### Transformer Hidden States

In a transformer, each layer produces a vector for each token. These hidden states encode information from the surrounding context through self-attention.

```text
Input tokens:
[CLS] the bank approved the loan [SEP]

Layer 1 states:
h_1, h_2, h_3, ...

Layer 12 states:
H_1, H_2, H_3, ...
```

These contextual token vectors are excellent internal representations, but they are not automatically good sentence embeddings. A vanilla BERT `[CLS]` vector is often not ideal for direct cosine similarity unless the model has been trained for sentence similarity or retrieval.

## 10. Sentence and Document Embeddings

Sentence/document embeddings represent a whole text span as one vector.

### Why BERT Alone Was Not Enough

The SBERT paper showed that using standard BERT as a cross-encoder for pairwise sentence comparison is computationally expensive. To compare 10,000 sentences pairwise, standard BERT would need around 50 million pair computations. SBERT instead produces independent sentence embeddings that can be compared with cosine similarity.

### Bi-Encoder Architecture

A bi-encoder embeds the query and document independently:

```text
Query:    "How do I reset my password?"  --> Encoder --> q vector
Document: "Steps for account recovery"   --> Encoder --> d vector

score = cosine_similarity(q, d)
```

Benefits:

- Document vectors can be precomputed.
- Search can use vector indexes.
- Latency is low.
- Works well for large-scale retrieval.

Limitations:

- Query and document do not interact token-by-token during scoring.
- It can miss fine-grained matching details.
- A reranker may be needed for best quality.

### Cross-Encoder Architecture

A cross-encoder processes the query and document together:

```text
[CLS] query tokens [SEP] document tokens [SEP] --> Transformer --> relevance score
```

Benefits:

- Higher accuracy for ranking small candidate sets.
- Full token-level interaction between query and document.

Limitations:

- Too expensive to run against every document.
- Cannot precompute a single reusable document vector.

### Retrieve and Rerank Pattern

Production search often uses both:

```text
1. Bi-encoder retrieves top 50-200 candidates quickly.
2. Cross-encoder reranks those candidates accurately.
3. The application returns or passes the best results to an LLM.
```

| Stage | Model type | Input size | Purpose |
| --- | --- | --- | --- |
| Retrieval | Bi-encoder embedding model | Entire corpus | High recall candidate generation |
| Reranking | Cross-encoder reranker | Top candidates only | Precision and final ordering |

## 11. Multimodal Embeddings

Multimodal embeddings place different data types in compatible vector spaces.

CLIP is a major example. It trains an image encoder and a text encoder so matching image-caption pairs are close together.

```text
Image of a dog  --> image encoder --> vector A
"a dog in a park" --> text encoder --> vector B

If A and B are close, the text describes the image.
```

### Multimodal Use Cases

| Use case | Query | Corpus | Result |
| --- | --- | --- | --- |
| Text-to-image search | Text | Images | Images matching description |
| Image-to-text retrieval | Image | Captions/docs | Relevant captions or descriptions |
| Visual question answering support | Text + image | Multimodal knowledge | Grounded context |
| Product search | Text or image | Product catalog | Visually/semantically similar products |
| Moderation | Text/image/video | Policy examples | Similar policy categories |

### Cross-Modal Alignment

Multimodal models rely on alignment. A text vector and image vector must be comparable. This only works if the model was trained to align those modalities. You cannot safely compare embeddings from unrelated models unless they share the same embedding space.

## 12. Dense, Sparse, and Hybrid Representations

### Dense Embeddings

Dense embeddings are floating-point vectors where most dimensions contain non-zero values.

Example:

```text
[0.014, -0.221, 0.087, ..., 0.006]
```

Strengths:

- Captures semantic similarity.
- Handles paraphrases well.
- Works across languages when trained for multilingual alignment.
- Useful for retrieval, clustering, and recommendations.

Weaknesses:

- Can miss exact terms.
- Harder to debug than keyword matching.
- Requires embedding generation and vector storage.
- Retrieval quality depends heavily on training data and chunk design.

### Sparse Representations

Sparse vectors have many zeros and usually correspond to vocabulary dimensions. TF-IDF and BM25 are classic sparse retrieval methods.

Strengths:

- Strong exact matching.
- Good for names, IDs, rare terms, and keywords.
- Interpretable.
- Mature indexing technology.

Weaknesses:

- Weak paraphrase handling.
- Vocabulary mismatch issues.
- Less semantic generalization.

### Learned Sparse Embeddings

Models such as SPLADE produce sparse lexical representations with learned expansion. They can combine neural semantic behavior with sparse inverted-index efficiency.

### Hybrid Search

Hybrid search combines sparse and dense retrieval.

Common score fusion:

```math
score = \alpha \cdot dense\_score + (1 - \alpha) \cdot sparse\_score
```

Another common method is reciprocal rank fusion:

```math
RRF(d) = \sum_{r \in rankings} \frac{1}{k + rank_r(d)}
```

Hybrid search is often better than pure dense retrieval for enterprise documents because it handles both semantic concepts and exact identifiers.

## 13. Similarity Search From Scratch

Before using a vector database, understand brute-force retrieval.

### Basic Algorithm

```text
Input:
  query text
  document texts
  embedding model
  top_k

Process:
  1. Embed every document.
  2. Embed the query.
  3. Compute similarity between query vector and each document vector.
  4. Sort by similarity descending.
  5. Return the top_k documents.
```

### Complexity

For `n` documents and `d` dimensions:

```text
Brute-force search cost: O(n * d)
```

This is fine for hundreds or thousands of vectors. It becomes expensive for millions or billions. At that scale, use approximate nearest-neighbor indexes or vector databases.

### Exact Search vs Approximate Search

| Method | Behavior | Best for |
| --- | --- | --- |
| Exact search | Compares query to every vector. | Small datasets, correctness baselines |
| Approximate nearest neighbor | Searches an index and may miss some exact nearest neighbors. | Large datasets, low latency |

Approximate search trades a small amount of recall for major speed and memory improvements.

## 14. Embeddings in Semantic Search

Semantic search uses embeddings to retrieve documents by meaning.

### Search Pipeline

```text
Offline indexing:
  Documents -> clean -> chunk -> embed -> store vectors + metadata

Online query:
  User query -> embed -> vector search -> retrieve top results -> return/rerank/use
```

### Diagram

```text
                         OFFLINE
┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────┐
│ Documents   │ -> │ Chunking  │ -> │ Embedding    │ -> │ Vector Store │
└─────────────┘    └───────────┘    └──────────────┘    └──────────────┘

                         ONLINE
┌─────────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────┐
│ User Query  │ -> │ Embed     │ -> │ Similarity   │ -> │ Top Results  │
└─────────────┘    └───────────┘    └──────────────┘    └──────────────┘
```

### Query and Document Encoders

Some embedding models use the same encoder for queries and documents. Others are trained with different prompts or heads.

Examples:

```text
Query prompt:    "query: how do I reset my password"
Document prompt: "passage: reset your password from account settings"
```

If a model was trained with such prefixes, use them. Ignoring required prompts can reduce retrieval quality.

### Symmetric vs Asymmetric Search

| Search type | Description | Example |
| --- | --- | --- |
| Symmetric | Query and documents are similar length/type. | Find duplicate questions. |
| Asymmetric | Query is short, document is longer. | Search documents with a user question. |

Models trained for semantic textual similarity may work well for symmetric similarity but underperform on asymmetric retrieval. For RAG, prefer models trained on query-document retrieval.

## 15. Embeddings in RAG

Retrieval-augmented generation uses embeddings to retrieve relevant context before generation.

### RAG Pipeline

```text
1. Collect documents.
2. Split documents into chunks.
3. Generate embeddings for chunks.
4. Store chunks, metadata, and vectors.
5. Embed user query.
6. Retrieve similar chunks.
7. Optionally rerank chunks.
8. Insert selected context into LLM prompt.
9. Generate answer with citations or grounded evidence.
```

### Why Embeddings Are Useful in RAG

LLMs do not automatically know your private documents, latest policies, database rows, or internal codebase. Embeddings allow the system to retrieve relevant external knowledge at query time.

### Chunking Matters

Embedding quality is tied to chunk quality.

| Chunking choice | Effect |
| --- | --- |
| Too small | Loses context; retrieved fragments may be incomplete. |
| Too large | Embedding becomes diluted; irrelevant text pollutes the vector. |
| No overlap | Boundary information may be lost. |
| Too much overlap | Higher cost and duplicate retrieval. |
| Structure-aware chunking | Often improves retrieval for docs, code, and markdown. |

Common starting points:

| Content type | Starting chunk strategy |
| --- | --- |
| Markdown/docs | Split by headings, then by token budget. |
| PDFs | Recover sections where possible; avoid arbitrary page splits. |
| Code | Split by function/class/module. |
| Support articles | One article or section per chunk. |
| Legal/contracts | Split by clause/section with metadata. |

### Metadata Matters

Store metadata with vectors:

```json
{
  "source": "refund-policy.md",
  "section": "International refunds",
  "url": "https://example.com/refunds",
  "created_at": "2026-08-09",
  "access_level": "internal",
  "document_id": "policy-123",
  "chunk_index": 7
}
```

Metadata enables filtering, citations, access control, freshness sorting, and debugging.

### Embeddings Are Not the Whole RAG System

Embeddings solve candidate retrieval. They do not solve:

- Whether the retrieved chunks are sufficient.
- Whether the LLM follows the evidence.
- Whether the answer cites sources correctly.
- Whether the corpus is up to date.
- Whether access control is enforced.
- Whether the user query needs decomposition.

RAG quality depends on the entire pipeline.

## 16. Common Use Cases

### Semantic Search

Find relevant documents by meaning.

Examples:

- Support knowledge base search.
- Internal policy search.
- Codebase search.
- Legal clause retrieval.
- Academic paper search.

### Recommendations

Embeddings can represent items and users.

```text
recommend(item) = nearest_neighbors(embedding(item), catalog_embeddings)
```

Examples:

- Similar articles.
- Related products.
- Candidate-job matching.
- Playlist continuation.
- Course recommendations.

### Clustering

Cluster embeddings to discover groups.

Examples:

- Group customer feedback into themes.
- Organize support tickets.
- Cluster product reviews.
- Detect topical communities in documents.

Common algorithms:

| Algorithm | Notes |
| --- | --- |
| K-means | Requires choosing number of clusters. |
| HDBSCAN | Finds variable-density clusters and outliers. |
| Agglomerative clustering | Useful for hierarchy and small datasets. |
| Spectral clustering | Useful for graph-like similarity structure. |

### Classification

Embeddings can be used for classification in two main ways.

Zero-shot style:

```text
1. Embed input text.
2. Embed label descriptions.
3. Choose label with highest similarity.
```

Supervised style:

```text
1. Embed training examples.
2. Train logistic regression, random forest, or another classifier.
3. Predict labels from embeddings.
```

### Duplicate Detection

Compare embeddings to find near-duplicates.

Examples:

- Duplicate support tickets.
- Duplicate FAQ entries.
- Similar product listings.
- Repeated bug reports.

### Anomaly Detection

Outliers in embedding space may represent unusual records.

Examples:

- Strange support tickets.
- Fraud-like behavior descriptions.
- Off-topic documents in a corpus.
- Misclassified data.

### Routing

Route a user query to the right handler.

Examples:

- Billing agent vs technical support agent.
- SQL tool vs web search tool.
- Cheap small model vs expensive large model.
- Human escalation queue.

### Memory

Agent memory systems often store past events as embeddings and retrieve the most relevant memories for the current context.

Important caution: relevance is not the same as truth, safety, or permission. Memory retrieval must still enforce access control and freshness.

## 17. Choosing an Embedding Model

There is no universally best embedding model. Choose based on task, data, latency, cost, privacy, language, and evaluation results.

### Decision Criteria

| Criterion | Questions to ask |
| --- | --- |
| Task | Search, RAG, clustering, classification, recommendation, multimodal? |
| Domain | General web text, legal, medical, code, finance, support, scientific papers? |
| Language | English only, multilingual, cross-lingual retrieval? |
| Input length | Short queries, long documents, code files, PDFs? |
| Dimension | What storage and latency can you afford? |
| Deployment | API, self-hosted GPU, CPU-only, edge device? |
| Privacy | Can text leave your infrastructure? |
| Cost | Per-token API pricing vs infrastructure cost? |
| Quality | Does it win on your own evaluation set? |
| Ecosystem | Does it integrate with your vector DB and framework? |

### API vs Open-Source Models

| Option | Strengths | Trade-offs |
| --- | --- | --- |
| API embeddings | Easy, scalable, low ops, strong general quality. | Per-call cost, data leaves environment, provider dependency. |
| Open-source local | Data control, no per-call API cost, customizable. | Requires hosting, optimization, monitoring, model selection. |
| Fine-tuned model | Better domain fit. | Requires labeled data, evaluation, training expertise. |

### Dimension Trade-Offs

Higher dimension can improve representational capacity, but increases cost.

Storage estimate for float32 vectors:

```math
storage\_bytes = num\_vectors \times dimensions \times 4
```

Examples:

| Vectors | Dimensions | Float32 storage |
| ---: | ---: | ---: |
| 100,000 | 384 | ~153 MB |
| 100,000 | 1536 | ~614 MB |
| 1,000,000 | 768 | ~3.1 GB |
| 1,000,000 | 1536 | ~6.1 GB |
| 10,000,000 | 1536 | ~61.4 GB |

This excludes metadata, index overhead, replicas, quantization tables, and database storage overhead.

### Practical Recommendation

Start simple:

1. Use a strong general-purpose model.
2. Build a small evaluation set from real queries.
3. Measure retrieval quality.
4. Inspect failures.
5. Improve chunking, metadata, hybrid search, or reranking before fine-tuning.
6. Fine-tune only when you have evidence that model mismatch is the bottleneck.

## 18. Model Families to Know

### OpenAI `text-embedding-3`

OpenAI provides API-based embedding models such as `text-embedding-3-small` and `text-embedding-3-large`. The OpenAI documentation states default dimensions of 1536 for `text-embedding-3-small` and 3072 for `text-embedding-3-large`, with an optional `dimensions` parameter for supported shortening. OpenAI docs also note that their embeddings are normalized to length 1, making cosine similarity and dot product rankings equivalent.

Strengths:

- Simple API usage.
- Published benchmark results in the OpenAI documentation.
- Good for fast prototyping and production systems that can use external APIs.

Trade-offs:

- Requires sending input text to the API provider.
- Per-token cost.
- External dependency.

### Sentence Transformers

Sentence Transformers is a major open-source ecosystem for sentence, text, sparse, cross-encoder, and multimodal embeddings. It popularized easy bi-encoder usage for semantic similarity and retrieval.

Common model example:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model is small and often used for tutorials. For production, compare newer models on your task.

### BGE

BGE models from BAAI are widely used open-source embedding models. They include general embedding models, rerankers, and multilingual variants. Many BGE models use query/document instructions or prompts, so follow the model card.

### E5

E5 models are trained for text embeddings and commonly use prefixes such as:

```text
query: ...
passage: ...
```

Using the correct prefix can significantly affect quality.

### Nomic Embed

Nomic Embed models are open embedding models commonly used for local and privacy-aware embedding workflows.

### Cohere Embed

Cohere provides API embedding models with multilingual and retrieval-oriented options. It is commonly considered when multilingual retrieval is important.

### Domain-Specific Models

Examples:

- Code embedding models for code search.
- Biomedical embedding models for medical literature.
- Legal embedding models for contracts and case law.
- Scientific embedding models for papers and citations.

Use domain-specific models when your corpus vocabulary and relevance patterns differ strongly from general web text.

## 19. Evaluation

Embedding quality must be evaluated on the task you care about.

### Why Generic Benchmarks Are Not Enough

Benchmarks such as MTEB are useful for comparing general capabilities, but your production task may differ:

- Your documents may be noisy PDFs.
- Your queries may contain internal acronyms.
- Your relevance criteria may depend on policy, date, or permissions.
- Your users may search with incomplete or misspelled terms.
- Your corpus may contain many near-duplicate pages.

Always create a domain-specific evaluation set.

### MTEB

MTEB, the Massive Text Embedding Benchmark, evaluates embedding models across tasks such as classification, clustering, pair classification, reranking, retrieval, semantic textual similarity, and summarization. It has expanded to multilingual and multimodal benchmarks.

Use MTEB to shortlist models, not to blindly choose the final production model.

### Retrieval Metrics

| Metric | Meaning | Good for |
| --- | --- | --- |
| Recall@K | Fraction of queries where a relevant document appears in top K. | RAG candidate retrieval |
| Precision@K | Fraction of top K results that are relevant. | Search quality |
| MRR | Mean reciprocal rank of first relevant result. | First-answer ranking |
| nDCG@K | Ranking metric that accounts for graded relevance and position. | Search/ranking evaluation |
| Hit Rate@K | Whether at least one relevant result appears in top K. | Simple RAG retrieval check |

### Semantic Textual Similarity Metrics

For sentence similarity tasks, models are often evaluated by correlation between predicted similarity and human labels.

| Metric | Meaning |
| --- | --- |
| Spearman correlation | Rank correlation between model scores and human scores. |
| Pearson correlation | Linear correlation between model scores and human scores. |

### Classification Metrics

When embeddings feed a classifier:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix

### Clustering Metrics

If labels exist:

- Adjusted Rand Index
- Normalized Mutual Information
- V-measure

If labels do not exist:

- Silhouette score
- Manual cluster inspection
- Topic coherence analysis

### Building a Retrieval Eval Set

Create examples like:

```json
{
  "query": "Can customers get a refund after 30 days?",
  "relevant_doc_ids": ["refund_policy_section_4"],
  "hard_negatives": ["refund_policy_section_2", "shipping_policy_section_1"]
}
```

Include:

- Real user queries.
- Paraphrases.
- Exact-term queries.
- Acronyms and internal names.
- Edge cases.
- Negative examples.
- Queries with no answer.

## 20. Production Engineering

Embeddings are simple conceptually, but production systems need careful engineering.

### Indexing Architecture

```text
Source systems
  |
  v
Extract/load documents
  |
  v
Normalize and chunk
  |
  v
Embed in batches
  |
  v
Store vectors + metadata + source text
  |
  v
Serve retrieval API
```

### Batch Embedding

Batching improves throughput and reduces overhead.

Consider:

- Provider batch limits.
- Token limits.
- Retry behavior.
- Rate limits.
- Idempotent writes.
- Partial failures.
- Backpressure.

### Caching

Cache embeddings by content hash:

```text
cache_key = sha256(model_name + model_version + normalized_text)
```

Include model name and version. If the embedding model changes, old cached vectors should not be mixed with new vectors.

### Versioning

Track:

- Embedding model name.
- Model version or provider release.
- Dimension.
- Normalization behavior.
- Chunking strategy version.
- Preprocessing version.
- Index build timestamp.

Changing any of these can change retrieval behavior.

### Do Not Mix Embedding Spaces

Vectors from different models are usually not comparable.

Bad:

```text
Documents embedded with Model A.
Queries embedded with Model B.
```

Unless the models were explicitly trained to share a space, this breaks similarity search.

### Updating Embeddings

When documents change:

1. Detect changed source content.
2. Recompute chunks.
3. Delete old chunks for that document.
4. Embed new chunks.
5. Upsert vectors and metadata.
6. Verify retrieval.

For model upgrades:

1. Build a parallel index with the new model.
2. Run offline evaluation.
3. Shadow traffic if possible.
4. Compare online metrics.
5. Cut over gradually.
6. Keep rollback path.

### Latency Budget

Online retrieval latency includes:

- Query preprocessing.
- Query embedding call.
- Vector search.
- Metadata filtering.
- Reranking.
- Prompt assembly.
- LLM generation.

Query embedding can dominate latency if it uses a remote API. Cache frequent queries and batch background workloads where possible.

### Storage Optimization

Options:

| Technique | Effect | Trade-off |
| --- | --- | --- |
| Lower dimensions | Less storage and faster search. | May reduce quality. |
| Float16 | Half storage vs float32. | Small precision loss. |
| Int8 quantization | Much smaller vectors. | More quality loss; needs calibration. |
| Binary quantization | Very compact. | Larger quality trade-off. |
| Product quantization | Compresses vectors for ANN. | Approximate distances. |
| Matryoshka truncation | Use prefix dimensions from trained model. | Requires compatible model/training. |

### Security and Privacy

Embeddings can leak information about source text. They are not safe just because they are numeric.

Risks:

- Sensitive text sent to external embedding APIs.
- Unauthorized retrieval of private chunks.
- Embedding inversion or membership inference research risk.
- Logs containing raw text or vectors.
- Multi-tenant data leakage through filters or IDs.

Controls:

- Enforce metadata-based access control before returning chunks.
- Minimize sensitive text sent to providers.
- Encrypt stores and backups.
- Avoid logging raw sensitive content.
- Separate tenant indexes when appropriate.
- Treat embeddings as derived sensitive data.

### Observability

Log and monitor:

- Query text or safe redacted representation.
- Retrieved document IDs.
- Similarity scores.
- Reranker scores.
- Selected context.
- User feedback.
- No-result rates.
- Latency per stage.
- Embedding provider errors.
- Index freshness.

For privacy-sensitive systems, log IDs and metrics instead of full content.

## 21. Advanced Topics

### Contrastive Learning

Contrastive learning trains embeddings by pulling positive pairs together and pushing negative pairs apart.

Example pairs:

```text
Positive: query = "reset password", document = "How to reset your password"
Negative: query = "reset password", document = "Shipping refund policy"
```

A common contrastive loss is InfoNCE:

```math
L_i = -\log \frac{\exp(sim(q_i, d_i^+) / \tau)}{\sum_{j=1}^{N}\exp(sim(q_i, d_j) / \tau)}
```

Where:

| Symbol | Meaning |
| --- | --- |
| `q_i` | Query vector |
| `d_i^+` | Positive document vector |
| `d_j` | Candidate documents, often including in-batch negatives |
| `sim` | Similarity function |
| `tau` | Temperature parameter |

### In-Batch Negatives

In a batch of query-positive pairs, other positives in the batch can be used as negatives for each query.

This is efficient but can introduce false negatives. For example, two different passages may both answer the same query.

### Hard Negatives

Hard negatives are incorrect documents that look similar to the query.

Example:

```text
Query: "How do I cancel my subscription?"
Positive: "Cancel your paid plan"
Hard negative: "Pause your subscription"
Easy negative: "Chocolate cake recipe"
```

Hard negatives make training more useful because the model learns fine distinctions.

### Dual Encoders for Retrieval

Dense Passage Retrieval uses a dual-encoder architecture for open-domain QA. The query encoder and passage encoder independently map text into dense vectors, enabling efficient retrieval over large corpora.

### Late Interaction: ColBERT

ColBERT keeps multiple token-level vectors for queries and documents and uses a late interaction scoring function. Instead of compressing a document into one vector, it preserves token-level matching while still allowing document representations to be precomputed.

Simplified scoring:

```math
score(q, d) = \sum_{i \in q} \max_{j \in d} q_i \cdot d_j
```

Strength:

- Better fine-grained retrieval than single-vector bi-encoders.

Trade-off:

- More storage and more complex indexing.

### Matryoshka Embeddings

Matryoshka embeddings are trained so earlier dimensions contain a useful compressed representation. This allows truncating a vector to smaller dimensions while preserving much of its quality.

Example:

```text
Full vector:      1024 dimensions
Truncated vector: 768, 512, 256, or 128 dimensions
```

This is useful when one model must support different cost/latency tiers.

### Quantization

Quantization reduces vector precision.

| Format | Bytes per dimension | Notes |
| --- | ---: | --- |
| float32 | 4 | Standard high precision |
| float16 | 2 | Common compression with small quality impact |
| int8 | 1 | Strong compression; may need calibration |
| binary | 0.125 | Very compact; larger accuracy trade-off |

Quantization is especially important for large-scale retrieval.

### Dimensionality Reduction

Methods like PCA, UMAP, and t-SNE can reduce dimensions.

Use cases:

- Visualization.
- Compression.
- Noise reduction.

Cautions:

- t-SNE and UMAP are mainly visualization tools, not general retrieval optimizers.
- PCA can reduce storage but may hurt retrieval.
- Always evaluate after reducing dimensions.

### Approximate Nearest Neighbor Indexes

Common ANN approaches:

| Index type | Idea | Notes |
| --- | --- | --- |
| HNSW | Graph-based navigation. | High recall, fast search, memory-heavy. |
| IVF | Partition vectors into clusters, search selected clusters. | Good for large datasets; needs training. |
| PQ | Compress vectors into product quantized codes. | Saves memory, approximate distances. |
| ScaNN | Optimized vector search with partitioning/quantization. | Used in high-performance retrieval. |
| DiskANN | Disk-based ANN for very large indexes. | Useful when vectors exceed RAM. |

Faiss is a widely used library for efficient similarity search and clustering of dense vectors. Vector databases often wrap or implement related ANN techniques and add metadata filtering, persistence, replication, and operational features.

### Embedding Fine-Tuning

Fine-tune when:

- Generic embeddings fail on domain-specific language.
- You have labeled query-document pairs.
- Your evaluation set shows consistent retrieval errors.
- Chunking, hybrid search, and reranking are not enough.

Data formats:

| Format | Example |
| --- | --- |
| Positive pairs | `(query, relevant_document)` |
| Triplets | `(query, positive_document, negative_document)` |
| Graded relevance | `(query, document, relevance_score)` |
| Click logs | Query, clicked result, skipped results |

Cautions:

- Click logs are biased by previous ranking.
- Fine-tuning can overfit small datasets.
- Bad negatives can damage model behavior.
- Always compare against a strong baseline.

## 22. Common Pitfalls

### Pitfall 1: Treating Embeddings as Magic Meaning

Embeddings capture patterns learned from data. They are not symbolic understanding and do not guarantee truth.

### Pitfall 2: Ignoring Exact Terms

Pure vector search may miss product IDs, legal citations, rare names, versions, and error codes. Use hybrid search or metadata filters.

### Pitfall 3: Mixing Models

Do not compare vectors from different embedding models unless they were designed to share a space.

### Pitfall 4: Bad Chunking

Poor chunks create poor embeddings. Arbitrary chunks can split definitions, tables, code, or policies in ways that make retrieval fail.

### Pitfall 5: Over-Relying on Similarity Score Thresholds

Similarity scores are model-specific and corpus-specific. A threshold that works in one domain may fail in another.

### Pitfall 6: No Evaluation Set

Without labeled queries and expected results, model selection becomes guesswork.

### Pitfall 7: Ignoring Access Control

Vector search can retrieve private chunks unless filters are applied correctly. Always enforce permissions before returning content to the user or LLM.

### Pitfall 8: Assuming Larger Dimensions Always Win

Large vectors cost more. Smaller models with good training can outperform larger generic models on specific tasks.

### Pitfall 9: Embedding Raw Noisy Text

Headers, footers, navigation menus, repeated disclaimers, OCR noise, and broken tables can pollute embeddings.

### Pitfall 10: Forgetting Reindexing

When the embedding model, preprocessing, or chunking changes, rebuild the index or maintain parallel versions.

## 23. Code Examples

### Raw Cosine Similarity with NumPy

```python
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 4.0])

print(cosine_similarity(a, b))
```

### Brute-Force Semantic Search with Precomputed Vectors

```python
import numpy as np


documents = [
    "Reset your password from account settings.",
    "Refunds are available within 30 days.",
    "International shipping takes 5 to 10 business days.",
]

# Placeholder vectors for demonstration. In real code, generate these with an embedding model.
doc_vectors = np.array([
    [0.90, 0.10, 0.05],
    [0.05, 0.88, 0.10],
    [0.10, 0.12, 0.91],
], dtype=np.float32)

query_vector = np.array([0.85, 0.12, 0.08], dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


doc_vectors = l2_normalize(doc_vectors)
query_vector = query_vector / np.linalg.norm(query_vector)

scores = doc_vectors @ query_vector
top_indices = np.argsort(scores)[::-1][:2]

for idx in top_indices:
    print(float(scores[idx]), documents[idx])
```

### Generate Embeddings with OpenAI

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Embeddings represent text as vectors for semantic search.",
)

embedding = response.data[0].embedding
print(len(embedding))
print(embedding[:5])
```

### Generate Embeddings with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "How do I reset my password?",
    "I forgot my login credentials.",
    "The refund policy lasts 30 days.",
]

embeddings = model.encode(texts, normalize_embeddings=True)

print(embeddings.shape)
print(embeddings[0] @ embeddings[1])
```

### Simple Retrieval with Sentence Transformers

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "To reset your password, open account settings and choose security.",
    "Refunds are available for most purchases within 30 days.",
    "Shipping times depend on destination and carrier.",
    "You can update your email address from your profile page.",
]

query = "I forgot my password. What should I do?"

doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode([query], normalize_embeddings=True)[0]

scores = doc_embeddings @ query_embedding
top_k = 2
top_indices = np.argsort(scores)[::-1][:top_k]

for idx in top_indices:
    print(f"score={scores[idx]:.3f} document={documents[idx]}")
```

### Zero-Shot Classification with Label Embeddings

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

labels = [
    "billing or payment issue",
    "technical login problem",
    "shipping and delivery question",
]

text = "I cannot sign in because my password does not work."

label_embeddings = model.encode(labels, normalize_embeddings=True)
text_embedding = model.encode([text], normalize_embeddings=True)[0]

scores = label_embeddings @ text_embedding
best = int(np.argmax(scores))

print(labels[best], float(scores[best]))
```

### Clustering Embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

texts = [
    "Reset my password",
    "Forgot login credentials",
    "Change account email",
    "Refund request",
    "Return my order",
    "Cancel purchase",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

for text, label in zip(texts, labels):
    print(label, text)
```

### Token Counting Before Embedding with OpenAI Tokenizer

```python
import tiktoken


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


print(count_tokens("Embeddings are useful for semantic search."))
```

## 24. Practical Project

Build a small embedding search system before using a vector database.

### Goal

Embed a set of documents, compute cosine similarity between a user query and every document, and return the top matches.

### Dataset

Start with 20 to 100 short documents such as:

- Support FAQ entries.
- Notes from this repository.
- Product descriptions.
- Course lesson summaries.
- Markdown sections.

### Requirements

1. Load documents from local files.
2. Split them into reasonable chunks.
3. Generate embeddings.
4. Store embeddings in a local JSON, CSV, or NumPy file.
5. Accept a query.
6. Embed the query.
7. Compute cosine similarity with raw NumPy.
8. Return top 5 chunks with scores and source metadata.
9. Manually inspect whether results are relevant.
10. Write 10 test queries and expected results.

### Stretch Goals

- Add keyword search and compare against vector search.
- Add hybrid score fusion.
- Add a reranker.
- Add metadata filters.
- Add chunk size experiments.
- Add evaluation metrics such as Recall@5.
- Visualize embeddings with UMAP or t-SNE.

## 25. Quick Reference

### Core Definitions

| Term | Meaning |
| --- | --- |
| Embedding | Numeric vector representation of an object. |
| Embedding model | Model that maps inputs to vectors. |
| Dimension | Number of values in a vector. |
| Cosine similarity | Similarity based on vector angle. |
| Dot product | Sum of element-wise products. |
| Normalization | Rescaling vector to unit length. |
| Vector search | Finding nearest vectors to a query vector. |
| ANN | Approximate nearest-neighbor search for scale. |
| Bi-encoder | Encodes query and document independently. |
| Cross-encoder | Scores query-document pair jointly. |
| Reranker | Model that reorders retrieved candidates. |
| Hybrid search | Combines dense vector and sparse keyword retrieval. |

### When to Use Embeddings

Use embeddings when:

- Meaning matters more than exact wording.
- You need semantic search.
- You need RAG retrieval.
- You want to cluster or classify text.
- You need recommendations or similarity matching.
- You need to route queries by intent.

Be careful when:

- Exact identifiers matter.
- The corpus has many rare terms.
- Access control is complex.
- You lack evaluation data.
- Source text is noisy.
- The embedding model is not trained for your language/domain.

### Best Practices

- Evaluate on real queries.
- Normalize vectors when using cosine/dot product search.
- Keep model, dimension, preprocessing, and chunking versions.
- Do not mix vectors from different models.
- Use metadata filters for access control and scoping.
- Use hybrid search for exact terms and rare identifiers.
- Use reranking when top results need high precision.
- Cache embeddings by content hash.
- Rebuild indexes when model or chunking changes.
- Inspect retrieval failures manually.

## 26. Self-Check Questions

1. What is the difference between a sparse representation and a dense embedding?

Answer: Sparse representations usually map terms to mostly zero vocabulary-based vectors and are strong for exact matching. Dense embeddings are compact learned vectors with most dimensions non-zero and are strong for semantic similarity.

2. Why does cosine similarity ignore vector magnitude?

Answer: It divides the dot product by both vector norms, so it measures direction rather than length.

3. Why should you not compare embeddings from two unrelated models?

Answer: Each model learns its own vector space. Dimensions and geometry are not aligned unless the models were explicitly trained to be compatible.

4. Why is a bi-encoder faster than a cross-encoder for large-scale search?

Answer: A bi-encoder precomputes document vectors and compares them cheaply to query vectors. A cross-encoder must process each query-document pair jointly.

5. Why might hybrid search outperform pure vector search?

Answer: Hybrid search combines semantic matching from embeddings with exact lexical matching from sparse search, helping with names, IDs, rare terms, and paraphrases.

6. What changes require reindexing embeddings?

Answer: Changes to the embedding model, model version, dimensions, preprocessing, chunking, normalization, or source documents can require reindexing.

7. What is the role of a reranker in RAG?

Answer: A reranker improves the ordering of a small candidate set retrieved by the embedding search stage, usually increasing precision before context is passed to the LLM.

8. Why can too-large chunks hurt retrieval?

Answer: Large chunks can mix multiple topics, causing the embedding to represent an average of unrelated content and reducing retrieval precision.

## 27. Resources and References

### Official Documentation and Practical Guides

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - API usage, model dimensions, use cases, normalization notes, and examples.
- [Hugging Face: Getting Started With Embeddings](https://huggingface.co/blog/getting-started-with-embeddings) - Practical tutorial using Sentence Transformers and semantic search.
- [Sentence Transformers Documentation](https://www.sbert.net/) - Sentence embeddings, semantic search, reranking, sparse encoders, quantization, evaluation, and training.
- [Faiss Documentation](https://faiss.ai/) - Efficient similarity search and clustering of dense vectors.
- [MTEB Benchmark Repository](https://github.com/embeddings-benchmark/mteb) - Evaluation framework and benchmark for embeddings.

### Foundational Papers

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Word2Vec paper by Mikolov et al.
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) - Stanford GloVe project and paper resources.
- [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) - Early sequence representation work.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) - Contextual transformer representations.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) - Efficient sentence embeddings for similarity search.
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) - Dual-encoder dense retrieval for QA.
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) - Multi-vector late interaction retrieval.
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - CLIP and image-text embedding alignment.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) - Benchmarking text embeddings across tasks.

### Related Roadmap Sections

- [Phase 4 - Vector Databases](../04.Vector_Databases/README.md) - Scaling similarity search with vector indexes and databases.
- [Phase 5 - Retrieval-Augmented Generation](../05.RAG/README.md) - Using embeddings to retrieve context for LLM generation.
- [Phase 13 - Multimodal and Voice Agents](../13.Multimodal_and_Voice_Agents/README.md) - Multimodal embeddings and retrieval use cases.
