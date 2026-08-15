# Vector Search Fundamentals

Vector search is the retrieval technique that finds items whose numeric representations are close to a query vector. In AI applications, those vectors are usually embeddings produced by a model. Similar texts, images, audio clips, users, or products should land near each other in vector space.

## Mental Model

Traditional keyword search asks: "Which documents contain these words?"

Vector search asks: "Which stored items are closest in meaning, behavior, or representation to this query?"

For example:

- Query: "How do I cancel my plan?"
- Matching document text: "Subscription termination policy"
- Keyword search may miss the match because the words differ.
- Vector search can match the intent if the embedding model learned that "cancel" and "termination" are related.

Vector search is useful when the user may express an idea differently from the stored text. It is weaker when exact symbols matter, such as product SKUs, error codes, policy numbers, legal citations, function names, and version numbers. Professional retrieval systems usually combine vector search with lexical search and filters.

## Core Terms

| Term | Meaning |
| --- | --- |
| Embedding | A numeric vector that represents text, image, audio, or another object. |
| Dimension | The number of values in a vector. A 1536-dimensional vector has 1536 numbers. |
| Dense vector | Most dimensions contain non-zero values. Used for semantic similarity. |
| Sparse vector | Most dimensions are zero. Used for lexical retrieval such as BM25 or SPLADE-like representations. |
| Corpus | The collection being searched. |
| Query vector | The embedding generated from the user's query. |
| Top-k | Number of nearest neighbors returned. |
| Recall | How many relevant items were retrieved out of all relevant items. |
| Precision | How many retrieved items were actually relevant. |
| ANN | Approximate nearest-neighbor search. It trades perfect recall for speed and scale. |

## Similarity and Distance Metrics

Vector databases rank items using a metric. The most common metrics are cosine similarity, dot product, and Euclidean distance.

| Metric | Best Used When | Notes |
| --- | --- | --- |
| Cosine similarity | You care about direction more than magnitude. Common for text embeddings. | Often intuitive for semantic similarity. |
| Dot product / inner product | Embeddings are normalized or magnitude is meaningful. | For unit-normalized embeddings, dot product and cosine produce equivalent rankings. |
| Euclidean / L2 distance | Geometric distance matters. | Common in classical ML and some image/vector workloads. |
| Manhattan / L1 distance | Absolute coordinate differences matter. | Less common for modern text retrieval. |
| Hamming / Jaccard | Binary vectors or set-like similarity. | Useful for compressed or binary representations. |

OpenAI states that its embedding outputs are normalized to length 1, so cosine similarity and Euclidean distance produce identical rankings, and cosine can be computed through dot product. Source: <https://help.openai.com/en/articles/6824809-embeddings-faq>.

## Exact Search vs Approximate Search

Exact nearest-neighbor search compares the query vector against every stored vector. It gives perfect recall, but cost grows with corpus size.

Approximate nearest-neighbor search builds an index that narrows the search space. It is much faster at scale, but it may miss a relevant item. In production, you tune the speed-recall tradeoff against a labeled evaluation set.

| Approach | Strength | Weakness |
| --- | --- | --- |
| Exact scan | Highest recall, simple behavior. | Slow and expensive for large corpora. |
| HNSW | Strong speed-recall tradeoff, common default. | More memory and slower index builds. |
| IVFFlat | Faster builds and lower memory than HNSW in some cases. | Usually needs training data and careful list/probe tuning. |
| Quantized search | Smaller index and faster search. | Can reduce recall; often needs reranking with full-precision vectors. |

## HNSW in Plain English

HNSW stands for Hierarchical Navigable Small World. It builds a graph where each vector is connected to nearby vectors. Search starts at a coarse upper layer, walks toward closer candidates, then descends into lower layers for a more precise search.

Important knobs:

- `m`: roughly controls graph connectivity. Higher values can improve recall but increase memory and build cost.
- `ef_construction`: candidate list size during index construction. Higher values can improve index quality but slow builds.
- `ef_search`: candidate list size during queries. Higher values improve recall but increase latency.

pgvector documents HNSW as having better query performance than IVFFlat in speed-recall tradeoff, with slower builds and more memory use. Source: <https://github.com/pgvector/pgvector>.

## IVFFlat in Plain English

IVFFlat divides vectors into clusters or lists. At query time, the system searches only the most relevant lists instead of the whole corpus.

Important knobs:

- Number of lists: more lists can make search faster, but each query must probe enough lists to maintain recall.
- Number of probes: more probes means searching more lists. Recall improves, latency increases.
- Training data: IVFFlat works best when the index is created after representative data exists.

pgvector recommends creating IVFFlat indexes after loading data and starting list/probe tuning from corpus size. Source: <https://github.com/pgvector/pgvector>.

## Scores Are Not Universal

A common beginner mistake is treating similarity scores as universal confidence scores. They are not.

Scores depend on:

- the embedding model,
- the distance metric,
- vector normalization,
- index approximation,
- query length and style,
- chunk length,
- corpus density,
- filters,
- hybrid weighting.

A score of `0.82` may be excellent in one collection and mediocre in another. Thresholds should be tuned per dataset and query type.

## Dense vs Sparse Retrieval

Dense retrieval:

- Captures semantic meaning.
- Handles synonyms and paraphrases.
- Works well for natural language questions.
- May miss exact identifiers or rare terms.

Sparse retrieval:

- Captures exact word overlap and term importance.
- Works well for codes, names, citations, acronyms, and rare keywords.
- Does not understand synonyms unless expanded or modeled.

Hybrid retrieval combines both. For RAG systems, hybrid search is often more robust than pure vector search because user questions mix semantic intent with exact terms.

## What Makes a Good Embedding Model?

A good embedding model is not always the largest model. It is the model that retrieves the right evidence for your workload.

Evaluate:

- Domain fit: Does it understand your language, jargon, and entity names?
- Dimensionality: Larger vectors may capture more detail but cost more to store and search.
- Input limit: Can it embed your chunks without truncation?
- Multilingual behavior: Does it support the languages in your corpus?
- Modality: Text-only, image-text, audio, or multimodal.
- Latency and price: Embedding generation cost matters during ingestion and reindexing.
- Version stability: Changing the embedding model usually requires re-embedding the corpus.

## Retrieval Quality Metrics

Use retrieval metrics before judging the LLM response. If retrieval fails, generation cannot reliably succeed.

| Metric | Question Answered |
| --- | --- |
| Recall@k | Did we retrieve at least one relevant result in the top k? |
| Precision@k | How many top-k results were relevant? |
| MRR | How high was the first relevant result ranked? |
| nDCG | Did highly relevant results appear near the top? |
| Latency p50/p95/p99 | How fast is retrieval for typical and slow requests? |
| Filter selectivity | Are metadata filters narrowing results too much or too little? |

## Practical Rules

- Normalize IDs and metadata before indexing. Retrieval bugs often come from bad data modeling, not vector math.
- Store source text and source metadata with each vector. RAG needs citations, debugging, and reindexing.
- Do not use only top-1 retrieval for RAG. Use a candidate set, then rerank or compress.
- Use exact search on small collections if latency is acceptable. ANN complexity is not always worth it.
- Do not compare scores across different embedding models or distance metrics.
- Build a test set of real questions early. Retrieval tuning without evaluation data becomes guesswork.

## Study Checklist

- I can explain cosine similarity, dot product, and Euclidean distance.
- I know why exact search is simple but expensive.
- I can describe the speed-recall tradeoff of ANN search.
- I know when semantic search fails and why hybrid search helps.
- I understand that similarity scores are dataset-specific, not universal confidence values.

