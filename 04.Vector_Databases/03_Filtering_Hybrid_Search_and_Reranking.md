# Filtering, Hybrid Search, and Reranking

Pure vector search is rarely enough for professional systems. Real user queries often include both meaning and constraints:

- "refund policy for enterprise customers in Germany"
- "error E_CONN_492 in version 2.8"
- "SOC 2 report from last quarter"
- "Arabic onboarding documents for the Cairo office"

These queries need semantic retrieval, exact matching, metadata filters, and often reranking.

## Metadata Filtering

Metadata filters restrict the search space before or during retrieval.

Common filters:

- `tenant_id`
- permissions or access groups
- document type
- language
- date range
- product
- region
- status or version
- source system

Filtering prevents irrelevant or unauthorized results from being retrieved. In multi-tenant RAG, access-control filtering is not optional.

## Pre-Filtering vs Post-Filtering

| Strategy | How It Works | Risk |
| --- | --- | --- |
| Pre-filtering | Apply metadata constraints before vector ranking. | If filters are too strict, recall drops. |
| Post-filtering | Run vector search, then discard results that do not match filters. | May return too few results if top candidates are filtered out. |
| Hybrid strategy | Increase candidate pool, filter, then rerank. | More cost and complexity. |

For security filters, prefer database-supported filtering during retrieval. Do not retrieve unauthorized content and then hope later code removes it.

## Full-Text and Lexical Search

Lexical search ranks documents by words and terms. BM25 is a widely used ranking algorithm that considers term frequency, inverse document frequency, and document length.

Lexical retrieval is strong for:

- exact product names,
- error codes,
- legal citations,
- function names,
- named entities,
- rare terms,
- short queries.

It is weak for:

- synonyms,
- paraphrases,
- broad conceptual questions,
- multilingual semantic matching without preprocessing.

Qdrant documents BM25-style full-text search through sparse vectors. Source: <https://qdrant.tech/documentation/search/text-search/full-text-search/>.

## Hybrid Search

Hybrid search combines dense semantic retrieval with sparse or lexical retrieval.

Why it helps:

- Dense retrieval finds meaning.
- Sparse retrieval finds exact terms.
- Fusion combines candidates from both retrieval styles.
- Reranking can apply a stronger but slower relevance model.

Common hybrid patterns:

1. Store dense and sparse vectors in the same record and query both.
2. Use one dense vector index and one keyword/sparse index, then merge results.
3. Use a search engine with BM25 and vector fields in the same document schema.
4. Retrieve broad candidates from multiple retrievers, then rerank.

Pinecone documents both single-index dense+sparse hybrid search and separate-index hybrid search, with explicit weighting between dense and sparse signals. Source: <https://docs.pinecone.io/guides/search/hybrid-search>.

Qdrant supports dense vectors, sparse vectors, multivectors, and hybrid queries with fusion strategies such as Reciprocal Rank Fusion and Distribution-Based Score Fusion. Source: <https://qdrant.tech/documentation/search/hybrid-queries/>.

## Score Fusion

When different retrievers return different scores, you need a fusion strategy.

### Weighted Score Fusion

Weighted fusion combines normalized dense and sparse scores:

```text
combined_score = alpha * dense_score + (1 - alpha) * sparse_score
```

`alpha` controls which signal dominates:

- `1.0`: dense-only
- `0.0`: sparse-only
- `0.5`: balanced
- `0.75`: dense-leaning

Pinecone warns that sparse and dense score ranges may not be naturally comparable and recommends explicit weighting for single-index hybrid search. Source: <https://docs.pinecone.io/guides/search/hybrid-search>.

### Reciprocal Rank Fusion

RRF ignores raw scores and uses ranks. A document ranked highly by multiple retrievers rises in the final list.

Use RRF when:

- score scales are incompatible,
- you combine multiple retrievers,
- you need a simple robust baseline,
- you do not have enough data to train a ranker.

### Distribution-Based Fusion

Distribution-based approaches normalize scores based on each result set's distribution. These can work better than simple weighting when score distributions vary by query, but they are harder to reason about.

## Reranking

Reranking takes an initial candidate set and sorts it with a stronger relevance signal.

Common rerankers:

- cross-encoder text reranker,
- LLM-as-reranker,
- ColBERT or late-interaction model,
- domain-specific classifier,
- rule-based boost/demotion.

Typical pipeline:

```text
query
  -> dense retrieval top 50
  -> lexical retrieval top 50
  -> merge and deduplicate
  -> rerank top 80
  -> keep top 5-10
  -> context packing
  -> generation
```

Reranking improves precision, but it costs latency and money. Use it when retrieval noise hurts answer quality or when the first-stage retriever must cast a wide net.

## Late Interaction

Standard dense retrieval represents each document chunk as one vector. Late-interaction approaches represent text with multiple vectors and compare query-token vectors to document-token vectors. This can improve fine-grained matching while still being more scalable than running a full cross-encoder over the entire corpus.

Qdrant's hybrid reranking tutorial describes a pipeline with dense embeddings, sparse BM25-style embeddings, and late-interaction embeddings for reranking. Source: <https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/>.

## Query Transformation

Hybrid systems often improve retrieval by transforming the user query before search.

Useful transformations:

- rewrite conversational query into standalone query,
- generate multiple query variants,
- expand acronyms,
- extract filters from natural language,
- translate query to corpus language,
- create both semantic and keyword-focused queries,
- decompose multi-hop questions.

Example:

```text
User: "Can I still get reimbursed for that flight from last month?"

Standalone query:
"employee travel reimbursement policy flight expense submitted one month after travel"

Extracted filters:
document_type = "policy"
topic = "travel"
date_context = "after travel"
```

## Contextual Retrieval

Contextual retrieval adds document-level context to chunks before embedding or indexing. The motivation is that naive chunking can remove important context. A chunk that says "the limit is 30 days" may be useless unless the system knows the section is about expense reimbursement.

Anthropic's contextual retrieval article describes contextual embeddings and contextual BM25 as methods to reduce retrieval failure by giving each chunk explanatory context before embedding/search. Source: <https://www.anthropic.com/engineering/contextual-retrieval>.

## Evaluation for Hybrid Retrieval

Evaluate hybrid search by query category:

- exact identifier queries,
- conceptual questions,
- policy/procedure questions,
- multi-hop questions,
- date-filtered questions,
- permission-sensitive questions,
- ambiguous conversational follow-ups.

Track:

- recall@k before reranking,
- precision@k after reranking,
- MRR,
- answer faithfulness,
- latency,
- cost per query,
- empty result rate,
- security filter violations.

## Practical Tuning Order

1. Build a small labeled query set.
2. Evaluate dense-only retrieval.
3. Evaluate lexical-only retrieval.
4. Evaluate hybrid fusion.
5. Tune `top_k`, alpha/RRF settings, and filters.
6. Add reranking if precision is still weak.
7. Add query rewriting or multi-query retrieval if recall is weak.
8. Re-run the same evaluation set after every change.

## Study Checklist

- I know why metadata filters are part of retrieval, not an afterthought.
- I can explain dense, sparse, and hybrid retrieval.
- I know when to use weighted score fusion vs RRF.
- I understand why reranking improves precision but adds latency.
- I can design an evaluation set that separates exact-match failures from semantic-retrieval failures.

