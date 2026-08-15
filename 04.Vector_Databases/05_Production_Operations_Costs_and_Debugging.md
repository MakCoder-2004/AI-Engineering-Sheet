# Production Operations, Costs, and Debugging

The hard part of vector search is not creating the first collection. The hard part is keeping retrieval correct, fast, secure, observable, and affordable as the corpus and query traffic change.

## Production Readiness Checklist

Before shipping a vector-backed feature, verify:

- deterministic document and chunk IDs,
- source lineage metadata,
- tenant and permission filters,
- embedding model and dimension recorded,
- duplicate detection,
- deletion/reindexing workflow,
- evaluation set with expected sources,
- monitoring for latency and empty results,
- backup and restore plan,
- migration plan for embedding model changes,
- cost estimate for ingestion and query traffic.

## Observability

Log every retrieval request with:

- query text or safe redacted representation,
- embedding model,
- vector store/index/namespace,
- filters applied,
- `top_k`,
- score threshold,
- returned IDs,
- returned scores/distances,
- latency,
- reranker latency if used,
- final chunks sent to the LLM,
- answer ID/trace ID.

For sensitive data, log hashes, IDs, or redacted snippets instead of raw private content.

## Retrieval Debugging Workflow

When a RAG answer is bad, debug retrieval before debugging the prompt.

1. Was the correct source ingested?
2. Was the relevant text extracted cleanly?
3. Was the text split into a retrievable chunk?
4. Was the chunk embedded with the expected model?
5. Was the query embedded with the same compatible model?
6. Did filters remove the right source?
7. Did the candidate set include the right chunk?
8. Did reranking demote the right chunk?
9. Did context packing drop the right chunk?
10. Did the LLM ignore or contradict the context?

This sequence separates retrieval failures from generation failures.

## Common Failure Modes

| Failure | Symptom | Fix |
| --- | --- | --- |
| Bad chunking | Relevant source exists but never appears in top-k. | Use section-aware or semantic chunking; improve overlap. |
| Embedding mismatch | Query terms differ from document language. | Add query rewriting, domain embeddings, or hybrid search. |
| Metadata filter bug | Empty or wrong results for valid queries. | Log filters; test each filter path. |
| Permission leak | User sees unauthorized sources. | Enforce tenant/ACL filters inside retrieval. |
| Context noise | LLM receives many irrelevant chunks. | Rerank, lower final k, compress context. |
| Stale index | Deleted/updated docs still appear. | Add source versioning and delete-by-document workflow. |
| Score threshold too strict | System says "I do not know" too often. | Tune thresholds on labeled queries. |
| Index approximation miss | Correct chunk appears in exact search but not ANN. | Increase ANN search params or candidate count. |

## Performance Tuning

Measure separately:

- embedding latency,
- vector DB query latency,
- filter latency,
- reranking latency,
- context packing latency,
- LLM latency.

Tuning options:

- reduce vector dimension if supported and quality remains acceptable,
- use approximate indexes for large collections,
- tune HNSW `ef_search` or IVFFlat probes,
- add payload/metadata indexes,
- reduce candidate set size,
- cache embeddings for repeated queries,
- cache retrieval results for common questions,
- batch ingestion embeddings,
- use faster rerankers or only rerank when needed.

## Cost Model

Vector search cost comes from more than database storage.

Estimate:

```text
total_cost =
  document_extraction_cost
  + embedding_generation_cost
  + vector_storage_cost
  + metadata_storage_cost
  + index_memory_or_compute_cost
  + query_compute_cost
  + reranking_cost
  + LLM_context_token_cost
  + observability/logging_cost
  + engineering/operations_cost
```

Important variables:

- number of documents,
- average chunks per document,
- embedding dimension,
- bytes per vector value,
- replicas,
- index overhead,
- query volume,
- top-k candidate count,
- reranker usage,
- average context tokens sent to the LLM.

RAG costs often move from "model call only" to "model call plus retrieval infrastructure plus evaluation plus operations."

## Memory and Storage Estimation

Rough dense-vector storage:

```text
raw_vector_bytes = vector_count * dimensions * bytes_per_float
```

For float32, `bytes_per_float = 4`.

Example:

```text
10,000,000 vectors * 1536 dimensions * 4 bytes = 61,440,000,000 bytes
```

That is roughly 61 GB before metadata, indexes, replicas, and database overhead. HNSW and other indexes can add substantial memory overhead.

## Security

Vector databases can leak data through retrieval.

Security controls:

- tenant isolation,
- row/document-level permissions,
- filter enforcement in the query layer,
- encrypted transport,
- encrypted storage,
- secret-free metadata,
- audit logs,
- deletion workflows,
- rate limits,
- prompt injection defenses in retrieved content,
- allowlisted retrieval tools for agents.

Never rely on the LLM to hide unauthorized context. The context should not be retrieved in the first place.

## Backup and Restore

Plan for:

- full backups,
- point-in-time recovery if supported,
- snapshot restore tests,
- reindex-from-source fallback,
- source data checksums,
- embedding model availability,
- disaster recovery time objectives.

Because vector indexes are derived from source documents, many teams treat the source store as the system of record and the vector database as a rebuildable serving index. That only works if the ingestion pipeline is deterministic and all source data is recoverable.

## Reindexing Strategy

You need reindexing when:

- embedding model changes,
- chunking strategy changes,
- metadata schema changes,
- source extraction improves,
- database/index type changes,
- distance metric changes.

Safe reindex pattern:

1. Create new collection/index/table.
2. Backfill from source.
3. Run evaluation against old and new indexes.
4. Shadow production queries if possible.
5. Switch traffic through config.
6. Keep old index for rollback.
7. Delete old index after retention period.

## Service-Level Indicators

Track:

- retrieval p50/p95/p99 latency,
- retrieval error rate,
- empty result rate,
- average candidate count,
- percentage of answers with citations,
- percentage of answers with no supporting context,
- feedback downvotes by source type,
- stale result rate,
- permission filter violations,
- cost per query.

## Study Checklist

- I can debug a bad RAG answer by tracing ingestion, retrieval, reranking, context packing, and generation.
- I can estimate vector storage from vector count and dimensions.
- I know why source lineage and deletion workflows are production requirements.
- I can explain how retrieval observability helps evaluation and incident response.
- I understand that security filters must be enforced before unauthorized context reaches the model.

