# Indexing and Database Design

A vector database is not just a place to put embeddings. It is a retrieval system with data modeling, lifecycle management, filtering, ranking, observability, and operational constraints.

## The Record Model

A production vector record usually has:

```text
id: stable unique ID
embedding: dense vector
sparse_embedding: optional lexical/sparse vector
document_text: chunk text or pointer to text
metadata: source, tenant, permissions, dates, version, tags
parent_id: original document ID
chunk_index: chunk order inside the source document
embedding_model: model name/version used to create the vector
created_at / updated_at: lifecycle fields
```

Stable IDs matter because you need repeatable updates and deletes. Avoid random IDs unless you also store a deterministic lookup key.

Useful ID patterns:

- `doc_id#chunk_0001`
- `tenant_id:source_id:page_12:chunk_03`
- hash of canonical source URI plus chunk offset

## Chunk Identity

Every chunk should be traceable back to the source.

At minimum store:

- source URI or file path,
- document title,
- page number or section heading if available,
- chunk number,
- byte/character range if possible,
- ingestion version,
- permissions/tenant fields,
- timestamp of source extraction.

This metadata enables citations, debugging, access control, deduplication, and reindexing.

## Embedding Model Compatibility

All vectors in a searchable index must be comparable. You cannot safely mix embeddings from different models unless the database stores them in separate vector fields and you query the correct field.

Track:

- embedding provider,
- model name,
- embedding dimension,
- distance metric,
- preprocessing rules,
- chunking strategy,
- normalization behavior,
- date of embedding.

When you change any of these, plan a migration. The safest approach is usually a new index or collection, backfilled in the background, then switched through configuration.

## Metadata Design

Metadata powers filtering, access control, debugging, and analytics. Keep it structured and predictable.

Good metadata:

```json
{
  "tenant_id": "acme",
  "source_type": "policy_pdf",
  "source_uri": "s3://kb/hr/leave_policy.pdf",
  "document_id": "leave_policy_2026",
  "page": 12,
  "section": "Parental Leave",
  "created_at": "2026-04-02",
  "effective_date": "2026-05-01",
  "security_class": "internal",
  "language": "en"
}
```

Avoid:

- inconsistent key names like `docId`, `doc_id`, and `documentId` in the same system,
- unbounded nested objects when the database does not index them well,
- storing secrets in metadata,
- metadata that cannot be regenerated from source systems.

Pinecone documents records as IDs, vectors, and optional flat JSON metadata for filtering. Source: <https://docs.pinecone.io/guides/index-data/data-modeling>.

## Namespaces, Collections, and Tenancy

Vector databases use different names for logical partitions:

- Chroma: collections.
- Pinecone: indexes and namespaces.
- Qdrant: collections and payload filters.
- pgvector: PostgreSQL tables, schemas, and columns.
- Weaviate: collections/classes.
- Milvus: collections and partitions.

Design questions:

- Should each tenant get a separate namespace or one shared index with metadata filters?
- Do permission filters need to be enforced inside retrieval, or can they be applied before/after?
- Will tenants have different embedding models?
- Do tenants need separate backup and deletion policies?

For most SaaS RAG systems, a shared index with strict `tenant_id` and permission filters is cheaper and simpler at moderate scale. Separate indexes or namespaces are useful when tenants require isolation, different embedding models, different retention rules, or separate scaling.

## Indexing Pipeline

A reliable indexing pipeline has explicit stages:

1. Discover source documents.
2. Load raw files or API records.
3. Extract text and structure.
4. Normalize text.
5. Split into chunks.
6. Add metadata and stable IDs.
7. Generate embeddings.
8. Upsert vectors.
9. Verify counts and sample queries.
10. Record ingestion logs and version metadata.

Do not treat indexing as a one-time script. Production documents change.

## Upserts, Updates, and Deletes

Common lifecycle operations:

- Insert: add new chunks.
- Upsert: insert or replace known chunk IDs.
- Delete by ID: remove a specific chunk.
- Delete by filter: remove all chunks for a document, tenant, or version.
- Re-embed: replace vectors after changing model or preprocessing.
- Tombstone: mark records deleted before physical deletion.

Recommended update pattern:

1. Generate new chunks for the document.
2. Use deterministic IDs.
3. Upsert new chunk records.
4. Delete old chunk IDs that no longer exist.
5. Log the source version and ingestion run.

For high-integrity systems, write source-document state to a relational table and treat vector DB writes as a derived index.

## Choosing `top_k`

`top_k` controls how many initial candidates the vector database returns. The best value depends on retrieval strategy.

Typical starting points:

- Simple FAQ bot: `k = 3` to `5`.
- Documentation Q&A: `k = 5` to `12`.
- Hybrid retrieval before reranking: `k = 20` to `100`.
- Reranking with cross-encoder: retrieve `20` to `100`, then keep `3` to `10`.
- Agentic research: larger candidate sets may be useful, but summarize/compress before generation.

Higher `k` improves recall but increases:

- retrieval latency,
- token cost,
- reranking cost,
- risk of context noise.

## Score Thresholds

Score thresholds reject weak matches. They are useful for "I do not know" behavior, but they are dangerous if guessed.

Tune thresholds per:

- embedding model,
- corpus,
- query type,
- metric,
- chunk size,
- database implementation.

Use a validation set with relevant and irrelevant queries. Track false positives and false negatives.

## Index Build Strategy

For small local development, simple persistent Chroma or in-memory stores are enough.

For production:

- bulk load first, then build ANN indexes when the database recommends it,
- avoid blocking writes during index creation when supported,
- use background reindexing for model changes,
- create read aliases or config switches for cutover,
- benchmark exact vs approximate search before assuming ANN is required.

pgvector notes that exact nearest-neighbor search is the default and approximate indexes trade some recall for speed. It supports HNSW and IVFFlat, with different build and memory tradeoffs. Source: <https://github.com/pgvector/pgvector>.

## Data Quality Controls

Bad data quality becomes bad retrieval.

Validate:

- empty chunks,
- duplicate chunks,
- chunks over model token limit,
- chunks with mostly navigation/footer boilerplate,
- broken Unicode extraction,
- missing metadata,
- mismatched embedding dimensions,
- records with wrong tenant/permission fields,
- source documents that disappeared.

Useful ingestion checks:

```text
documents_discovered == documents_loaded
chunks_created > 0
chunks_with_empty_text == 0
embedding_dimension == expected_dimension
upserted_count == chunks_created
sample_queries_return_expected_sources
```

## Common Design Mistakes

| Mistake | Result | Better Approach |
| --- | --- | --- |
| Random chunk IDs | Updates create duplicates. | Use deterministic IDs from source and chunk position. |
| No source metadata | Cannot cite or debug. | Store source URI, page, title, and chunk range. |
| Mixing embedding models | Search quality becomes unstable. | Use separate indexes or fields per model. |
| No permission filters | Data leakage risk. | Enforce tenant and ACL filters at retrieval time. |
| Oversized chunks | Poor recall and wasted tokens. | Tune chunking by document type and evaluate. |
| Too many tiny chunks | Fragmented answers. | Use overlap, parent-child retrieval, or section-aware chunking. |
| Blind thresholding | Relevant results get dropped. | Tune on real queries. |

## Study Checklist

- I can design a vector record schema with IDs, embeddings, text, and metadata.
- I know why embedding model versioning matters.
- I can explain the lifecycle for updates, deletes, and reindexing.
- I can choose reasonable starting values for `top_k`.
- I know how to validate an ingestion pipeline before trusting retrieval.

