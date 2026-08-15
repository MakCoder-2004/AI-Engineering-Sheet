# Vector Database Tools and Tradeoffs

There is no universal best vector database. The right choice depends on scale, hosting model, filters, hybrid search, team skills, security requirements, and how much operational control you need.

## Selection Criteria

Evaluate every vector store against:

- data size: number of vectors, dimensions, metadata size,
- write pattern: batch indexing, streaming updates, deletes,
- query pattern: top-k, filters, hybrid search, reranking,
- tenancy: namespaces, partitions, row-level security,
- deployment: local, cloud, Kubernetes, managed service,
- consistency and durability,
- backup and restore,
- observability,
- cost,
- ecosystem integrations,
- migration path.

## Chroma

Chroma is a common local-development vector database for RAG learning and prototypes.

Strengths:

- simple Python developer experience,
- persistent local collections,
- easy integration with LangChain,
- useful for notebooks, demos, and small internal tools,
- supports metadata filters and document content filters.

Limitations:

- not usually the first choice for heavy multi-tenant production workloads,
- operational model differs from managed cloud vector databases,
- you still need a strategy for backups, access control, and reindexing.

Key concepts:

- Chroma stores records in collections.
- Collections can have embedding functions attached.
- Queries can use text or precomputed embeddings.
- `where` filters metadata.
- `where_document` filters document text.
- Query results can include IDs, documents, metadata, distances, and embeddings.

Chroma documents `.query` for similarity search and `.get` for direct retrieval by ID/filter. Source: <https://docs.trychroma.com/docs/querying-collections/query-and-get>.

## pgvector

pgvector adds vector similarity search to PostgreSQL.

Strengths:

- works inside the database many teams already operate,
- transactional metadata and vector records can live together,
- SQL querying and joins,
- good fit when vectors are part of a broader relational application,
- supports exact search and approximate indexes.

Limitations:

- high-scale vector workloads may require careful Postgres tuning,
- HNSW indexes can use significant memory,
- row-level security and filters are powerful but must be designed carefully,
- scaling reads/writes follows PostgreSQL scaling constraints.

Useful pgvector capabilities:

- vector, halfvec, sparsevec, and bit vector types,
- L2, inner product, cosine, L1, Hamming, and Jaccard distance operators depending on type/index,
- HNSW and IVFFlat indexes,
- hybrid search through Postgres full-text search plus vector search,
- SQL query analysis with `EXPLAIN (ANALYZE, BUFFERS)`.

pgvector notes that exact nearest-neighbor search is the default and approximate indexes trade recall for speed. Source: <https://github.com/pgvector/pgvector>.

## Pinecone

Pinecone is a managed vector database focused on production vector search.

Strengths:

- managed infrastructure,
- serverless and hosted operational model,
- namespaces,
- dense, sparse, and hybrid patterns,
- integrated embedding options,
- production-oriented search APIs.

Limitations:

- service cost must be modeled,
- architecture depends on Pinecone's API model,
- complex hybrid/rerank designs still need application-level evaluation.

Pinecone's docs distinguish single-index dense+sparse hybrid search from separate dense/sparse indexes and document-schema patterns. Source: <https://docs.pinecone.io/guides/search/hybrid-search>.

## Qdrant

Qdrant is an open-source vector search engine with managed cloud options.

Strengths:

- strong filtering support through JSON payloads,
- dense, sparse, and multivector search,
- REST and gRPC APIs,
- hybrid queries and fusion,
- good fit for advanced retrieval experiments.

Limitations:

- production operation requires understanding collections, payload indexes, snapshots, and resource tuning,
- hybrid search design still needs measurement,
- managed vs self-hosted choice affects operational burden.

Qdrant documents dense, sparse, and multi-vector search, payload filtering, hybrid search, and fusion strategies such as RRF and DBSF. Source: <https://github.com/qdrant/qdrant>.

## Weaviate

Weaviate is an open-source vector database with managed cloud options and a schema-centric data model.

Strengths:

- schema and object-oriented modeling,
- vector and hybrid search,
- modules/integrations,
- GraphQL and REST APIs,
- multi-tenant features.

Limitations:

- schema design is more explicit than simple key-value vector stores,
- operational model must be learned,
- module choices can create coupling.

Use it when schema-aware object search, hybrid search, and managed/self-hosted flexibility matter.

## Milvus and Zilliz

Milvus is an open-source vector database designed for large-scale vector search. Zilliz is the managed service from the Milvus ecosystem.

Strengths:

- built for large vector collections,
- multiple index types,
- high-scale similarity search,
- strong ecosystem for vector-native workloads.

Limitations:

- more infrastructure complexity than simple local stores,
- production setup has more moving parts,
- best suited when scale justifies the operational overhead.

## Elasticsearch and OpenSearch

Search engines now support vector fields and hybrid retrieval.

Strengths:

- mature lexical search,
- filters, aggregations, and operational tooling,
- hybrid search in systems teams may already run,
- useful when keyword search remains central.

Limitations:

- vector search capabilities and performance depend on version/configuration,
- dense-vector-only workloads may be simpler in dedicated vector stores,
- scoring can be complex when combining signals.

Use these when your app already needs full-text search, faceting, filtering, and operational search features.

## Tool Choice Matrix

| Need | Good Starting Choice |
| --- | --- |
| Notebook or local RAG prototype | Chroma |
| App already on PostgreSQL | pgvector |
| Managed vector search with low ops burden | Pinecone, Qdrant Cloud, Weaviate Cloud, Zilliz |
| Heavy metadata filtering | Qdrant, Weaviate, Elasticsearch/OpenSearch, pgvector |
| Strong SQL joins and transactions | pgvector |
| Large-scale vector-native workloads | Milvus/Zilliz, Pinecone, Qdrant |
| Keyword-first search plus vectors | Elasticsearch/OpenSearch, Pinecone document schema, Qdrant sparse vectors |
| Strict multi-tenant SaaS | Depends on isolation needs; test namespace/filter behavior carefully |

## Local Development vs Production

Local development priorities:

- easy setup,
- fast iteration,
- readable examples,
- no infrastructure.

Production priorities:

- access control,
- predictable latency,
- durability,
- backup/restore,
- reindexing strategy,
- monitoring,
- cost control,
- support model.

It is normal to prototype with Chroma and deploy with pgvector, Qdrant, Pinecone, Weaviate, or Milvus.

## Migration Strategy

Avoid hard-coding one vector database throughout your business logic. Keep a small retrieval abstraction around:

- upsert documents/chunks,
- delete by document ID,
- query by text/vector,
- query with filters,
- return standardized results with scores and metadata.

Do not over-abstract every database feature. Hybrid search, filters, reranking, and namespaces differ enough that a lowest-common-denominator wrapper can hide important capabilities. Abstract your application's needs, not every vendor API.

## Study Checklist

- I can explain when Chroma is enough and when it is not.
- I know why pgvector is attractive for PostgreSQL-heavy systems.
- I can compare managed vector databases with self-hosted systems.
- I understand that hybrid search behavior differs by vendor.
- I can choose a vector store based on workload rather than popularity.

