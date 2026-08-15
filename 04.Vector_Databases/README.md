# Phase 4 - Vector Databases

Vector databases store embeddings and make nearest-neighbor search practical at application scale. They are the storage and retrieval layer behind most RAG systems, semantic search products, recommendation systems, duplicate detection systems, and multimodal search applications.

## Study Goals

By the end of this phase, you should be able to:

- Explain dense vectors, sparse vectors, similarity metrics, and approximate nearest-neighbor search.
- Choose between Chroma, pgvector, Pinecone, Qdrant, Weaviate, Milvus, Elasticsearch/OpenSearch, and other stores based on scale, filtering, hosting, and operational needs.
- Design a vector schema with stable IDs, metadata, namespaces, source lineage, and deletion/update strategy.
- Tune HNSW, IVFFlat, top-k, score thresholds, filters, hybrid search, and reranking.
- Debug retrieval quality and latency with repeatable evaluation sets instead of guessing.
- Estimate the practical cost of storage, embedding generation, indexing, query traffic, reranking, and observability.

## Files in This Section

- [01_Vector_Search_Fundamentals.md](01_Vector_Search_Fundamentals.md) - embeddings, distance metrics, dense vs sparse retrieval, similarity scores, and recall.
- [02_Indexing_and_Database_Design.md](02_Indexing_and_Database_Design.md) - schema design, chunk identity, metadata, HNSW, IVFFlat, namespaces, and lifecycle operations.
- [03_Filtering_Hybrid_Search_and_Reranking.md](03_Filtering_Hybrid_Search_and_Reranking.md) - metadata filters, BM25, sparse vectors, score fusion, RRF, and reranking.
- [04_Vector_DB_Tools_and_Tradeoffs.md](04_Vector_DB_Tools_and_Tradeoffs.md) - Chroma, pgvector, Pinecone, Qdrant, Weaviate, Milvus, and selection criteria.
- [05_Production_Operations_Costs_and_Debugging.md](05_Production_Operations_Costs_and_Debugging.md) - observability, scaling, backups, migrations, cost model, and failure modes.

## Recommended Study Order

1. Learn the math and retrieval concepts in `01_Vector_Search_Fundamentals.md`.
2. Study how embeddings become durable records in `02_Indexing_and_Database_Design.md`.
3. Learn why real systems combine vector search with filtering, lexical search, and reranking in `03_Filtering_Hybrid_Search_and_Reranking.md`.
4. Compare database options in `04_Vector_DB_Tools_and_Tradeoffs.md`.
5. Finish with production operations in `05_Production_Operations_Costs_and_Debugging.md`.

## Source Trail

These notes were built from the course topics, the linked course repositories, and current public documentation:

- Chroma documentation: <https://docs.trychroma.com/>
- pgvector README: <https://github.com/pgvector/pgvector>
- Pinecone hybrid search docs: <https://docs.pinecone.io/guides/search/hybrid-search>
- Qdrant hybrid search docs: <https://qdrant.tech/documentation/search/hybrid-queries/>
- OpenAI embeddings FAQ: <https://help.openai.com/en/articles/6824809-embeddings-faq>
