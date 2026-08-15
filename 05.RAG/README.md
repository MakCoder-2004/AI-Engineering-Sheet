# Phase 5 - Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation connects LLMs to external knowledge at query time. A strong RAG system is not just "vector search plus a prompt"; it is a full pipeline for loading data, transforming it, indexing it, retrieving evidence, ranking it, packing context, generating grounded answers, evaluating quality, and operating the system securely in production.

## Study Goals

By the end of this phase, you should be able to:

- Explain why RAG exists and when long-context prompting is enough.
- Build an indexing pipeline with loaders, parsers, chunkers, metadata, embeddings, and vector storage.
- Build a query pipeline with query rewriting, retrieval, hybrid search, reranking, context packing, and answer generation.
- Diagnose RAG failures by separating ingestion, retrieval, reranking, context, and generation problems.
- Evaluate RAG with retrieval metrics, answer metrics, faithfulness checks, and human review.
- Design production RAG with security, observability, costs, scaling, and deployment in mind.
- Understand advanced patterns: contextual retrieval, late chunking, agentic RAG, GraphRAG, and multimodal RAG.

## Files in This Section

- [00_RAG_Overview_and_Study_Map.md](00_RAG_Overview_and_Study_Map.md) - the full RAG mental model, architecture map, and learning path.
- [02_Document_Loading_Chunking_and_Indexing.md](02_Document_Loading_Chunking_and_Indexing.md) - loaders, parsing, cleaning, chunking, embeddings, and indexing.
- [03_Retrieval_Reranking_and_Context_Packing.md](03_Retrieval_Reranking_and_Context_Packing.md) - similarity search, hybrid search, reranking, query transforms, and token budgeting.
- [04_Evaluation_Debugging_and_Observability.md](04_Evaluation_Debugging_and_Observability.md) - failure modes, metrics, test sets, LangSmith-style tracing, and debugging.
- [05_Production_RAG_Security_Scaling_and_Cost.md](05_Production_RAG_Security_Scaling_and_Cost.md) - APIs, hosting, security layer, scaling, and real costs.
- [06_Advanced_RAG_Patterns.md](06_Advanced_RAG_Patterns.md) - long context vs RAG, contextual retrieval, late chunking, agentic RAG, GraphRAG, and multimodal RAG.
- [01_Agentic_RAG_and_Multi_Step_Agents.md](01_Agentic_RAG_and_Multi_Step_Agents.md) - existing deep note on agentic RAG and multi-step agents.

## Course Timeline Mapping

The files above cover and extend these course sections:

- Full RAG overview
- Development environment setup
- Document loaders
- RAG indexing pipeline
- embedding dimensions
- Chroma vector database setup
- similarity search with scores
- basic RAG systems
- debugging RAG systems
- hybrid search
- token budgeting
- observability and LangSmith
- RAG optimization
- scaling and vector search costs
- production hosting with Supabase/pgvector
- production visibility
- security layer and security checklist
- long context vs RAG
- contextual retrieval
- late chunking vs early chunking
- agentic RAG
- GraphRAG
- multimodal RAG

## Source Trail

- Original RAG paper: <https://arxiv.org/abs/2005.11401>
- LangChain retrieval docs: <https://docs.langchain.com/oss/python/deepagents/retrieval>
- LangSmith observability docs: <https://docs.langchain.com/langsmith/observability-concepts>
- Ragas metrics docs: <https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/>
- Anthropic contextual retrieval: <https://www.anthropic.com/engineering/contextual-retrieval>
- Course repositories: <https://github.com/pdichone/production-course-main-code> and <https://github.com/pdichone/fcc-production-rag-part-6>
