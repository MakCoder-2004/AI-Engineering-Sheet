# RAG Overview and Study Map

Retrieval-Augmented Generation, usually called RAG, is an architecture where an application retrieves external information and provides it to an LLM as context before generating an answer.

The original RAG paper by Lewis et al. introduced retrieval-augmented generation for knowledge-intensive NLP tasks by combining parametric memory in a model with non-parametric memory in retrieved documents. Source: <https://arxiv.org/abs/2005.11401>.

## Why RAG Exists

LLMs have two major limits:

- They have finite context windows.
- Their trained knowledge is fixed at training time.

RAG helps by retrieving relevant knowledge at runtime. LangChain's retrieval docs describe retrieval as the foundation for giving LLMs access to context-specific information when answering. Source: <https://docs.langchain.com/oss/python/deepagents/retrieval>.

Use RAG when:

- the answer depends on private or proprietary data,
- the data changes frequently,
- answers need citations,
- the corpus is too large for the prompt,
- different users have access to different data,
- you need observability over which sources were used.

Do not use RAG by default when:

- the task is pure reasoning over the user's prompt,
- the necessary facts are already in the prompt,
- the corpus is tiny and can fit in the model context,
- data freshness and source citation do not matter,
- a structured database query is more reliable.

## The Two Pipelines

RAG has two major pipelines: indexing and query-time generation.

### Indexing Pipeline

```text
source documents
  -> load / parse
  -> clean / normalize
  -> split into chunks
  -> attach metadata
  -> embed
  -> store in vector DB / search index
  -> validate ingestion
```

This runs when documents are created, updated, deleted, or reindexed.

### Query Pipeline

```text
user question
  -> classify / rewrite / decompose
  -> retrieve candidates
  -> filter by permissions and metadata
  -> rerank / compress
  -> pack context into prompt
  -> generate answer
  -> cite sources
  -> evaluate / trace / log
```

This runs for each user request.

## Main RAG Architectures

| Architecture | Flow | Best For | Tradeoff |
| --- | --- | --- | --- |
| 2-step RAG | Always retrieve, then generate. | FAQs, documentation Q&A, policy bots. | Predictable but less flexible. |
| Conversational RAG | Rewrite follow-up question, retrieve, answer. | Chat over documents. | Requires conversation-aware query rewriting. |
| Hybrid RAG | Query transform, retrieve, validate, rerank, answer. | Production Q&A with quality controls. | More moving parts. |
| Agentic RAG | Agent decides when and how to retrieve. | Research, multi-source workflows, ambiguous tasks. | Higher latency and less predictable behavior. |
| GraphRAG | Retrieve over entity/relation graph plus text. | Multi-hop reasoning over connected knowledge. | Requires graph extraction and maintenance. |
| Multimodal RAG | Retrieve text, images, pages, slides, or tables. | PDFs, scans, screenshots, diagrams. | Harder parsing, indexing, and evaluation. |

LangChain categorizes RAG into 2-step, agentic, and hybrid patterns. Source: <https://docs.langchain.com/oss/python/deepagents/retrieval>.

## What "Good RAG" Means

A good RAG system:

- retrieves the right evidence,
- excludes irrelevant or unauthorized evidence,
- uses the evidence faithfully,
- admits when the answer is not in the corpus,
- cites sources,
- handles updates and deletes,
- is observable,
- has measurable quality,
- has predictable cost and latency.

A bad RAG system:

- retrieves random semantically similar chunks,
- stuffs too much context into the prompt,
- hides retrieval failures behind confident generation,
- has no test set,
- cannot explain why a source was retrieved,
- leaks data across users or tenants.

## The Five Common RAG Failure Modes

The linked production RAG course repository highlights five common failure modes: bad chunking, embedding mismatch, retrieval noise, context overflow, and hallucination. Source: <https://github.com/pdichone/fcc-production-rag-part-6>.

| Failure Mode | What It Looks Like | Primary Fix |
| --- | --- | --- |
| Bad chunking | Correct document exists, but relevant chunk is missing or split badly. | Better parsing, semantic/section chunking, overlap. |
| Embedding mismatch | Query and document use different vocabulary. | Query rewriting, hybrid search, domain-tuned embeddings. |
| Retrieval noise | Top results contain loosely related but useless chunks. | Reranking, filters, better chunk metadata. |
| Context overflow | Relevant content is retrieved but lost in too much context. | Context packing, compression, summarization, token budgeting. |
| Hallucination | LLM ignores context or invents facts. | Grounded prompts, citations, answer validation, refusal policy. |

## RAG vs Fine-Tuning

RAG and fine-tuning solve different problems.

Use RAG for:

- external knowledge,
- factual lookup,
- fresh data,
- citations,
- access-controlled data.

Use fine-tuning for:

- style,
- response format,
- task behavior,
- domain-specific classification,
- tool-use patterns.

Often you combine both: a fine-tuned or instructed model with a RAG pipeline.

## RAG vs Long Context

Long-context models reduce the need for retrieval in small or medium cases, but they do not remove the need for RAG.

Long context is useful when:

- the relevant corpus is small enough to fit,
- you need holistic reasoning over a whole document,
- latency and token cost are acceptable,
- access control is simple.

RAG is useful when:

- corpus is large,
- data changes,
- only a few sources are relevant,
- citations matter,
- user permissions differ,
- retrieval quality must be measured.

Advanced systems often use both: retrieve a subset, then use a long-context model to reason over richer source context.

## Study Path

1. Understand vector search in `04.Vector_Databases`.
2. Learn the end-to-end RAG architecture in this file.
3. Study indexing in `02_Document_Loading_Chunking_and_Indexing.md`.
4. Study retrieval and context packing in `03_Retrieval_Reranking_and_Context_Packing.md`.
5. Study debugging and evaluation in `04_Evaluation_Debugging_and_Observability.md`.
6. Study production security, scaling, and costs in `05_Production_RAG_Security_Scaling_and_Cost.md`.
7. Study advanced patterns in `06_Advanced_RAG_Patterns.md`.

## Professional Checklist

- I can draw the indexing and query pipelines from memory.
- I can explain which RAG failure mode caused a bad answer.
- I can choose between 2-step, hybrid, and agentic RAG.
- I know when long context is enough and when retrieval is still needed.
- I understand that RAG quality starts with data quality, not prompts.

