# Retrieval, Reranking, and Context Packing

The query pipeline turns a user question into a grounded answer. Retrieval gets candidate evidence; reranking improves precision; context packing decides what the LLM actually sees.

## Basic RAG Flow

```text
question
  -> retrieve top-k chunks
  -> format context
  -> prompt LLM
  -> answer with citations
```

This baseline is useful for learning. Production systems add query rewriting, filters, hybrid search, reranking, answer validation, and observability.

The linked course repository includes a basic LangChain RAG example using `OpenAIEmbeddings`, Chroma, a retriever, `ChatPromptTemplate`, and a chain that formats retrieved documents before generation. Source: <https://github.com/pdichone/production-course-main-code>.

## Retrieval Inputs

The retriever should receive a search-optimized query, not always the raw user message.

Examples:

| User Message | Retrieval Query |
| --- | --- |
| "What about refunds?" | "refund policy eligibility conditions deadlines" |
| "Can I do that from Egypt?" | "remote work policy Egypt location eligibility" |
| "Does this apply to contractors?" | "policy applicability contractors employees vendors" |

Conversational RAG usually needs a standalone-question rewrite step.

## Query Transformation Techniques

| Technique | Purpose |
| --- | --- |
| Standalone rewrite | Convert follow-up into full query. |
| Multi-query retrieval | Generate several query variants to improve recall. |
| Step-back prompting | Retrieve broader conceptual context first. |
| HyDE | Generate a hypothetical answer/document, then embed that for retrieval. |
| Query decomposition | Split multi-hop questions into subquestions. |
| Filter extraction | Convert natural language constraints into metadata filters. |
| Acronym/entity expansion | Add known synonyms and domain names. |

Use transformations when baseline retrieval misses relevant evidence. Do not add them blindly; measure the effect.

## Similarity Search with Scores

Similarity scores help debug retrieval, but they are not answer confidence.

Use scores to ask:

- Are top results close together or is there a clear winner?
- Are relevant chunks below irrelevant chunks?
- Did filtering remove high-scoring sources?
- Does a query type consistently produce low scores?
- Does changing chunk size improve score separation?

Do not compare scores across embedding models, vector databases, distance metrics, or corpora.

## Hybrid Retrieval

Hybrid retrieval combines semantic and lexical search.

Recommended when:

- exact terms matter,
- users search codes, IDs, names, or citations,
- dense retrieval misses rare vocabulary,
- documents contain technical language,
- query/document vocabulary differs.

Common implementation:

```text
dense top 50
  + sparse/BM25 top 50
  -> merge and deduplicate
  -> rerank
  -> keep top 5-10
```

Pinecone and Qdrant both document dense+sparse hybrid retrieval patterns. Sources: <https://docs.pinecone.io/guides/search/hybrid-search> and <https://qdrant.tech/documentation/search/hybrid-queries/>.

## Reranking

First-stage retrieval should optimize recall. Reranking should optimize precision.

Reranking candidates:

- cross-encoder reranker,
- LLM relevance grader,
- ColBERT/late-interaction reranker,
- business-rule ranker,
- recency/source-authority boosts.

Typical settings:

- Retrieve 20-100 candidates.
- Rerank them.
- Keep 3-10 for the final context.

Reranking is most useful when the top-k retrieval set contains the answer somewhere, but not near the top.

## Context Packing

Context packing is the process of deciding what retrieved content goes into the prompt.

The goal is not to include the most text. The goal is to include the best evidence in a format the model can use.

Good context packing:

- prioritizes high-quality evidence,
- groups chunks by source,
- removes duplicates,
- preserves headings and citations,
- includes metadata useful for answering,
- avoids unrelated chunks,
- fits the token budget,
- leaves enough room for the model's answer.

## Token Budgeting

Token budgeting prevents context overflow.

Example budget:

```text
model_context_window = 128000
system_prompt = 1500
developer_instructions = 1000
conversation_history = 6000
retrieved_context = 25000
tool/output buffer = 5000
answer_budget = 4000
reserved_safety_margin = 3000
```

For smaller models, the retrieved context budget may be much tighter.

Professional systems often set:

- max chunks,
- max tokens per source,
- max total context tokens,
- minimum score,
- diversity constraints,
- citation requirements.

## Lost-in-the-Middle Risk

LLMs may underuse information buried in the middle of a long context. Reduce this risk by:

- reranking aggressively,
- putting the most important evidence first,
- grouping evidence under clear source labels,
- summarizing long sources,
- asking the model to cite source IDs,
- avoiding excessive weak context.

## Prompt Pattern for Grounded Answers

Use explicit constraints:

```text
Answer using only the provided sources.
If the sources do not contain the answer, say you do not know.
Cite the source IDs used for every factual claim.
Do not use outside knowledge for policy, pricing, legal, or account-specific answers.
```

This helps, but prompt instructions cannot fix missing or noisy retrieval.

## Answer Modes

Different queries need different answer modes:

| Query Type | Best Answer Mode |
| --- | --- |
| Factual lookup | Short answer with citation. |
| Procedure | Step-by-step answer with source references. |
| Comparison | Table with evidence per row. |
| Multi-document synthesis | Summary plus cited supporting points. |
| Unknown answer | Clear refusal: not found in sources. |
| Ambiguous query | Ask clarifying question or show assumptions. |

## Study Checklist

- I can distinguish raw user message from retrieval query.
- I can explain when to use query rewriting, multi-query, and decomposition.
- I know why similarity scores are useful for debugging but not universal confidence.
- I can design a retrieve-rerank-pack pipeline.
- I can budget context tokens and preserve citation metadata.

