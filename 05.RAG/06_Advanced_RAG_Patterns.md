# Advanced RAG Patterns

Advanced RAG patterns exist because basic retrieve-then-generate pipelines fail on complex documents, ambiguous questions, multi-hop reasoning, and multimodal sources.

## Long Context Models vs RAG

Long-context models can read much larger inputs, but larger context does not automatically mean better answers.

Use long context when:

- the user provides a small set of documents,
- the answer requires holistic reading,
- you need fewer moving parts,
- latency and token cost are acceptable.

Use RAG when:

- the corpus is too large,
- documents change frequently,
- user permissions matter,
- citations and retrieval logs matter,
- only a small subset is relevant,
- query-time filtering is required.

Use both when:

- retrieval narrows the corpus,
- long context lets the model reason over richer selected evidence.

## Contextual Retrieval

Naive chunking can strip context from a passage. Contextual retrieval adds explanatory context to each chunk before embedding and/or lexical indexing.

Pattern:

```text
document + chunk
  -> generate short chunk-specific context
  -> prepend or store context with chunk
  -> embed contextualized chunk
  -> search contextualized representation
```

Anthropic reports that contextual embeddings and contextual BM25 reduce retrieval failures in their experiments. Source: <https://www.anthropic.com/engineering/contextual-retrieval>.

Best used for:

- policies,
- legal docs,
- long reports,
- technical docs with repeated phrases,
- chunks that depend on section headings.

Risks:

- extra preprocessing cost,
- generated context can be wrong,
- reindexing required when context-generation prompt changes,
- extra text can bias retrieval if too verbose.

## Late Chunking

Late chunking tries to preserve broader document context before producing chunk-level representations.

Why it matters:

- Early chunking embeds isolated text.
- Isolated chunks may not contain enough context.
- Late chunking lets chunk representations benefit from surrounding content.

Use late chunking when:

- a document's local statements depend heavily on global context,
- chunks repeat generic language,
- section headings are essential,
- retrieval misses relevant passages because chunks are too context-poor.

## Self-Correcting Retrieval

Self-correcting RAG adds validation loops.

```text
retrieve
  -> grade relevance/sufficiency
  -> if weak, rewrite query and retrieve again
  -> generate answer
  -> grade groundedness
  -> if weak, revise or refuse
```

Useful validators:

- relevance grader,
- answer-supported-by-context checker,
- citation checker,
- no-answer detector,
- policy/safety checker.

Do not let loops run unbounded. Set iteration limits, timeouts, and cost budgets.

## Agentic RAG

Agentic RAG gives an agent retrieval tools and lets it decide when to search, which source to search, and whether more evidence is needed.

Good fit:

- research assistants,
- multi-source workflows,
- troubleshooting assistants,
- tasks requiring tool use plus retrieval,
- questions where one retrieval pass is not enough.

Weak fit:

- low-latency FAQ bots,
- strict deterministic workflows,
- tasks with simple known retrieval path,
- systems where every tool call must be tightly controlled.

LangChain describes agentic RAG as a pattern where the agent reasons step by step and decides when and how to retrieve. Source: <https://docs.langchain.com/oss/python/deepagents/retrieval>.

## GraphRAG

GraphRAG uses graph structure along with text retrieval. The system extracts entities and relationships, stores them in a graph or graph-like index, and retrieves connected evidence for multi-hop questions.

Example question:

```text
Which suppliers are affected by the policy changes that apply to products manufactured in region X?
```

This may require:

- supplier entities,
- product entities,
- manufacturing regions,
- policy documents,
- relationships between them.

GraphRAG helps when the answer depends on connections, not just similar text.

Costs:

- entity extraction,
- relation extraction,
- graph maintenance,
- ontology/schema decisions,
- graph retrieval evaluation,
- more complicated debugging.

## Multimodal RAG

Multimodal RAG retrieves from non-text sources:

- scanned PDFs,
- images,
- charts,
- diagrams,
- slide decks,
- forms,
- screenshots,
- audio/video transcripts.

Strategies:

- OCR plus text RAG,
- image captioning plus text RAG,
- table extraction,
- page-level screenshot embeddings,
- vision-language retrieval,
- ColPali-style document page retrieval,
- multimodal answer generation.

Use multimodal retrieval when text extraction loses important visual meaning, such as charts, layout, forms, handwritten notes, or diagrams.

## Choosing an Advanced Pattern

| Problem | Pattern |
| --- | --- |
| Chunks lack context | Contextual retrieval or late chunking. |
| Dense search misses exact terms | Hybrid search. |
| Retrieved chunks are noisy | Reranking. |
| Question needs multiple searches | Query decomposition or agentic RAG. |
| Answer depends on relationships | GraphRAG. |
| Documents are visual | Multimodal RAG. |
| Model ignores relevant text | Better context packing and answer validation. |
| System hallucinates | Grounded prompts, citation checks, refusal logic. |

## Advanced RAG Evaluation

Advanced systems need tagged evaluations.

Create test subsets:

- `exact_match`,
- `semantic`,
- `multi_hop`,
- `table`,
- `visual`,
- `no_answer`,
- `permission_sensitive`,
- `requires_current_doc`,
- `ambiguous`.

Measure each subset separately. A change that improves semantic search may hurt exact-match queries.

## Study Checklist

- I can decide between long context and RAG for a workload.
- I understand contextual retrieval and why it helps.
- I know when self-correcting or agentic RAG is worth the latency.
- I can explain why GraphRAG helps multi-hop reasoning.
- I can identify when document layout requires multimodal retrieval.

