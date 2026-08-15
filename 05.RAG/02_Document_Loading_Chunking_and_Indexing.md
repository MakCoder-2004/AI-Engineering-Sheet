# Document Loading, Chunking, and Indexing

The indexing pipeline decides what the retriever can find later. Most RAG failures start before the user asks a question.

## Source Loading

Documents can come from:

- PDFs,
- HTML pages,
- Markdown files,
- Word documents,
- spreadsheets,
- slide decks,
- databases,
- ticketing systems,
- chat systems,
- wikis,
- object storage,
- APIs.

The loader's job is not only to read bytes. It must preserve enough structure to support retrieval.

Useful loader outputs:

- text content,
- source ID,
- title,
- headings,
- page number,
- table boundaries,
- image references,
- author or owner,
- created/updated timestamps,
- permissions,
- source URL.

## Parsing and Cleaning

Parsing quality matters. A PDF parser that mixes headers, footers, page numbers, and table columns into broken text will produce poor embeddings.

Clean:

- repeated headers and footers,
- page numbers,
- navigation menus,
- legal boilerplate repeated on every page,
- broken hyphenation,
- duplicate whitespace,
- OCR artifacts,
- hidden text,
- irrelevant sidebars.

Preserve:

- section headings,
- page numbers,
- table captions,
- figure captions,
- ordered lists,
- source links,
- timestamps,
- document hierarchy.

## Chunking

Chunking splits documents into retrievable units.

The chunk must be:

- small enough to fit retrieval and context budgets,
- large enough to contain a complete idea,
- traceable to the source,
- meaningful when read alone,
- compatible with the embedding model's input limit.

## Chunking Strategies

| Strategy | How It Works | Use When |
| --- | --- | --- |
| Fixed-size character/token chunks | Split every N characters or tokens. | Fast baseline, simple text. |
| Recursive splitting | Split by paragraphs, sentences, then smaller separators. | General-purpose docs. |
| Markdown/header-aware splitting | Split by headings and sections. | Technical docs, policies, tutorials. |
| Semantic chunking | Split where semantic topic changes. | Long narrative or mixed-topic docs. |
| Parent-child retrieval | Retrieve small child chunks, return larger parent sections. | Need precise search plus rich context. |
| Table-aware chunking | Keep table rows/headers together. | Financial, operational, tabular docs. |
| Sliding window chunks | Overlapping windows across text. | Avoid losing facts across boundaries. |

## Chunk Size

There is no universal best chunk size.

Starting points:

- FAQ answers: 100-300 tokens.
- Documentation paragraphs: 300-700 tokens.
- Policies and manuals: 500-1000 tokens.
- Legal or scientific text: section-aware chunks, often 700-1500 tokens.
- Tables: preserve logical table units rather than forcing token size.

Overlap is useful when answers span boundaries. Too much overlap increases storage, cost, duplicate retrieval, and prompt noise.

## Early Chunking vs Late Chunking

Early chunking:

```text
document -> chunks -> embed each chunk
```

Strengths:

- simple,
- cheap,
- works with standard embedding models,
- easy to update individual chunks.

Weaknesses:

- each chunk may lose global document context.

Late chunking:

```text
document -> model representation with broader context -> chunk-level embeddings
```

Strengths:

- chunks can carry information from surrounding document context,
- useful when local chunks depend on global section meaning.

Weaknesses:

- model/support requirements are more specialized,
- can be more expensive,
- implementation is less universal.

## Contextual Chunk Enrichment

Contextual retrieval adds a short explanation to each chunk before embedding or indexing.

Example:

```text
Original chunk:
"The deadline is 30 days after the transaction date."

Contextualized chunk:
"This chunk is from the employee travel reimbursement policy, section 'Expense submission deadlines'. The deadline is 30 days after the transaction date."
```

This helps retrieval because the chunk now contains the missing context. Anthropic describes contextual embeddings and contextual BM25 as techniques for improving retrieval over chunked documents. Source: <https://www.anthropic.com/engineering/contextual-retrieval>.

## Metadata Enrichment

Every chunk should include metadata.

Recommended fields:

```json
{
  "document_id": "travel_policy_2026",
  "chunk_id": "travel_policy_2026#chunk_0012",
  "source_uri": "s3://company-policies/travel_policy.pdf",
  "title": "Travel Policy",
  "section": "Expense Submission Deadlines",
  "page": 14,
  "chunk_index": 12,
  "language": "en",
  "tenant_id": "acme",
  "access_group": "employees",
  "updated_at": "2026-06-10",
  "embedding_model": "text-embedding-3-small"
}
```

Metadata is not decoration. It powers citations, filtering, access control, deletion, reindexing, and debugging.

## Embedding

Embedding transforms each chunk into a vector.

Decisions:

- model provider,
- vector dimension,
- input token limit,
- text normalization,
- batching,
- retry behavior,
- rate limit handling,
- caching,
- storage format.

Do not silently truncate chunks. If a chunk is longer than the embedding model supports, split it or log it as an ingestion error.

OpenAI's embeddings FAQ states that vector databases are recommended for fast nearest-neighbor retrieval over many vectors and that OpenAI embedding outputs are normalized to length 1. Source: <https://help.openai.com/en/articles/6824809-embeddings-faq>.

## Indexing into a Store

Common store options:

- Chroma for local development and prototypes,
- pgvector for PostgreSQL-based applications,
- Pinecone/Qdrant/Weaviate/Zilliz for managed production vector search,
- Elasticsearch/OpenSearch when full-text search is already central.

Store:

- stable ID,
- vector,
- original chunk text or pointer,
- metadata,
- sparse vector if using hybrid search,
- ingestion version.

## Ingestion Validation

After indexing, verify:

- number of documents loaded,
- number of chunks created,
- number of vectors inserted,
- expected embedding dimension,
- no empty chunks,
- no missing permission metadata,
- no duplicate IDs unless intended,
- sample queries retrieve expected sources.

Example validation table:

| Check | Why It Matters |
| --- | --- |
| `chunks_created > 0` | Detects parser failures. |
| `empty_chunks == 0` | Prevents garbage vectors. |
| `embedding_dimension == expected` | Prevents query-time errors. |
| `all_chunks_have_source_uri` | Enables citations. |
| `all_chunks_have_tenant_id` | Enables secure retrieval. |
| `sample_recall@5` | Confirms retrieval works before deployment. |

## Incremental Indexing

Production data changes. Indexing must handle:

- new documents,
- edited documents,
- deleted documents,
- permission changes,
- source moves,
- schema changes.

Recommended flow:

1. Track source document version.
2. Re-parse changed documents.
3. Generate chunks with deterministic IDs.
4. Upsert new chunks.
5. Delete stale chunks for that document/version.
6. Run sample retrieval checks.
7. Log the ingestion run.

## Study Checklist

- I can explain why parsing quality affects retrieval.
- I can choose chunking strategy based on document type.
- I can define metadata needed for citations, filtering, and reindexing.
- I know why embedding model changes require reindexing.
- I can validate an ingestion run before using it in a RAG system.

