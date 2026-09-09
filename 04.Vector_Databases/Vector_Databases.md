# Vector Databases

> What they are, why they exist, and a full deep dive into the two databases used in the Production RAG course: **Chroma** (development) and **pgvector on PostgreSQL** (production).
> Companion handbook: [`05.RAG/RAG.md`](../05.RAG/RAG.md)

## Table of Contents

1. [What Is a Vector Database?](#1-what-is-a-vector-database)
2. [Why We Use Them — and Why They Matter](#2-why-we-use-them--and-why-they-matter)
3. [Core Mechanics — Similarity, KNN vs ANN, Indexes](#3-core-mechanics--similarity-knn-vs-ann-indexes)
4. [Chroma DB — Full Deep Dive](#4-chroma-db--full-deep-dive)
5. [pgvector — PostgreSQL as a Vector Database](#5-pgvector--postgresql-as-a-vector-database)
6. [Chroma vs pgvector — Choosing](#6-chroma-vs-pgvector--choosing)
7. [Study Checklist](#7-study-checklist)
8. [Source Trail](#8-source-trail)

---

## 1. What Is a Vector Database?

A **vector database** stores high-dimensional numerical vectors (embeddings) and answers one question fast: *"which stored vectors are closest to this query vector?"*

Embeddings turn text, images, or audio into points in a vector space where **distance means semantic similarity** — "How do I cancel my subscription?" lands near "termination policy" even with zero shared words. A vector DB is the engine that exploits this: it indexes millions of vectors so nearest-neighbor search takes milliseconds.

```mermaid
flowchart LR
    subgraph Write["Write path (indexing)"]
        D["Documents"] --> E["Embedding model"] --> V["Vectors<br/>+ metadata"] --> IX[(Vector DB<br/>ANN index)]
    end
    subgraph Read["Read path (query)"]
        Q["Query"] --> QE["Embed same model"] --> S["Nearest-neighbor<br/>search"] --> R["Top-k chunks<br/>+ scores"]
        IX -.-> S
    end
```

**Where it sits in AI systems:** it is the retrieval layer behind RAG, semantic search, recommendations, duplicate detection, and multimodal search. In RAG specifically, retrieval quality *is* answer quality — the vector DB is where most "the bot gives wrong answers" bugs actually live.

---

## 2. Why We Use Them — and Why They Matter

### The naive approach breaks

You *can* store embeddings in a NumPy array and compute distances to every vector:

```python
scores = [cosine_similarity(query_vec, v) for v in all_vectors]   # O(n) per query
```

This brute-force **k-nearest-neighbor (KNN)** scan is fine for hundreds of vectors. At 1M vectors × 1536 dimensions it means billions of float operations per query — seconds of latency, no concurrency story, no persistence, no filtering. It does not scale.

### What a vector DB adds over a plain array

| Capability | Why it matters |
|------------|----------------|
| **ANN indexes** (HNSW, IVF) | Sub-linear search — ~millisecond queries at millions of vectors |
| **Persistence & durability** | Index survives restarts; backups, replication |
| **Metadata filtering** | `WHERE topic = 'database'` *before* vector search — powers access control and multi-tenancy |
| **CRUD** | Update/delete documents without rebuilding everything |
| **Hybrid search** | Combine vector + keyword (BM25) scoring |
| **Scaling** | Sharding, replication, managed hosting |

### Why it is important (the one-liner for interviews)

> LLM knowledge is frozen and generic; the vector DB is what connects it to your **private, current** data at query time. Without fast similarity search, RAG at production scale is impossible.

---

## 3. Core Mechanics — Similarity, KNN vs ANN, Indexes

### Similarity metrics

| Metric | Formula / Operator | Range | Use when |
|--------|--------------------|-------|----------|
| **Cosine similarity** | `cos(θ) = A·B / (‖A‖‖B‖)` | −1…1 (1 = same direction) | Default for text embeddings; ignores magnitude |
| **Euclidean (L2)** | `√Σ(aᵢ−bᵢ)²` | 0…∞ (0 = identical) | When magnitude matters; Chroma's default |
| **Dot product** | `A·B` | −∞…∞ | OpenAI embeddings are normalized → dot product ≡ cosine, cheaper |

> **Gotcha:** some stores return *similarity* (higher = better), others return *distance* (lower = better). Chroma returns distance — convert with `similarity = 1 / (1 + distance)`.

### KNN vs ANN

- **KNN (exact):** compare against *every* vector → perfect recall, O(n) latency.
- **ANN (approximate nearest neighbor):** build an index that trades a little recall (e.g., 95–99%) for orders-of-magnitude speed. This is the production trade-off — nobody needs the 3rd-best match if you can get a near-best match 100x faster.

### Index types

| Index | Idea | Tuning | Trade-off |
|-------|------|--------|-----------|
| **Flat / brute force** | Exact scan | none | Perfect recall; only for small data |
| **IVF (Inverted File)** | Cluster vectors; search only the `nprobe` nearest clusters | `nlist` (clusters), `nprobe` (clusters searched) | Fast; can miss neighbors in bordering clusters |
| **HNSW (Hierarchical Navigable Small World)** | Multi-layer proximity graph; greedy traversal from coarse top layer to fine bottom layer | `M` (edges/node), `ef_construction` (build), `ef_search` (query) | Best recall/latency balance; more memory, slower builds |

```mermaid
flowchart TB
    Q["Query vector"] --> L3["Layer 3 (few nodes)<br/>long-range jumps"]
    L3 --> L2["Layer 2"]
    L2 --> L1["Layer 1 (all nodes)<br/>precise local search"]
    L1 --> R["Approximate top-k"]
```

**Recall levers:** higher `ef_search`/`nprobe` → better recall, higher latency. Always measure recall on a small labeled test set before shipping.

---

## 4. Chroma DB — Full Deep Dive

**Chroma** is an open-source embedding database designed for developer ergonomics: it runs **embedded in your Python process** (no server), persists to a local directory, and is the default vector store of the LangChain tutorials — the course uses it for all development work.

### Architecture

```mermaid
flowchart LR
    subgraph YourApp["Your Python process"]
        LC["LangChain / app code"] --> C["Chroma client<br/>(embedded)"]
        C --> COLL["Collection<br/>(name + metadata schema)"]
        COLL --> EF["Embedding function<br/>(e.g. OpenAI)"]
        COLL --> IDX["ANN index<br/>(HNSW)"]
    end
    IDX --> DISK[("persist_directory<br/>sqlite + parquet")]
```

Key concepts:

- **Collection** ≈ a table: holds vectors + documents + metadata under one name.
- **Embedding function** — attach it once; Chroma embeds automatically on add and query. If you add raw vectors instead, Chroma stores them as-is.
- **Persistence** — everything lands in `persist_directory`; reopen by pointing a new client at the same path.
- **Distance** — default L2; scores are distances (lower = better).

### 4.1 Setup and first collection

```bash
pip install chromadb langchain-chroma
```

```python
import chromadb

# persistent client (survives restarts)
client = chromadb.PersistentClient(path="./chroma_db")
# client = chromadb.EphemeralClient()          # in-memory, for tests

collection = client.get_or_create_collection(name="docs")
collection.add(
    ids=["doc1", "doc2"],                       # IDs are required and unique
    documents=["LangChain is an LLM framework.",
               "Chroma is an embedding database."],
    metadatas=[{"topic": "framework"}, {"topic": "database"}],
)

res = collection.query(query_texts=["What is Chroma?"], n_results=2)
```

### 4.2 The LangChain integration (course pattern)

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# create + index in one call
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db/",
)
```

> Code: `code/production-course-main-code/vector_stores.py` — `chroma_basics()`

> Course map (LangChain lessons 45-46): `WebBaseLoader` → `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)` → `OpenAIEmbeddings(model="text-embedding-3-small")` → `Chroma.from_documents` → `as_retriever(k=4)`. The course mentions Pinecone — use Chroma here (same LangChain interface, local persist, no key). Full ingest + naive vs 2-step flow: `05.RAG/RAG.md` §25.5; agent wiring: `07.LangChain_Ecosystem/LangChain/LangChain.md` §6.
> `init_embeddings("openai:text-embedding-3-small")` (LangChain handbook §6) is the provider-agnostic alias for `OpenAIEmbeddings(model=...)` above — same vectors, first form switches provider with one string.

### 4.3 Similarity search with scores

```python
# distance-based (Chroma default: L2, lower = better)
results = vectorstore.similarity_search_with_score("Explain vector stores.", k=3)
for doc, distance in results:
    print(f"similarity={1 / (1 + distance):.4f} | {doc.page_content[:60]}")
```

> Code: `vector_stores.py` — `similarity_search_with_scores()`

### 4.4 Metadata filtering

Metadata is stored alongside vectors and filtered **before** vector search:

```python
# exact match
vectorstore.similarity_search("What databases are available?", k=5,
                              filter={"topic": "database"})

# raw chromadb operators
collection.query(
    query_texts=["databases"],
    where={"topic": {"$eq": "database"}},        # $eq $ne $gt $gte $lt $lte $in $nin
    where_document={"$contains": "vector"},      # substring on document text
)
```

> Code: `vector_stores.py` — `metadata_filtering()`

### 4.5 as_retriever — plugging into RAG

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",          # or "mmr" for diversity
    search_kwargs={"k": 3},            # mmr: also "fetch_k": 20
)
docs = retriever.invoke("How do I build AI applications?")
```

> Code: `vector_stores.py` — `as_retriever()`

### 4.6 Persistence — survive the restart

```python
# index once
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                    persist_directory="./chroma_db/")

# "restart": reload from disk — no re-embedding, count is intact
reloaded = Chroma(embedding_function=embeddings, persist_directory="./chroma_db/")
print(reloaded._collection.count())
reloaded.similarity_search("LangChain", k=2)
```

> Code: `vector_stores.py` — `persist_chroma()`

### 4.7 CRUD essentials (raw client)

```python
collection.update(ids=["doc1"], documents=["updated text"], metadatas=[{"topic": "new"}])
collection.delete(ids=["doc1"])
collection.delete(where={"topic": "obsolete"})     # delete by filter
```

### 4.8 When to use Chroma — and its limits

| Great for | Not designed for |
|-----------|------------------|
| Local development & prototyping | Massive multi-tenant production fleets |
| Course projects, notebooks, tests | Strict SLAs / horizontal scale-out |
| Embedded, zero-ops simplicity | Advanced access control (use Postgres RLS instead) |
| Datasets up to ~millions of vectors | Need for SQL joins with relational data |

**Course pattern:** prototype everything in Chroma → move to pgvector/Supabase for production. The LangChain code barely changes (swap `Chroma` for `PGVector`).

---

## 5. pgvector — PostgreSQL as a Vector Database

**pgvector** is a Postgres *extension* that adds a `vector` column type, distance operators, and ANN indexes. Your relational database becomes your vector database.

### Why it matters

- **One database for everything** — vectors live next to your users, permissions, and application tables → real SQL joins, foreign keys, transactions.
- **ACID + backups + ops you already have** — no new infrastructure to run.
- **Row-Level Security** — multi-tenant RAG where tenant isolation is enforced by the database, not app code.
- **Supabase** (the course's production host) is managed Postgres with pgvector preinstalled.

### 5.1 Setup

```sql
-- once per database
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1536),          -- dimension must match the model:
    created_at  TIMESTAMPTZ DEFAULT now()  -- text-embedding-3-small = 1536
);
```

### 5.2 Distance operators

| Operator | Distance | Note |
|----------|----------|------|
| `<->` | Euclidean (L2) | |
| `<=>` | Cosine distance | `1 − cosine similarity`; **default choice for text** |
| `<#>` | Negative inner product | Use with normalized (OpenAI) embeddings |

```sql
-- top-5 by cosine similarity ($1 = query embedding passed from the app)
SELECT id, content, metadata,
       embedding <=> $1 AS distance
FROM documents
WHERE metadata->>'tenant_id' = 'acme'      -- filter BEFORE vector scan
ORDER BY embedding <=> $1
LIMIT 5;
```

### 5.3 ANN indexes (the production step)

Without an index, pgvector does exact KNN — fine up to ~100K rows, then add one:

```sql
-- HNSW (recommended: best recall/speed; no training)
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- IVFFlat (alternative: faster builds, needs data first, train once)
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);          -- ~rows/1000; query with SET ivfflat.probes = 10;
```

Tune recall with query-time `SET hnsw.ef_search = 40;` (default 40) — higher = better recall, slower.

### 5.4 Hybrid search in pure SQL

Postgres has full-text search built in — combine lexical + semantic with Reciprocal Rank Fusion:

```sql
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
    FROM documents ORDER BY embedding <=> $1 LIMIT 20
),
lexical AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', content),
                                                    websearch_to_tsquery('english', $2)) DESC) AS rank
    FROM documents
    WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', $2)
    LIMIT 20
)
SELECT id, SUM(1.0 / (60 + rank)) AS rrf_score          -- k = 60
FROM (SELECT * FROM semantic UNION ALL SELECT * FROM lexical) t
GROUP BY id
ORDER BY rrf_score DESC
LIMIT 5;
```

### 5.5 LangChain integration

```python
# pip install langchain-postgres
from langchain_postgres import PGVector

vectorstore = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="docs",
    connection="postgresql+psycopg://user:pass@localhost:5432/mydb",
    use_jsonb=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 5.6 Supabase setup (course's production path)

1. Create project → open SQL editor.
2. Enable the extension: `create extension vector;`
3. Create the `documents` table + `match_documents` RPC function (Supabase docs provide the template).
4. Query from Python with `supabase-py` passing the query embedding → get top-k rows.
5. Enable **Row-Level Security** so each tenant can only search its own rows — the security layer from the course's production project.

---

## 6. Chroma vs pgvector — Choosing

| Criterion | Chroma | pgvector (Postgres) |
|-----------|--------|---------------------|
| Setup | `pip install`, embedded | Run/rent Postgres + extension |
| Best stage | Development, prototypes | Production |
| Scale comfort | Up to ~millions of vectors | Millions+ with HNSW; Postgres scaling path |
| Metadata filtering | `where` with `$eq/$gt/...` | Full SQL + JSONB + RLS |
| Relational joins | No | Yes — killer feature |
| Multi-tenancy | App-level | Database-level (Row-Level Security) |
| Hybrid search | Via LangChain EnsembleRetriever | Native (tsvector) in SQL |
| Backups/HA | Manual (copy dir) | Mature tooling / managed (Supabase) |
| Course usage | All hands-on sections | Production hosting + production project |

**Decision rule:** start in Chroma for iteration speed; move to pgvector when you need joins, tenant isolation, transactions, or managed ops — which in practice means *production*.

---

## 7. Study Checklist

- [ ] I can define a vector DB and draw its read/write paths
- [ ] I can explain why brute-force KNN fails at scale and what ANN trades away
- [ ] I know cosine vs L2 vs dot product, and which operators map to which store
- [ ] I can explain HNSW and IVFFlat and name their tuning knobs
- [ ] I can create a Chroma collection, add/query/filter, persist, and reload it
- [ ] I can convert Chroma distances to similarities and explain the convention
- [ ] I can write pgvector SQL: table, `<=>` query, HNSW index, metadata pre-filter
- [ ] I can write a hybrid SQL query combining tsvector and vectors via RRF
- [ ] I can state when to pick Chroma vs pgvector and why production favors Postgres
- [ ] I know how Supabase + RLS enables secure multi-tenant RAG

## 8. Source Trail

- Chroma docs: <https://docs.trychroma.com/>
- pgvector GitHub: <https://github.com/pgvector/pgvector>
- Supabase pgvector guide: <https://supabase.com/docs/guides/database/extensions/pgvector>
- OpenAI embeddings FAQ (normalized vectors, DB recommendations): <https://help.openai.com/en/articles/6824809-embeddings-faq>
- Companion: [`05.RAG/RAG.md`](../05.RAG/RAG.md)
