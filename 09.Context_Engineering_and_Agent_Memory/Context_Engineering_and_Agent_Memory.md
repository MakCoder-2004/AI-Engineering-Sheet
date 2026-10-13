# Context Engineering and Agent Memory — A Practical Handbook

> **Central idea:** A model does not automatically know which parts of your application, documents, conversation, tools, and user state matter *now*. Context engineering is the design of the pipeline that selects, authorizes, orders, budgets, and verifies that information for each model call. Memory is the subset of that pipeline concerned with carrying useful state across turns and sessions.

## Contents

1. [Context Window Deep Dive](#1-context-window-deep-dive)
2. [Context Assembly Framework](#2-context-assembly-framework)
3. [RAG Done Right](#3-rag-done-right)
4. [Memory Systems](#4-memory-systems)
5. [Context for Agents](#5-context-for-agents)
6. [Production Patterns](#6-production-patterns)
7. [Enterprise Context Patterns](#7-enterprise-context-patterns)
8. [End-to-end architecture and study plan](#8-end-to-end-architecture-and-study-plan)

### Prerequisites and using the examples

Understand messages/roles, basic Python, retrieval, and API calls. The examples use Python 3.11+ and, where indicated, require a model provider API key or optional local embedding and reranking models. Each Python block reproduces code from the implementation for that topic; it may use imports, globals, or methods defined elsewhere in that implementation. Read it alongside the surrounding explanation. Some examples are teaching demonstrations with keyword heuristics or in-memory stores; their limitations are explained where they appear. Model IDs, quotas, API syntax, and published prices change: verify current provider settings before running or estimating costs. Numerical outcomes are not claimed as benchmark results.

---

## 1. Context Window Deep Dive

### 1.1 What is inside a context window?

The context window is the model's finite input capacity for a call, subject to the provider's limit and sometimes shared with generated output. In an application the assembled input may contain system/developer instructions, current user message, selected conversation turns, retrieved passages, tool definitions, tool responses, structured output requirements, and multimodal payloads. Each has different trust, priority, volatility, and cost. **A large window is capacity, not an instruction to fill it.** Long contexts can fail because relevant evidence is absent, buried, contradictory, stale, or unauthorised, even when the request technically fits.

Define a call budget before retrieval: `usable_input = model_limit - reserved_output - provider_overhead - safety_margin`. Measure exact token counts using the provider's tokenizer/counting endpoint when available. Character counts (`len(text) // 4`) and `1.3 * word_count` are rough English-text estimates; code, non-Latin text, JSON, images, and tool schemas vary. Count *the actual serialized request* and inspect the provider's returned usage after generation. If the provider applies separate constraints, respect those as well.

An actual token count and a quick character estimate illustrate the difference:

```python
def count_tokens(text: str, model: str = model) -> int:
    """Count tokens in a text string using the Anthropic API"""
    response = client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens


def estimate_tokens(text: str) -> int:
    """Quick estimate: ~4 characters per token for English"""
    return len(text) // 4
```

### 1.2 Token Economics

The total cost of an answer is not just its prompt: indexing/embedding, retrieval and reranking, compression, multiple model calls, cached tokens, generated output, and retries all count. If input costs `p_in` and output costs `p_out` USD per million tokens, `call_cost = input_tokens * p_in / 1_000_000 + output_tokens * p_out / 1_000_000`. A more useful metric is `total_spend / number_of_acceptable_answers`: a cheap answer that is wrong is expensive to fix. Be careful that a summarization or judge call adds extra expense. For capacity planning, estimate average and high-percentile tokens per request, concurrency, provider token-per-minute limits, and cache hit rate.

The cost calculator uses this function; `PRICING` holds configurable model rates:

```python
def calculate_request_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet"
) -> float:
    """Calculate cost for a single request"""
    pricing = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return input_cost + output_cost
```

Use a token-counting endpoint to compare real usage to rough estimates. Before optimizing, record: task success, input/output tokens, retrieval overhead, retries, response time, and cost per successful answer. Optimize the bottleneck, not token count alone.

### 1.3 Lost in the Middle — Deep Dive

Long-context experiments have shown position sensitivity on some models and tasks: relevant evidence in the middle can be harder to use than evidence near the beginning or end. This is a **tendency to test**, not a universal U-shaped law or a reason to discard middle passages. Causes include competition among similar passages, conflicting instructions, and the model's difficulty integrating distant evidence. Different models, lengths, tasks, and citation formats may produce different effects.

Run a controlled probe: place the same answer-bearing passage at early/middle/late positions; keep question, total length, distractors, model, and decoding configuration fixed; rotate distractor order and repeat each position. Measure exact-match or semantic correctness **and citation to the right passage**. If position matters, surface the most critical constraint early, put the immediate question near the end, use section boundaries and citations, reduce distractors, and test ordering on your own workload.

The position experiment inserts the target among identically formatted distractor records:

```python
records = generate_records(num_records, exclude_id=TARGET_ID)
target_record = f"Customer {TARGET_ID}: refund window {TARGET_WINDOW} days."

if position == "beginning":
    records.insert(0, target_record)
elif position == "end":
    records.append(target_record)
else:  # middle
    records.insert(len(records) // 2, target_record)
```

The recommended arrangement is a **starting hypothesis**, not a guaranteed optimum. Repeat each position test rather than interpreting one answer as proof of an effect.

### 1.4 Attention Dilution

Attention dilution means relevant information competes with irrelevant or conflicting material. Example: retrieving 20 passages about generic shipping policies when the question asks for a tenant-specific warranty exception. More text raises the risk that the model follows the wrong policy or mixes incompatible facts. Distinguish **retrieval recall** (did candidates contain the answer?) from **final context precision** (how much injected text is answer-bearing?). A bigger initial candidate set can improve recall, but only a smaller, verified subset should reach the generator.

This anti-pattern sends every document instead of selecting relevant chunks:

```python
def antipattern_document_dump(question: str, all_docs: str) -> str:
    """
    BAD: Dump entire documentation into context
    Problems: Expensive, slow, attention dilution
    """
    # This might be 50K-100K tokens!
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": f"{all_docs}\n\nQuestion: {question}"}],
    )
    return response.content[0].text
```

Five recurring failures are whole-document dumps, static context for every user, important information buried in noise, no explicit budget, and no measurement. Evidence selection must also preserve necessary exceptions and negations; shorter is not automatically safer.

### 1.5 Quality vs Quantity

Good context is **relevant, sufficient, current, trustworthy, attributable, and affordable**. Bad context can be shorter (missing the exception) or longer (drowning it). Ask: Does the chosen passage support every requested claim? Does it include the policy effective date? Is there conflicting evidence? Can the answer cite a specific source? Compare candidate assemblies on the same labeled questions rather than declaring one layout superior from a single demo.

The noisy scenario generator cycles through irrelevant sentences to stress-test an otherwise answer-bearing context:

```python
def generate_noise(sentences: int = 50) -> str:
    """Generate irrelevant filler content"""
    noise_templates = [
        "The company was established in the early 2000s.",
        "Our headquarters are located in San Francisco.",
        "We have a diverse team of professionals.",
        "Customer satisfaction is our top priority.",
        "We operate in multiple regions globally.",
        "Our technology stack is modern and scalable.",
        "The board of directors meets quarterly.",
        "We sponsor various community events.",
        "Our office features an open floor plan.",
        "Employee wellness programs are available.",
    ]

    return " ".join(noise_templates[i % len(noise_templates)] for i in range(sentences))
```

Useful ratio: `signal_density = relevant_evidence_units / all_injected_evidence_units`, but it does not tell you whether the one missing fact was critical. Track sufficiency/coverage separately.

### 1.6 Measuring Performance

Create a fixed evaluation set with factual questions, unanswerable questions, exceptions, conflicting revisions, and malicious retrieved text. Record the expected answer **and the supporting passage ID**; evaluate retrieval before generation. Retrieval metrics: recall@k, precision@k, MRR/nDCG where ranking matters, and zero-hit/filtered-hit rates. Generation metrics: answer correctness, evidence support, citation accuracy, appropriate abstention, task completion, latency, total cost, and quality across user cohorts. Report confidence intervals or multiple runs when outputs vary; keep a human-reviewed sample to calibrate an LLM judge.

The analyzer calculates an explicit signal-density proxy before comparing repeated runs:

```python
def signal_density(context: str, keywords: list[str]) -> float:
    """Fraction of sentences in the context that mention any expected keyword.

    A deterministic proxy for 'how much of this context is actually on-topic'.
    Not perfect — but defensible, reproducible, and free.
    """
    sentences = [s for s in re.split(r"[.!?\n]+", context) if s.strip()]
    if not sentences:
        return 0.0
    keywords_lc = [k.lower() for k in keywords]
    relevant = sum(1 for s in sentences if any(k in s.lower() for k in keywords_lc))
    return relevant / len(sentences)
```

Compare minimal/structured/noisy/bloated variants across repeated runs and use a second model to judge answers if desired. Judge scores are estimates, not ground truth: use blinded reviews, explicit rubrics, and audit judge mistakes. Separate retriever failures from assembler and generator failures in traces.

### 1.7 Project Context Audit

**Goal:** replace a static information dump with a query-specific context builder and demonstrate any gains rather than assuming them. Compare both versions against the same questions and measure usage, latency, answer correctness, and citations.

The focused assembler retrieves a policy and only adds product information if relevant:

```python
def build_good_context(question: str) -> str:
    """
    GOOD: Engineered context assembly.
    - Only relevant sections retrieved
    - Structured with markdown headers
    - Question placed at the END (recency primacy)
    """
    parts = ["## Relevant Policy", retrieve_relevant_policy(question)]

    products = retrieve_relevant_products(question)
    if products:
        parts.extend(["\n## Related Products", products])

    parts.extend(
        [
            "\n## Customer Question",
            question,
            "\n## Instructions",
            "Answer based on the policy above. Be concise and helpful.",
        ]
    )

    return "\n".join(parts)
```

**Build procedure:** (1) inventory static context and sources; (2) classify by owner, sensitivity, change frequency, and task relevance; (3) set an input/output budget; (4) retrieve only the authorized current policy and any relevant product; (5) assemble labeled evidence and the question; (6) run both versions against the same test set; (7) inspect errors, not just mean token savings. Include a counterexample where extra context is necessary (e.g., electronics follow a shorter return window). Deliver an audit table with `query`, `needed_source`, `selected_source`, `tokens`, `correct?`, `supported?`, `latency`, `cost`.

---

## 2. Context Assembly Framework

### 2.1 Six Context Types

Application context can be divided into six composable kinds; the *current task* is the final request, not a seventh stored information category.

| Type | Purpose | Typical origin | Typical failure |
|---|---|---|---|
| Instructional | Role, behavior, constraints | Application configuration | User data impersonates an instruction |
| Example | Demonstrate desired behavior | Curated few-shot library | Stale or misleading examples |
| Knowledge | Evidence for the current task | Authorized retrieval | Wrong version or ungrounded citation |
| Memory | Relevant user/session state | Conversation/store | Stale, false, or cross-user fact |
| Tool | Available operations and schemas | Tool registry | Too many tools or uncontrolled results |
| Output | Response schema/style | Product contract | Invalid shape or lost constraints |

This assembler joins the six kinds and the current task:

```python
def assemble_full_context(
    instructional: str,
    examples: str,
    knowledge: str,
    memory: str,
    tools: str,
    output: str,
    current_task: str
) -> str:
    parts = []
    if instructional:
        parts.append(instructional)
    if examples:
        parts.append(examples)
    if knowledge:
        parts.append(knowledge)
    if memory:
        parts.append(memory)
    if tools:
        parts.append(tools)
    if output:
        parts.append(output)
    parts.append(f"## Current Task\n{current_task}")
    return "\n\n---\n\n".join(parts)
```

Track each component's `source_id`, scope, version, trust level, expiry, token cost, and whether it is mandatory. A header is helpful for navigation but is **not** a security boundary; role separation and server-side authorization remain necessary.

### 2.2 Context Assembly Order

An initial layout is **instructions → examples → knowledge → memory → tools → output format → current task**. Keep stable instructions at the appropriate high-priority message role, then add short, query-relevant evidence; place the live question so it is unmistakable. API-native tool schemas generally belong in the API's `tools` field, not a hand-written paragraph; the linear layout describes conceptual ordering, not a mandate to flatten every API field into one user message.

One assembler explicitly reserves a head and tail around optional middle sections:

```python
full_text = self._join_with_budget(
    head=[instruction],
    middle=[p for p in (examples_text, knowledge, memory) if p],
    tail=[output, task],
)
```

This implementation drops entire middle sections when the budget is exceeded. This demonstrates prioritization but can silently omit essential evidence and its `components_used` list can still name a dropped section. A robust assembler returns actual included component IDs, verifies required evidence survived, counts real tokens, and either compresses intelligently or abstains/escalates when evidence cannot fit. Test positions experimentally (Chapter 1).

### 2.3 Right Info, Right Time

Retrieve in response to the **current intent**, not on every turn. For a support question, fetch the relevant policy; for an agent about to write a file, give the relevant interface and write-tool schema; for a follow-up, carry the unresolved referent and applicable constraints. Treat history as a source to query, not an append-only prompt. Refresh volatile facts (prices, permissions, live account data) at point of use. Persist durable preferences separately from temporary state; never elevate retrieved text into application authority.

The following query-dependent selector returns only documents with matching terms:

```python
def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
    query_lower = query.lower()

    scored = []
    for doc in self.documents:
        score = sum(
            1 for word in query_lower.split() if word in doc.content.lower()
        )
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
```

This is a lexical baseline; authorize and filter documents *before* scoring them. A practical decision rule: include a component if removing it would harm task success, grounding, authorization, or required format more than its token/latency cost. Validate with ablation tests, not intuition alone.

### 2.4 Context Templates

Templates are reusable **contracts**, not static walls of text. Define required slots, type/length limits, escaping/quoting for external content, citation requirements, and behavior when a slot is empty. Common patterns include support, RAG Q&A, code assistant, agent, and content generator. For example, a RAG template should say how to abstain and reference source labels; a code template needs relevant existing code and compatibility constraints.

The Q&A template displays each retrieved passage with its source and relevance score:

```python
def rag_qa_template(
    documents: List[RetrievedDocument], question: str, include_confidence: bool = True
) -> str:
    """Template for RAG-based Q&A systems"""

    docs_section = []
    for i, doc in enumerate(documents, 1):
        docs_section.append(f"""### Document {i}
Source: {doc.source}
Relevance: {doc.score:.2f}

{doc.content}""")

    docs_text = "\n\n".join(docs_section)

    confidence_section = (
        """
## Confidence
Rate your confidence: HIGH (answer clearly in docs), MEDIUM (partially covered), LOW (mostly inference)"""
        if include_confidence
        else ""
    )

    return f"""## Task
Answer the question based ONLY on the provided documents.
If the answer isn't in the documents, say "I don't have information about that in my knowledge base."

## Retrieved Documents

{docs_text}

## Important
- Cite sources using [Source: X] format
- Don't make up information not present in documents
- Be concise and direct
{confidence_section}

## Question
{question}

## Answer
Provide your answer, then list sources used."""
```

Template slots are still untrusted if filled from retrieval or user input. This template prints `Relevance` and asks the model for confidence, but a retriever score is not calibrated answer confidence. Validate that every citation points to actually provided evidence.

### 2.5 Project Assembly Pipeline

**Architecture:** request → identity/permissions → query interpretation → retrieval → history/memory selection → example/tool selection → budget/assembly → generation → evidence and format checks → metrics. A small support-agent implementation joins keyword `KnowledgeRetriever`, `ContextAssembler`, `SupportPipeline`, cost/latency reporting, and optional inspection of the assembled text.

The pipeline assembles the request before making its model call:

```python
context = self.assembler.assemble(
    query=query,
    user=user,
    history=history,
    examples=self.default_examples if use_examples else None,
)
```

**Build exercise:** add source IDs and authorization to documents; reserve output tokens before retrieval; drop irrelevant examples first; protect an answer-bearing policy exception; enforce a tokenizer-based input cap; record actual included components and explicit `insufficient_evidence` when necessary. Compare retrieved chunks, assembled prompt, final answer, usage, and citations per trace. Keyword matching is a teaching simplification, not semantic search.

---

## 3. RAG Done Right

### 3.1 Why RAG Fails

RAG is a sequence of independent failure opportunities: missing/obsolete or unauthorised content at ingestion, bad text extraction, poor chunk boundaries, mismatched query vocabulary, aggressive filters, low recall in first-stage search, poor rerank, lost negation during compression, context overfill, and unsupported generation. The generator cannot cite evidence it never sees. Diagnose from left to right: **Did the corpus contain the answer? Did any candidate contain it? Did it survive reranking/compression? Was it injected? Did the answer use and cite it correctly?**

The pipeline processes documents and initializes the retriever, compressor and generator:

```python
def __init__(self, documents: List[Dict]):
    # Process documents
    processor = DocumentProcessor()
    self.chunks = processor.process(documents)

    # Initialize components
    self.retriever = HybridRetriever(self.chunks)
    self.compressor = ContextCompressor()
    self.client = anthropic.Anthropic()
```

Do not confuse answer fluency with evidence support. Add an abstention path when authorized evidence is missing or conflicting, and investigate expired document versions.

### 3.2 Chunking Strategies

**Fixed-size** chunks are easy and predictable but can split a table or a sentence. **Structure-aware** chunks preserve headings/paragraphs; **recursive** splitting falls back from section to paragraph to sentence/word when sections are too long; **sentence windows** retrieve a precise hit then expand neighboring sentences. Code often favors functions/classes; policies favor clauses plus headings; tabular data may need row/header preservation. Consider parent-child retrieval: index small units for search, inject a larger parent for context.

This sentence-window chunker stores a sentence for search and its neighbors for answering:

```python
def sentence_window_chunk(text: str, window_size: int = 3):
    import nltk

    sentences = nltk.sent_tokenize(text)

    chunks = []
    for i, sentence in enumerate(sentences):
        # Store the sentence as the chunk
        # But keep surrounding context as metadata
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        chunks.append(
            {
                "chunk": sentence,  # What we embed and search
                "context": " ".join(sentences[start:end]),  # What we send to LLM
                "position": i,
            }
        )

    return chunks
```

Other chunking implementations split on **characters**, not tokens; do not apply a token-size recommendation directly to them. For fixed-size chunking set `0 <= overlap < chunk_size` and guard against a final-chunk loop that fails to advance; preserve structural metadata and evaluate with real query-answer pairs. Sentence windows require the tokenizer's sentence resources.

### 3.3 Chunk Size, Overlap & Metadata

Smaller chunks improve granularity but can omit the qualifying sentence; larger chunks preserve context but inject more noise. Starting *token* ranges are 200–400 for focused Q&A, 500–1000 for summarization, 300–500 for code, and 400–800 for legal clauses; these are heuristics to tune, not universal optima. For document length `D`, chunk size `S`, overlap `O`, step `S-O`, approximate count for `D > S` is `1 + ceil((D-S)/(S-O))`; the large-document storage multiplier tends toward `S/(S-O)`. Overlap improves boundary coverage but duplicates indexing, costs, and sometimes redundant evidence. Track `doc_id`, `chunk_id`, section, page/offset, revision/effective date, tenant, ACL, and embedding model/version.

An enriched chunk preserves source position and retrieval metadata:

```python
@dataclass
class EnrichedChunk:
    content: str                                  # The actual text

    # --- Structural metadata: where this chunk lives in the document ---
    source: str = ""                              # Document name / path
    section: str = ""                             # e.g. "Refund Policy"
    subsection: str = ""                          # e.g. "Enterprise Customers"
    position: int = 0                             # Chunk N of M

    # --- Semantic metadata: what this chunk is about ---
    summary: str = ""                             # LLM-generated summary
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    # --- Retrieval metadata: how we find and stamp it ---
    embedding: List[float] = field(default_factory=list)  # Vector embedding
    created_at: datetime = field(default_factory=datetime.now)

    def to_context(self) -> str:
        """Format for the LLM context window — note the source breadcrumb,
        which is what enables accurate citation."""
        return f"[Source: {self.source} > {self.section}]\n{self.content}"
```

The simple overlap calculation uses continuous arithmetic to illustrate cost; actual chunks require integer boundary handling. Evaluate boundary questions and deduplicate overlapping candidates before assembly.

### 3.4 Semantic, Keyword & Hybrid Search

Dense/vector retrieval can bridge vocabulary gaps (`terminate` ↔ `cancel`) but can miss exact IDs, negation, or rare strings. BM25/keyword retrieval catches error codes and exact terminology but may miss paraphrases. Hybrid candidate generation takes the **union** of both; combine normalized scores only after understanding their distributions, or use reciprocal rank fusion (RRF): `RRF(d) = sum_r 1 / (k + rank_r(d))`. Apply mandatory ACL/tenant filters **before** retrieval returns results; never rely on the prompt to hide unauthorized hits.

Reciprocal rank fusion combines independently ranked lists without normalizing scores:

```python
def reciprocal_rank_fusion(result_lists: List[List[str]], k: int = 60):
    scores = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Dense retrieval can use local SentenceTransformer embeddings and cosine similarity, while keyword search can use BM25. Prebuild the index once. For production, retain stable document IDs rather than using full chunk text as a deduplication key; attach filters and version stamps.

### 3.5 Reranking

First-stage retrieval optimizes fast recall; a cross-encoder jointly reads `(question, candidate)` and scores relevance more accurately, at higher cost. Typical flow: take 20–100 candidates (tune empirically), rerank authorized candidates, keep the few that fit the evidence budget. Measure recall of the correct passage *before* reranking and ranking quality *after*: reranking cannot recover a missing candidate. Guard for empty results, long documents that exceed reranker input limits, duplicates, and score interpretations (raw scores are not necessarily probabilities).

The reranker receives only the candidate texts from the first retrieval stage:

```python
candidates = self.retriever.search(query, top_k=initial_k)
candidate_texts = [c["chunk"] for c in candidates]
return self.reranker.rerank(query, candidate_texts, top_k=final_k)
```

The local cross-encoder scores each `(query, text)` pair together. The integrated RAG example later uses **heuristic scoring instead**, so do not mistake its scores for a real cross-encoder. If chunks repeat, keep IDs attached to candidates while scoring so citations remain unambiguous.

### 3.6 Context Compression

**Extractive** compression keeps source sentences; **abstractive** compression paraphrases/summarizes and can lose numbers, negation, or attribution. Prefer source-preserving extraction for policies and contracts; preserve surrounding context needed to interpret exceptions. Compress *after* recall and authorization, before prompt assembly; reserve enough context to verify that an extracted sentence is not misleading. Measure `evidence preservation` as well as token reduction.

This compressor counts shared keywords and then restores the original sentence order:

```python
def compress(self, query: str, text: str, keep_ratio: float = 0.5) -> str:
    """
    Extract most relevant sentences.

    Args:
        query: The search query
        text: Text to compress
        keep_ratio: Fraction of sentences to keep (0.5 = 50%)
    """
    # Split into sentences
    sentences = self._split_sentences(text)
    if len(sentences) <= 2:
        return text

    # Score each sentence by keyword overlap with the query.
    # (Stopwords/punctuation stripped so ranking reflects real signal.)
    query_words = keywords(query)
    scores = []
    for sent in sentences:
        overlap = len(query_words & keywords(sent))
        scores.append(overlap)

    # Keep top sentences (maintain original order)
    num_keep = max(1, int(len(sentences) * keep_ratio))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :num_keep
    ]
    top_indices.sort()  # Restore original order

    return " ".join([sentences[i] for i in top_indices])
```

The `LLMCompressor` demonstration is actually a **simulated keyword extractor**, despite its name. A production LLM compressor needs claim-to-source verification, a budget for its own call, and a fallback when it loses required facts.

### 3.7 Project: Engineered RAG

The integrated pipeline composes section-aware paragraph chunking, lexical first-stage retrieval called `HybridRetriever`, heuristic rerank, extractive context, a model call, and a naive-vs-engineered comparison. Its `overlap` constructor argument is not applied by its paragraph chunker; `chunk_size` there is **characters**, not tokens. Treat its comparison as a local demonstration, not proof of production accuracy.

Its engineered route reranks a larger candidate pool before compression:

```python
def query(self, question: str, use_engineering: bool = True) -> RAGResponse:
    """Answer a question using RAG."""
    start_time = time.time()

    # Retrieve
    if use_engineering:
        results = self.retriever.retrieve(
            question, initial_k=15, final_k=5, use_reranking=True
        )
        context = self.compressor.compress(question, results, target_chars=1200)
    else:
        # Naive RAG
        results = self.retriever.retrieve(
            question, initial_k=5, final_k=5, alpha=1.0, use_reranking=False
        )
        context = "\n\n".join([r.chunk.content for r in results])

    # Generate answer
    response = self.client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": f"""Answer based ONLY on the provided context.
If the answer isn't in the context, say "I don't have that information."
Cite your sources using [Source] format.

Context:
{context}

Question: {question}

Answer:""",
            }
        ],
    )

    latency = (time.time() - start_time) * 1000

    return RAGResponse(
        answer=response.content[0].text,
        sources=list(set(r.chunk.source for r in results)),
        context_chars=len(context),
        context_tokens_est=len(context) // 4,
        retrieval_count=len(results),
        engineered=use_engineering,
        latency_ms=latency,
    )
```

**Build exercise:** ingest a small versioned policy corpus; generate queries with synonyms, IDs, exceptions, and out-of-scope questions; replace heuristic scoring with embeddings, BM25 and a real cross-encoder; return stable source IDs and offsets; reject unauthorized/stale chunks; test retrieval recall, citation support, abstention, total cost and latency. Compare `top_k`, compression on/off, and chunk sizes with the same evaluation set. Success means *supported answers on answerable queries* and *appropriate refusal/abstention on unsupported ones*, not just fewer tokens.

---

## 4. Memory Systems

### 4.1 Why Memory Matters

A model call is not a reliable durable store of prior interactions. History must be sent again or reconstructed from authorized storage. Useful memory reduces repeated questions, preserves choices and preferences, and allows work to resume. Unmanaged memory can fossilize incorrect facts, mix users, over-personalize, expose sensitive data, or turn past malicious text into future instructions. Distinguish **conversation log** (event record), **working/session memory** (current task facts), **long-term memory** (selected durable facts), and **retrieval of memory** (choose what is relevant now). Model weights are not your application's writable user memory.

The persistent fact record separates content, category, importance and timestamp:

```python
@dataclass
class Memory:
    content: str
    category: str
    importance: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
```

For real use also record `tenant_id`, `user_id`, source turn, consent/purpose, confidence, last verification, expiry, and supersession/deletion status. Memory is a fact candidate until verified; do not store arbitrary assistant claims as user truth.

### 4.2 Conversation History

Four strategies are **sliding window** (simple, loses old constraints); **summarization** (keeps gist, can lose exact details); **selective pruning** (keeps salient turns, may mis-score); and **hybrid** (recent verbatim turns plus an older summary/important facts). Summarize only when the history needs it; keep the raw log separately so summaries can be audited or rebuilt. Update an existing summary incrementally while preserving decisions, unresolved tasks, exact figures, source IDs, and explicit user corrections. An offline fallback that truncates text is **not** genuine summarization.

The sliding-window implementation retains the most recent pairs of messages:

```python
class SlidingWindowHistory:
    """Keep only the last N turns."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        # Keep only last N turns (1 turn = user + assistant)
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_context(self) -> List[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def token_estimate(self) -> int:
        return sum(len(m.content) for m in self.messages) // 4
```

Sliding by pairs is convenient for demos; real conversations also have tool messages and partial turns. Maintain valid role/tool-call ordering for your provider. A summary should be labeled as historical evidence, not silently turned into a fresh user command.

### 4.3 Short-Term Memory

Session memory holds current goals, pending decisions, entities and constraints (e.g., “compare two laptops under $1,000”), often with a short TTL. Extract candidate facts from explicit user statements; mark provenance and uncertainty; update when corrected; evict transient state after task completion. Ask “what will the next turn need?” rather than “what words can I remember?” Working memory is distinct from raw dialogue and from persistent profile.

Session facts retain the turn they came from:

```python
@dataclass
class Fact:
    content: str
    category: str  # preference, constraint, context, state, identity
    confidence: float = 0.5
    extracted_at: datetime = field(default_factory=datetime.now)
    source_turn: int = 0
```

Pattern matching can over-extract and misclassify phrases. A production extractor should return typed candidates, validate against the original utterance, detect negation/corrections, and apply per-category retention rules. Do not promote a temporary shopping budget into permanent identity without a reason and appropriate controls.

### 4.4 Long-Term Memory

Long-term memory stores *selected* cross-session information: confirmed preferences, accessibility needs, ongoing projects, or explicitly saved facts. A lifecycle is `capture → validate → normalize → deduplicate/merge → store → retrieve → verify/update → expire/delete`. Use versioned records or an event log to handle “I used to prefer X; now I prefer Y.” Split facts by scope (user vs organization vs task), and provide mechanisms for correction and deletion. Storage must be authenticated and isolated by tenant/user; a local JSON file is a learning scaffold only.

The profile's deduplication rule illustrates a simple first step:

```python
def add_memory(self, memory: Memory):
    """Add memory if not duplicate."""
    existing = [m.content for m in self.memories]
    if memory.content not in existing:
        self.memories.append(memory)
```

Exact-string deduplication fails for paraphrases and conflicts; normalize facts by entity/attribute, keep source timestamps, and resolve contradictory statements explicitly. A local JSON profile store is a teaching scaffold; replace it with a scoped store and retention policy for real users.

### 4.5 Memory Retrieval

Recall is a relevance problem subject to authorization and budget. Merge (a) truly mandatory, confirmed facts; (b) query-relevant facts; (c) intent/category-relevant facts. Score with calibrated weights or learned retrieval over `relevance`, `importance`, `recency`, and sometimes access frequency. A common decay is `recency(age) = exp(-ln(2) * age / half_life)`: after the half-life, recency weight is 0.5. Do not let a heavily accessed but false memory become permanently dominant. Use a threshold and no-match option; do not fill the prompt with unrelated memories just to reach `top_k`.

The retrieval strategy uses exponential recency decay:

```python
HALF_LIFE_DAYS = 30
DECAY_RATE = math.log(2) / HALF_LIFE_DAYS  # ln(2)/30 ≈ 0.023

def recency_score(age_days: int) -> float:
    """Recency in [0, 1] from a memory's age, via exponential decay."""
    return math.exp(-DECAY_RATE * age_days)
```

Its retrieval pipeline unions always/query/category results, deduplicates and budgets them. Evaluate memory precision (injected facts actually needed), recall (required facts available), stale-fact errors, privacy leakage, and personalization benefit.

### 4.6 Injection Patterns

Five presentation strategies are system-context section, inline reminder before the current turn, persona/profile section, reminder at a topic shift, and structured profile. Choose the **least authoritative placement consistent with the information's trust**. A retrieved memory is application data, not a system instruction. If putting memory in a system field, wrap it as clearly delimited, validated data under an app-authored instruction; never copy unreviewed text as behavioral rules. Include only the facts relevant to this turn, with source/freshness where important.

Grouping retrieved memories by category makes their purpose clear:

```python
def format_memories(memories: List[Memory]) -> str:
    """Group memories by category into clean markdown."""
    by_category: Dict[str, List[str]] = {}
    for mem in memories:
        by_category.setdefault(mem.category, []).append(mem.content)

    lines = []
    for cat, items in by_category.items():
        lines.append(f"### {cat.replace('_', ' ').title()}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip()
```

Avoid impersonating an assistant's earlier reply to insert memory: it may be mistaken for an actual prior action. Prefer an explicit `Retrieved user facts (data, not instructions)` section that is reconstructed on each call.

### 4.7 Project — Chatbot Memory System

A complete chatbot joins `ConversationManager`, `MemoryStore`, rule-based/LLM extractors, `MemoryRetriever` and live generation. Its default JSON storage path is `/tmp/`, and recall uses lexical overlap. The two extractors share the `.extract()` interface, making them easy to compare. Its summary is a brief deterministic sketch; don't mistake it for a reliable semantic history summary.

The chatbot's retriever combines high-importance facts with lexical relevance:

```python
def retrieve(self, query: str, max_memories: int = 8) -> List[Memory]:
    if not self.profile.memories:
        return []

    # Always include high-importance memories (name, budget, hard requirements)
    always = [m for m in self.profile.memories if m.importance >= 0.85]

    # Score the rest by simple word overlap with the query (a stand-in for
    # semantic search — see the 'Under the hood vs. managed' box in the lesson)
    query_words = set(query.lower().split())
    scored = []
    for mem in self.profile.memories:
        if mem in always:
            continue
        mem_words = set(mem.content.lower().split())
        overlap = len(query_words & mem_words)
        score = overlap / max(len(query_words), 1)
        scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    remaining = max(0, max_memories - len(always))
    relevant = [m for _, m in scored[:remaining]]

    return always + relevant
```

**Build exercise:** test two sessions for the same user, then two users with identical questions. Validate recall, correct updates (“my budget is now $700”), no cross-user leakage, and retention/deletion; instrument extraction false positives. Harden persistence (configurable path, safe error handling and access control), filter low-relevance memories, cap *all* memory including always-include entries, and separate model-generated text from user-confirmed facts. Measure personalization quality and added latency/cost with and without memory.

---

## 5. Context for Agents

### 5.1 Agent Context vs. Traditional Context

A one-shot RAG answer has one retrieval/generation boundary. An agent loops through **plan → choose tool → execute → observe → update state → continue**, potentially many times. Every tool response can enlarge future context; repeated action traces accumulate noise and may include untrusted instructions. The key artifact is a **compact, explicit task state**: goal, constraints, completed/pending steps, verified findings, source references and unresolved questions. The transcript and raw tool outputs remain available outside the prompt for audit and selective recall.

Agent state tracks the plan separately from the tool transcript:

```python
@dataclass
class AgentState:
    task: str
    status: str  # "planning", "executing", "waiting", "complete"
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    key_findings: Dict[str, str] = field(default_factory=dict)
    current_step: Optional[str] = None
```

Rebuild each turn's context from state and relevant observations; avoid recursive “summary of a summary of a summary” without verification against source artifacts.

### 5.2 Managing Tool Context

Tool descriptions, schema fields, arguments, outputs and errors all consume attention and sometimes tokens. Select only authorized, task-relevant tools; allowlist by agent role and user permission **outside** the model. Return structured, paginated, bounded results. Store full raw output in a referenced artifact and inject only selected fields and a citation/handle. Errors should preserve actionable cause and retryability, not stack-trace dumps or secrets. “Tool discovery” does not make an untrusted tool trustworthy.

Tool selection loads only categories needed for the current task:

```python
def get_tools_for_task(self, task: str) -> List[Tool]:
    """Get relevant tools based on task analysis."""
    needed_categories = self._analyze_task(task)
    tools = []
    for category in needed_categories:
        tools.extend(self.tools_by_category.get(category, []))
    return tools
```

This category selector is keyword-based and compression is rule-based unless given a client. Validate actual tool permissions separately; supply API-native schemas where supported. Count schemas when computing input budget.

### 5.3 Chaining Tool Results

An agent may search → read → extract → calculate → write. Each step should hand the next step a structured **delta** (`result_id`, key facts, provenance, uncertainty, what remains) rather than the concatenation of all earlier raw results. Keep large tables/documents in a store and pass typed references. If intermediate facts are critical, verify them against raw results before irreversible actions. Establish an explicit maximum carried-forward budget and a way to re-fetch details on demand.

Each chain step stores raw output and a bounded handoff representation:

```python
@dataclass
class ChainStep:
    step_id: str
    tool_name: str
    arguments: dict
    raw_result: str
    compressed_result: str
    key_data: dict
    tokens_original: int
    tokens_compressed: int
```

This implementation keeps raw and compressed results in process memory and appends each compressed step into `get_chain_context()`. For long-running agents, even compressed steps need summarization, eviction or indexed retrieval; persisted artifacts need access control and TTL.

### 5.4 Multi-Turn Agent Memory

Working state answers: “What is the task? What has been tried? Which outcomes were verified? What must happen next?” Persist checkpoints at meaningful transitions for durability; maintain idempotency keys around side-effecting tools. Summaries should preserve exact IDs and status transitions rather than replacing facts with vague prose. In longer runs, use a sliding recent-action window plus durable state and indexed findings. Bound each turn independently, rather than assuming early turns are cheap enough for the whole task.

Task initialization records a plan before execution begins:

```python
def set_task(self, task: str, steps: Optional[List[str]] = None):
    """Initialize memory with task."""
    self.state.task = task
    self.state.status = "planning"
    self.state.pending_steps = steps or []
```

Record transitions as events so a resumed agent can distinguish “planned” from “tool actually succeeded.”

### 5.5 Sharing Context Across Multi-Agent Systems

A handoff should include recipient, subtask, permitted context, verified findings, source handles, unresolved questions, and output contract. Do not hand an entire unrestricted transcript to every agent. Use a shared store with scoped read/write and per-record provenance; promote agent proposals to trusted shared facts only after validation. Explicitly define whether agents may overwrite a result and how version conflicts are resolved.

The handoff carries a recipient-specific task and compressed facts:

```python
@dataclass
class AgentHandoff:
    """Context handoff between agents."""

    from_agent: str
    to_agent: str
    task: str
    context: str   # Compressed context
    key_data: dict  # Structured data
    confidence: float = 0.8
```

The in-memory `SharedMemoryStore` and visibility flags are illustrative. Enforce scope on both writes and reads, and prevent a fallback path from returning an entire findings dictionary when a recipient lacks permission.

### 5.6 Context Isolation

Isolation has two dimensions: **information** (which tenant/user/agent may see a fact) and **capability** (which tool/action may be executed). Implement both in the application and storage layer, not just by telling the model “do not reveal.” Treat webpages, retrieved documents and tool outputs as untrusted data; delimit them and validate structured fields. Regex “prompt-injection removal” catches obvious examples but is bypassable and can damage legitimate text; trust boundaries, authorization, minimal tool scope, and output verification are the core controls.

A sandbox specification defines the agent's accessible tools and memory scopes:

```python
@dataclass
class ContextSandbox:
    """Defines boundaries for an agent's context."""

    agent_id: str
    allowed_tools: Set[str]
    max_context_tokens: int
    can_access_persistent_memory: bool
    can_write_shared_memory: bool
    isolation_level: str
```

`SandboxedAgent` and `ContextFirewall` model the idea, but an in-process filter is not an OS sandbox or security boundary. Verify with tests where a retrieved passage contains instructions to exfiltrate a memory or call a disallowed tool.

### 5.7 Project: Agent Context System

An integrated agent combines dynamic tool selection, result compression, task state and isolation around a live model. It demonstrates wiring, not a complete durable/secure production agent: results may be truncated with first/last-character heuristics, and its injection-pattern removal is not a defense by itself.

The agent reconstructs a compact per-turn prompt from state and recent actions:

```python
def get_context(self) -> str:
    """Build context for LLM."""
    sections = []

    # Task
    sections.append(f"## Task\n{self.state.task}")

    # Status
    sections.append(f"## Status\n{self.state.status}")

    # Summaries of old turns
    if self.summaries:
        sections.append(f"## Earlier\n{'; '.join(self.summaries[-2:])}")

    # Recent turns
    if self.turns:
        recent = "\n".join(
            [f"- Turn {t.turn_number}: {t.summary}" for t in self.turns[-5:]]
        )
        sections.append(f"## Recent Actions\n{recent}")

    # Findings
    if self.state.findings:
        findings = "\n".join(
            [f"- {k}: {v}" for k, v in self.state.findings.items()]
        )
        sections.append(f"## Key Findings\n{findings}")

    return "\n\n".join(sections)
```

**Build exercise:** define a research task with search/read/analyze/write tools; compare naive transcript carry-forward with structured state; save raw results separately with IDs; allowlist capabilities; simulate a large irrelevant webpage and an adversarial snippet; add checkpoint/resume and audit logs. Evaluate completion, correct tool choice, state consistency after restart, tool-result tokens per turn, blocked cross-scope access, and unsupported final claims.

---

## 6. Production Patterns

### 6.1 Context Caching

Cache *deterministic and properly scoped work* before caching final answers: parsed chunks, embeddings keyed by `(text, model/version, preprocessing version)`, retrieved candidate IDs keyed by `(query, corpus revision, ACL scope, filters)`, and then eligible responses keyed by the complete request (messages, model/version, parameters, tenant/user permission context and evidence revision). Set TTL by freshness requirement; invalidate on policy edits, revocation, deletions, embedding upgrades and prompt-version changes. A response cache may be inappropriate for sensitive, personalized, or rapidly changing answers. Provider-side prompt-prefix caching is a distinct optimization: keep stable prefix content stable and verify provider-specific billing semantics.

A response-cache key hashes the request's system text and messages:

```python
def _make_key(self, messages: list, system: str = "") -> str:
    content = json.dumps({"system": system, "messages": messages}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:32]
```

This response key excludes tenant, model, parameters and document revision, so use it only as an illustration. `EmbeddingCache` normalizes and hashes text, but embeddings change when the embedding model or preprocessing changes. Track hit rate, staleness, wrong-scope cache hits, latency saved and invalidation frequency.

### 6.2 Error Handling

Classify failures by **stage** (identity, retrieval, memory, assembly, model, validation, persistence) and **response** (retryable, permanent, degraded, needs human). Authentication or permission errors should fail closed; lack of evidence should not be disguised as a confident answer. Timeouts and transient provider errors may be retried; malformed context or over-budget prompts need repair, not identical retries. Return a clear status internally, an honest user-visible explanation, and trace IDs; redact secrets from logs.

Errors are classified by whether to retry, stop or degrade:

```python
class ErrorSeverity(Enum):
    TRANSIENT = "transient"  # Retry
    PERMANENT = "permanent"  # Stop
    PARTIAL = "partial"  # Degrade
```

This implementation classifies by substrings in error messages and may return a generic fallback passage. In production use typed errors/status codes; a generic fallback cannot supply missing evidence. Instrument both the original failure and any fallback failure.

### 6.3 Rate Limiting

Enforce request-per-minute, token-per-minute and concurrency caps at appropriate scopes: provider account, model, tenant, user and operation. A token bucket permits small bursts up to capacity; a sliding window offers a tighter requests/window constraint. Estimate tokens before reserving capacity, reconcile with actual usage, and queue or reject with retry timing when full. Coordinate state across application instances: a process-local lock does not enforce a distributed quota. Separate rate limits from daily spend limits.

The token bucket refills before atomically granting a reservation:

```python
def acquire(self, tokens: int = 1) -> bool:
    """Try to acquire tokens. Returns True if successful."""
    with self.lock:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

The `MultiTierRateLimiter` illustrates multiple levels; validate atomic reservation across tiers in a shared backend when deployed across workers. Rate-limited retries must not multiply the load spike.

### 6.4 Cost Management

Budget by user, tenant, operation and time period. Include generator input/output, embedding, rerank, judge, memory-extraction, compression, caching, and failed/retried calls. Compute per-request estimated cost *before* issuing calls and reconcile provider usage *after*. Alert on unusual token growth, repeated retries, cache misses and cost per accepted answer. Route simpler tasks to an appropriate lower-cost model only when evaluation shows quality remains adequate; example prices are illustrative.

The tracker calculates usage against configured per-model rates:

```python
def calculate_cost(
    self, model: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost for a request."""
    pricing = self.PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

The fallback pricing above is a demo convenience, not a safe billing assumption. In production, reject unknown price configurations or tag costs as unknown, distinguish cached and uncached rates, and enforce budgets before optional work.

### 6.5 Graceful Degradation

Design ordered fallbacks **per feature** and define what quality is lost. Example: reranker unavailable → authorized first-stage retrieval with a stricter evidence threshold; vector store unavailable → authorized keyword index; both unavailable → say evidence cannot be verified rather than inventing a policy. A lower-tier model may still handle a simple response, but must not silently bypass required source or privacy checks. Display internal degradation state in metrics and (where material) tell the user that the answer is limited.

Degradation levels make a fallback's reduced capabilities visible:

```python
class DegradationLevel(Enum):
    FULL = 1  # Everything works
    REDUCED = 2  # Some features limited
    FALLBACK = 3  # Using backup systems
    MINIMAL = 4  # Basic functionality only
    FAILED = 5  # Nothing works
```

The retrieval fallbacks in this example generate illustrative strings, not sourced document results. Define acceptance gates at every level: source provenance, tenant scope, freshness, and evidence sufficiency. Test fallback-to-fallback failures and recovery back to full service.

### 6.6 Retry Patterns

Retry only idempotent or safely keyed operations after **transient** failures. Use exponential backoff with jitter and a maximum elapsed time; honor server `Retry-After` when supplied. A circuit breaker opens after sustained failures, probes recovery in half-open mode, and prevents an unhealthy dependency from consuming all resources. Avoid retrying 400/permission errors or a rejected tool action without an idempotency key. Measure attempt count and accumulated cost.

The backoff implementation computes a bounded, jittered delay:

```python
def _calculate_delay(self, attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = self.base_delay * (self.exponential_base**attempt)
    delay = min(delay, self.max_delay)

    if self.jitter:
        delay = delay * (1 + random.random() * 0.5)

    return delay
```

`RobustRetry` detects retryability via strings such as `"timeout"` and `"429"`; substitute provider-specific typed exceptions/status and clear retry budgets. A fallback cannot make an unauthorized action safe.

### 6.7 Production composition

An integrated system wires the cache, rate limiter, cost manager, degrader and retry utilities together. An effective request flow is **authenticate → check scope/quota/budget → fetch authorized evidence → assemble with an exact budget → call model under bounded retry → verify/output → reconcile spend/cache/audit**. Treat this as an in-process integration exercise; it does not by itself supply distributed quotas or durable encrypted state.

Its context assembler ranks knowledge snippets by query overlap and packs them to budget:

```python
def assemble(self, query: str, token_budget: int, max_snippets: int = 4) -> str:
    """Rank snippets by word overlap with the query, then pack to budget."""
    q_words = set(query.lower().replace("?", "").split())
    ranked = sorted(
        self.KNOWLEDGE,
        key=lambda s: len(q_words & set(s.lower().split())),
        reverse=True,
    )

    chosen, used = [], 0
    for snippet in ranked[:max_snippets]:
        cost = self.count_tokens(snippet)
        if used + cost > token_budget:
            break
        chosen.append(snippet)
        used += cost

    return "\n".join(chosen)
```

Inject failures (retrieval timeout, 429, cache staleness, corrupted memory), observe fallback paths, and assert that every returned factual claim still has an authorized source. The example uses a simulated generator, not a real provider response.

---

## 7. Enterprise Context Patterns

### 7.1 Context Engineering Standards

Standardize component contracts, metadata, source IDs, versions, token budgets, security scope, logging, citation conventions and evaluation gates. An organization-wide `ContextResult` can include content, source IDs, retrieval method, timestamp, freshness/version and an authorization decision. Keep standards executable: schema validation, lint rules, traces and regression suites. Name owners for every template, index and memory store; require an explicit path for deprecating stale context.

A standard retrieval result keeps its content, sources, version and timestamp together:

```python
@dataclass
class ContextResult:
    """Standard result format for context retrieval."""

    content: str
    sources: List[str]
    metadata: Dict[str, Any]
    version: str
    timestamp: str
    quality_score: Optional[float] = None
```

`ContextProvider` defines `retrieve`, `validate`, and `get_metadata` operations; explicit sources and testability are core design principles. Extend the interface with tenant/ACL scope and structured failure reasons rather than passing unrestricted arbitrary dictionaries.

### 7.2 Multi-Tenancy Patterns

Partition context by tenant *at ingestion, indexing, retrieval, memory, caches, logs and deletion*. For shared storage, every query must enforce tenant and document ACL filters server-side; pooled or dedicated stores can provide stronger isolation. Tie rate/spend quotas, encryption keys and retention to the tenant. IDs in a prompt are **not** an access-control mechanism. Verify negative tests: user A cannot infer tenant B's documents through search, cache, shared memory or citation IDs.

A shared store places each tenant in a separate logical namespace:

```python
class SharedContextStore(TenantContextStore):
    """Shared storage with logical tenant isolation."""

    def __init__(self):
        self.store_data: Dict[str, Dict[str, Any]] = {}

    def store(self, tenant_id: str, key: str, context: Any) -> None:
        if tenant_id not in self.store_data:
            self.store_data[tenant_id] = {}
        self.store_data[tenant_id][key] = context

    def retrieve(self, tenant_id: str, key: str) -> Optional[Any]:
        return self.store_data.get(tenant_id, {}).get(key)

    def delete(self, tenant_id: str, key: str) -> None:
        if tenant_id in self.store_data:
            self.store_data[tenant_id].pop(key, None)

    def list_keys(self, tenant_id: str) -> List[str]:
        return list(self.store_data.get(tenant_id, {}).keys())
```

The teaching implementation uses nested dictionaries and demonstrates logical isolation, not durable enforcement. Authenticate `tenant_id` from the server-side identity, apply ACLs before candidate selection and reranking, and include scope in every cache key.

### 7.3 Compliance & Security

Build a data map: which fields enter prompts, which provider receives them, where logs/cache/memories are stored, how long they remain, and who can delete them. Apply least privilege, purpose limitation, retention and encryption according to your jurisdiction and contract. Minimize sensitive content before model calls; redact or tokenize only when the task still works, and keep reversible mappings in a restricted store. Audit **who accessed which source and why** without logging raw secrets by default. Treat retrieved instructions as data; constrain tool execution and validate final disclosures.

PII detection distinguishes several categories before redaction:

```python
class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
```

Regex detection cannot find every sensitive entity and may misidentify harmless patterns; test with realistic data. `PIIRedactor`, access policy and audit components are not a certification of compliance. Cover deletion in derived embeddings, summaries, caches, logs and backups to the extent required by your data policy.

### 7.4 Context Versioning

Version prompts, templates, schemas, retrieval configuration, embedding model, indexes, source documents and memory record schema together. Record an **assembly manifest** for each call: `prompt_version`, `retriever_version`, `index_revision`, `model_id`, `memory_schema`, `evidence_ids`, `tenant_scope`, `budget`. A content-hash or immutable document revision helps reconstruct which policy the model saw. Roll out changes using offline evals then canaries; make rollback and migration explicit. Semantic versioning is useful for compatibility contracts but not a proof of behavioral compatibility.

Version increments distinguish breaking, additive and corrective changes:

```python
def bump_major(self) -> "Version":
    return Version(self.major + 1, 0, 0)

def bump_minor(self) -> "Version":
    return Version(self.major, self.minor + 1, 0)

def bump_patch(self) -> "Version":
    return Version(self.major, self.minor, self.patch + 1)
```

An in-memory `ArtifactStore` can illustrate versioned items; production manifests should be persistent and use correct semantic-version ordering, especially when release candidates exist. Reindex and invalidate caches on incompatible chunking/embedding changes.

### 7.5 Reusable Context Libraries

Define components with typed inputs/outputs, failure modes, scopes, versions, provenance and an execution budget: `Retriever → ACL filter → Reranker → Compressor → Assembler → Validator`. A registry makes cross-team reuse easier, but avoid a component that silently changes tenant scope or strips attribution. Compare library defaults to task-specific evaluation; publish clear migration instructions when a component changes behavior.

Reusable components declare versioned input and output schemas:

```python
@dataclass
class ComponentMetadata:
    """Metadata for registered components."""

    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    author: str = ""
    tags: List[str] = field(default_factory=list)
```

A registry and pipeline builder can compose retriever/ranker/filter components, though basic implementations often use lexical heuristics. Add schema validation and an explicit `AuthContext` passed through every stage.

### 7.6 Testing & Validation

Use a pyramid: **unit** tests for budget accounting, ACL filters, deduplication, schema validation and citation IDs; **retrieval/quality** tests for recall, coverage, source freshness and token/latency thresholds; **integration** tests for model calls, fallback paths and store permissions; **regression** tests for known failures across prompt/index versions. Include adversarial content, stale memories, conflicting policy revisions, missing sources, Unicode, huge tool output, and concurrency. Quality thresholds must be measured on representative data; the example numbers are placeholders.

The quality gate expresses explicit metric thresholds:

```python
@dataclass
class QualityThresholds:
    min_relevance: float = 0.7
    min_coverage: float = 0.8
    max_tokens: int = 50000
    max_latency_ms: float = 1000
```

Require an evaluator to supply real labels for `relevance` and `coverage`; a passed threshold is meaningless if those fields are guessed. In CI, gate on security/correctness invariants and compare quality with prior baselines; in production, sample traces for human review and measure drift by tenant and source revision.

### 7.7 Enterprise composition

An offline integration joins tenant config, regex PII redaction, role checks, audit events, version storage and quality metrics. Its in-memory audit list is not immutable, its simplified version lookup can order versions lexically, and its policies are demonstrations. Use it to map interfaces and transitions, then replace in-memory implementations with properly enforced services.

The platform constructs its enterprise components together:

```python
class EnterpriseContextPlatform:
    """Complete enterprise context platform."""

    def __init__(self):
        # Core components
        self.tenants: Dict[str, Tenant] = {}
        self.pii_handler = PIIHandler()
        self.audit_logger = AuditLogger()
        self.access_controller = AccessController()
        self.version_manager = VersionManager()
        self.quality_tracker = QualityTracker()

        # Usage tracking
        self.usage: Dict[str, Dict] = {}
```

**Acceptance exercise:** create two tenants, two document revisions, one deleted memory, a stale cache key and a hostile retrieved passage. Assert no cross-tenant retrieval, correct policy version and citation, deletion across derived stores, rejection of untrusted tool instructions, and a trace containing only the minimal audit metadata.

---

## 8. End-to-end architecture and study plan

The chapters fit together as a *context compiler*: it turns a user/task and authorized data sources into a bounded request and returns a validated answer plus trace. A reference sequence:

```text
Incoming request
  → authenticate identity, tenant, permissions, purpose
  → classify task and load versioned template/tool allowlist
  → retrieve authorized, current documents and relevant memories
  → rerank/deduplicate; compress with source spans preserved
  → reserve output budget; assemble role-separated, typed context
  → model/tool loop with bounded retries and explicit agent state
  → validate evidence, citations, schema, disclosure and uncertainty
  → update memory only from eligible confirmed facts
  → log redacted manifest/usage; reconcile cost; cache if eligible
```

The capstone request path checks limits and permissions, retrieves context, assembles a prompt, and records usage. Its generation is simulated:

```python
def research(self, request: ResearchRequest) -> ResearchResponse:
    start = datetime.now()
    self.metrics.increment("requests.total")

    # Check limits
    if not self.tenants.check_limits(request.tenant_id, 10000)["allowed"]:
        return ResearchResponse(request.request_id, False, error="Limit exceeded")

    # Check permission
    if not self.access.check_permission(request.user_id, Permission.READ):
        self.audit.log(
            request.tenant_id, request.user_id, "research", "denied:permission"
        )
        return ResearchResponse(
            request.request_id, False, error="Permission denied"
        )

    # Redact PII
    clean_query = self.pii.redact(request.query)

    # Retrieve context
    ctx_result = self.context_pipeline.retrieve(clean_query)
    self.metrics.record("context.quality", ctx_result.quality_score)

    # Assemble prompt
    prompt = self.prompt_manager.assemble(
        "research", {"context": ctx_result.context, "query": clean_query}
    )

    # Generate response (simulated)
    response_text = f"Based on the context about '{clean_query[:30]}...': "
    response_text += "Context engineering is a systematic approach to designing optimal context for LLMs."
    tokens_used = prompt.token_estimate + len(response_text.split()) * 2

    # Record
    latency = (datetime.now() - start).total_seconds() * 1000
    self.tenants.record_usage(request.tenant_id, tokens_used)
    self.metrics.record("request.latency", latency)
    self.metrics.record("tokens.used", tokens_used)

    self.audit.log(
        request.tenant_id,
        request.user_id,
        "research",
        "success",
        {"tokens": tokens_used},
    )

    return ResearchResponse(
        request_id=request.request_id,
        success=True,
        response=response_text,
        sources=ctx_result.sources,
        tokens_used=tokens_used,
        latency_ms=latency,
    )
```

Study the offline concepts first: context types, chunking, memory retrieval, tool-result management, caching, tenant isolation and validation. Next, experiment with real local embeddings and cross-encoder reranking. Finally, build the projects that use a live model provider **with an API key and expected spend**: context audit, assembly pipeline, engineered RAG, memory-enabled chatbot and multi-turn agent. Compare what each demonstration *actually does* with what its design intends. Integrate caching, resilience, enterprise boundaries and evaluation in the final capstone.

### Implementation checklist

- [ ] Define task, answerability, evidence and citation requirements; build a labeled evaluation set.
- [ ] Authenticate and scope user, tenant, tool permissions and document access before retrieval.
- [ ] Ingest versioned documents with source IDs, ACLs, dates, sections and stable offsets.
- [ ] Measure first-stage recall; add hybrid retrieval, reranking and evidence-preserving compression as needed.
- [ ] Partition stable instructions, dynamic evidence, relevant memory, tool schemas and output contract.
- [ ] Reserve generated output and count serialized input with the provider's actual tokenizer/endpoint.
- [ ] Fail honestly on missing, conflicting, unauthorized or over-budget evidence.
- [ ] Keep agent state and raw artifacts outside the prompt; inject bounded findings per turn.
- [ ] Govern memory capture, verification, update, TTL, consent/purpose and deletion.
- [ ] Scope caches and quotas by identity/revision; verify fallbacks still satisfy source and permission checks.
- [ ] Log a redacted assembly manifest, citations, exact usage, latency, cost and version IDs.
- [ ] Regression-test correctness, abstention, position sensitivity, adversarial input and cross-scope leakage.

### Glossary

| Term | Working definition |
|---|---|
| Context assembly | Selecting, ordering and formatting information for a specific call |
| Context budget | Tokens allotted to required and optional input plus reserved output |
| Grounding | Ensuring factual claims are supported by the evidence supplied |
| Provenance | Source identity, revision and span for a fact or passage |
| Recall@k | Share of queries whose needed evidence appears among first `k` candidates |
| Reranking | More accurate scoring of a smaller first-stage candidate set |
| Compaction | Replacing long history/tool output with a shorter representation |
| Working memory | State useful for an active session or task |
| Long-term memory | Selected facts carried across sessions under a retention policy |
| Trust boundary | Separation between app-authored instructions and external/user/tool data |
| Assembly manifest | Versioned record of which components and evidence were used in a call |
