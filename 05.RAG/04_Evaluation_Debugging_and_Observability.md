# Evaluation, Debugging, and Observability

RAG systems need measurement. Without evaluation and tracing, every improvement is a guess.

## What to Evaluate

Evaluate each stage separately:

- ingestion quality,
- retrieval quality,
- reranking quality,
- context packing quality,
- answer faithfulness,
- answer usefulness,
- citation accuracy,
- latency,
- cost,
- security behavior.

End-to-end answer quality matters, but it does not tell you which component failed.

## Retrieval Metrics

| Metric | Measures |
| --- | --- |
| Recall@k | Whether relevant evidence appears in top-k. |
| Precision@k | How much of top-k is relevant. |
| MRR | How high the first relevant result appears. |
| nDCG | Whether more relevant results rank higher. |
| Hit rate | Whether at least one relevant result appears. |
| Empty result rate | How often retrieval returns nothing useful. |

Retrieval evaluation requires a dataset of queries and expected relevant sources or chunks.

## Generation Metrics

| Metric | Measures |
| --- | --- |
| Faithfulness / groundedness | Whether answer claims are supported by context. |
| Answer relevance | Whether the answer addresses the question. |
| Correctness | Whether the answer matches reference answer or facts. |
| Citation accuracy | Whether cited sources support the cited claims. |
| Refusal correctness | Whether the system says "not found" when needed. |
| Helpfulness | Whether the answer is usable for the target user. |

Ragas lists RAG metrics including context precision, context recall, response relevancy, faithfulness, multimodal faithfulness, and multimodal relevance. Source: <https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/>.

## Build a RAG Test Set

Start small but representative.

Include:

- 10 common questions,
- 10 edge cases,
- 10 queries requiring exact terms,
- 10 queries requiring semantic matching,
- 5 questions with no answer in the corpus,
- 5 permission-sensitive questions,
- 5 multi-hop questions if your app claims to support them.

For each example, record:

- user question,
- expected answer,
- relevant source IDs,
- forbidden source IDs if permissions matter,
- tags such as `exact_match`, `policy`, `multi_hop`, `no_answer`,
- notes explaining what failure would look like.

## Debugging Bad Answers

Use this sequence:

1. Check if the correct source exists in the corpus.
2. Check if parsing extracted the relevant text.
3. Check if chunking preserved enough context.
4. Check if the chunk was embedded and indexed.
5. Run retrieval without filters.
6. Run retrieval with filters.
7. Compare dense-only, sparse-only, and hybrid retrieval.
8. Inspect reranker input and output.
9. Inspect final context sent to the LLM.
10. Inspect the prompt and final answer.

This avoids blaming the LLM for an indexing or retrieval problem.

## Observability

A trace should show:

- user input,
- rewritten query,
- extracted filters,
- retrieved chunks,
- retrieval scores,
- reranker scores,
- context sent to the model,
- model input/output,
- citations,
- latency per step,
- token usage,
- errors and retries.

LangSmith structures observability around projects, traces, runs, and threads. A trace records one operation, runs represent individual units of work inside that operation, and threads link traces across multi-turn conversations. Source: <https://docs.langchain.com/langsmith/observability-concepts>.

## Three Pillars of Production Visibility

### Quality

Track:

- answer correctness,
- groundedness,
- retrieval recall,
- citation accuracy,
- user feedback,
- no-answer behavior.

### Performance

Track:

- total latency,
- retrieval latency,
- embedding latency,
- reranking latency,
- LLM latency,
- timeout rate,
- retry count.

### Cost

Track:

- embedding tokens,
- prompt tokens,
- completion tokens,
- vector DB read/write costs,
- reranker cost,
- trace/log volume,
- cost by route, tenant, and feature.

## Offline vs Online Evaluation

Offline evaluation:

- runs before deployment,
- uses curated datasets,
- compares versions,
- helps tune chunking, retrieval, prompts, and models.

Online evaluation:

- runs on production traces,
- detects drift and regressions,
- uses user feedback, heuristics, and LLM judges,
- supports monitoring and incident response.

LangSmith's evaluation docs distinguish offline evaluations on datasets from online evaluations on production runs and threads. Source: <https://docs.langchain.com/langsmith/evaluation-concepts>.

## Human Review

Human evaluation is still important for:

- domain correctness,
- legal/policy interpretation,
- tone and usefulness,
- citation trust,
- ambiguous questions,
- failure analysis.

Use human labels to seed automated evaluations. Do not rely only on LLM-as-judge metrics for high-stakes domains.

## Regression Testing

Every RAG change should run against a stable test set:

- embedding model change,
- chunk size change,
- parser change,
- prompt change,
- vector DB change,
- hybrid weighting change,
- reranker change,
- security filter change.

Track whether the new version improves target metrics without breaking critical examples.

## Study Checklist

- I can define separate metrics for retrieval and generation.
- I can build a small labeled RAG test set.
- I know how to inspect a trace from query rewrite through final answer.
- I can separate retrieval failure from generation failure.
- I understand why online monitoring and offline evaluation solve different problems.

