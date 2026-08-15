# LangSmith Observability, Evaluation, and Debugging

LangSmith is the observability and evaluation layer of the LangChain ecosystem. It is useful for LangChain apps, LangGraph apps, agents, RAG systems, and even non-LangChain LLM applications when instrumented.

## Why Observability Matters

LLM applications fail through many interacting parts:

- prompt,
- model choice,
- retrieved context,
- tool call arguments,
- tool result,
- parser,
- memory,
- state transition,
- user input,
- latency or timeout.

Without traces, you only see the final bad answer. With traces, you can see which step caused it.

## LangSmith Data Model

LangSmith observability docs define:

- Project: container for traces for an application/service.
- Trace: one operation, such as a user request.
- Run: one unit of work inside a trace, such as model call, retrieval call, or parser step.
- Thread: linked traces across a multi-turn conversation.

Source: <https://docs.langchain.com/langsmith/observability-concepts>.

## What to Trace

Trace:

- model calls,
- tool calls,
- retrieval calls,
- prompt inputs,
- prompt outputs,
- parser failures,
- graph state transitions,
- latency,
- token usage,
- cost metadata,
- user/session IDs after redaction.

Be careful with sensitive data. Use redaction and retention rules appropriate for your domain.

## Debugging with Traces

For a RAG failure, inspect:

1. User question.
2. Rewritten query.
3. Metadata filters.
4. Retrieved chunks.
5. Scores/ranks.
6. Reranker output.
7. Final context.
8. Model prompt.
9. Final answer.
10. Citations.

For an agent failure, inspect:

1. System prompt.
2. Tool list.
3. Model tool decision.
4. Tool arguments.
5. Tool result.
6. Next model decision.
7. Stop condition.
8. Error/retry path.

## Evaluation Concepts

LangSmith evaluations use:

- datasets,
- examples,
- target functions,
- evaluators,
- experiments,
- feedback.

LangSmith evaluation docs describe offline evaluations on datasets and online evaluations on production runs/threads. Source: <https://docs.langchain.com/langsmith/evaluation-concepts>.

## Datasets

A dataset is a collection of test examples.

Example fields:

```json
{
  "inputs": {
    "question": "What is the expense submission deadline?"
  },
  "reference_outputs": {
    "answer": "30 days after the transaction date."
  },
  "metadata": {
    "category": "policy",
    "expected_source": "travel_policy_2026"
  }
}
```

For RAG, include expected source IDs when possible. For agents, include expected tool calls or outcome constraints.

## Evaluators

Evaluator types:

- code evaluators: deterministic checks,
- LLM-as-judge evaluators,
- human feedback,
- pairwise comparison,
- composite evaluators.

Examples:

- exact match,
- schema validity,
- contains citation,
- retrieved expected source,
- answer grounded in context,
- tool call accuracy,
- toxicity/safety check,
- latency under threshold.

## Experiments

An experiment is a run of a specific application version against a dataset.

Use experiments to compare:

- prompt versions,
- model versions,
- chunk sizes,
- retrievers,
- vector databases,
- rerankers,
- agent policies.

Do not trust one-off manual tests. Keep a dataset and run experiments before important changes.

## Online Monitoring

Online evaluation watches production behavior.

Track:

- error rate,
- latency,
- token usage,
- cost,
- user feedback,
- refusal rate,
- empty retrieval rate,
- low groundedness,
- tool failure rate.

Online metrics should feed your next offline test cases. When users report failures, turn those traces into dataset examples.

## Study Checklist

- I can explain projects, traces, runs, and threads.
- I know what to inspect for RAG and agent failures.
- I can create a dataset for offline evaluation.
- I can choose code vs LLM-as-judge evaluators.
- I understand how production traces become future regression tests.

