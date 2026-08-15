# Production Patterns and Ecosystem Tradeoffs

LangChain is useful, but production quality depends on how you structure the application around it. Frameworks do not replace system design.

## Recommended Project Structure

```text
app/
  api/
    routes.py
    schemas.py
  llm/
    models.py
    prompts.py
    structured_outputs.py
  rag/
    ingestion.py
    retrievers.py
    chunking.py
    context.py
  agents/
    tools.py
    graphs.py
    policies.py
  evals/
    datasets/
    evaluators.py
  observability/
    tracing.py
  config.py
```

Keep prompts, tools, retrieval, API schemas, and evaluation code separate enough to test and change independently.

## Dependency Management

LangChain uses core packages plus provider/integration packages.

General practice:

- pin dependency versions in production,
- separate optional integrations,
- read migration guides before major upgrades,
- keep direct provider SDK knowledge for critical paths,
- avoid importing every integration into the runtime image.

## Configuration

Use explicit configuration for:

- model provider,
- model name,
- temperature,
- max tokens,
- embedding model,
- vector store connection,
- collection/index name,
- retrieval `k`,
- score thresholds,
- reranker settings,
- tracing enabled/disabled,
- timeout and retry policy.

Do not hide production behavior in notebook defaults.

## Testing Strategy

Test layers:

- unit tests for tools and pure functions,
- schema tests for structured outputs,
- prompt rendering tests,
- retriever tests with fixture documents,
- graph routing tests,
- integration tests with sandbox providers,
- offline evals for RAG/agent quality.

Mock model calls for deterministic unit tests. Use real models for evaluation and integration tests where behavior matters.

## Security Patterns

For LangChain tools:

- validate tool input schemas,
- enforce authorization inside the tool,
- use allowlists for URLs/actions,
- avoid arbitrary code execution,
- rate-limit tool calls,
- log tool inputs/outputs with redaction.

For RAG:

- enforce tenant and ACL filters before generation,
- separate retrieved data from instructions,
- cite sources,
- detect prompt injection in retrieved content,
- do not expose hidden chain or system prompts to users.

For agents:

- set iteration limits,
- require approval for destructive actions,
- restrict tool permissions,
- trace every tool call,
- fail closed on security errors.

## Performance Patterns

- Stream final model responses when useful.
- Batch embeddings during ingestion.
- Cache deterministic query rewrites and embeddings.
- Run independent retrievers in parallel.
- Use small models for classification/routing.
- Use larger models only where quality requires them.
- Time out slow tools.
- Keep context concise.

## When to Use Each Ecosystem Piece

| Need | Use |
| --- | --- |
| Simple model call | LangChain model interface or direct SDK. |
| Linear pipeline | Runnables/LCEL-style chain. |
| RAG prototype | LangChain loaders, splitters, vector stores, retrievers. |
| Production RAG | LangChain components plus explicit app architecture and evals. |
| Stateful workflow | LangGraph. |
| Agent with tools | LangChain `create_agent`; LangGraph for lower-level control. |
| Trace/debug/evaluate | LangSmith. |
| Visual prototype | LangFlow or notebooks. |

## Common Mistakes

| Mistake | Consequence | Better Approach |
| --- | --- | --- |
| Using an agent for a fixed workflow | More latency and nondeterminism. | Use a chain or graph. |
| Hiding business logic in prompts | Hard to test and audit. | Put deterministic logic in code. |
| No tracing | Failures are opaque. | Enable LangSmith or equivalent tracing early. |
| No eval dataset | Changes are guesswork. | Build small representative datasets. |
| Over-abstracting providers | Important provider behavior disappears. | Abstract app needs, not every SDK feature. |
| Trusting tool arguments blindly | Security and data integrity risk. | Validate and authorize every tool call. |

## Professional Workflow

1. Build the simplest working pipeline.
2. Add tracing immediately.
3. Create a small evaluation dataset.
4. Measure baseline quality and latency.
5. Add retrieval, reranking, tools, or agents only when a measured failure requires it.
6. Move complex branching to LangGraph.
7. Add security controls before production traffic.
8. Keep production traces feeding future evals.

## Study Checklist

- I can structure a LangChain project for maintainability.
- I know which configuration values must be explicit.
- I can choose the right LangChain ecosystem component for a workload.
- I understand the security risks of tools and RAG.
- I can build a test/evaluation loop around LangChain applications.

