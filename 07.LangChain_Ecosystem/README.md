# Phase 7 - LangChain Ecosystem

The LangChain ecosystem is a set of tools for building, orchestrating, tracing, evaluating, and deploying LLM applications. Modern LangChain is best understood as a family:

- LangChain: model, tool, prompt, retriever, and agent framework.
- LangGraph: low-level orchestration runtime for stateful, durable, long-running workflows and agents.
- LangSmith: tracing, observability, datasets, evaluation, prompt management, and debugging.
- Integrations: model providers, vector stores, retrievers, document loaders, tools, and app connectors.
- LangFlow and visual builders: useful for prototyping and demonstrations, but production systems usually need code-level control.

## Study Goals

By the end of this phase, you should be able to:

- Explain the role of LangChain, LangGraph, LangSmith, and integrations.
- Build simple model calls, chains, tools, structured outputs, and retrievers.
- Use runnables/LCEL-style composition for predictable pipelines.
- Build RAG with document loaders, text splitters, embeddings, vector stores, and retrievers.
- Decide when to use a simple chain, a LangChain agent, or a LangGraph workflow.
- Trace, debug, and evaluate LLM applications with LangSmith concepts.
- Avoid common framework misuse: over-agentizing, hiding business logic in prompts, and skipping tests/evals.

## Files in This Section

- [01_LangChain_Core_Concepts.md](01_LangChain_Core_Concepts.md) - ecosystem map, models, messages, prompts, tools, agents, and integrations.
- [02_Runnables_LCEL_Chains_and_Structured_Output.md](02_Runnables_LCEL_Chains_and_Structured_Output.md) - runnables, composition, streaming, batching, output parsing, and structured response design.
- [03_Retrieval_and_RAG_with_LangChain.md](03_Retrieval_and_RAG_with_LangChain.md) - loaders, splitters, embeddings, vector stores, retrievers, and RAG pipeline patterns.
- [04_LangGraph_Agents_State_and_Workflows.md](04_LangGraph_Agents_State_and_Workflows.md) - state graphs, nodes, edges, persistence, human-in-the-loop, and agent workflows.
- [05_LangSmith_Observability_Evaluation_and_Debugging.md](05_LangSmith_Observability_Evaluation_and_Debugging.md) - traces, runs, threads, datasets, evaluators, experiments, monitoring, and debugging.
- [06_Production_Patterns_and_Ecosystem_Tradeoffs.md](06_Production_Patterns_and_Ecosystem_Tradeoffs.md) - deployment patterns, package choices, security, testing, and when not to use LangChain.

## Source Trail

- LangChain overview: <https://docs.langchain.com/oss/python/langchain/overview>
- LangChain models docs: <https://docs.langchain.com/oss/python/langchain/models>
- LangChain agents docs: <https://docs.langchain.com/oss/python/langchain/agents>
- LangChain retrieval docs: <https://docs.langchain.com/oss/python/deepagents/retrieval>
- LangGraph overview: <https://langchain-ai.github.io/langgraph/>
- LangSmith observability docs: <https://docs.langchain.com/langsmith/observability-concepts>
- LangSmith evaluation docs: <https://docs.langchain.com/langsmith/evaluation-concepts>
- LangChain reference docs: <https://reference.langchain.com/>
