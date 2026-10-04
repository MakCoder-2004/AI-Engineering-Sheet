# LangChain Handbook

This folder contains the Python-first handbook for LangChain 1.x: the high-level APIs for model applications, prompts, tools, agents, runnables, and retrieval. It expands the existing guide with API-level explanations, practical design constraints, current package boundaries, and a topic map of the supplied production-course examples.

## Files in this folder

| File | Content |
| --- | --- |
| [`LangChain.md`](./LangChain.md) | The complete handbook in one navigable Markdown file. Its opening **Document map** links to every chapter. |

## Handbook structure

The handbook starts with framework selection and core usage, then proceeds through model/message/prompt/tool APIs and runnable composition; retrieval and RAG; integration/package boundaries; memory, reliability, security, testing, operations, and migration; finally it maps the course repository sources and defines API/reference coverage.

### Chapters at a glance

1. Mental model and framework selection
2. Models and messages
3. Prompt templates
4. Structured output
5. Tools and function calling
6. Agents and the agent execution loop
7. Middleware, runtime context, guardrails, and human approval
8. LCEL and runnables
9. Retrieval and RAG
10. Integration catalog
11. Memory, streaming, MCP, and multi-agent systems
12. Testing, security, and operations
13. Migration map
14. Complete support-agent example
15. Further references
16. Package architecture and installation
17. Model API field guide
18. Message and prompt API field guide
19. Runnable and LCEL field guide
20. Agent, tool, and middleware internals
21. Document ingestion and text splitting
22. Embeddings, vector stores, and retrievers
23. RAG patterns and evaluation
24. Conversation state and memory boundaries
25. Reliability, performance, and cost
26. Security and data governance
27. Testing and debugging playbook
28. Course repository topic map
29. API coverage and reference policy

## Scope and reference policy

- Language: Python.
- Examples target current LangChain 1.x APIs. Provider packages and integrations are independently versioned; the handbook calls out behavior that must be verified against the selected provider.
- Core public API concepts are documented in depth. The handbook does not duplicate every provider-specific class among the 1,000+ integrations; it links to the live integration catalog and generated API references.
- Course-repository files are used to identify topics and inspire adapted examples. Links pin to the inspected source snapshot. The repository tree has no visible license file, so substantial source snippets are not reproduced verbatim.
- Do not treat example credentials, mock data, course versions, or demo implementations as production configuration.

## Start here

Read the [Document map in `LangChain.md`](./LangChain.md#document-map-this-files-structure), then use the relevant chapter. Check the official [LangChain Python guide](https://docs.langchain.com/oss/python/langchain/overview), [Python API reference](https://reference.langchain.com/python/langchain/), [Core API reference](https://reference.langchain.com/python/langchain-core/), and [integration catalog](https://docs.langchain.com/oss/python/integrations) when package versions change.
