# LangGraph Handbook

This folder contains the Python handbook for LangGraph 1.x, the low-level orchestration runtime for explicit, stateful workflows and agents. It covers graph construction, functional workflows, state/reducer semantics, persistence, human interrupts, streaming, failure recovery, multi-agent patterns, testing, and deployment.

## Files in this folder

| File | Content |
| --- | --- |
| [`LangGraph.md`](./LangGraph.md) | The complete handbook in one Markdown document. Its opening **Document map** is the chapter outline and navigation index. |

## Handbook structure

1. Mental model
2. Choosing the Graph or Functional API
3. State schemas and reducers
4. Nodes, edges, and control flow
5. Tools and model-directed tool loops
6. Persistence, threads, checkpointers, stores, and memory
7. Interrupts and human-in-the-loop workflows
8. Streaming modes and event consumption
9. Subgraphs and persistence scoping
10. Retries, timeouts, error handling, and fault tolerance
11. Replay, fork, and time travel
12. Workflow and agent design choices
13. Graph testing
14. Studio, LangSmith observability, CLI, and deployment
15. Migration from legacy APIs
16. Complete local mini-project
17. Execution model and graph lifecycle
18. State/channel contracts and runtime context
19. Graph API method guide
20. Routing, `Command`, `Send`, fan-out, and joins
21. Functional API and durable task semantics
22. Checkpoint/store operations and subgraph persistence
23. Interrupt/resume and approval protocol
24. Streaming and observability field guide
25. Fault handling and durable side effects
26. Handoffs, supervisors, hierarchical agents, and research workflows
27. Testing and debugging playbook
28. Production operations, security, and cost
29. Course repository topic map
30. API coverage and reference policy

## Scope and reference policy

- Language: Python. The supplied course source is Python-based.
- Examples use the current LangGraph 1.x model; LangChain `create_agent` is referenced as the higher-level agent harness built on LangGraph.
- Core public runtime APIs are explained in depth. Checkpointer/store integrations and CLI/deployment options are linked to their own current references because their packages and product surfaces evolve separately.
- Course examples are topic sources and are adapted with attribution. The linked repository snapshot has no visible license file, so substantial snippets are not reproduced verbatim.
- In-memory saver/store implementations are for learning and tests; production guidance calls out durable storage, identity, replay safety, and idempotent side effects.

## Start here

Open the [Document map in `LangGraph.md`](./LangGraph.md#document-map-this-files-structure), then start with chapters 1–4 before reading persistence or multi-agent material. The [official Python overview](https://docs.langchain.com/oss/python/langgraph/overview), [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), and [Python reference](https://reference.langchain.com/python/langgraph/) are the primary sources for version-sensitive details.
