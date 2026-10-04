# LangSmith Handbook

This folder contains the Python-first handbook for LangSmith tracing, evaluation, datasets, feedback, prompt versioning, and observability/deployment workflows. It describes how to inspect and improve LLM application behavior while separating telemetry from application authorization, business state, and deterministic tests.

## Files in this folder

| File | Content |
| --- | --- |
| [`LangSmith.md`](./LangSmith.md) | The complete handbook in one navigable Markdown file. The opening **Document map** is its full chapter outline. |

## Handbook structure

1. What LangSmith solves
2. Setup and environment
3. Runs, traces, threads, datasets, experiments, feedback, and prompts
4. Automatic framework tracing
5. `@traceable` and arbitrary Python instrumentation
6. Tags, metadata, feedback, and privacy
7. Dataset creation and management
8. Deterministic, LLM, pairwise, and human evaluators
9. Offline `evaluate` / `aevaluate` experiments
10. Online run/thread/trajectory monitoring
11. Prompt commits, tags, and model configurations
12. Studio, Agent Server, and deployment choices
13. Python SDK and CLI surface
14. A traced support-evaluation project
15. Operating checklist
16. Product architecture and trace lifecycle
17. Python SDK `Client` field guide
18. Trace contexts, wrappers, processors, and redaction
19. Trace topology, querying, metrics, and threads
20. Dataset lifecycle, examples, provenance, and versions
21. Evaluator contracts, calibration, experiments, and statistics
22. Online evaluation, user feedback, and human annotation
23. Prompt release and rollback workflow
24. Privacy, security, and governance
25. Operations, cost controls, and troubleshooting
26. Course repository topic map
27. API coverage and reference policy

## Scope and reference policy

- Language: Python, matching the supplied course code and current local handbook.
- Platform availability and details can differ for LangSmith Cloud, regional offerings, hybrid, and self-hosted deployments.
- The handbook deeply explains common SDK workflows and links the generated API reference for exact method signatures, advanced resource methods, and full REST coverage.
- Course examples are used to expand the topic inventory. The source repository has no visible license file, so substantial code is not copied verbatim; source files are linked at the inspected commit.
- Traces can contain prompts, user data, tool arguments/results, and errors. Follow the handbook’s redaction, retention, access-control, and endpoint guidance before enabling production capture.

## Start here

Use the [Document map in `LangSmith.md`](./LangSmith.md#document-map-this-files-structure) to jump directly to a subject. For core concepts, see the official [observability guide](https://docs.langchain.com/langsmith/observability-concepts), [evaluation guide](https://docs.langchain.com/langsmith/evaluation-concepts), [Python SDK reference](https://reference.langchain.com/python/langsmith/), and [LangSmith REST API reference](https://docs.langchain.com/langsmith/smith-api-ref).
