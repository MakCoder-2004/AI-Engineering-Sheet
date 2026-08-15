# LangChain Core Concepts

LangChain is a framework for building applications around language models. Its main value is not that it makes one model call shorter. Its value is that it gives you reusable interfaces for models, prompts, tools, structured outputs, agents, retrievers, and integrations.

## Ecosystem Map

| Component | Role |
| --- | --- |
| LangChain | High-level framework for model calls, tools, agents, middleware, and app composition. |
| LangGraph | Low-level graph runtime for durable, stateful, controllable workflows and agents. |
| LangSmith | Observability, tracing, evaluation, datasets, experiments, and debugging. |
| Integrations | Provider packages for models, vector stores, loaders, tools, and services. |
| Deep Agents | Higher-level agent harness built on LangGraph for complex agent workflows. |
| LangFlow | Visual/prototyping layer for flows and demos. |

LangChain's current overview describes `create_agent` as a configurable agent harness composed from model, tools, prompt, and middleware, and notes that LangChain agents are built on LangGraph. Source: <https://docs.langchain.com/oss/python/langchain/overview>.

## When to Use LangChain

Use LangChain when:

- you need provider-agnostic model calls,
- you want standardized tools and messages,
- you are building RAG with loaders, splitters, vector stores, and retrievers,
- you need agents with tools,
- you want LangSmith tracing/evaluation integration,
- you want to swap models or stores during experiments.

Avoid or minimize LangChain when:

- your app is a single direct API call,
- your team needs full control over every request body,
- the framework abstraction hides important provider-specific behavior,
- you are debugging low-level protocol issues,
- you are tempted to solve deterministic business logic with agent loops.

## Models

Models are the reasoning engine of a LangChain app. LangChain provides standard interfaces across providers, so you can use a similar calling pattern for different model vendors.

Model capabilities may include:

- text generation,
- chat messages,
- tool calling,
- structured output,
- multimodal input/output,
- reasoning behavior,
- streaming,
- batching,
- token usage metadata.

LangChain's model docs state that models can be used with agents or standalone for tasks like generation, classification, and extraction. Source: <https://docs.langchain.com/oss/python/langchain/models>.

## Messages

Chat applications usually operate on messages rather than raw strings.

Common message roles:

- system: highest-level behavior and constraints,
- developer: application-level rules,
- user: user request,
- assistant: model response,
- tool: tool result returned to the model.

Keep business policy out of user messages. Put stable application behavior in system/developer instructions and keep dynamic context clearly separated.

## Prompts

Prompts are templates for model inputs.

Good prompts:

- define role and task,
- include input variables,
- state constraints,
- specify output format,
- define refusal behavior,
- separate instructions from retrieved data,
- avoid hidden assumptions.

Bad prompts:

- contain large unstructured instruction dumps,
- ask the model to enforce security without backend checks,
- mix user text and trusted instructions,
- require exact JSON without validation,
- include outdated examples.

## Tools

Tools let a model or agent call external functions.

Examples:

- search documentation,
- query a database,
- retrieve RAG context,
- call a calculator,
- open a ticket,
- send an email draft,
- fetch account status.

Good tools are:

- narrow,
- typed,
- documented,
- validated,
- permission-aware,
- observable,
- idempotent when possible.

Do not expose broad tools like "run arbitrary SQL" or "make any HTTP request" to an agent unless you have strong sandboxing and approval controls.

## Agents

An agent is a loop where the model can decide:

- whether to call a tool,
- which tool to call,
- what arguments to pass,
- how to use the result,
- when to stop.

LangChain's agents docs describe agents as systems that combine language models with tools and run until a stop condition is met, such as a final answer or an iteration limit. Source: <https://docs.langchain.com/oss/python/langchain/agents>.

Use agents for:

- tasks with variable steps,
- tool selection,
- information gathering,
- multi-step workflows,
- ambiguous goals.

Prefer deterministic chains/workflows for:

- fixed transformations,
- compliance-sensitive flows,
- payments or account changes,
- predictable ETL,
- simple RAG Q&A.

## Middleware

Middleware shapes model or agent behavior around the core loop.

Use middleware for:

- retries,
- guardrails,
- model routing,
- dynamic system prompts,
- context trimming,
- tool call control,
- logging,
- error handling.

Middleware is powerful, but it can make behavior harder to understand. Trace it.

## Integrations

LangChain integrations connect to:

- model providers,
- embedding providers,
- vector stores,
- document loaders,
- retrievers,
- search APIs,
- databases,
- observability systems.

Use integration packages deliberately. Pin versions in real projects and read provider-specific docs when behavior matters.

## Study Checklist

- I can explain LangChain vs LangGraph vs LangSmith.
- I know when a model call, chain, agent, or graph is appropriate.
- I can define a narrow tool with a clear schema.
- I understand why security cannot be delegated only to prompts.
- I can identify when LangChain is useful and when direct provider SDKs are simpler.

