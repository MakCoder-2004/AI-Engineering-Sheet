# LangGraph Agents, State, and Workflows

LangGraph is the LangChain ecosystem's low-level orchestration framework for stateful, controllable, long-running agent and workflow applications.

## Why LangGraph Exists

Simple chains are linear. Agents are flexible but can be unpredictable. Many production workflows need both deterministic control and LLM-powered decisions.

LangGraph helps model workflows as graphs:

```text
state -> node -> state -> edge -> next node
```

The LangGraph overview describes it as a low-level orchestration framework and runtime for long-running, stateful agents, focused on durable execution, streaming, human-in-the-loop, and persistence. Source: <https://langchain-ai.github.io/langgraph/>.

## Core Concepts

| Concept | Meaning |
| --- | --- |
| State | Shared data that moves through the graph. |
| Node | A function or runnable that reads state and returns updates. |
| Edge | A transition from one node to another. |
| Conditional edge | Routing logic based on current state. |
| Checkpointer | Persistence layer for graph state. |
| Interrupt | A pause point, often for human approval. |
| Tool node | Executes tool calls requested by a model. |

## When to Use LangGraph

Use LangGraph when you need:

- branching workflows,
- loops,
- durable execution,
- human approval,
- resumability,
- multi-agent coordination,
- long-running tasks,
- explicit state,
- precise control over agent behavior.

Use a simple LangChain chain when:

- the workflow is linear,
- there is no state beyond input/output,
- no approval or checkpointing is needed,
- you can express the logic cleanly as normal code.

## State Design

State should be explicit and typed.

Example state fields:

```text
messages
user_id
tenant_id
retrieved_documents
draft_answer
approval_status
tool_results
error
iteration_count
```

Good state design:

- separates trusted system data from user/model text,
- stores source IDs, not only rendered text,
- tracks errors,
- limits history growth,
- records decisions needed for audit.

## Nodes

Nodes should do one clear job:

- classify request,
- rewrite query,
- retrieve documents,
- grade retrieved context,
- call model,
- execute tool,
- validate answer,
- request human approval.

Avoid giant nodes that hide the whole app. Small nodes are easier to trace, test, and replace.

## Edges and Routing

Conditional edges let you route based on state:

```text
retrieve
  -> if context sufficient: generate
  -> if context weak: rewrite_query
  -> if permission error: deny
  -> if needs approval: human_review
```

This is especially useful for self-correcting RAG and agent workflows.

## Human-in-the-Loop

Human-in-the-loop is useful when:

- tool calls change external state,
- answer has legal/compliance risk,
- model confidence is low,
- user requests a sensitive operation,
- the system needs approval before sending an email or making a purchase.

Design approval points as explicit graph states, not hidden prompt instructions.

## Persistence and Durability

Durable execution lets workflows survive process restarts, pauses, and long-running operations.

Use persistence when:

- workflows last longer than one request,
- users return later,
- tools are slow,
- approval is asynchronous,
- audit trails matter.

## LangGraph for Agentic RAG

Agentic RAG graph shape:

```text
start
  -> analyze_question
  -> retrieve
  -> grade_context
  -> if insufficient: rewrite_query -> retrieve
  -> if sufficient: generate_answer
  -> validate_grounding
  -> if invalid: revise_or_refuse
  -> final
```

This gives more control than a free-form agent loop while preserving adaptive retrieval.

## Multi-Agent Workflows

Use multiple agents when roles are genuinely distinct:

- research agent,
- retrieval agent,
- critic/evaluator,
- planner,
- domain specialist,
- final answer writer.

Do not split into agents just for style. Multi-agent systems add latency, cost, and debugging complexity.

## Study Checklist

- I can explain nodes, edges, state, and conditional routing.
- I know when LangGraph is better than a simple chain.
- I can design a state object for a RAG workflow.
- I understand how human approval fits into a graph.
- I can avoid unnecessary multi-agent complexity.

