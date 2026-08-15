# Runnables, LCEL, Chains, and Structured Output

LangChain's composition layer is built around runnables: units of work that can be invoked, streamed, batched, and composed.

## Runnable Mental Model

A runnable is anything that follows a standard interface:

```text
input -> runnable -> output
```

Examples:

- prompt template,
- chat model,
- output parser,
- retriever,
- function wrapper,
- chain of multiple steps,
- parallel branch.

The LangChain reference says runnable objects can be composed declaratively, with `RunnableSequence` and `RunnableParallel` as main composition primitives. Source: <https://reference.langchain.com/python/langchain-core/runnables/base/Runnable>.

## Why Runnables Matter

Runnables give you:

- consistent `invoke` behavior,
- async invocation,
- batch processing,
- streaming,
- tracing,
- retries,
- composition,
- easier testing of individual steps.

This matters because real LLM applications are pipelines, not isolated prompts.

## RunnableSequence

A sequence runs steps one after another.

Conceptual example:

```text
input
  -> prompt template
  -> model
  -> output parser
  -> final result
```

Use sequences for predictable workflows:

- classification,
- extraction,
- summarization,
- RAG answer generation,
- rewriting queries.

## RunnableParallel

A parallel runnable sends the same input to multiple branches.

Use it for:

- retrieving from multiple indexes,
- generating multiple query rewrites,
- running independent classifiers,
- combining structured extraction with summarization,
- preparing context and preserving original question at the same time.

Example RAG shape:

```text
question
  -> {
       context: retriever,
       question: passthrough
     }
  -> prompt
  -> model
  -> parser
```

This keeps the original question while also retrieving context.

## LCEL-Style Composition

LCEL-style composition uses pipe-like flow:

```python
chain = prompt | model | parser
```

Benefits:

- readable data flow,
- easy step replacement,
- built-in tracing support,
- testable components.

Limits:

- complex branching can become hard to read,
- business logic may be clearer as normal Python functions,
- deeply nested runnable dictionaries can hide behavior.

Use LCEL for simple and medium pipelines. Use LangGraph or explicit Python orchestration for stateful, branching, long-running workflows.

## Output Parsers

Output parsers convert model text into structured application data.

Use parsers when:

- the model returns text but your app needs JSON,
- you need validation or post-processing,
- provider-native structured output is unavailable,
- you are maintaining older chains.

LangChain reference docs note that output parsers remain useful when models do not support native structured output or when extra processing/validation is needed. Source: <https://reference.langchain.com/python/langchain-core/output_parsers>.

## Structured Output

Structured output asks the model to return data matching a schema, such as a Pydantic model, dataclass, TypedDict, or JSON schema.

Use structured output for:

- extraction,
- classification,
- routing,
- tool arguments,
- API responses,
- UI rendering,
- evaluation results.

LangChain's structured output docs describe predictable schema-based responses and returning structured agent responses through the `structured_response` key when configured on agents. Source: <https://docs.langchain.com/oss/python/langchain/structured-output>.

## Parser vs Structured Output

| Need | Prefer |
| --- | --- |
| Provider supports strict structured output | Provider-native structured output. |
| Agent should return schema | Agent `response_format`. |
| Model only returns text | Output parser. |
| Need custom validation | Pydantic or custom parser. |
| Need robust production API | Schema validation plus error handling. |

## Error Handling

LLM pipelines fail in specific ways:

- model timeout,
- provider rate limit,
- invalid JSON,
- schema validation error,
- tool failure,
- retrieval empty result,
- context too long,
- unsafe output.

Handle errors at the step that owns them.

Examples:

- retry transient provider errors,
- return no-answer when retrieval has no evidence,
- ask a repair model to fix malformed JSON only if safe,
- fail closed on permission errors,
- log trace IDs for debugging.

## Testing Chains

Test each piece:

- prompt formatting with known inputs,
- parser behavior on valid and invalid outputs,
- retriever output shape,
- full chain output contract,
- failure paths.

Do not only test the happy path. Production failures often come from malformed model output or missing retrieval context.

## Study Checklist

- I can explain a runnable as an input-output component.
- I know the difference between sequence and parallel composition.
- I can decide when LCEL is enough and when LangGraph is clearer.
- I understand structured output vs output parsing.
- I can design error handling for each chain step.

