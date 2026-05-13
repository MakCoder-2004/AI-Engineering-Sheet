# Sending a Simple Response with the OpenAI API

## Overview

The most fundamental operation with the OpenAI API is sending a message and receiving a response. This document covers how to set up the client, structure a request, and handle the response properly.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Library | `openai` Python package |
| Authentication | OpenAI API Key |
| Model | `gpt-4o-mini` (or any available chat model) |

Install the library:
```bash
pip install openai
```

---

## Core Concepts

### The Chat Completions API

OpenAI's chat API follows a **conversational message format** where each message has:

| Role | Description |
|------|-------------|
| `system` | Sets the assistant's behavior and personality |
| `user` | The human's input or question |
| `assistant` | The AI's previous responses (for multi-turn conversations) |

### Key Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `model` | `str` | — | Which model to use (e.g. `gpt-4o-mini`) |
| `messages` | `list` | — | List of message dicts with `role` and `content` |
| `temperature` | `float` | `[0, 2]` | Controls randomness; 0 = deterministic, 2 = very creative |
| `max_tokens` | `int` | > 0 | Maximum number of tokens in the response |

---

## Basic Implementation

```python
from openai import OpenAI

# Initialize the client with your API key
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

def get_response(prompt: str) -> str:
    """
    Send a prompt to the OpenAI API and return the response.
    
    Args:
        prompt: The user's question or instruction
    
    Returns:
        The model's response text, or an error message
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that gives concise and practical answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,   # Controls answer randomness from [0, 2]
            max_tokens=100,    # Controls response length
        )

        # Extract the text content from the first (and usually only) choice
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


# Example usage
prompt = "Quick productivity tip."
response = get_response(prompt)
print(response)
```

---

## Response Object Structure

When the API call is successful, the response object has the following structure:

```
ChatCompletion
├── id                    — Unique completion ID
├── object                — "chat.completion"
├── created               — Unix timestamp
├── model                 — Model used
├── choices[]             — List of generated responses
│   ├── index             — Choice index (0 for first)
│   ├── message
│   │   ├── role          — "assistant"
│   │   └── content       — The actual text response
│   └── finish_reason     — "stop", "length", etc.
└── usage
    ├── prompt_tokens     — Tokens in your input
    ├── completion_tokens — Tokens in the response
    └── total_tokens      — Total tokens consumed
```

Extracting the text:
```python
# The standard way to get the response text
text = response.choices[0].message.content.strip()
```

---

## Understanding `temperature`

```
temperature = 0.0  → Very deterministic, consistent answers
temperature = 0.8  → Slightly creative, good for general use (default-ish)
temperature = 1.5  → Highly creative, more varied responses
temperature = 2.0  → Maximum randomness — can be incoherent
```

> **Tip:** For factual tasks (code generation, Q&A), use `temperature=0.0–0.3`. For creative tasks (storytelling, brainstorming), use `0.7–1.2`.

---

## Error Handling Best Practices

```python
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError

client = OpenAI(api_key="<YOUR_KEY>")

def safe_get_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    except AuthenticationError:
        return "Error: Invalid API key."
    except RateLimitError:
        return "Error: Rate limit reached. Please wait before retrying."
    except OpenAIError as e:
        return f"OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
```

---

## Best Practices

- ✅ **Always use a `system` message** to define the assistant's role and behavior
- ✅ **Wrap calls in try/except** for robust error handling
- ✅ **Use `.strip()`** on the response content to remove leading/trailing whitespace
- ✅ **Set `max_tokens`** to avoid runaway costs on long responses
- ❌ **Never hardcode API keys** — use environment variables instead

### Using Environment Variables (Recommended)

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

---

## Quick Reference

```python
# Minimal complete example
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=50
)

print(response.choices[0].message.content)
# Output: "The capital of France is Paris."
```
