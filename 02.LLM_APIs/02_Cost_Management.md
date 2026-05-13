# Calculating API Costs with OpenAI

## Overview

Before deploying AI features at scale, it is **essential to estimate costs**. OpenAI charges based on the number of tokens consumed — both input (your prompt) and output (the model's response). Understanding this helps you budget accurately and optimize your usage.

---

## How Pricing Works

### Token Pricing Model

> **Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)**

OpenAI prices differ by model. Below are example prices for `gpt-4o-mini`:

| Token Type | Price per 1M tokens | Price per token |
|------------|---------------------|-----------------|
| Input (Prompt) | $0.15 | $0.00000015 |
| Output (Completion) | $0.60 | $0.00000060 |

> ⚠️ Prices change over time. Always check the [OpenAI Pricing Page](https://openai.com/pricing) for the latest rates.

### What is a Token?

A token is roughly **4 characters** or **¾ of a word** in English.

| Text | Approximate Tokens |
|------|--------------------|
| "Hello, world!" | ~4 tokens |
| 1 English sentence (~10 words) | ~13 tokens |
| 1 paragraph (~100 words) | ~130 tokens |
| 1 page (~500 words) | ~650 tokens |

---

## Implementation

```python
from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Define your prompt and set a max token limit for the response
prompt = "This is a prompt"
max_completion_tokens = 100

# Make the API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=max_completion_tokens
)

# --- Pricing constants (gpt-4o-mini as of writing) ---
input_token_price  = 0.15 / 1_000_000   # $0.15 per million input tokens
output_token_price = 0.60 / 1_000_000   # $0.60 per million output tokens

# --- Extract actual token usage from response ---
input_tokens  = response.usage.prompt_tokens       # Tokens in your prompt
output_tokens = response.usage.completion_tokens   # Tokens in the response

# --- Calculate cost ---
cost = (input_tokens * input_token_price) + (output_tokens * output_token_price)

print(f"Input tokens:  {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Total tokens:  {input_tokens + output_tokens}")
print(f"Estimated cost: ${cost:.8f}")
```

---

## The `usage` Object

The response object contains a `usage` field that gives you precise token counts:

```python
response.usage.prompt_tokens      # Tokens consumed by your input
response.usage.completion_tokens  # Tokens in the model's response
response.usage.total_tokens       # Sum of both
```

---

## Cost Estimation Tool (Reusable Function)

```python
def estimate_cost(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_completion_tokens: int = 500,
    input_price_per_million: float = 0.15,
    output_price_per_million: float = 0.60
) -> dict:
    """
    Estimates the cost of an OpenAI API call.
    
    Returns a dict with:
        - input_tokens:  actual prompt tokens used
        - output_tokens: tokens in the response
        - total_tokens:  combined total
        - cost_usd:      estimated cost in US dollars
        - response_text: the actual response
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_completion_tokens
    )

    input_price  = input_price_per_million  / 1_000_000
    output_price = output_price_per_million / 1_000_000

    input_t  = response.usage.prompt_tokens
    output_t = response.usage.completion_tokens
    cost     = (input_t * input_price) + (output_t * output_price)

    return {
        "input_tokens":  input_t,
        "output_tokens": output_t,
        "total_tokens":  response.usage.total_tokens,
        "cost_usd":      round(cost, 8),
        "response_text": response.choices[0].message.content
    }

# Usage
result = estimate_cost("Summarize machine learning in 3 bullet points.", max_completion_tokens=100)
print(f"Cost: ${result['cost_usd']}")
print(f"Response: {result['response_text']}")
```

---

## Cost Comparison Table — Common Models

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Use Case |
|-------|---------------------|----------------------|----------|
| `gpt-4o-mini` | $0.15 | $0.60 | Cost-effective, everyday tasks |
| `gpt-4o` | $2.50 | $10.00 | Complex reasoning, multimodal |
| `gpt-3.5-turbo` | $0.50 | $1.50 | Legacy, still widely used |

---

## Cost Optimization Tips

### 1. Minimize Input Tokens
- Be concise in your system prompt — don't repeat instructions
- Trim unnecessary whitespace and boilerplate from prompts

### 2. Limit Output Tokens
```python
# Use max_completion_tokens to cap response length
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    max_completion_tokens=50  # Only need a short answer
)
```

### 3. Use the Right Model
- For simple classification or extraction → `gpt-4o-mini`
- For complex reasoning or code generation → `gpt-4o`
- Match model capability to task complexity

### 4. Batch Processing
- Combine multiple short tasks into a single API call when possible

### 5. Cache Frequent Responses
```python
from functools import lru_cache

@lru_cache(maxsize=256)
def cached_response(prompt: str) -> str:
    # Only make API call if not cached
    ...
```

---

## Example Cost Scenarios

| Scenario | Prompt Tokens | Response Tokens | Cost (gpt-4o-mini) |
|----------|--------------|-----------------|---------------------|
| Classify a tweet | 30 | 5 | ~$0.0000075 |
| Summarize a paragraph | 150 | 80 | ~$0.000070 |
| Analyze a full document | 2,000 | 500 | ~$0.000600 |
| 1,000 classifications/day | 30,000 | 5,000 | ~$0.0075/day |

---

## Key Takeaways

- 💡 Cost = Input tokens × input price + Output tokens × output price
- 💡 Always extract `response.usage` to see actual token consumption
- 💡 `gpt-4o-mini` is highly cost-effective for most tasks
- 💡 Set `max_completion_tokens` to control output costs
- 💡 Monitor usage to detect unexpected spikes
