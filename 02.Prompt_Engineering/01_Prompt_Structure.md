# Structuring Prompts & Controlling Output Format

## Overview

One of the most powerful techniques in prompt engineering is controlling **how** the model formats its output. By including explicit instructions and output templates in your prompt, you can make the model's responses consistent, parseable, and ready for downstream use.

---

## The Problem with Unstructured Outputs

Without structure, the model may respond in any format:
```
User: "Give me a title for this text."
Model: "How about 'The Adventures of Young David'? Or maybe 'David's Curious Journey'?"
```

With a structured prompt:
```
User: [formatted instructions + format template]
Model:
  - Text: Once upon a time...
  - Title: The Adventures of Young David
```

---

## Core Pattern: Instructions + Format + Input

The most effective way to get structured output is to split your prompt into three parts:

```python
instructions = "Do this task..."
output_format = "Use this exact format for the output: ..."
input_data    = "Here is the data: ..."

prompt = instructions + output_format + input_data
```

---

## Implementation

```python
from openai import OpenAI

client = OpenAI("<YOUR_API_KEY>")

def get_response(prompt: str) -> str:
    """Send a structured prompt and return the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# --- Define each component separately ---

instructions = (
    "You will be provided with a text delimited by triple backticks. "
    "Generate a suitable title for it."
)

output_format = """
Use the following format for the output:
    - Text: <text we want to title>
    - Title: <the generated title>
"""

text = "Once upon a time in a quaint little village, there lived a curious young boy named David. David was [...]"

# Combine into one structured prompt
prompt = instructions + output_format + f'"{text}"'

print(get_response(prompt))
```

**Expected output:**
```
- Text: Once upon a time in a quaint little village...
- Title: The Adventures of Young David
```

---

## Techniques for Structured Output

### 1. Delimiter-Delimited Input

Use delimiters (triple backticks, XML tags, etc.) to clearly separate the input data from the instructions:

```python
# Triple backtick delimiters
prompt = f"""
Generate a title for the following text:

```{text}```

Respond with only the title, nothing else.
"""

# XML-style delimiters
prompt = f"""
Generate a title for the text between the <text> tags.

<text>
{text}
</text>

Output only the title.
"""
```

### 2. Conditional Output Format

Force the model to behave differently based on conditions in the input:

```python
prompt = """
You will be given a product description. 

If the product is electronic:
    - Category: Electronics
    - Key Feature: <most important feature>
    
If the product is clothing:
    - Category: Fashion
    - Key Feature: <most important feature>

Otherwise:
    - Category: Other
    - Key Feature: N/A

Product: "Wireless noise-cancelling headphones with 30-hour battery"
"""
```

### 3. JSON Output Format

Getting structured data for programmatic use:

```python
prompt = """
Extract information from the following customer review and return it as JSON:

Review: "The delivery was fast, but the product quality is disappointing. 
The color is not as described and it broke after a week."

Return JSON with this exact schema:
{
    "delivery_sentiment": "positive" | "negative" | "neutral",
    "product_sentiment": "positive" | "negative" | "neutral",
    "issues": ["list", "of", "issues"],
    "overall_rating": 1-5
}

Return only the JSON, no other text.
"""
```

### 4. Numbered Lists and Tables

```python
prompt = """
Summarize the following article in a structured format:

Article: [...]

Use exactly this format:
1. Main Topic: <one sentence>
2. Key Points:
   • Point 1
   • Point 2
   • Point 3
3. Conclusion: <one sentence>
"""
```

---

## Output Format Comparison

| Format Type | Use Case | Example |
|-------------|----------|---------|
| Bullet list | Human-readable summaries | `• Key point 1` |
| Key-value pairs | Field extraction | `- Name: John` |
| JSON | Programmatic parsing | `{"name": "John"}` |
| Markdown table | Comparative data | `\| Col1 \| Col2 \|` |
| Numbered list | Ordered steps | `1. First step` |
| Plain answer | Simple Q&A | `Paris` |

---

## Best Practices

### ✅ Do's
- Use **delimiters** to separate instructions from data
- Specify **exact output format** when you need consistency
- Include **an example** if the format is complex
- Ask for "only X, nothing else" to avoid extra chatter

### ❌ Don'ts
- Don't leave the output format ambiguous for structured data
- Don't use overly complex formats in a single prompt (break it up)
- Don't rely on consistent casing/spacing without explicitly requesting it

---

## Complete Reusable Function

```python
from openai import OpenAI

client = OpenAI(api_key="<YOUR_API_KEY>")

def structured_prompt(
    instructions: str,
    input_data: str,
    output_format: str,
    delimiter: str = '"""',
    temperature: float = 0.3,  # Lower for structured output
    max_tokens: int = 200
) -> str:
    """
    Sends a structured prompt with clear instructions and output format.
    
    Args:
        instructions:  What you want the model to do
        input_data:    The text/data to process
        output_format: How you want the output formatted
        delimiter:     String to wrap the input data (default: triple quotes)
        temperature:   Lower = more consistent formatting
        max_tokens:    Response length limit
    
    Returns:
        The model's formatted response
    """
    prompt = (
        f"{instructions}\n\n"
        f"Output format:\n{output_format}\n\n"
        f"Input:\n{delimiter}\n{input_data}\n{delimiter}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content.strip()


# Example
result = structured_prompt(
    instructions="Generate a suitable title for the provided text.",
    input_data="Once upon a time in a quaint little village...",
    output_format="- Text: <text>\n- Title: <generated title>"
)
print(result)
```
