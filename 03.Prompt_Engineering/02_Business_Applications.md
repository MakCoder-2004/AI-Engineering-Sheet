# Prompt Engineering for Business Applications

## Overview

Prompt engineering for real business workflows goes beyond general Q&A. This document covers four high-value enterprise use cases: **text summarization**, **text transformation**, **text analysis/classification**, and **code generation from examples**. Each includes a reusable prompt pattern that can be adapted to your specific needs.

---

## Setup

```python
from openai import OpenAI

client = OpenAI("<YOUR_API_KEY>")

def get_response(prompt: str, temperature: float = 0.8, max_tokens: int = 100) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## Use Case 1: Text Summarization

**Business Context:** Product catalogs, reports, customer feedback, research papers — all need concise summaries for decision-makers.

### Prompt Pattern

```python
product_description = """
The UltraComfort Pro Chair is engineered for professionals who spend long hours 
at their desks. Featuring a patented lumbar support system, adjustable armrests 
at 4 axes, breathable mesh back, and a hydraulic height adjustment range of 
16-21 inches. The chair weighs 28 lbs and supports up to 330 lbs. 
Built with recycled materials and designed for 12+ hour daily use.
"""

prompt = f"""
Summarize the following product description in no more than five bullet points.

Requirements:
- Use bullet points only
- Keep each point concise (max 10 words)
- Do not exceed 5 bullets
- Focus on key features, benefits, and main purpose

Product Description:
{product_description}
"""

response = get_response(prompt, temperature=0.3, max_tokens=150)
print("Summarized description:\n", response)
```

**Expected output:**
```
• Designed for 12+ hour professional desk use
• Patented lumbar support and 4-axis armrests
• Breathable mesh back with hydraulic height adjustment
• Supports up to 330 lbs, weighs 28 lbs
• Built with recycled, eco-friendly materials
```

### Key Prompt Elements for Summarization

| Element | Purpose | Example |
|---------|---------|---------|
| Output format | Control the structure | "bullet points only" |
| Length constraint | Prevent verbose output | "no more than 5 bullets" |
| Focus directive | Guide what to include | "key features and benefits" |
| Negative constraint | Prevent extra content | "Do not add information not in the text" |

---

## Use Case 2: Text Transformation

**Business Context:** Rewriting customer-facing communications, adapting tone for different audiences, converting casual language to formal.

### Tone Transformation

```python
sample_email = """
Hey, so your order is late because of some problems on our end. 
Sorry about that. It should arrive eventually. 
Call us if you want I guess.
"""

prompt = f"""
Rewrite the following email to improve its tone.

Requirements:
- Make it professional, positive, and customer-centric
- Maintain the original meaning and intent
- Improve clarity and structure if needed
- Do not add new information
- Keep it suitable for a business context
- Add a proper greeting and closing

Email:
{sample_email}
"""

response = get_response(prompt, temperature=0.5, max_tokens=200)
print("Before transformation:\n", sample_email)
print("After transformation:\n", response)
```

**Before:**
> "Hey, so your order is late because of some problems on our end. Sorry about that. It should arrive eventually."

**After:**
> "Dear Valued Customer, We sincerely apologize for the delay in your order. Our team is working diligently to resolve this and ensure delivery as soon as possible. Please don't hesitate to contact us if you need any assistance. Warm regards, Customer Support Team"

### Transformation Use Cases

| Use Case | Input | Output |
|----------|-------|--------|
| Tone improvement | Casual → Professional | Customer emails |
| Language simplification | Technical → Plain | User documentation |
| Localization | English → target style | Regional marketing |
| Format conversion | Paragraphs → Bullet list | Slide decks |
| Length reduction | Long report → Summary | Executive briefs |

---

## Use Case 3: Text Analysis & Classification

**Business Context:** Automatically categorize support tickets, classify product feedback, route customer requests to the correct team.

### Support Ticket Classification

```python
ticket = "I've been charged twice for the same order last Tuesday. Order #12345."

prompt = f"""
Classify the following customer support ticket into exactly ONE of the following categories:

- Technical Issue
- Billing Inquiry
- Product Feedback

Requirements:
- Output ONLY the category name
- Do NOT include explanations, punctuation, or extra text
- Do NOT add any additional words

Ticket:
{ticket}
"""

response = get_response(prompt, temperature=0.0, max_tokens=10)  # temperature=0 for determinism
print(f"Ticket: {ticket}")
print(f"Category: {response}")
```

### Multi-Field Analysis

For more complex classification with additional insights:

```python
def analyze_ticket(ticket_text: str) -> dict:
    """
    Analyze a support ticket for category, urgency, and sentiment.
    Returns structured output.
    """
    prompt = f"""
Analyze the following customer support ticket and return ONLY a JSON object.

Ticket: "{ticket_text}"

Return this exact JSON structure (no other text):
{{
    "category": "Technical Issue" | "Billing Inquiry" | "Product Feedback" | "General Inquiry",
    "urgency": "High" | "Medium" | "Low",
    "sentiment": "Frustrated" | "Neutral" | "Positive",
    "summary": "One sentence description of the issue"
}}
"""
    import json
    response = get_response(prompt, temperature=0.0, max_tokens=150)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Could not parse response", "raw": response}


# Example
result = analyze_ticket("I've been charged twice for my order #12345 from last Tuesday!")
print(result)
# → {'category': 'Billing Inquiry', 'urgency': 'High', 'sentiment': 'Frustrated', 'summary': '...'}
```

### Prompt Tips for Classification

| Tip | Why It Matters |
|-----|----------------|
| `temperature=0.0` | Maximum consistency for classification |
| List exact category names | Prevents model from creating new categories |
| "Output ONLY the category name" | Eliminates explanation boilerplate |
| Provide 3-5 clear categories | More categories = more confusion |

---

## Use Case 4: Code Generation from Examples

**Business Context:** Infer transformation logic from input-output examples. Useful for data transformation rules, business logic inference, and test generation.

### Pattern: Learn a Function from Examples

```python
examples = """
input = [10, 5, 8]  -> output = 23
input = [5, 2, 4]   -> output = 11
input = [2, 1, 3]   -> output = 6
input = [8, 4, 6]   -> output = 18
"""

# Note: The pattern here is sum(list) + min(list)
# 10+5+8=23, 5+2+4=11, 2+1+3=6, 8+4+6=18

prompt = f"""
You are given several input-output examples. Your task is to infer the underlying 
Python function that maps inputs to outputs.

Examples:
{examples}

Requirements:
- Identify the transformation pattern
- Write a Python function that implements this rule
- Return ONLY the function code
- Do NOT include explanations, comments, or extra text

Function signature:
def transform(input_list):
"""

response = get_response(prompt, temperature=0.2, max_tokens=100)
print(response)
```

**Expected output:**
```python
def transform(input_list):
    return sum(input_list) + min(input_list)
```

### When to Use This Pattern

```
✅ Data transformation rules from business logic
✅ Test case generation (input → expected output)
✅ Reverse-engineering undocumented transformations
✅ Generating regex or validation rules from examples
```

---

## Business Application Matrix

| Task | Temperature | max_tokens | Key Prompt Constraint |
|------|-------------|-----------|----------------------|
| Summarization | 0.3 | 200–400 | "no more than N bullets" |
| Tone transformation | 0.5–0.7 | 300–500 | "maintain original meaning" |
| Classification | 0.0 | 5–20 | "output ONLY the category" |
| Code generation | 0.2 | 200–400 | "return ONLY the code" |
| Data extraction | 0.0 | 100–300 | "return valid JSON only" |

---

## Common Prompt Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| No output constraint | Model writes paragraphs for a one-word answer | "Output ONLY the category name" |
| Vague instructions | Model makes up its own rules | Be explicit about format and constraints |
| High temperature for classification | Inconsistent labels across runs | Use `temperature=0.0` for deterministic tasks |
| Missing negative constraint | Model adds explanations and caveats | "Do NOT include explanations" |

---

## Putting It All Together: A Business Pipeline

```python
from openai import OpenAI
import json

client = OpenAI(api_key="<YOUR_KEY>")

def process_customer_feedback(feedback_text: str) -> dict:
    """
    Complete pipeline: Analyze feedback, classify it, and generate a summary.
    """

    # Step 1: Classify
    classify_prompt = f"""
    Classify this feedback: "{feedback_text}"
    Category options: Positive, Negative, Feature Request, Bug Report
    Output ONLY the category.
    """
    category = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": classify_prompt}],
        temperature=0.0, max_tokens=10
    ).choices[0].message.content.strip()

    # Step 2: Summarize
    summarize_prompt = f"""
    Summarize this customer feedback in one sentence: "{feedback_text}"
    """
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summarize_prompt}],
        temperature=0.3, max_tokens=60
    ).choices[0].message.content.strip()

    # Step 3: Generate response
    respond_prompt = f"""
    Write a professional, empathetic response to this customer feedback:
    "{feedback_text}"
    Keep it to 2-3 sentences. Be specific to their feedback.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": respond_prompt}],
        temperature=0.7, max_tokens=100
    ).choices[0].message.content.strip()

    return {
        "original": feedback_text,
        "category": category,
        "summary": summary,
        "suggested_response": response
    }


# Test it
result = process_customer_feedback(
    "The app crashes every time I try to upload a photo. Very frustrating!"
)
for key, value in result.items():
    print(f"{key}: {value}\n")
```
