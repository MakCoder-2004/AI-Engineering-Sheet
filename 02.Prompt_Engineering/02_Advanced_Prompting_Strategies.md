# Advanced Prompt Engineering Strategies

## Overview

Beyond basic prompting, advanced techniques dramatically improve the quality, reasoning, and consistency of LLM responses. This document covers six key strategies with code examples, explanations, and when to use each.

---

## Strategy Overview

| Strategy | Best For | Example Task |
|----------|----------|--------------|
| [One-Shot](#1-one-shot-prompting) | Pattern-following with one example | Extract odd numbers |
| [Few-Shot](#2-few-shot-prompting) | Classification with labeled examples | Sentiment analysis |
| [Multi-Step](#3-multi-step-prompting) | Tasks requiring multiple checks | Code review |
| [Chain-of-Thought](#4-chain-of-thought-cot-prompting) | Logical/mathematical reasoning | Math word problems |
| [One-Shot CoT](#5-one-shot-chain-of-thought) | Reasoning with one worked example | Sum of even numbers |
| [Self-Consistency](#6-self-consistency-prompting) | High-stakes decisions | Inventory calculation |

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

## 1. One-Shot Prompting

> **Idea:** Provide **one example** so the model learns the desired pattern before answering.

### When to Use
- The task has a specific format or style you want replicated
- Zero-shot doesn't produce the right structure

### How It Works
```
Prompt:
  Example: [example input] → [example output]
  Now do the same for: [your actual input]
```

### Code Example

```python
prompt = """
Extract the odd numbers from the following set.

Example:
Input: {1, 3, 7, 12, 19}
Output: {1, 3, 7, 19}

Now do the same for the following set:
Input: {3, 5, 11, 12, 16}
Output:
"""

response = get_response(prompt)
print(response)  # → {3, 5, 11}
```

---

## 2. Few-Shot Prompting

> **Idea:** Provide **multiple labeled examples** so the model learns a classification or transformation behavior.

### When to Use
- Classification tasks (sentiment, intent, category)
- When you want output in a very specific format
- When zero-shot or one-shot isn't reliable enough

### How It Works
Use `user`/`assistant` turns to simulate labeled training examples before the real query.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        # Example 1: Positive → label 1
        {"role": "user",      "content": "The product quality exceeded my expectations"},
        {"role": "assistant", "content": "1"},

        # Example 2: Negative → label -1
        {"role": "user",      "content": "I had a terrible experience with this product's customer service"},
        {"role": "assistant", "content": "-1"},

        # Actual query to classify
        {"role": "user",      "content": "The price of the product is really fair given its features"}
    ],
    temperature=0
)

print(response.choices[0].message.content)  # → 1 (positive)
```

### Comparison: Zero-Shot vs Few-Shot

| Approach | Behavior | Reliability |
|----------|----------|-------------|
| Zero-shot | Model uses general knowledge | Variable |
| One-shot | Model mimics one example | Better |
| Few-shot (3-5 examples) | Model learns the pattern | Most consistent |

---

## 3. Multi-Step Prompting

> **Idea:** Force the model to check multiple conditions or perform multiple steps **sequentially** before answering.

### When to Use
- Code review, validation tasks
- When you need the model to verify several criteria
- When you want to prevent the model from jumping to conclusions

### Code Example

```python
code = '''
def calculate_rectangle_area(length, width):
    area = length * width
    return area
'''

prompt = f"""
Analyze the following Python function:

{code}

Check each of the following in order:
1. Is the syntax correct? (Yes/No + explanation)
2. How many input parameters does it take?
3. How many values does it return?
4. Is there any potential issue with the logic?
5. Overall verdict: Is this function correct?
"""

response = get_response(prompt, temperature=0.2, max_tokens=300)
print(response)
```

---

## 4. Chain-of-Thought (CoT) Prompting

> **Idea:** Encourage step-by-step reasoning by explicitly asking the model to show its work. This dramatically improves accuracy on multi-step logical tasks.

### When to Use
- Math word problems
- Logical deduction
- Any task where reasoning steps matter
- When you need the model to not make "leaps" in logic

### The Core Insight
Without CoT:
```
Q: If John is 20, his father is twice his age, how old will his father be in 10 years?
A: 50
```
With CoT:
```
Step 1: John is 20 years old
Step 2: Father's current age = 20 × 2 = 40
Step 3: Father's age in 10 years = 40 + 10 = 50
Answer: 50
```

### Code Example

```python
prompt = """
A friend is 20 years old. Their father is twice their age.

Calculate the father's age in 10 years.

Show your reasoning step-by-step:
1. Current age of the friend
2. Current age of the father
3. Father's age in 10 years
4. Final answer
"""

response = get_response(prompt, temperature=0.2)
print(response)
```

### When NOT to Use CoT
- Simple factual questions ("What is the capital of France?")
- Classification where reasoning adds noise
- When token cost is a concern

---

## 5. One-Shot Chain-of-Thought

> **Idea:** Combine few-shot learning with chain-of-thought reasoning. Provide **one worked example with full reasoning**, then ask the model to solve a new problem.

### When to Use
- Math or logic tasks where the reasoning pattern is transferable
- When you want to "teach" the model your preferred reasoning style

### Code Example

```python
# Provide ONE fully worked example with reasoning
example = """Q: Sum the even numbers in {9, 10, 13, 4, 2}.
A: Even numbers are {10, 4, 2}. Sum = 16 (10 + 4 + 2)"""

# Ask a new question in the same format
question = """Q: Sum the even numbers in {15, 13, 82, 7, 14}.
A:"""

prompt = example + "\n\n" + question

response = get_response(prompt, temperature=0.2)
print(response)
# → Even numbers are {82, 14}. Sum = 96 (82 + 14)
```

---

## 6. Self-Consistency Prompting

> **Idea:** Generate **multiple independent reasoning paths** and take a majority vote. This is like running a committee of experts and picking the consensus answer.

### When to Use
- High-stakes calculations
- When single-pass reasoning gives inconsistent results
- Complex word problems with many steps

### How It Works

```
Prompt: "Solve this problem using three independent experts.
         Each expert solves it independently.
         Compare answers and take majority vote."
```

### Code Example

```python
self_consistency_instruction = """
Solve this problem using three independent experts.

Each expert:
1. Solves independently, step-by-step
2. States their final answer clearly

Then:
- Compare all three answers
- If there's a majority, that's the final answer
- If all three differ, re-examine and pick the most rigorous
- Output the final agreed result
"""

problem = """
Store starts with 50 devices (60% are mobile phones).
3 clients each buy 1 mobile phone.
1 client buys a laptop.
10 laptops and 5 mobile phones are added to stock.

Find the final count of mobile phones and laptops.
"""

prompt = self_consistency_instruction + "\n\nProblem:\n" + problem

response = get_response(prompt, temperature=0.5, max_tokens=400)
print(response)
```

**Expected solution:**
```
Initial: 50 devices total
  → 30 mobile phones (60% of 50), 20 laptops

After sales:
  → Mobile phones: 30 - 3 = 27
  → Laptops: 20 - 1 = 19

After restocking:
  → Mobile phones: 27 + 5 = 32
  → Laptops: 19 + 10 = 29

Final: 32 mobile phones, 29 laptops
```

---

## 7. Iterative Prompt Engineering

> **Idea:** Treat prompt writing like software development — write, test, analyze, refine, repeat.

### The Process

```
┌─────────────────────────┐
│   1. Write initial       │
│      prompt              │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   2. Test with model     │
│      (5-10 runs)         │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   3. Analyze output      │
│      quality             │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   4. Identify weaknesses │
│      (ambiguity, gaps)   │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   5. Refine prompt       │
│      and repeat          │
└─────────────────────────┘
```

### Common Weakness Patterns

| Problem | Symptom | Fix |
|---------|---------|-----|
| Ambiguity | Model interprets differently each run | Add constraints and examples |
| Too broad | Model adds irrelevant content | Add "only respond with X" |
| Too restrictive | Model refuses or gives empty output | Relax constraints |
| Wrong format | Output doesn't match expected shape | Add explicit format example |
| Off-topic | Model goes beyond the task | Add "do not include X" |

---

## Strategy Selection Guide

```
Is the task a classification or pattern-matching task?
  YES → Use Few-Shot Prompting
  NO  ↓

Does the task require multi-step reasoning?
  YES → Use Chain-of-Thought
  NO  ↓

Do you have one good example to show the model?
  YES → Use One-Shot Prompting
  NO  ↓

Is the answer high-stakes or prone to errors?
  YES → Use Self-Consistency
  NO  ↓

Does the task have multiple validation steps?
  YES → Use Multi-Step Prompting
  NO  → Use Zero-Shot with clear instructions
```

---

## Quick Reference

```python
# Zero-shot
prompt = "Task description. Input: {...} Output:"

# One-shot
prompt = "Example: Input→Output\n\nNow: Input:"

# Few-shot (using message history)
messages = [
    {"role": "user", "content": "example_1_input"},
    {"role": "assistant", "content": "label_1"},
    {"role": "user", "content": "example_2_input"},
    {"role": "assistant", "content": "label_2"},
    {"role": "user", "content": "actual_query"},
]

# Chain-of-Thought
prompt = "Problem. Show your work step by step:\n1.\n2.\n3.\nFinal answer:"

# Self-Consistency
prompt = "Solve using 3 independent experts. Majority vote = final answer. Problem: ..."
```
