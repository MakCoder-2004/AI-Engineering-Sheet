# Building Conversation History with the OpenAI API

## Overview

A single-turn request is stateless — the model doesn't remember what was said before. To build **multi-turn conversations** (like a chatbot), you must maintain and pass the full conversation history with each new API call.

---

## How Conversation History Works

```
User: "Explain what pi is."
  → API Call with: [system msg, user msg]
  ← Assistant responds with explanation
  
User: "Can you give me an example?"
  → API Call with: [system msg, user msg, assistant msg, new user msg]
  ← Assistant responds in context
```

> The OpenAI API is **stateless** — you are responsible for maintaining the conversation history and sending it with each request.

---

## Message Roles

| Role | When to Use | Example |
|------|-------------|---------|
| `system` | Set the assistant's behavior once at the start | `"You are a helpful math tutor."` |
| `user` | Human's message in the conversation | `"What is pi?"` |
| `assistant` | Model's previous response (stored and re-sent) | Model's last answer |

---

## Flow Diagram

```
Start
  │
  ▼
Build initial messages list:
  [system_message, first_user_message]
  │
  ▼
Send to API ──► Get response
  │
  ▼
Extract assistant message from response
  │
  ▼
Append assistant message to messages list
  │
  ▼
User sends next message
  │
  ▼
Append new user message to messages list
  │
  ▼
Send full messages list to API ──► ...
```

---

## Basic Implementation

```python
from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

# Start with a system message and first user message
messages = [
    {"role": "system", "content": "You are a helpful math tutor that speaks concisely."},
    {"role": "user",   "content": "Explain what pi is."}
]

# --- Turn 1: Send the first message ---
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_completion_tokens=100
)

# Extract the assistant's response as a message dict
assistant_dict = {
    "role":    "assistant",
    "content": response.choices[0].message.content
}

# Append the assistant's reply to the history
messages.append(assistant_dict)

print("Conversation so far:")
for msg in messages:
    print(f"  [{msg['role']}]: {msg['content'][:60]}...")
```

---

## Full Multi-Turn Conversation Example

```python
from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

def chat(messages: list, user_input: str) -> tuple[str, list]:
    """
    Send a user message and get a response.
    
    Args:
        messages:    The full conversation history so far
        user_input:  The new user message to send
    
    Returns:
        (response_text, updated_messages_list)
    """
    # Add user's new message to history
    messages.append({"role": "user", "content": user_input})

    # Call the API with full conversation context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=200
    )

    # Get the assistant's reply
    reply = response.choices[0].message.content.strip()

    # Add assistant's reply to history
    messages.append({"role": "assistant", "content": reply})

    return reply, messages


# Initialize with system prompt
conversation = [
    {"role": "system", "content": "You are a helpful math tutor that speaks concisely."}
]

# Turn 1
reply, conversation = chat(conversation, "Explain what pi is.")
print(f"User: Explain what pi is.")
print(f"Assistant: {reply}\n")

# Turn 2 — context from Turn 1 is preserved!
reply, conversation = chat(conversation, "Can you give me a real-world example?")
print(f"User: Can you give me a real-world example?")
print(f"Assistant: {reply}\n")

# Turn 3
reply, conversation = chat(conversation, "What is 2 * pi roughly equal to?")
print(f"User: What is 2 * pi roughly equal to?")
print(f"Assistant: {reply}")
```

---

## Real-World Example: AI Math Tutor

This pattern is exactly what you'd use to build an AI tutor application. Here's a more complete implementation:

```python
from openai import OpenAI
import json

client = OpenAI(api_key="<YOUR_API_KEY>")

class MathTutor:
    """
    A stateful AI math tutor using OpenAI's Chat Completions API.
    Maintains full conversation history across turns.
    """

    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": (
                    "You are an AI math tutor for high school students. "
                    "Explain concepts clearly with step-by-step solutions. "
                    "Encourage the student and be patient. "
                    "If a student is confused, try a different approach."
                )
            }
        ]

    def ask(self, question: str) -> str:
        """Send a question and get a response in context."""
        # Add student's question
        self.history.append({"role": "user", "content": question})

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            temperature=0.5,
            max_completion_tokens=300
        )

        answer = response.choices[0].message.content.strip()

        # Store the answer in history
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def reset(self):
        """Clear conversation history (keep system message)."""
        self.history = [self.history[0]]


# Example session
tutor = MathTutor()

questions = [
    "I don't understand quadratic equations.",
    "Can you solve x² - 5x + 6 = 0 step by step?",
    "Wait, what does 'factoring' mean again?"
]

for question in questions:
    print(f"\nStudent: {question}")
    answer = tutor.ask(question)
    print(f"Tutor: {answer}")
```

---

## Managing Long Conversations (Token Limits)

Every model has a **context window** (maximum total tokens). If the conversation grows too long, you need to trim it.

```
Model          Context Window
─────────────────────────────
gpt-4o-mini    128,000 tokens
gpt-4o         128,000 tokens
gpt-3.5-turbo   16,385 tokens
```

### Strategy: Sliding Window

```python
def trim_history(messages: list, max_messages: int = 20) -> list:
    """
    Keep the system message and last N messages to stay within token limits.
    """
    system_msg = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    # Keep only the last max_messages
    trimmed = conversation[-max_messages:]

    return system_msg + trimmed
```

---

## Conversation State: What It Looks Like

After two turns, `messages` looks like this:

```python
[
    {"role": "system",    "content": "You are a helpful math tutor..."},
    {"role": "user",      "content": "Explain what pi is."},
    {"role": "assistant", "content": "Pi (π) is a mathematical constant..."},
    {"role": "user",      "content": "Can you give me a real-world example?"},
    {"role": "assistant", "content": "Sure! Pi is used when calculating..."},
]
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always include the system message | Defines assistant behavior for the whole conversation |
| Store the full assistant response | Needed for accurate context in follow-up turns |
| Trim history when approaching limits | Prevents token limit errors |
| Track `response.usage.total_tokens` | Monitor costs in long conversations |
| Use `temperature=0.3–0.7` for tutors | Consistent, focused answers |

---

## Summary

```
1. Start with:  [system_msg]
2. User speaks: append {"role": "user",      "content": "..."}
3. API responds: append {"role": "assistant", "content": "..."}
4. Repeat from step 2 with full history
```

The key insight: **the model has no memory** — you carry the memory by sending the full conversation history every time.
