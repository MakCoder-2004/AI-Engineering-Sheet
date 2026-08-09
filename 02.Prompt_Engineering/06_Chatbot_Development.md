# Prompt Engineering for Chatbot Development

## Overview

Building effective chatbots requires more than just hooking up an API. You need to define the chatbot's **purpose**, **audience**, **tone**, and **behavior rules** through carefully crafted system prompts. This document covers the key patterns for building robust, production-ready chatbots.

---

## The Chatbot Prompt Architecture

A production chatbot system prompt should define three layers:

```
System Prompt
│
├── 1. Purpose           — What the chatbot is and does
├── 2. Audience          — Who it talks to and how they think
└── 3. Tone / Behavior   — How it communicates
```

---

## Pattern 1: Dual-Prompt `get_response()` Function

The standard chatbot helper function accepts both a **system prompt** (fixed) and a **user message** (dynamic):

```python
from openai import OpenAI

client = OpenAI("<YOUR_API_KEY>")

def get_response(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """
    Standard chatbot response function.
    
    Args:
        system_prompt:  The chatbot's role, rules, and behavior instructions
        user_message:   The user's input
        temperature:    Response creativity (0=deterministic, 1=creative)
    
    Returns:
        The assistant's response as a string
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=temperature,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
```

---

## Pattern 2: Composing the System Prompt

Break the system prompt into **named components** for easier maintenance:

```python
# Component 1: Define the chatbot's purpose
chatbot_purpose = """
You are a customer support chatbot for an electronics e-commerce platform.
Your main role is to assist customers with inquiries related to electronic gadgets,
including troubleshooting, product information, order support, and general guidance.
"""

# Component 2: Define the target audience
audience_guidelines = """
The target audience consists of tech-savvy individuals who are interested in purchasing
and using electronic gadgets such as smartphones, headphones, laptops, and accessories.
"""

# Component 3: Define tone and communication style
tone_guidelines = """
Always respond in a professional, clear, and user-friendly tone.
Be helpful, concise, and solution-oriented while maintaining a polite and supportive attitude.
"""

# Combine all components into one system prompt
system_prompt = chatbot_purpose + audience_guidelines + tone_guidelines

# Use it
response = get_response(
    system_prompt,
    "My new headphones aren't connecting to my device"
)
print(response)
```

---

## Pattern 3: Role-Playing Prompts

Role-playing prompts give the chatbot a **persona** and explicit behavioral rules. This is especially powerful for expert advisors.

```python
# Base persona
base_system_prompt = (
    "Act as a learning advisor who receives queries from users mentioning their "
    "background, experience, and goals, and accordingly provides a response that "
    "recommends a tailored learning path of textbooks, including both beginner-level "
    "and more advanced options."
)

# Behavioral rules: What to do when info is missing
behavior_guidelines = """
Behavior guidelines:
- If the user does NOT provide information about their background, experience level, 
  or learning goals, ask follow-up questions to obtain this information before 
  making recommendations.
- Always try to understand the user's current level (beginner, intermediate, advanced) 
  and objectives.
- Be proactive in clarifying missing details to give more personalized advice.
"""

# Response format rules
response_guidelines = """
Response guidelines:
- Recommend no more than three textbooks in total.
- Ensure recommendations are relevant to the user's stated or inferred level.
- If possible, include a mix of beginner and more advanced resources.
"""

system_prompt = base_system_prompt + behavior_guidelines + response_guidelines

user_prompt = "Hey, I'm looking for courses on Python and data visualization. What do you recommend?"
response = get_response(system_prompt, user_prompt)
print(response)
```

---

## Pattern 4: Incorporating External Context (RAG-lite)

Inject specific, real facts into the chatbot's context so it can answer domain-specific questions accurately — without hallucinating.

```python
# System prompt defines persona
system_prompt = """
You are a helpful and gentle customer support chatbot for a delivery service.
Your purpose is to clearly and politely answer user questions about delivery services.
Always respond in a calm, friendly, and reassuring tone.
"""

# Known fact from your knowledge base
context_question = "What types of items can be delivered using MyPersonalDelivery?"
context_answer = (
    "We deliver everything from everyday essentials such as groceries, medications, "
    "and documents to larger items like electronics, clothing, and furniture. "
    "However, please note that we currently do not offer delivery for hazardous "
    "materials or extremely fragile items requiring special handling."
)

# Inject the known Q&A as prior context, then ask a follow-up question
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": context_question},     # Seed question
        {"role": "assistant", "content": context_answer},       # Pre-filled answer
        {"role": "user",      "content": "Do you deliver furniture?"}  # Real user question
    ]
)

print(response.choices[0].message.content)
# The chatbot answers confidently based on the injected fact
```

### How Context Injection Works

```
┌─────────────────────────────────────────────────┐
│ System: "You are a support bot..."              │  ← Fixed persona
├─────────────────────────────────────────────────┤
│ User:      "What types of items can..."         │  ← Seeded question
│ Assistant: "We deliver groceries, documents..." │  ← Known fact injected
├─────────────────────────────────────────────────┤
│ User: "Do you deliver furniture?"               │  ← Real user query
│ Assistant: "Yes, we do deliver furniture..."    │  ← Grounded in facts above
└─────────────────────────────────────────────────┘
```

> 💡 This technique is a lightweight version of **Retrieval-Augmented Generation (RAG)**. Instead of searching a database, you pre-populate the conversation with relevant facts.

---

## Full Multi-Turn Chatbot Implementation

Combining all patterns for a production-ready chatbot:

```python
from openai import OpenAI

client = OpenAI(api_key="<YOUR_API_KEY>")

class Chatbot:
    """
    A configurable chatbot with persistent conversation history.
    """

    def __init__(
        self,
        purpose: str,
        audience: str,
        tone: str,
        behavior_rules: str = "",
        model: str = "gpt-4o-mini"
    ):
        self.model = model
        self.system_prompt = purpose + audience + tone + behavior_rules
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self, user_message: str, temperature: float = 0.7) -> str:
        """Process a user message and return the chatbot's response."""
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Get response
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=temperature,
            max_tokens=300
        )

        reply = response.choices[0].message.content.strip()

        # Store reply in history
        self.history.append({"role": "assistant", "content": reply})

        return reply

    def inject_context(self, question: str, answer: str):
        """Inject a known Q&A pair as prior context."""
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

    def reset(self):
        """Reset conversation while preserving system prompt."""
        self.history = [self.history[0]]


# --- Example: Electronics Support Chatbot ---

bot = Chatbot(
    purpose=(
        "You are a customer support chatbot for TechStore, an electronics retailer. "
        "Help customers with product questions, troubleshooting, and order support."
    ),
    audience=(
        "Customers range from tech beginners to enthusiasts. Adjust your language "
        "based on their apparent technical level."
    ),
    tone=(
        "Be professional, friendly, and concise. Always offer a next step or solution. "
        "If you can't help, direct them to human support."
    ),
    behavior_rules=(
        "Never make up product specifications. If you don't know, say so clearly. "
        "Always offer to escalate to a human agent for complex issues."
    )
)

# Inject known facts
bot.inject_context(
    question="What's your return policy?",
    answer="TechStore offers a 30-day return policy on all items. Electronics must be unopened or defective."
)

# Start chatting
print(bot.chat("My Bluetooth headphones keep dropping the connection."))
print(bot.chat("I've already tried that. Is there anything else?"))
```

---

## System Prompt Checklist

Use this checklist when building a new chatbot:

- [ ] **Purpose**: What is this chatbot for? What can it do?
- [ ] **Scope**: What is explicitly OUT OF SCOPE? (prevent over-answering)
- [ ] **Audience**: Who is the user? What is their background/level?
- [ ] **Tone**: Formal, friendly, casual, technical?
- [ ] **Behavior rules**: What to do when info is missing? When to escalate?
- [ ] **Response format**: Should answers be concise? Detailed? Numbered?
- [ ] **Domain facts**: Inject key facts the model must know

---

## Common Chatbot Patterns Reference

| Pattern | Use Case | Key Technique |
|---------|---------|---------------|
| Purpose + Audience + Tone | General assistant | Compose system prompt from parts |
| Role-playing persona | Expert advisor | "Act as a [role] who..." |
| Context injection | Domain-specific Q&A | Pre-fill assistant messages with known facts |
| Behavior rules | Guard rails | Explicit "If X then Y" rules in system prompt |
| Multi-turn history | Conversation memory | Append each turn to message history |
| Ask before recommending | Missing info guard | "If user hasn't provided X, ask first" |

---

## Tone Examples

| Tone | System Prompt Addition | Good For |
|------|----------------------|----------|
| Formal | "Maintain formal, precise language at all times." | Legal, finance |
| Friendly | "Be warm, encouraging, and use simple language." | Education, retail |
| Expert | "Communicate at an advanced technical level." | Developer tools |
| Empathetic | "Acknowledge emotions before providing solutions." | Healthcare, support |
| Concise | "Never use more than 3 sentences per response." | Mobile apps |
