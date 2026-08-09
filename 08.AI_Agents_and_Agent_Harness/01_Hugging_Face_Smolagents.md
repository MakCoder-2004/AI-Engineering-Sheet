# Hugging Face `smolagents`: A Comprehensive Guide

> A complete guide to building AI agents with Hugging Face's `smolagents` framework — covering `CodeAgent`, `ToolCallingAgent`, custom tools, and production-ready agent deployment.

---

## Table of Contents

1. [What Are AI Agents?](#1-what-are-ai-agents)
2. [Introducing smolagents](#2-introducing-smolagents)
3. [Agent Types: CodeAgent vs ToolCallingAgent](#3-agent-types-codeagent-vs-toolcallingagent)
4. [How Function-Calling Works](#4-how-function-calling-works)
5. [How Code Agents Work](#5-how-code-agents-work)
6. [The Code Agent Flow](#6-the-code-agent-flow)
7. [Getting Started with CodeAgent](#7-getting-started-with-codeagent)
8. [Built-in Tools](#8-built-in-tools)
9. [Using Community Tools from Hugging Face Hub](#9-using-community-tools-from-hugging-face-hub)
10. [Why Create Custom Tools?](#10-why-create-custom-tools)
11. [Creating Custom Tools](#11-creating-custom-tools)
12. [Best Practices for Custom Tools](#12-best-practices-for-custom-tools)
13. [Registering Custom Tools with Your Agent](#13-registering-custom-tools-with-your-agent)
14. [Advanced Agent Configuration](#14-advanced-agent-configuration)
15. [The Agent Execution Cycle](#15-the-agent-execution-cycle)
16. [Real-World Use Cases](#16-real-world-use-cases)
17. [Production Deployment](#17-production-deployment)
18. [Troubleshooting & Optimization](#18-troubleshooting--optimization)
19. [Quick Reference](#19-quick-reference)

---

## 1. What Are AI Agents?

An **AI agent** is a system that uses a Large Language Model (LLM) to interact with its environment to achieve a user-defined objective.

### From Chatbots to Agents

| Capability | Chatbots | AI Agents |
|---|---|---|
| Response type | Natural language only | Actions + language |
| Initiative | Passive and reactive | Actively reasons toward a goal |
| Tool usage | None | Web search, file reading, data analysis, API calls |
| Execution | Single response | Iterative cycle of reasoning |

### The Agent Reasoning Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                  The Agent Reasoning Cycle                   │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐         │
│  │  THOUGHT │───▶│    ACTION    │───▶│OBSERVATION│         │
│  │          │    │              │    │           │         │
│  │"I need   │    │"Call search_ │    │"Found 3   │         │
│  │ to search│    │ company()"   │    │ results"  │         │
│  └──────────┘    └──────────────┘    └───────────┘         │
│        ▲                                     │              │
│        └─────────────────────────────────────┘              │
│                   (Repeat until goal met)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Introducing smolagents

`smolagents` is a **lightweight Python framework** developed by Hugging Face for building AI agents with minimal overhead.

### Key Features

| Feature | Description |
|---|---|
| Lightweight | Minimal dependencies, fast startup |
| Two agent types | `CodeAgent` and `ToolCallingAgent` |
| Hugging Face integration | Seamless Hub integration |
| Tool ecosystem | Built-in, community, and custom tools |
| Multi-model support | Works with any Hugging Face inference endpoint |

### Installation

```bash
pip install smolagents
```

### Verification

```python
import smolagents
print(smolagents.__version__)
```

---

## 3. Agent Types: CodeAgent vs ToolCallingAgent

`smolagents` supports two distinct agent architectures optimized for different use cases.

### Comparison Matrix

| Feature | ToolCallingAgent | CodeAgent |
|---|---|---|
| Action format | Structured JSON function calls | Python code execution |
| Complexity handling | Simple, linear actions | Complex, multi-step reasoning |
| State management | Explicit via conversation | Implicit via code variables |
| Debugging | Easy to trace JSON logs | Standard Python debugging |
| Performance overhead | Low (per-call parsing) | Moderate (code execution) |
| Best for | Sequential tool chains | Analytical workflows, data processing |

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    smolagents Architecture                     │
│                                                               │
│              ┌──────────────────────────────┐                │
│              │       User Input/Task         │                │
│              └──────────────┬───────────────┘                │
│                             │                                 │
│             ┌───────────────┴───────────────┐                │
│             ▼                               ▼                │
│  ┌─────────────────────┐     ┌─────────────────────┐        │
│  │   ToolCallingAgent   │     │     CodeAgent        │        │
│  │                      │     │                      │        │
│  │  LLM → JSON          │     │  LLM → Python code   │        │
│  │  {"tool": "x",       │     │  block               │        │
│  │   "params": {...}}   │     │                      │        │
│  │         │            │     │         │            │        │
│  │         ▼            │     │         ▼            │        │
│  │  Execute Tool        │     │  exec() code         │        │
│  └─────────────────────┘     └─────────────────────┘        │
│             │                               │                 │
│             └───────────────┬───────────────┘                │
│                             ▼                                 │
│              ┌──────────────────────────────┐                │
│              │        Final Response         │                │
│              └──────────────────────────────┘                │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. How Function-Calling Works

`ToolCallingAgent` uses structured JSON to invoke tools, making it predictable and easy to debug.

### The JSON Tool Call Flow

```
┌────────────┐  ┌────────────┐  ┌─────────────────┐  ┌─────────────┐
│ Developer  │─▶│   Agent    │─▶│   JSON Call      │─▶│    Tool     │
│            │  │            │  │                  │  │  Execution  │
│ Defines    │  │ Selects    │  │ {"tool":          │  │             │
│ Tools      │  │ Tool       │  │  "search",        │  │             │
│            │  │            │  │  "query": "..."}  │  │             │
└────────────┘  └────────────┘  └─────────────────┘  └─────────────┘
```

### Real-World Example: Competitor Analysis

```json
Action 1: {"tool": "search_company", "company": "Competitor A"}
Action 2: {"tool": "get_pricing",    "company": "Competitor A", "plan": "Basic"}
Action 3: {"tool": "get_pricing",    "company": "Competitor A", "plan": "Pro"}
Action 4: {"tool": "search_company", "company": "Competitor B"}
Action 5: {"tool": "get_pricing",    "company": "Competitor B", "plan": "Basic"}
```

### JSON Schema for Tool Definitions

```python
tool_schema = {
    "name": "search_company",
    "description": "Search for company information by name",
    "parameters": {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "The exact company name to search for"
            }
        },
        "required": ["company"]
    }
}
```

---

## 5. How Code Agents Work

`CodeAgent` writes and executes Python code, enabling complex multi-step reasoning in a single action.

### Code Agent Execution Example

```python
# The agent generates and executes this code:
competitors = ["Competitor A", "Competitor B", "Competitor C"]
pricing_data = {}

for company in competitors:
    company_info = search_company(company)
    plans = extract_pricing_plans(company_info)
    pricing_data[company] = plans

most_affordable_option = min(
    pricing_data,
    key=lambda x: pricing_data[x]['basic_plan']
)
```

### Why Code Agents Are Powerful

| Advantage | Explanation |
|---|---|
| Composition | Chain multiple operations without multiple LLM calls |
| Control flow | Use loops, conditionals, and functions |
| State | Variables persist across operations |
| Libraries | Leverage Python's ecosystem (`pandas`, `numpy`, etc.) |
| Efficiency | One LLM call for complex workflows |

---

## 6. The Code Agent Flow

```
┌───────────────────────────────────────────────────────────────┐
│                   Code Agent Execution Flow                    │
│                                                               │
│  ┌─────────────┐                                             │
│  │ INPUT TASK  │  "Calculate the average sales per region..."│
│  └──────┬──────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────┐                                             │
│  │  PYTHON     │  def analyze_sales(data):                   │
│  │  CODE       │      regions = data['region'].unique()      │
│  │  GENERATION │      averages = {}                          │
│  │             │      for region in regions:                 │
│  │             │          avg = data[data['region']==region] │
│  │             │                    ['sales'].mean()         │
│  │             │          averages[region] = avg             │
│  │             │      return averages                        │
│  └──────┬──────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────┐                                             │
│  │    CODE     │  Safe execution environment                 │
│  │  EXECUTION  │  with permission controls                   │
│  └──────┬──────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────┐                                             │
│  │  REASONING  │  "The average for North is $42,000,         │
│  │             │   South is $38,000..."                      │
│  └──────┬──────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌─────────────┐                                             │
│  │    TASK     │  Final response delivered to user           │
│  │  COMPLETED  │                                             │
│  └─────────────┘                                             │
└───────────────────────────────────────────────────────────────┘
```

---

## 7. Getting Started with CodeAgent

### Basic CodeAgent (No Tools)

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize the model
model = InferenceClientModel()

# Create agent with no tools (pure Python reasoning)
agent = CodeAgent(
    tools=[],
    model=model
)

# Run a task
result = agent.run("Calculate the average of the list [23, 45, 67, 89]")
print(result)
```

### CodeAgent with Web Search Tool

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[WebSearchTool()]
)

result = agent.run("What is the current stock price of Tesla?")
```

### Using Different Models

```python
from smolagents import CodeAgent, HfApiModel, TransformersModel

# Option 1: Hugging Face Inference API
model = HfApiModel(model_id="meta-llama/Llama-3.2-3B")

# Option 2: Local transformers model
model = TransformersModel(model_id="microsoft/phi-2")

# Option 3: Custom endpoint
model = InferenceClientModel(
    base_url="http://localhost:8080",
    api_key="your-key"
)

agent = CodeAgent(tools=[], model=model)
```

---

## 8. Built-in Tools

`smolagents` comes with a rich set of pre-built tools organized by category.

### Tool Categories

| Category | Tools | Use Case |
|---|---|---|
| Information Retrieval | `ApiWebSearchTool`, `DuckDuckGoSearchTool`, `GoogleSearchTool`, `WebSearchTool`, `WikipediaSearchTool` | Search the web, APIs, or knowledge bases |
| Web Interaction | `VisitWebpageTool` | Scrape and interact with web pages |
| Code Execution | `PythonInterpreterTool` | Execute Python code safely |
| User Interaction | `UserInputTool` | Ask user for input during execution |
| Speech Processing | `SpeechToTextTool` | Convert audio to text |
| Workflow Control | `FinalAnswerTool` | Signal task completion |

### Tool Usage Examples

```python
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    UserInputTool
)

agent = CodeAgent(
    model=HfApiModel(),
    tools=[
        DuckDuckGoSearchTool(),   # Web search
        VisitWebpageTool(),       # Read web pages
        WikipediaSearchTool(),    # Wikipedia lookup
        UserInputTool()           # Ask user for clarification
    ]
)

agent.run("Research the history of artificial intelligence and summarize key milestones")
```

### Tool Parameters Reference

```python
# WebSearchTool options
WebSearchTool(
    max_results=5,
    safe_search=True,
    timeout=30
)

# VisitWebpageTool options
VisitWebpageTool(
    user_agent="smolagents-bot",
    timeout=30,
    extract_text_only=True
)
```

---

## 9. Using Community Tools from Hugging Face Hub

The Hugging Face Hub hosts thousands of community-contributed tools that you can load and use instantly.

### Loading a Remote Tool

```python
from smolagents import load_tool, CodeAgent, HfApiModel

# Load a tool from the Hub
model_downloads_tool = load_tool(
    "huggingface-tools/hf-model-downloads",
    trust_remote_code=True
)

# Create agent with remote + built-in tools
agent = CodeAgent(
    tools=[model_downloads_tool, WebSearchTool()],
    model=HfApiModel()
)

agent.run("Find the most downloaded image classification model on Hugging Face")
```

### Popular Community Tools

| Tool ID | Description | Category |
|---|---|---|
| `huggingface-tools/sentiment-analysis` | Text sentiment classification | NLP |
| `huggingface-tools/text-to-image` | Generate images from text | Generation |
| `huggingface-tools/image-segmentation` | Segment objects in images | Vision |
| `huggingface-tools/summarization` | Text summarization | NLP |
| `huggingface-tools/translation` | Language translation | NLP |

### Using Tool Collections

```python
from smolagents import ToolCollection

# Load multiple tools at once
tools = ToolCollection.from_hub(
    "huggingface-tools/nlp-toolkit",
    trust_remote_code=True
)

agent = CodeAgent(tools=tools.tools, model=HfApiModel())
```

---

## 10. Why Create Custom Tools?

If code agents can already write Python and use built-in tools, why create custom tools?

### The Advantages of Custom Tools

| Benefit | Description |
|---|---|
| **Reliability** | Write and test logic explicitly instead of relying on the agent to guess |
| **Reusability** | Use tools across multiple projects and agents |
| **Consistency** | Predictable behavior across runs (essential for debugging) |
| **Controlled Access** | Expose only what you want: specific files, API endpoints, database queries, cloud resources |

### When to Create Custom Tools

| Scenario | Built-in Solution | Custom Tool Need |
|---|---|---|
| Database access | None | Wrap database connection |
| Internal API | None | Authenticate and query |
| Proprietary logic | Could be guessed | Explicit implementation |
| File system access | Limited | Controlled file operations |
| Business rules | Might hallucinate | Enforced logic |

---

## 11. Creating Custom Tools

### Scenario: Retail Store Inventory

You have inventory data in a CSV file with columns: `product_name`, `size`, `color`, `quantity`, `price`.

```python
from smolagents import tool
import pandas as pd

@tool
def check_inventory(product_name: str) -> int:
    """
    Check the available quantity of a product in the inventory CSV.

    Args:
        product_name (str): The name of the product to look up.

    Returns:
        int: The quantity in stock. Returns 0 if the product is not found.
    """
    df = pd.read_csv("store_inventory.csv")
    match = df[df["product_name"] == product_name]
    stock_quantity = int(match.iloc[0]["quantity"]) if not match.empty else 0
    return stock_quantity
```

### Advanced Custom Tool with Multiple Parameters

```python
from smolagents import tool
import pandas as pd

@tool
def get_product_details(
    product_name: str,
    attribute: str = "quantity"
) -> dict:
    """
    Get detailed information about a product from inventory.

    Args:
        product_name (str): The product name to look up.
        attribute (str): What to return: 'quantity', 'price', 'size', or 'all'

    Returns:
        dict: Dictionary containing requested product information.
    """
    df = pd.read_csv("store_inventory.csv")
    match = df[df["product_name"] == product_name]

    if match.empty:
        return {"error": f"Product '{product_name}' not found"}

    product = match.iloc[0]

    if attribute == "all":
        return {
            "product_name": product["product_name"],
            "quantity": int(product["quantity"]),
            "price": float(product["price"]),
            "size": product["size"],
            "color": product["color"]
        }

    return {attribute: product[attribute]}
```

### Custom Tool with External API

```python
from smolagents import tool
import requests
import os

@tool
def get_weather(city: str, units: str = "metric") -> dict:
    """
    Get current weather for a city using OpenWeatherMap API.

    Args:
        city (str): City name (e.g., "London", "Tokyo")
        units (str): 'metric' for Celsius or 'imperial' for Fahrenheit

    Returns:
        dict: Weather information including temperature and conditions.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "units": units, "appid": api_key}

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    return {
        "city": city,
        "temperature": data["main"]["temp"],
        "condition": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"]
    }
```

---

## 12. Best Practices for Custom Tools

### The Anatomy of a Well-Formed Tool

```python
from smolagents import tool
from typing import Optional

@tool                                    # 1. @tool decorator
def check_inventory(
    product_name: str,                   # 2. Type hints for all parameters
    size: Optional[str] = None           # 3. Default values for optional params
) -> int:                                # 4. Return type hint
    """
    Check the available quantity of a product in the inventory CSV.

    Args:                                # 5. Clear docstring with Args section
        product_name (str): The name of the product to look up.
        size (str, optional): Filter by size (S, M, L, XL)

    Returns:                             # 6. Returns section
        int: The quantity in stock. Returns 0 if product is not found.

    Raises:                              # 7. Document exceptions
        FileNotFoundError: If inventory CSV doesn't exist
    """
    try:                                 # 8. Implementation with error handling
        df = pd.read_csv("store_inventory.csv")
        match = df[df["product_name"] == product_name]

        if size:
            match = match[match["size"] == size]

        return int(match.iloc[0]["quantity"]) if not match.empty else 0
    except FileNotFoundError:
        return 0  # Graceful fallback
```

### Best Practices Checklist

| Practice | Why It Matters |
|---|---|
| Type hints | Enables LLM to understand expected input/output |
| Descriptive docstring | LLM reads this to decide when to use the tool |
| Error handling | Prevents agent from crashing on edge cases |
| Simple signatures | LLM handles 1–3 parameters best |
| Descriptive names | `calculate_shipping_cost` > `calc` |
| Return structured data | Easier for agent to parse than raw strings |
| Idempotent where possible | Safer for retries and debugging |

### Tool Naming Convention

```python
# ✅ Good naming (verb_noun or get_noun)
@tool
def calculate_tax(amount: float) -> float: ...

@tool
def get_customer_email(customer_id: str) -> str: ...

# ❌ Avoid ambiguous names
@tool
def process(data: str) -> str: ...   # Too vague

@tool
def do_it(x: int) -> int: ...        # Meaningless
```

---

## 13. Registering Custom Tools with Your Agent

### Basic Registration

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(
    tools=[check_inventory, get_weather, calculate_shipping],
    model=HfApiModel()
)

agent.run("Do we have any large t-shirts in stock?")
```

### Managing External Dependencies

```python
agent = CodeAgent(
    tools=[check_inventory],
    model=HfApiModel(),
    additional_authorized_imports=[
        "pandas",    # For CSV handling
        "numpy",     # For numerical operations
        "requests",  # For API calls
        "datetime"   # For date handling
    ]
)
```

### Registering Tools with Different Access Levels

```python
# Production agent with restricted tools
production_agent = CodeAgent(
    tools=[
        read_only_database_tool,   # SELECT only
        cache_lookup_tool,         # Read-only cache
        email_notification_tool    # Write but controlled
    ],
    model=HfApiModel(),
    additional_authorized_imports=["pandas"]
)

# Admin agent with full access
admin_agent = CodeAgent(
    tools=[
        read_write_database_tool,
        file_system_tool,
        deployment_tool
    ],
    model=HfApiModel(),
    additional_authorized_imports=["pandas", "psycopg2", "subprocess"]
)
```

---

## 14. Advanced Agent Configuration

### Memory and Context Management

```python
from smolagents import CodeAgent, HfApiModel, ChatMemory

memory = ChatMemory(
    max_messages=50,
    system_prompt="You are a helpful data analysis assistant."
)

agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    memory=memory,
    max_iterations=15,
    verbose=True
)
```

### Custom System Prompts

```python
agent = CodeAgent(
    tools=[check_inventory],
    model=HfApiModel(),
    system_prompt="""
    You are an inventory management assistant for a retail store.

    Guidelines:
    1. Always verify stock levels before suggesting purchases.
    2. When inventory is low (<10 units), recommend reordering.
    3. Use the check_inventory tool for all stock queries.
    4. Format responses as: "Product: X | Stock: Y | Status: Z"
    """
)
```

### Controlling Agent Behavior

```python
agent = CodeAgent(
    tools=[database_tool, email_tool],
    model=HfApiModel(),

    # Execution limits
    max_iterations=10,         # Prevent infinite loops
    max_execution_time=60,     # Timeout in seconds

    # Safety
    safe_mode=True,            # Block dangerous operations

    # Logging
    log_level="INFO",          # DEBUG, INFO, WARNING, ERROR
    log_file="agent_logs.json" # Save conversation history
)
```

---

## 15. The Agent Execution Cycle

### Detailed Flow Diagram

```
User Input: "What's the cheapest laptop in stock?"
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 1: ANALYZE                                              │
│ - Intent: Find cheapest laptop                               │
│ - Required: Inventory data access                            │
│ - Output format: Price comparison                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: MATCH TO TOOL                                        │
│ ✓ check_inventory(product_name, size) → quantity/price       │
│ ✓ get_product_details(product_name)   → full details         │
│                                                              │
│ Selected: check_inventory("laptop")                          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: EXECUTE TOOL                                         │
│ Input:  {"product_name": "laptop", "size": null}             │
│ Output: {                                                    │
│   "laptop_basic": {"price": 599,  "quantity": 12},          │
│   "laptop_pro":   {"price": 1299, "quantity": 5}            │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: FORMULATE RESPONSE                                   │
│ "The cheapest laptop in stock is the Laptop Basic at $599.   │
│  There are 12 units available."                              │
└──────────────────────────────────────────────────────────────┘
```

### Multi-Step Reasoning Example

```python
# Agent internal reasoning trace:
#
# Iteration 1:
#   Thought: Need to find all available laptops first
#   Action:  get_all_products(category="laptop")
#   Result:  Found 5 laptop models
#
# Iteration 2:
#   Thought: Need prices for comparison
#   Action:  get_pricing(models=["Laptop A", "Laptop B", ...])
#   Result:  A=$599, B=$799, C=$1299, D=$899, E=$699
#
# Iteration 3:
#   Thought: Need to check stock availability for cheapest (A)
#   Action:  check_inventory("Laptop A")
#   Result:  12 units in stock
#
# Iteration 4:
#   Thought: Have all required data
#   Action:  FinalAnswer("Laptop A is cheapest at $599, 12 in stock")
```

---

## 16. Real-World Use Cases

### Use Case 1: Customer Support Automation

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_order_status(order_id: str) -> dict:
    """Get current status of a customer order."""
    return {"status": "shipped", "tracking": "USPS123456"}

@tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for an order."""
    return {"refund_id": "REF-123", "amount": 49.99}

@tool
def search_knowledge_base(query: str) -> list:
    """Search internal FAQ/knowledge base."""
    return [{"question": "...", "answer": "..."}]

support_agent = CodeAgent(
    tools=[get_order_status, process_refund, search_knowledge_base],
    model=HfApiModel(),
    system_prompt="You are a customer support agent. Be helpful and polite."
)

support_agent.run("I haven't received my order #12345, can I get a refund?")
```

### Use Case 2: Financial Data Analysis

```python
from smolagents import CodeAgent, HfApiModel, tool
import yfinance as yf
import pandas as pd

@tool
def get_stock_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch historical stock price data."""
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

@tool
def calculate_metrics(data: pd.DataFrame) -> dict:
    """Calculate financial metrics from price data."""
    return {
        "volatility":   data["Close"].pct_change().std(),
        "max_drawdown": (data["Close"] / data["Close"].cummax() - 1).min(),
        "sharpe":       data["Close"].pct_change().mean() / data["Close"].pct_change().std(),
        "current_price":data["Close"].iloc[-1]
    }

@tool
def get_company_news(ticker: str) -> list:
    """Fetch recent news for a company."""
    stock = yf.Ticker(ticker)
    return stock.news[:5]

finance_agent = CodeAgent(
    tools=[get_stock_data, calculate_metrics, get_company_news],
    model=HfApiModel(),
    additional_authorized_imports=["yfinance", "pandas", "numpy"]
)

finance_agent.run("Analyze TSLA stock performance and risks for the past 3 months")
```

### Use Case 3: Database Query Assistant

```python
from smolagents import CodeAgent, HfApiModel, tool
import sqlite3
from typing import List, Dict

@tool
def query_database(sql_query: str) -> List[Dict]:
    """
    Execute a SQL query on the company database.

    Args:
        sql_query (str): SELECT query to execute (no INSERT/UPDATE/DELETE)

    Returns:
        List[Dict]: Query results as list of dictionaries
    """
    conn = sqlite3.connect("company.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Safety: block write operations
    if any(word in sql_query.upper() for word in ["INSERT", "UPDATE", "DELETE", "DROP"]):
        return [{"error": "Write operations are not allowed"}]

    cursor.execute(sql_query)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

@tool
def get_table_schema(table_name: str) -> Dict:
    """Get column information for a database table."""
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
    conn.close()
    return {"table": table_name, "columns": columns}

db_agent = CodeAgent(
    tools=[query_database, get_table_schema],
    model=HfApiModel(),
    system_prompt="You are a SQL expert. Generate safe, efficient queries."
)

db_agent.run("Show me the top 10 customers by total purchase amount")
```

### Use Case 4: Research & Report Generation

```python
from smolagents import CodeAgent, HfApiModel, WebSearchTool, VisitWebpageTool, WikipediaSearchTool

research_agent = CodeAgent(
    tools=[
        WebSearchTool(),
        VisitWebpageTool(),
        WikipediaSearchTool()
    ],
    model=HfApiModel(),
    max_iterations=20,
    system_prompt="""
    You are a research assistant. When generating reports:
    1. Gather information from multiple sources
    2. Cross-verify facts
    3. Cite your sources
    4. Structure output with headings and bullet points
    """
)

research_agent.run("""
Research the current state of AI in healthcare. Include:
- Major players and their technologies
- Regulatory landscape
- Adoption statistics
- Future projections for 2025–2030
""")
```

---

## 17. Production Deployment

### Production Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                 Production Agent Architecture                  │
│                                                               │
│  ┌─────────────┐                                             │
│  │   Client    │                                             │
│  │ (Web/Mobile)│                                             │
│  └──────┬──────┘                                             │
│         │ HTTPS                                               │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         API Gateway (FastAPI / Flask)                 │   │
│  │  - Authentication & Rate Limiting                     │   │
│  │  - Request Validation                                 │   │
│  │  - Load Balancing                                     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Agent Orchestrator                         │   │
│  │  - Agent Pool Management                              │   │
│  │  - Task Queue (Redis / RabbitMQ)                      │   │
│  │  - State Persistence                                  │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         ▼                ▼                ▼                  │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐             │
│  │  Agent 1  │   │  Agent 2  │   │  Agent N  │             │
│  │  (Sales)  │   │ (Support) │   │(Research) │             │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘             │
│        │               │               │                     │
│        └───────────────┼───────────────┘                     │
│                        ▼                                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  PostgreSQL  │ │    Redis    │ │   Object    │           │
│  │   (State)   │ │   (Cache)   │ │   Storage   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└───────────────────────────────────────────────────────────────┘
```

### Deployment: FastAPI Wrapper

```python
# app.py - Production API Server
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
from smolagents import CodeAgent, HfApiModel
import asyncio, uuid
from datetime import datetime

app = FastAPI(title="AI Agent API", version="1.0.0")

# Security
api_key_header = APIKeyHeader(name="X-API-Key")
VALID_API_KEYS = {"prod_key_123", "dev_key_456"}

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Request/Response Models
class AgentRequest(BaseModel):
    task: str
    session_id: Optional[str] = None
    max_iterations: Optional[int] = 10
    timeout_seconds: Optional[int] = 60

class AgentResponse(BaseModel):
    session_id: str
    result: str
    iterations_used: int
    execution_time_ms: float
    timestamp: str

# Agent Pool
class AgentPool:
    def __init__(self, pool_size: int = 5):
        self.pool = []
        self.pool_size = pool_size
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            agent = CodeAgent(
                tools=[check_inventory, get_weather],
                model=HfApiModel(),
                max_iterations=15
            )
            self.pool.append(agent)

    async def execute(self, task: str):
        agent = self.pool.pop()
        try:
            start_time = datetime.now()
            result = agent.run(task)
            exec_time = (datetime.now() - start_time).total_seconds() * 1000
            return result, agent.iteration_count, exec_time
        finally:
            self.pool.append(agent)

agent_pool = AgentPool(pool_size=5)
sessions = {}

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest, api_key: str = Depends(verify_api_key)):
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result, iterations, exec_time = await asyncio.wait_for(
            agent_pool.execute(request.task),
            timeout=request.timeout_seconds
        )

        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({
            "task": request.task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        return AgentResponse(
            session_id=session_id,
            result=result,
            iterations_used=iterations,
            execution_time_ms=exec_time,
            timestamp=datetime.now().isoformat()
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Agent execution timeout")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pool_size": len(agent_pool.pool)}
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# requirements.txt
smolagents>=1.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0
redis>=5.0.0
pandas>=2.0.0
requests>=2.31.0
python-multipart>=0.0.6
```

### Docker Compose for Multi-Service Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HF_API_TOKEN=${HF_API_TOKEN}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agents
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2GB
        reservations:
          memory: 1GB

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agents
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - agent-api

volumes:
  redis-data:
  postgres-data:
```

### Monitoring & Observability

```python
# monitoring.py — Prometheus metrics integration
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

agent_requests = Counter('agent_requests_total', 'Total agent requests', ['agent_type'])
agent_duration = Histogram('agent_duration_seconds', 'Agent execution duration')
agent_errors   = Counter('agent_errors_total', 'Total agent errors', ['error_type'])
active_agents  = Gauge('active_agents', 'Number of active agents')

class MonitoringMiddleware:
    async def __call__(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        agent_requests.labels(agent_type='code_agent').inc()
        agent_duration.observe(duration)

        return response

@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Production Best Practices

| Aspect | Recommendation | Rationale |
|---|---|---|
| State Management | Use Redis for session storage | Fast, distributed, supports TTL |
| Rate Limiting | 100 requests/minute per API key | Prevent abuse and cost spikes |
| Timeouts | 60 seconds max per task | Prevent hung agents |
| Retry Logic | Exponential backoff (3 attempts) | Handle transient failures |
| Logging | Structured JSON logs to ELK | Debugging and audit trails |
| Security | API keys + IP whitelisting | Defense in depth |
| Scaling | Horizontal with Kubernetes | Handle variable load |
| Cost Control | Token budget per session | Prevent runaway costs |

---

## 18. Troubleshooting & Optimization

### Common Issues and Solutions

| Problem | Symptoms | Solution |
|---|---|---|
| Infinite Loops | Agent repeats same actions | Set `max_iterations=15`, add diversity prompt |
| Tool Misuse | Agent calls wrong tool | Improve tool descriptions, add examples |
| Token Overflow | Context length exceeded | Implement sliding window memory |
| Slow Execution | >30 seconds per task | Reduce `max_iterations`, optimize tools |
| Hallucination | False information returned | Add verification tools, lower temperature |
| API Rate Limits | 429 errors | Implement retry with backoff and caching |

### Debugging Tools

```python
import logging

logging.basicConfig(level=logging.DEBUG)

agent = CodeAgent(
    tools=[my_tool],
    model=HfApiModel(),
    verbose=True,       # Print all internal steps
    log_level="DEBUG"
)

# Save execution trace
agent.run("complex task", log_file="trace.json")

# Inspect trace
import json
with open("trace.json") as f:
    trace = json.load(f)
    for step in trace["steps"]:
        print(f"Step {step['iteration']}: {step['thought']}")
        print(f"Action:      {step['action']}")
        print(f"Observation: {step['observation']}\n")
```

### Performance Optimization

```python
# 1. Tool Caching
from functools import lru_cache
from smolagents import tool

@tool
@lru_cache(maxsize=100)
def expensive_api_call(query: str) -> dict:
    """Cached API call for repeated queries."""
    import requests
    response = requests.get(f"https://api.example.com/search?q={query}")
    return response.json()


# 2. Batch Processing
@tool
def batch_process(items: list) -> list:
    """Process multiple items in one call instead of N calls."""
    return [process_item(item) for item in items]


# 3. Async Tools (Python 3.8+)
@tool
async def async_web_search(query: str) -> list:
    """Async web search for better concurrency."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.duckduckgo.com/?q={query}") as resp:
            return await resp.json()
```

### Performance Benchmarking

```python
import time
from statistics import mean, median

def benchmark_agent(agent, tasks, iterations=5):
    """Benchmark agent performance across multiple tasks."""
    results = {}

    for task in tasks:
        times = []
        for _ in range(iterations):
            start = time.time()
            agent.run(task)
            times.append(time.time() - start)

        results[task] = {
            "mean":            mean(times),
            "median":          median(times),
            "min":             min(times),
            "max":             max(times),
            "iterations_used": agent.iteration_count
        }

    return results

# Usage
agent = CodeAgent(tools=[my_tools], model=HfApiModel())
tasks = [
    "What's the weather in London?",
    "Check inventory for product ABC",
    "Calculate average of [1,2,3,4,5]"
]

benchmarks = benchmark_agent(agent, tasks, iterations=3)
for task, stats in benchmarks.items():
    print(f"{task}: {stats['mean']:.2f}s avg")
```

---

## 19. Quick Reference

### One-Liner Agent Creation

```python
from smolagents import CodeAgent, HfApiModel

# Minimal agent
agent = CodeAgent(tools=[], model=HfApiModel())

# With web search
agent = CodeAgent(tools=[WebSearchTool()], model=HfApiModel())

# Production-ready
agent = CodeAgent(
    tools=[custom_tool],
    model=HfApiModel(),
    max_iterations=15,
    verbose=False,
    additional_authorized_imports=["pandas", "numpy"]
)

result = agent.run("Your task here")
```

### Quick Tool Template

```python
from smolagents import tool
from typing import Optional, Dict

@tool
def my_custom_tool(
    required_param: str,
    optional_param: Optional[int] = None
) -> Dict[str, any]:
    """
    Brief description of what the tool does.

    Args:
        required_param (str): Description of required param.
        optional_param (int, optional): Description of optional param.

    Returns:
        Dict: Description of return value.
    """
    # Your implementation here
    return {"result": "value"}
```

### Common Patterns

```python
# Pattern 1: Data Analysis Pipeline
@tool
def analyze_data(df: pd.DataFrame, analysis_type: str) -> dict:
    """Run statistical analysis on a dataframe."""
    return {"mean": df.mean(), "median": df.median(), "std": df.std()}


# Pattern 2: API Wrapper
@tool
def call_external_api(endpoint: str, payload: dict) -> dict:
    """Safely call external API with auth."""
    headers = {"Authorization": f"Bearer {os.getenv('API_TOKEN')}"}
    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


# Pattern 3: File Processor
@tool
def process_file(file_path: str, operation: str) -> any:
    """Safely process files with validation."""
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    if not file_path.endswith(('.csv', '.json', '.txt')):
        return {"error": "Unsupported file type"}
    # Process file...
    return result


# Pattern 4: Database Query
@tool
def query_database(sql: str, params: tuple = ()) -> list:
    """Execute read-only SQL query."""
    if any(word in sql.upper() for word in ['INSERT', 'UPDATE', 'DELETE', 'DROP']):
        return {"error": "Write operations not allowed"}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
```

### Environment Variables Reference

```bash
# Required
export HF_API_TOKEN="your_huggingface_token"

# Optional
export OPENWEATHER_API_KEY="your_weather_api_key"
export REDIS_URL="redis://localhost:6379"
export DATABASE_URL="postgresql://user:pass@localhost/db"
export LOG_LEVEL="INFO"
export AGENT_MAX_ITERATIONS="15"
export AGENT_TIMEOUT_SECONDS="60"
```

### CLI Commands

```bash
# Run agent from command line
python -c "from smolagents import CodeAgent, HfApiModel; \
  agent = CodeAgent(tools=[], model=HfApiModel()); \
  print(agent.run('Your task'))"

# Test tool locally
python -c "from my_tools import check_inventory; print(check_inventory('laptop'))"

# Benchmark performance
python -m smolagents.benchmark --tasks tasks.json --iterations 5

# Validate tool schema
python -m smolagents.validate --tool my_tools.py
```

### Error Codes Quick Reference

| Error Code | Meaning | Solution |
|---|---|---|
| E001 | Tool not found | Check tool registration |
| E002 | Invalid parameters | Verify type hints match |
| E003 | Max iterations exceeded | Increase `max_iterations` or simplify task |
| E004 | Timeout | Increase timeout or optimize code |
| E005 | Unauthorized import | Add to `additional_authorized_imports` |
| E006 | API rate limit | Implement retry with backoff |
| E007 | Model not responding | Check HF API token and network |

---

## Summary

### Key Takeaways

- **Agents > Chatbots**: Agents actively reason and take actions, not just respond.
- **Two Agent Types**: `ToolCallingAgent` for simple sequential tool use; `CodeAgent` for complex, multi-step reasoning with Python.
- **Tool Ecosystem**: Built-in + Community + Custom tools give you full flexibility.
- **Custom Tools**: Provide reliability, reusability, consistency, and controlled access.
- **Production Ready**: Deploy with FastAPI, Docker, Redis, and proper monitoring.

### When to Use smolagents vs Alternatives

| Framework | Best For | Learning Curve |
|---|---|---|
| `smolagents` | Lightweight, Hugging Face ecosystem | Low |
| LangChain | Complex chains, many integrations | Medium |
| AutoGPT | Autonomous, open-ended tasks | Medium |
| Custom | Maximum control, specific requirements | High |

### Getting Help

- [Official Documentation](https://huggingface.co/docs/smolagents)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [GitHub Issues](https://github.com/huggingface/smolagents/issues)
- [Hugging Face Discord](https://discord.gg/huggingface)

---

*This guide covers smolagents as of 2026. For the latest updates, refer to the [official documentation](https://huggingface.co/docs/smolagents).*