
website : https://docs.langchain.com/oss/python/deepagents/customization#skills

Public claude code system prompt : https://gist.github.com/agokrani/919b536246dd272a55157c21d46eda14

# AI Agents — Types, Architectures & Frameworks

## What Is an AI Agent?

An AI agent is an autonomous software system that can perceive its environment, reason, make decisions, and take actions to achieve goals — without step-by-step human instruction. Unlike a simple chatbot that responds to one prompt at a time, an agent can plan multi-step tasks, use tools, and self-correct.


```
User Goal → Agent → [Think → Act → Observe] → ... → Result
                      (loop until task is done)
```

---

## Core Components

Every AI agent is built from four components:

| Component | Description | Example |
|-----------|-------------|---------|
| **LLM (Brain)** | Reasoning engine that interprets tasks and decides actions | GPT-4, Claude, Llama |
| **Memory** | Short-term (conversation) + long-term (persistent knowledge) | Chat history, vector DB |
| **Tools** | External capabilities the agent can invoke | Web search, code execution, APIs |
| **Planning** | Strategy for breaking goals into steps | ReAct loop, task decomposition |

---

## Agent Types (Classic AI)

### 1. Simple Reflex Agent

Acts on **current input only** using if-then rules. No memory, no planning.

```
IF temperature > 30°C → turn on AC
IF email contains "urgent" → flag as high priority
```

- **Pros**: Fast, predictable
- **Cons**: No context, fails in complex environments
- **Example**: Thermostat, spam filter

### 2. Model-Based Reflex Agent

Maintains an **internal model** of the world to handle partially observable environments.

```
Internal state: "Door was locked 5 minutes ago"
Observation: "No unlock event detected"
Action: "Door is still locked" → skip re-checking
```

- **Pros**: Handles incomplete information
- **Cons**: Internal model can become outdated

### 3. Goal-Based Agent

Evaluates actions based on whether they achieve a **desired goal state**.

```
Goal: "Book a flight from Delhi to London under $500"
→ Search flights → Compare prices → Select cheapest → Book
```

- **Pros**: Flexible, can plan ahead
- **Cons**: Slower, needs goal specification

### 4. Utility-Based Agent

Like goal-based, but ranks actions by a **utility function** to pick the best option.

```
Flight A: $450, 1 stop, 14 hours → Utility: 0.72
Flight B: $480, direct, 9 hours  → Utility: 0.89  ← picks this
```

- **Pros**: Optimizes for quality, not just success
- **Cons**: Designing utility functions is hard

### 5. Learning Agent

Improves over time by learning from **feedback and experience**.

```
Action: Recommended Product A → User ignored it
Learning: Reduce weight for that category
Next time: Recommend Product B → User purchased
Learning: Increase weight for this category
```

- **Pros**: Gets better with use
- **Cons**: Needs training data, can learn bad patterns

---

## LLM Agent Architectures

### 1. ReAct (Reasoning + Acting)

The most common LLM agent pattern. Alternates between **Thought → Action → Observation** in a loop.

```
User: What's the weather in Tokyo and should I bring an umbrella?

Thought: I need to check Tokyo's weather forecast.
Action: weather_api("Tokyo")
Observation: Tokyo — 22°C, 80% chance of rain

Thought: High rain probability, I should recommend an umbrella.
Action: respond("It's 22°C in Tokyo with 80% chance of rain. Yes, bring an umbrella.")
```

- **Pros**: Transparent reasoning, auditable, works with tools
- **Cons**: Can loop endlessly, expensive (many LLM calls)
- **Frameworks**: LangChain, LlamaIndex, CrewAI

### 2. Plan-and-Execute

First creates a **full plan**, then executes each step. Separates planning from execution.

```
User: Write a blog post about AI agents and publish it.

Plan:
  1. Research AI agent types
  2. Write outline
  3. Draft blog post
  4. Review and edit
  5. Publish to blog platform

Execute: Step 1 → Step 2 → ... → Step 5
```

- **Pros**: Better for long tasks, can re-plan if a step fails
- **Cons**: Initial plan may be wrong, rigid if environment changes
- **Frameworks**: LangGraph, AutoGen

### 3. ReWOO (Reasoning Without Observation)

Plans all tool calls **upfront** before executing any, then synthesizes results.

```
Plan:
  #E1 = search("population of France")
  #E2 = search("population of Germany")
  #E3 = compare(#E1, #E2)

Execute all, then combine results in one LLM call.
```

- **Pros**: Fewer LLM calls (cheaper, faster), parallelizable
- **Cons**: Can't adapt mid-execution if early results change the plan

### 4. Multi-Agent Systems

Multiple specialized agents **collaborate** on a task.

**Vertical (Manager-Worker):**
```
Manager Agent
  ├── Research Agent → gathers information
  ├── Writer Agent   → drafts content
  └── Reviewer Agent → checks quality
```

**Horizontal (Peer-to-Peer):**
```
Agent A ←→ Agent B ←→ Agent C
(all see shared message thread, contribute equally)
```

- **Pros**: Division of labor, specialized expertise per agent
- **Cons**: Complex coordination, higher cost
- **Frameworks**: CrewAI, AutoGen, LangGraph

### 5. Reflective / Self-Critique Agent

Generates a response, then **critiques and refines** its own output.

```
Draft: "Python is a compiled language..."
Self-critique: "Incorrect — Python is interpreted, not compiled."
Revised: "Python is an interpreted language..."
```

- **Pros**: Self-correcting, higher quality output
- **Cons**: Double the LLM calls, slower
- **Frameworks**: Reflexion, LangGraph

### 6. Tool-Use Agent

An LLM that can **call external tools** (APIs, functions, databases) when needed.

```
User: "What's 847 × 293?"

LLM decides: This needs a calculator.
Tool call: calculator(847, 293)
Result: 248,171
Response: "847 × 293 = 248,171"
```

- **Pros**: Extends LLM beyond text generation, grounded results
- **Cons**: Tool selection errors, security risks
- **Frameworks**: OpenAI function calling, Claude tool use, LangChain tools

---

## Popular Frameworks

| Framework | Type | Language | Best For |
|-----------|------|----------|----------|
| **LangChain** | General agent framework | Python, JS | ReAct agents, tool chains |
| **LangGraph** | Stateful agent graphs | Python | Complex multi-step workflows |
| **CrewAI** | Multi-agent | Python | Role-based agent teams |
| **AutoGen** | Multi-agent (Microsoft) | Python | Conversational agent groups |
| **LlamaIndex** | Data-focused agents | Python | RAG + agent pipelines |
| **Semantic Kernel** | Enterprise agents (Microsoft) | C#, Python | Enterprise AI integration |
| **Haystack** | Pipeline-based | Python | Production RAG + agents |
| **Phidata** | Function-calling agents | Python | Quick tool-use agents |
| **Claude Agent SDK** | Anthropic's agent SDK | Python | Claude-powered agents |
| **OpenAI Agents SDK** | OpenAI's agent SDK | Python | GPT-powered agents |

---

## Comparison Table

| Architecture | LLM Calls | Cost | Flexibility | Best For |
|-------------|-----------|------|-------------|----------|
| ReAct | Many | High | High | General tool-use tasks |
| Plan-and-Execute | Medium | Medium | Medium | Long multi-step tasks |
| ReWOO | Few | Low | Low | Parallelizable tool tasks |
| Multi-Agent | Many | Very High | Very High | Complex collaborative tasks |
| Reflective | Double | High | High | Quality-critical outputs |
| Tool-Use | Few | Low | Medium | Simple tool augmentation |

---

## Choosing the Right Architecture

```
Simple task + tools (search, calculator)?
  └─ Tool-Use Agent or ReAct

Long multi-step workflow?
  └─ Plan-and-Execute

Need to minimize LLM calls / cost?
  └─ ReWOO

Complex task needing specialization?
  └─ Multi-Agent (CrewAI / AutoGen)

Quality-critical, must be accurate?
  └─ Reflective Agent

Just starting out?
  └─ ReAct (simplest, most supported)
```

---

## Key Challenges

1. **Hallucination** — agents can fabricate tool results or reasoning steps
2. **Infinite loops** — ReAct agents may loop without converging
3. **Tool overload** — performance drops beyond 8-10 tools
4. **Cost** — each reasoning step is an LLM call
5. **Security** — tool calls can have real-world side effects (emails, payments)
6. **Error propagation** — early mistakes cascade through the plan

---

## Shallow vs Deep Agents

### 1. Shallow Agents (Basic Agent)

A basic agent is the simplest form of LLM-powered agent with a direct input-output flow.

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌────────┐
│  Input  │ ──► │   LLM   │ ──► │  Tools  │ ──► │ Output │
│         │     │ (Brain) │     │(Serper  │     │        │
│         │     │         │     │ DDG API)│     │        │
└─────────┘     └─────────┘     └─────────┘     └────────┘
```

**Architecture:**
- **Input** → User prompt or task
- **LLM (Brain)** → Reasoning engine (GPT-4, Claude, etc.)
- **Tools** → External APIs (Serper, DuckDuckGo, etc.)
- **Output** → Generated response

**Limitations (Cons):**
- No explicit planning capabilities
- Cannot handle complex queries effectively
- Limited context retention (short-term memory only)
- Single-step reasoning without decomposition
- No self-correction mechanism

---

### 2. Multi-Agent Systems (LangGraph)

For more complex tasks, frameworks like **LangGraph** enable sophisticated multi-agent orchestration.

**Features:**
- Connect multiple AI agents to collaborate on tasks
- Add **Human-in-the-Loop** for approval at key decision points
- Maintain **memory context** across agent interactions
- Stateful workflows with persistent conversation history
- Conditional routing between agents

**Use Cases:**
- Complex workflows requiring multiple specialized agents
- Scenarios needing human oversight
- Long-running tasks with memory requirements

---

### 3. Deep Agents

Deep agents represent the most advanced agent architecture, capable of sophisticated planning, reasoning, and task decomposition. Examples: **Deep Search Agent** in Claude, ChatGPT, Manus AI.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEEP AGENT                               │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │   Planning   │    │           Sub Agents                │   │
│  │    Tool      │    │  ┌─────────┐ ┌─────────┐ ┌───────┐ │   │
│  │              │───►│  │ Agent 1 │ │ Agent 2 │ │  ...  │ │   │
│  │  (Todo List) │    │  └─────────┘ └─────────┘ └───────┘ │   │
│  └──────────────┘    └─────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────────────────────────┐   │
│  │   System     │    │          File System                 │   │
│  │   Prompt     │    │  ┌─────────────────────────────────┐ │   │
│  │              │    │  │   Persistent Memory             │ │   │
│  │              │───►│  │   (Shared across all agents)    │ │   │
│  └──────────────┘    │  └─────────────────────────────────┘ │   │
│                      └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Four Key Properties:**

| Property | Description |
|----------|-------------|
| **1. Planning Tool** | Breaks down complex tasks into a todo list/action plan |
| **2. Sub Agents** | Each todo item is delegated to a specialized sub-agent for execution |
| **3. System Prompt** | Defines agent behavior, constraints, and role definitions |
| **4. File System** | Persistent memory accessible to all sub-agents (shared context, documents, history) |

**How Deep Agents Work:**

1. **Planning** → The agent creates a structured todo list based on the user's goal
2. **Decomposition** → Each todo item is assigned to a dedicated sub-agent
3. **Execution** → Sub-agents work in parallel or sequence, using their specialized capabilities
4. **Memory Sharing** → All sub-agents can access the file system for shared context and information
5. **Synthesis** → Results are combined into a coherent final output

**Advantages over Shallow Agents:**
- Handles complex, multi-step queries
- Maintains long-term memory across sessions
- Self-correcting through planning and reflection
- Specialized expertise through sub-agents
- Scalable for enterprise-level tasks
