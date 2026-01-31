# Prompt Engineering Techniques

## What Is Prompt Engineering?

Prompt engineering is the practice of crafting effective instructions for LLMs to get accurate, relevant, and structured outputs. The same model can give wildly different results depending on how you phrase your prompt.

---

## 1. Zero-Shot Prompting

Ask the model to perform a task **without any examples**.

```
Prompt:
  Classify the sentiment of this review as positive, negative, or neutral.
  Review: "The food was amazing but the service was terrible."

Output:
  Neutral
```

- **When to use**: Simple tasks the model already understands
- **Pros**: No examples needed, quick
- **Cons**: May not follow specific output formats

---

## 2. Few-Shot Prompting

Provide **a few examples** before the actual task so the model learns the pattern.

```
Prompt:
  Classify the sentiment:

  Review: "Loved the ambiance!" → Positive
  Review: "Worst experience ever." → Negative
  Review: "It was okay, nothing special." → Neutral
  Review: "The pasta was delicious but overpriced." →

Output:
  Neutral
```

- **When to use**: When zero-shot gives inconsistent results
- **Pros**: Guides format and reasoning without fine-tuning
- **Cons**: Uses up context window, example quality matters a lot

---

## 3. Chain-of-Thought (CoT) Prompting

Ask the model to **think step by step** before giving the final answer.

```
Prompt:
  Q: A store has 45 apples. They sell 12 in the morning and receive
  a shipment of 30. How many apples do they have now?
  Think step by step.

Output:
  Step 1: Start with 45 apples
  Step 2: Sell 12 → 45 - 12 = 33
  Step 3: Receive 30 → 33 + 30 = 63
  Answer: 63 apples
```

- **When to use**: Math, logic, reasoning, multi-step problems
- **Pros**: Dramatically improves accuracy on reasoning tasks
- **Cons**: Longer responses, higher token usage

---

## 4. Zero-Shot Chain-of-Thought

Simply add **"Let's think step by step"** without providing examples.

```
Prompt:
  If a train travels at 60 km/h for 2.5 hours, how far does it go?
  Let's think step by step.

Output:
  Distance = Speed × Time
  Distance = 60 × 2.5 = 150 km
  The train travels 150 km.
```

- **When to use**: Quick reasoning boost without crafting examples
- **Pros**: One magic phrase improves reasoning significantly
- **Cons**: Less reliable than few-shot CoT on complex problems

---

## 5. Role Prompting (Persona)

Assign a **role or persona** to the model to shape its response style and expertise.

```
Prompt:
  You are a senior Python developer with 15 years of experience.
  Review this code and suggest improvements:

  def add(a, b):
      return a + b

Output:
  The function works but could benefit from:
  1. Type hints: def add(a: float, b: float) -> float
  2. A docstring explaining the parameters
  3. Input validation if used at a system boundary
```

- **When to use**: When you need domain expertise or a specific tone
- **Pros**: Controls style, depth, and perspective
- **Cons**: Model may overfit to the role and add unnecessary complexity

---

## 6. System Prompting

Set persistent instructions via a **system message** that frames all subsequent interactions.

```
System: You are a recipe assistant. Only respond with recipes.
        Format: Recipe name, ingredients list, numbered steps.
        Keep responses under 150 words.

User: I have eggs and cheese.

Output:
  **Cheese Omelette**
  Ingredients: 2 eggs, 50g cheese, salt, pepper, butter
  1. Beat eggs with salt and pepper
  2. Melt butter in a pan over medium heat
  3. Pour eggs, cook until edges set
  4. Add cheese, fold in half
  5. Serve immediately
```

- **When to use**: Chatbots, APIs, any multi-turn application
- **Pros**: Consistent behavior across turns
- **Cons**: Can be overridden by strong user prompts (prompt injection risk)

---

## 7. Tree-of-Thought (ToT) Prompting

Model explores **multiple reasoning paths** and evaluates which is best.

```
Prompt:
  I need a name for a coffee shop that is cozy and book-themed.
  Generate 3 different ideas. For each, evaluate its strengths
  and weaknesses. Then pick the best one.

Output:
  Idea 1: "The Reading Brew" — catchy, clear theme. Might sound generic.
  Idea 2: "Chapter & Cup" — elegant, memorable. Easy to brand.
  Idea 3: "Bookbean Café" — fun, playful. Might not feel premium.

  Best choice: "Chapter & Cup" — memorable, brandable, conveys both themes.
```

- **When to use**: Creative tasks, strategic decisions, complex problem-solving
- **Pros**: Better answers through self-evaluation
- **Cons**: Very high token usage, slower

---

## 8. ReAct (Reasoning + Acting)

Model alternates between **thinking** (reasoning) and **acting** (using tools/searching).

```
Prompt:
  Question: What is the population of the capital of France?

Output:
  Thought: I need to find the capital of France first.
  Action: Search "capital of France"
  Observation: The capital of France is Paris.
  Thought: Now I need the population of Paris.
  Action: Search "population of Paris"
  Observation: The population of Paris is approximately 2.1 million.
  Answer: The population of Paris is approximately 2.1 million.
```

- **When to use**: Tasks requiring external data, multi-step tool use
- **Pros**: Grounds responses in real data, reduces hallucination
- **Cons**: Requires tool integration (search, APIs, code execution)

---

## 9. Self-Consistency Prompting

Generate **multiple answers** using CoT, then pick the most common (majority vote).

```
Prompt (run 3 times with temperature > 0):
  Q: If 3 machines take 3 minutes to make 3 widgets,
  how long do 100 machines take to make 100 widgets?

  Run 1 → 3 minutes
  Run 2 → 3 minutes
  Run 3 → 100 minutes

  Majority answer: 3 minutes ✓
```

- **When to use**: High-stakes reasoning where correctness matters
- **Pros**: More reliable than single-pass CoT
- **Cons**: Multiple API calls, higher cost

---

## 10. Structured Output Prompting

Force the model to respond in a **specific format** (JSON, Markdown, CSV, etc.).

```
Prompt:
  Extract the following from this text and return as JSON:
  - name, age, occupation

  Text: "Maria is a 28-year-old software engineer from Berlin."

Output:
  {
    "name": "Maria",
    "age": 28,
    "occupation": "software engineer"
  }
```

- **When to use**: API responses, data extraction, pipeline input
- **Pros**: Machine-parseable output, consistent format
- **Cons**: Model may occasionally break format

---

## 11. Retrieval-Augmented Generation (RAG)

**Retrieve** relevant context from a knowledge base and inject it into the prompt before generation.

```
Prompt:
  Context (retrieved from vector DB):
  "Our return policy allows returns within 30 days with a receipt.
   Electronics have a 15-day return window."

  User question: Can I return a laptop I bought 20 days ago?

Output:
  Unfortunately, no. Electronics have a 15-day return window,
  and 20 days exceeds that limit.
```

- **When to use**: Q&A over documents, customer support, knowledge bases
- **Pros**: Grounded answers, reduces hallucination, no fine-tuning needed
- **Cons**: Depends on retrieval quality, adds latency

---

## 12. Prompt Chaining

Break a complex task into **sequential prompts**, where each step's output feeds the next.

```
Step 1 Prompt: "Summarize this 5-page article in 3 bullet points."
Step 1 Output: • Point A  • Point B  • Point C

Step 2 Prompt: "Based on these points, write a tweet thread."
Step 2 Output: 🧵 1/3 Point A explained...

Step 3 Prompt: "Translate this thread to Spanish."
Step 3 Output: 🧵 1/3 Punto A explicado...
```

- **When to use**: Multi-step workflows, content pipelines
- **Pros**: Each step is focused, easier to debug, better quality
- **Cons**: Multiple API calls, error propagation between steps

---

## 13. Directional Stimulus Prompting

Add a **hint or keyword** to nudge the model toward a specific direction.

```
Prompt:
  Write a poem about the ocean.
  Hint: Include themes of solitude and mystery.

Output:
  Beneath the waves where silence dwells,
  The ocean keeps its ancient spells...
```

- **When to use**: Creative writing, when you want a specific angle without being prescriptive
- **Pros**: Light guidance without over-constraining
- **Cons**: Hints may be ignored if too subtle

---

## 14. Reverse Prompting

Ask the model to **generate the ideal prompt** for a given task.

```
Prompt:
  I want an AI to write professional LinkedIn posts about tech topics.
  What would be the best system prompt for this?

Output:
  "You are a LinkedIn content strategist specializing in technology.
   Write engaging, professional posts that are 150-200 words.
   Start with a hook, include a personal insight, end with a
   question to drive engagement. Use 3-5 relevant hashtags."
```

- **When to use**: When you're unsure how to prompt, bootstrapping prompt design
- **Pros**: Leverages the model's understanding of itself
- **Cons**: May need iteration

---

## 15. Constrained / Negative Prompting

Tell the model what **NOT to do** to avoid common failure modes.

```
Prompt:
  Explain quantum computing to a 10-year-old.
  Do NOT use jargon, do NOT mention qubits by name,
  do NOT exceed 100 words.

Output:
  Imagine a regular computer is like flipping a coin — it's
  either heads or tails. A quantum computer is like spinning
  the coin — while it spins, it's kind of both at once! This
  lets it try many answers at the same time instead of one by
  one, making it super fast at solving tricky puzzles.
```

- **When to use**: Controlling verbosity, avoiding hallucination, enforcing boundaries
- **Pros**: Prevents common failure patterns
- **Cons**: Too many negatives can confuse the model

---

## Comparison Table

| Technique | Complexity | Token Usage | Best For |
|-----------|-----------|-------------|----------|
| Zero-Shot | Low | Low | Simple, well-known tasks |
| Few-Shot | Low | Medium | Format control, classification |
| Chain-of-Thought | Medium | Medium | Math, logic, reasoning |
| Zero-Shot CoT | Low | Medium | Quick reasoning boost |
| Role Prompting | Low | Low | Domain expertise, tone control |
| System Prompting | Low | Low | Chatbots, consistent behavior |
| Tree-of-Thought | High | High | Creative, strategic decisions |
| ReAct | High | High | Tool use, grounded answers |
| Self-Consistency | Medium | Very High | High-stakes accuracy |
| Structured Output | Low | Low | APIs, data extraction |
| RAG | High | Medium | Q&A over documents |
| Prompt Chaining | Medium | High | Multi-step workflows |
| Directional Stimulus | Low | Low | Creative nudging |
| Reverse Prompting | Low | Low | Prompt design bootstrapping |
| Constrained/Negative | Low | Low | Avoiding failure modes |

---

## Choosing the Right Technique

```
Simple task, model knows it well?
  └─ Zero-Shot

Need specific output format?
  └─ Few-Shot + Structured Output

Reasoning or math problem?
  └─ Chain-of-Thought (or Zero-Shot CoT for quick wins)

Creative brainstorming?
  └─ Tree-of-Thought

Q&A over your own documents?
  └─ RAG

Complex multi-step workflow?
  └─ Prompt Chaining

Building a chatbot?
  └─ System Prompting + Role Prompting

Need tool use (search, code, APIs)?
  └─ ReAct

High-stakes, must be correct?
  └─ Self-Consistency
```

---

## Tips for Better Prompts

1. **Be specific** — "List 5 benefits" beats "Tell me about benefits"
2. **Set the format** — "Respond in JSON" / "Use bullet points"
3. **Give context** — include relevant background information
4. **Set constraints** — word limits, audience level, what to avoid
5. **Iterate** — prompt engineering is experimental; refine based on output
6. **Use delimiters** — separate sections with `---`, `"""`, or XML tags
7. **Put instructions first** — models pay more attention to the beginning
8. **One task per prompt** — split complex tasks via prompt chaining
