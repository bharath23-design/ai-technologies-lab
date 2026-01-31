# RAG (Retrieval-Augmented Generation) — Complete Guide

## What Is RAG?

RAG combines a **retrieval system** (search) with a **generative model** (LLM) to produce answers grounded in actual data. Instead of relying solely on what the LLM memorized during training, RAG fetches relevant documents first, then generates a response based on them.

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Embed      │────▶│  Vector DB    │────▶│  Retrieved   │
│   Query      │     │  Search       │     │  Documents   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                    ┌────────────▼────────────┐
                                    │  LLM generates answer    │
                                    │  using retrieved context  │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                           Final Answer
```

---

## Why RAG?

| Problem | How RAG Solves It |
|---------|-------------------|
| LLM hallucination | Grounds answers in real documents |
| Outdated knowledge | Retrieves current data (no retraining) |
| Domain-specific Q&A | Searches your own documents |
| Cost of fine-tuning | No model training needed, just index your data |
| Traceability | Can cite source documents |

---

## RAG Pipeline — Step by Step

### Ingestion Phase (Offline)

```
Documents → Chunk → Embed → Store in Vector DB
```

1. **Load** — read PDFs, web pages, databases, APIs
2. **Chunk** — split into smaller pieces (e.g., 500 tokens each)
3. **Embed** — convert each chunk to a vector using an embedding model
4. **Store** — save vectors + metadata in a vector database

### Query Phase (Online)

```
User Query → Embed → Search Vector DB → Retrieve Top-K → LLM → Answer
```

1. **Embed query** — same embedding model as ingestion
2. **Retrieve** — find top-K most similar chunks from vector DB
3. **Augment** — inject retrieved chunks into the LLM prompt as context
4. **Generate** — LLM produces an answer based on the context

---

## Types of RAG

### 1. Naive RAG (Basic)

Simple retrieve-then-generate pipeline. No optimization.

```
Query → Embed → Top-K retrieval → Stuff into prompt → LLM → Answer
```

- **Pros**: Easy to build, works for simple Q&A
- **Cons**: Irrelevant chunks, lost context in long docs, no self-correction

### 2. Advanced RAG

Adds **pre-retrieval and post-retrieval** optimizations.

```
Query → Rewrite query → Embed → Retrieve → Rerank → Filter → LLM → Answer
```

Techniques:
- **Query rewriting** — rephrase for better retrieval
- **Hybrid search** — combine vector search + keyword search (BM25)
- **Reranking** — use a cross-encoder to re-score retrieved chunks
- **Metadata filtering** — filter by date, source, category before search

### 3. Modular RAG

Flexible pipeline where components can be **swapped, added, or removed**.

```
Query → [Router] → [Retriever A or B] → [Reranker] → [Compressor] → LLM
```

- Mix and match retrievers, rankers, generators
- Add/remove steps based on the task

---

## Advanced RAG Techniques

### 4. Self-RAG (Self-Reflective RAG)

The LLM **decides whether to retrieve** and evaluates its own output for relevance and support.

```
Query → LLM decides: "Do I need retrieval?"
  ├─ No  → Generate directly
  └─ Yes → Retrieve → Generate → Self-evaluate:
           "Is this relevant?" "Is this supported by the source?"
           → Accept or retry
```

- **Pros**: Avoids unnecessary retrieval, self-correcting
- **Cons**: More LLM calls, complex implementation

### 5. Corrective RAG (CRAG)

Evaluates retrieved documents and **falls back to web search** if retrieval quality is low.

```
Query → Retrieve → Evaluate relevance:
  ├─ Relevant → Use for generation
  ├─ Ambiguous → Refine and re-retrieve
  └─ Irrelevant → Web search fallback → Generate
```

- **Pros**: Handles retrieval failures gracefully
- **Cons**: Web search adds latency and cost

### 6. Adaptive RAG

Dynamically chooses the **retrieval strategy** based on query complexity.

```
Query → Classify complexity:
  ├─ Simple  → Direct LLM answer (no retrieval)
  ├─ Medium  → Single-step RAG
  └─ Complex → Multi-step RAG with decomposition
```

### 7. GraphRAG

Uses a **knowledge graph** instead of (or alongside) vector search.

```
Documents → Extract entities & relationships → Build knowledge graph
Query → Graph traversal → Retrieve subgraph → LLM → Answer
```

- **Pros**: Captures relationships between entities, global reasoning
- **Cons**: Graph construction is expensive, complex setup
- **Frameworks**: Microsoft GraphRAG, Neo4j + LangChain

### 8. Long RAG

Retrieves **longer chunks** (sections or full documents) instead of small passages.

```
Standard RAG: 100-500 token chunks → loses context
Long RAG:     2000-4000 token chunks → preserves context
```

- **Pros**: Better context preservation, fewer retrieval misses
- **Cons**: Needs larger context window models

### 9. Multi-Modal RAG

Retrieves and reasons over **text + images + tables + audio**.

```
Query → Search text chunks + image embeddings + table data
     → Multi-modal LLM generates answer using all modalities
```

- **Pros**: Handles documents with figures, charts, diagrams
- **Cons**: Needs multi-modal embedding models and LLMs

### 10. Agentic RAG

An **AI agent** orchestrates the RAG pipeline, deciding when to retrieve, what to search, and whether to re-query.

```
Agent receives query
  → Thought: "I need financial data from 2024"
  → Action: search_financial_db("revenue 2024")
  → Observation: [results]
  → Thought: "I also need competitor data"
  → Action: search_web("competitor revenue 2024")
  → Observation: [results]
  → Generate final answer from combined context
```

- **Pros**: Dynamic, multi-source, self-directing
- **Cons**: Complex, expensive, needs good tool design

---

## Chunking Strategies

| Strategy | Chunk Size | Best For |
|----------|-----------|----------|
| **Fixed-size** | 500-1000 tokens with overlap | Simple docs, general purpose |
| **Sentence-based** | Split on sentence boundaries | Preserving complete thoughts |
| **Paragraph-based** | Split on paragraphs | Well-structured documents |
| **Semantic** | Split when topic changes (using embeddings) | Mixed-topic documents |
| **Recursive** | Split by headings → paragraphs → sentences | Hierarchical documents |
| **Document-level** | Entire document as one chunk | Short docs, Long RAG |

**Overlap**: Always use 10-20% overlap between chunks to avoid cutting off important context at boundaries.

---

## Retrieval Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| **Dense retrieval** | Embed query + docs, cosine similarity | Semantic search |
| **Sparse retrieval (BM25)** | Keyword matching with TF-IDF | Exact term matching |
| **Hybrid** | Combine dense + sparse scores | Best of both worlds |
| **Reranking** | Cross-encoder rescores top-K results | Precision improvement |
| **Multi-query** | Generate multiple query variants, merge results | Recall improvement |
| **HyDE** | Generate hypothetical answer, use it as query | Better retrieval alignment |

---

## Popular Frameworks & Tools

| Tool | Category | Description |
|------|----------|-------------|
| **LangChain** | Framework | End-to-end RAG chains with many integrations |
| **LlamaIndex** | Framework | Data ingestion, indexing, and querying |
| **Haystack** | Framework | Production-ready RAG pipelines |
| **DSPy** | Optimizer | Declarative prompt/retrieval optimization |
| **ChromaDB** | Vector DB | Lightweight, embedded vector store |
| **LanceDB** | Vector DB | Serverless, local vector DB |
| **Pinecone** | Vector DB | Managed cloud vector DB |
| **Weaviate** | Vector DB | Hybrid search vector DB |
| **Qdrant** | Vector DB | Filtered vector search |
| **Cohere Rerank** | Reranker | API-based reranking model |
| **Unstructured** | Loader | Parse PDFs, DOCX, HTML, images |

---

## Comparison Table

| RAG Type | Retrieval | Self-Correct | Multi-Hop | Best For |
|----------|-----------|-------------|-----------|----------|
| Naive | Basic vector | No | No | Simple Q&A |
| Advanced | Hybrid + rerank | No | No | Production Q&A |
| Self-RAG | Conditional | Yes | No | Quality-critical |
| Corrective RAG | With fallback | Yes | No | Unreliable corpus |
| Adaptive | Dynamic | Partial | Optional | Mixed query types |
| GraphRAG | Knowledge graph | No | Yes | Relationship queries |
| Long RAG | Long chunks | No | No | Context-heavy docs |
| Multi-Modal | Text + images | No | No | Rich documents |
| Agentic RAG | Agent-driven | Yes | Yes | Complex research |

---

## Choosing the Right RAG Approach

```
Simple Q&A over a few documents?
  └─ Naive RAG (LangChain + ChromaDB)

Production system needing accuracy?
  └─ Advanced RAG (hybrid search + reranking)

Documents with lots of relationships?
  └─ GraphRAG

Need self-correction and reliability?
  └─ Self-RAG or Corrective RAG

Mixed query complexity?
  └─ Adaptive RAG

Documents with images, tables, charts?
  └─ Multi-Modal RAG

Complex multi-source research?
  └─ Agentic RAG
```

---

## Common Pitfalls

1. **Chunks too small** — lose context, irrelevant matches
2. **Chunks too large** — exceed context window, dilute relevant info
3. **No overlap** — important info split across chunk boundaries
4. **Wrong embedding model** — domain mismatch (e.g., general model for medical texts)
5. **No reranking** — top-K results may not be the most relevant
6. **Ignoring metadata** — filtering by date/source improves precision
7. **Stuffing too many chunks** — overwhelming the LLM reduces quality
8. **No evaluation** — measure retrieval recall and answer accuracy
