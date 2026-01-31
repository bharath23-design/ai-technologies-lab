# Embedding Methods & Vector Databases

## What Are Embeddings?

Embeddings convert text, images, or other data into dense numerical vectors that capture semantic meaning. Similar items end up close together in vector space.

```
"king" → [0.21, -0.45, 0.89, ...]   (384 or 768+ dimensions)
"queen" → [0.19, -0.42, 0.91, ...]  (close to "king")
"car"  → [-0.73, 0.12, -0.34, ...]  (far from "king")
```

---

## Embedding Models

### 1. Word2Vec

Learns word vectors by predicting surrounding words (CBOW) or target word from context (Skip-gram).

- **Dimensions**: 100-300
- **Pros**: Fast training, captures word analogies (king - man + woman = queen)
- **Cons**: One vector per word (no context), can't handle unseen words
- **Use case**: Legacy systems, simple word similarity

### 2. GloVe (Global Vectors)

Builds word vectors from a global word co-occurrence matrix.

- **Dimensions**: 50-300
- **Pros**: Captures global statistics, pre-trained vectors widely available
- **Cons**: Static embeddings (no context), large memory for big vocabularies
- **Use case**: Text classification, word analogy tasks

### 3. FastText

Extends Word2Vec by embedding **subword n-grams**, then averaging them per word.

- **Dimensions**: 100-300
- **Pros**: Handles misspellings and unseen words (via subwords), good for morphologically rich languages
- **Cons**: Still static (no context), larger model size than Word2Vec
- **Use case**: Typo-tolerant search, multilingual tasks

### 4. Sentence-Transformers (SBERT)

Fine-tuned BERT/transformer models that produce **sentence-level** embeddings via mean pooling.

- **Dimensions**: 384-1024
- **Pros**: Semantic sentence similarity, fast inference, many pre-trained models on HuggingFace
- **Cons**: Needs GPU for large-scale encoding
- **Popular models**: `all-MiniLM-L6-v2` (384d, fast), `all-mpnet-base-v2` (768d, better quality)
- **Use case**: Semantic search, clustering, duplicate detection

**Used in this repo**: `task1-name-matching` with `all-MiniLM-L6-v2`

### 5. OpenAI Embeddings

API-based embedding models from OpenAI.

- **Dimensions**: 1536 (`text-embedding-ada-002`), 256-3072 (`text-embedding-3-small/large`)
- **Pros**: High quality, no local GPU needed, simple API
- **Cons**: Paid, data leaves your machine, API latency
- **Use case**: Production RAG systems, enterprise search

### 6. Cohere Embed

API-based embeddings with built-in search optimization.

- **Dimensions**: 384-1024
- **Pros**: Optimized for search (separate `search_document` and `search_query` types), multilingual
- **Cons**: Paid API
- **Use case**: Multilingual semantic search

### 7. BERT Embeddings (CLS / Mean Pooling)

Extract embeddings from BERT's hidden states directly (without fine-tuning for similarity).

- **Dimensions**: 768 (base), 1024 (large)
- **Pros**: Strong contextual understanding, handles polysemy (same word, different meanings)
- **Cons**: Not optimized for similarity out-of-the-box (use Sentence-Transformers instead)
- **Use case**: Feature extraction for downstream classifiers

### 8. Instructor Embeddings

Task-aware embeddings — you provide an **instruction** describing the task along with the text.

- **Dimensions**: 768
- **Pros**: One model adapts to many tasks via instructions, strong zero-shot performance
- **Cons**: Slower inference (processes instruction + text)
- **Use case**: Multi-task retrieval where queries vary in nature

```python
model.encode([["Represent the query for retrieval: ", "best pizza recipe"]])
```

### 9. BGE (BAAI General Embedding)

Open-source embeddings from BAAI, fine-tuned for retrieval tasks.

- **Dimensions**: 384-1024
- **Pros**: Top MTEB benchmark scores, open-source, multiple sizes
- **Cons**: Larger models need more memory
- **Popular models**: `bge-small-en-v1.5` (384d), `bge-large-en-v1.5` (1024d)
- **Use case**: RAG, document retrieval

### 10. E5 (EmbEddings from bidirEctional Encoder rEpresentations)

Microsoft's embedding models trained with contrastive learning on large text pairs.

- **Dimensions**: 384-1024
- **Pros**: Strong retrieval performance, prefix-based (`query:` / `passage:`)
- **Cons**: Requires correct prefixes for best results
- **Use case**: Asymmetric search (short query → long document)

---

## Vector Databases

### 1. LanceDB

Serverless, embedded vector DB built on Lance columnar format.

- **Pros**: No server setup, local storage, fast ANN search, Python-native
- **Cons**: Less mature ecosystem than Pinecone/Weaviate
- **Best for**: Local projects, prototyping, edge deployment

**Used in this repo**: `task1-name-matching`

### 2. ChromaDB

Open-source embedding database designed for AI applications.

- **Pros**: Simple API, runs in-memory or persistent, built-in embedding functions
- **Cons**: Not ideal for very large datasets (100M+ vectors)
- **Best for**: Prototyping, small-to-medium RAG systems

### 3. FAISS (Facebook AI Similarity Search)

Library (not a database) for efficient similarity search and clustering of dense vectors.

- **Pros**: Extremely fast, GPU support, battle-tested at scale
- **Cons**: No built-in persistence/metadata, just a search library
- **Best for**: High-performance search at scale, research

### 4. Pinecone

Fully managed cloud vector database.

- **Pros**: Zero ops, scales automatically, low latency, metadata filtering
- **Cons**: Paid, data in cloud, vendor lock-in
- **Best for**: Production systems needing managed infrastructure

### 5. Weaviate

Open-source vector database with built-in vectorization modules.

- **Pros**: GraphQL API, hybrid search (vector + keyword), modular vectorizers
- **Cons**: Heavier setup than LanceDB/Chroma
- **Best for**: Production apps needing hybrid search

### 6. Milvus

Open-source vector database built for scalable similarity search.

- **Pros**: Handles billions of vectors, distributed architecture, GPU indexing
- **Cons**: Complex setup (requires etcd, MinIO), overkill for small projects
- **Best for**: Large-scale enterprise deployments

### 7. Qdrant

Open-source vector database with advanced filtering.

- **Pros**: Rich filtering, payload indexing, gRPC + REST API, Rust-based (fast)
- **Cons**: Needs separate server process
- **Best for**: Production apps needing complex filtered vector search

---

## Similarity Metrics

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| **Cosine Similarity** | cos(A, B) = A·B / (\|A\|\|B\|) | -1 to 1 | Text similarity (direction matters, not magnitude) |
| **Euclidean (L2)** | \|\|A - B\|\| | 0 to ∞ | When magnitude matters |
| **Dot Product** | A · B | -∞ to ∞ | Normalized vectors, retrieval ranking |
| **Manhattan (L1)** | Σ\|Aᵢ - Bᵢ\| | 0 to ∞ | High-dimensional sparse vectors |

---

## Comparison Table — Embedding Models

| Model | Dimensions | Type | Context | Speed | Quality |
|-------|-----------|------|---------|-------|---------|
| Word2Vec | 100-300 | Static | None | Very Fast | Low |
| GloVe | 50-300 | Static | None | Very Fast | Low |
| FastText | 100-300 | Static (subword) | None | Very Fast | Medium |
| SBERT (MiniLM) | 384 | Contextual | Sentence | Fast | Good |
| SBERT (mpnet) | 768 | Contextual | Sentence | Medium | Very Good |
| BGE | 384-1024 | Contextual | Sentence | Medium | Very Good |
| E5 | 384-1024 | Contextual | Sentence | Medium | Very Good |
| Instructor | 768 | Task-aware | Sentence | Slow | Very Good |
| OpenAI Ada-002 | 1536 | API | 8191 tokens | API | Excellent |
| Cohere Embed | 384-1024 | API | 512 tokens | API | Excellent |

## Comparison Table — Vector Databases

| Database | Type | Setup | Scale | Best For |
|----------|------|-------|-------|----------|
| LanceDB | Embedded | None | Small-Med | Local/prototyping |
| ChromaDB | Embedded/Server | Minimal | Small-Med | RAG prototyping |
| FAISS | Library | None | Large | Research/performance |
| Pinecone | Managed Cloud | None | Large | Production (managed) |
| Weaviate | Self-hosted/Cloud | Medium | Large | Hybrid search |
| Milvus | Self-hosted | Complex | Very Large | Enterprise |
| Qdrant | Self-hosted/Cloud | Medium | Large | Filtered search |

---

## Choosing the Right Setup

```
Local project, quick prototype?
  └─ LanceDB or ChromaDB + Sentence-Transformers

Production RAG with managed infra?
  └─ Pinecone or Weaviate + OpenAI/Cohere embeddings

High-performance research?
  └─ FAISS + BGE or E5

Need hybrid (vector + keyword) search?
  └─ Weaviate or Qdrant

Typo-tolerant name/text matching?
  └─ LanceDB + Sentence-Transformers (this repo's approach)
```
