# Task 1 — Structural Report Parameter Extraction (RAG Pipeline)

## Objective

Extract specific structural/seismic design parameters from a PDF structural report using a **Retrieval-Augmented Generation (RAG)** pipeline backed by an LLM.

---

## Parameters Extracted

| Parameter        | Description                                          |
|------------------|------------------------------------------------------|
| Ground Snow Load | Pg — ground-level snow load (psf)                   |
| Roof Snow Load   | Pf — flat-roof snow load (psf, ASCE 7)              |
| Wind Speed       | Design wind speed (mph, building code)              |
| SS               | Seismic spectral response — short period            |
| S1               | Seismic spectral response — 1 second                |
| SDS              | Design spectral acceleration — short period         |
| SD1              | Design spectral acceleration — 1 second             |

---

## Pipeline Steps

```
PDF File
   │
   ▼
[Step 1] Text Extraction         (PyMuPDF + Tesseract OCR fallback for scanned pages)
   │
   ▼
[Step 2] Text Chunking           (RecursiveCharacterTextSplitter, 500 chars / 50 overlap)
   │
   ▼
[Step 3] Embedding + Vector Store (HuggingFace MiniLM → ChromaDB)
   │
   ▼
[Step 4] Semantic Retrieval      (Top-5 chunks per parameter query)
   │
   ▼
[Step 5] LLM Extraction          (GPT-4o-mini → structured output)
   │
   ▼
JSON Output
```

---

## Where the LLM Is Used

**Function:** `extract_parameter()` — lines 179–186

```python
def extract_parameter(parameter: str, context_chunks: list[str]) -> dict:
    context = "\n\n---\n\n".join(context_chunks)

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)
    chain  = _PROMPT | llm.with_structured_output(ParameterResult, method="function_calling")
    result: ParameterResult = chain.invoke({"context": context, "parameter": parameter})

    return result.model_dump()
```

### What the LLM does

- Receives a **system prompt** framing it as a structural engineering document analyst.
- Receives the **top-5 retrieved text chunks** as context (from ChromaDB similarity search).
- Receives the **parameter name** to extract (e.g., `"SDS"`).
- Returns a **structured JSON object** via OpenAI function calling:

```json
{
  "parameter":      "SDS",
  "value":          "0.833",
  "unit":           "g",
  "source_snippet": "SDS = 2/3 × SMS = 2/3 × 1.25g = 0.833g"
}
```

### LLM Configuration

| Setting        | Value                  |
|----------------|------------------------|
| Model          | `gpt-4o-mini`          |
| Temperature    | `0.1` (near-deterministic) |
| Output method  | OpenAI function calling (structured) |
| Output schema  | `ParameterResult` (Pydantic)         |

---

## Structured Output Schema

Defined as a Pydantic model (`ParameterResult`):

| Field           | Type                          | Description                              |
|-----------------|-------------------------------|------------------------------------------|
| `parameter`     | `str`                         | Name of the extracted parameter          |
| `value`         | `Optional[str]`               | Extracted value, or `null` if not found  |
| `unit`          | `Optional[str]`               | Unit of measurement, or `null`           |
| `source_snippet`| `Optional[str]`               | Short quote from the PDF supporting the answer |

---

## How ChromaDB Querying Works

ChromaDB is the vector store in this pipeline. Querying happens in two phases.

---

### Phase 1 — Indexing (storing chunks into ChromaDB)

**Code:** `build_vector_store()` — called once when `rebuild_store=True`

```python
Chroma.from_texts(
    texts=chunks,               # 41 text chunks from the PDF
    embedding=_get_embeddings(), # HuggingFace all-MiniLM-L6-v2
    persist_directory=CHROMA_DIR # saved to disk: chroma_store/
)
```

**What happens inside:**

```
chunk[0]  "Ground Snow Load, Pg = 60 psf..."
chunk[1]  "Flat Roof snow load = Pf = 0.7..."
chunk[2]  "Basic Wind Speed = 115 mph..."
chunk[3]  "SS = 0.284, S1 = 0.068..."
...
chunk[40] "..."
        │
        ▼  all-MiniLM-L6-v2 runs on each chunk
        │
chunk[0]  → vector[0]  = [0.23, -0.11,  0.87, ...]  ← 384 numbers
chunk[1]  → vector[1]  = [0.31,  0.44, -0.12, ...]
chunk[2]  → vector[2]  = [0.12, -0.08,  0.55, ...]
chunk[3]  → vector[3]  = [0.41,  0.09, -0.45, ...]
...
chunk[40] → vector[40] = [...]
        │
        ▼  all 41 vectors + original texts saved to chroma_store/ on disk
```

ChromaDB stores three things per chunk:
- The **original text** of the chunk
- The **384-dimensional vector** of the chunk
- An internal **ID**

---

### Phase 2 — Querying (retrieving relevant chunks at runtime)

**Code:** `retrieve_chunks()` — called once per parameter

```python
def retrieve_chunks(vector_store: Chroma, query: str, top_k: int = 5):
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]
```

**What `similarity_search` does step by step:**

```
STEP 1 — Embed the query string
────────────────────────────────
query = "seismic SS S_S spectral response acceleration short period Table 1604"
        │
        ▼  all-MiniLM-L6-v2 (same model used during indexing)
        │
query_vector = [0.38, 0.07, -0.41, 0.29, ...]   ← 384 numbers


STEP 2 — Compute cosine similarity between query_vector and all 41 chunk vectors
──────────────────────────────────────────────────────────────────────────────────
Cosine similarity measures how closely two vectors point in the same direction:
  score = 1.0  →  identical meaning
  score = 0.0  →  completely unrelated

chunk[0]  "Ground Snow Load = 60 psf..."     → score = 0.21  (low)
chunk[1]  "Flat Roof snow load = 0.7..."     → score = 0.18  (low)
chunk[2]  "Basic Wind Speed = 115 mph..."    → score = 0.19  (low)
chunk[3]  "SS = 0.284, S1 = 0.068..."        → score = 0.87  (HIGH)
chunk[4]  "SDS = 2/3 x SMS = 0.303..."       → score = 0.81  (HIGH)
chunk[5]  "SD1 = 2/3 x SM1 = 0.109..."       → score = 0.76  (HIGH)
...


STEP 3 — Rank and return top 5 chunks
───────────────────────────────────────
Rank 1: chunk[3]  "SS = 0.284 (Table 1604.11)..."       score = 0.87
Rank 2: chunk[4]  "SDS = 2/3 x SMS = 0.303..."          score = 0.81
Rank 3: chunk[5]  "SD1 = 2/3 x SM1 = 0.109..."          score = 0.76
Rank 4: chunk[6]  "Spectral Response Acceleration..."    score = 0.71
Rank 5: chunk[7]  "Seismic Load: S_S = 0.284, S_1..."   score = 0.68

These 5 chunks are joined and sent to GPT-4o-mini as context.
```

---

### Why PARAM_QUERIES matters

The retrieval quality depends entirely on the query string used.

```
BAD query:   "What is SS?"
  → Short, vague → query vector points in a generic direction
  → chunk[3] "SS = 0.284..." gets LOW similarity score → NOT returned
  → LLM receives wrong context → returns null

GOOD query:  "seismic SS S_S spectral response acceleration short period Table 1604"
  → Rich with domain keywords → query vector points toward seismic content
  → chunk[3] "SS = 0.284..." gets HIGH similarity score → returned as rank 1
  → LLM receives correct context → returns 0.284
```

`PARAM_QUERIES` maps each parameter to a rich query containing the actual words
that appear near the value in the PDF:

```python
PARAM_QUERIES = {
    "SS":               "seismic SS S_S spectral response acceleration short period Table 1604",
    "S1":               "seismic S1 S_1 spectral response acceleration 1 second Table 1604",
    "SDS":              "SDS design spectral acceleration short period IBC Equation 16-22",
    "SD1":              "SD1 design spectral acceleration 1 second IBC Equation 16-23",
    "ground snow load": "ground snow load Pg psf building code",
    "roof snow load":   "roof snow load flat roof Pf psf ASCE",
    "wind speed":       "wind speed mph building code",
}
```

---

### ChromaDB files on disk

After indexing, the following files are created at `chroma_store/`:

```
chroma_store/
  ├── chroma.sqlite3        ← metadata + original chunk texts
  └── <uuid>/
        ├── data_level0.bin ← 384-dim vectors stored as HNSW index
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

On the next run (`rebuild_store=False`), ChromaDB loads from these files directly —
no need to re-embed the PDF.

---

### Retrieval Strategy

For parameters that use symbol/subscript notation in PDFs (e.g. `SS`, `SDS`), plain name lookup would miss matches. Custom retrieval queries are defined in `PARAM_QUERIES` to improve recall:

```python
"SS":  "seismic SS S_S spectral response acceleration short period Table 1604"
"SDS": "SDS design spectral acceleration short period IBC Equation 16-22"
```

---

## Output

Results are saved to:
```
data/structural_report_extracted.json
```

Each key is a parameter name; each value is the structured extraction result.

---

## Tech Stack

| Component       | Library                          |
|-----------------|----------------------------------|
| PDF parsing     | PyMuPDF (`fitz`)                 |
| OCR fallback    | Tesseract via `pytesseract`      |
| Text splitting  | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings      | HuggingFace `all-MiniLM-L6-v2`  |
| Vector store    | ChromaDB                         |
| LLM             | OpenAI GPT-4o-mini via LangChain |
| Structured output | Pydantic + OpenAI function calling |
