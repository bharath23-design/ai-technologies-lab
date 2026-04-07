# Task 1 — Structural Report PDF Parameter Extraction

## Objective

Analyze the structural report PDF and extract the following design parameters:

| # | Parameter | Description |
|---|-----------|-------------|
| 1 | Ground snow load | Ground-level snow load in psf |
| 2 | Roof snow load | Flat roof snow load calculated from ground load |
| 3 | Wind speed | Design wind speed in mph |
| 4 | SS | Seismic spectral response acceleration — short period |
| 5 | S1 | Seismic spectral response acceleration — 1 second |
| 6 | SDS | Design spectral acceleration — short period |
| 7 | SD1 | Design spectral acceleration — 1 second |

**Input PDF:** `data/Structural Report.pdf`
**Output:** `data/structural_report_extracted.json`

---

## How It Works — Full Explanation

### Step 1: Text Extraction

PyMuPDF (`fitz`) opens the PDF and reads each page.

- If a page has native text → read directly
- If a page has less than 50 characters (scanned page) → render as image at 200 DPI and run `pytesseract` OCR

Result: all pages merged into one raw string (~15,000 characters for this report).

---

### Step 2: Chunking

The full text is too large to send to an LLM in one go. It is split into smaller
overlapping pieces using `RecursiveCharacterTextSplitter`:

- **Chunk size:** 500 characters
- **Overlap:** 50 characters — so values sitting at a chunk boundary appear in both chunks and are not lost
- **Result:** 41 chunks

Example of one chunk:
```
Ground Snow Load, Pg: City of Leominster, MA = 60 psf
(Mass Building Code, 780 CMR, 10th Ed.)
Roof Snow Load: Flat Roof snow load = Pf = 0.7 Ce Ct I pg ...
```

---

### Step 3: Indexing — Embeddings + ChromaDB

Each of the 41 chunks is converted into a **vector** (list of 384 numbers) using the
HuggingFace embedding model `all-MiniLM-L6-v2`. This is called an embedding —
it captures the meaning of the text as numbers.

```
"Ground Snow Load = 60 psf..."      →  [0.23, -0.11, 0.87, ...]   ← 384 numbers
"SS = 0.284 (Table 1604.11)..."     →  [0.41,  0.09, -0.45, ...]
"SDS = 2/3 x SMS = 0.303..."        →  [0.18,  0.33,  0.71, ...]
... (41 vectors total)
```

All 41 vectors are stored in **ChromaDB** on disk (`chroma_store/`).
This is called **indexing** — done once, reused on every run.

---

### Step 4: Querying — Retrieval

For each of the 7 parameters, a descriptive retrieval query is defined in `PARAM_QUERIES`:

```python
"SS"  → "seismic SS S_S spectral response acceleration short period Table 1604"
"SDS" → "SDS design spectral acceleration short period IBC Equation 16-22"
...
```

The query string is embedded into a vector using the same model.
That query vector is then compared against all 41 stored chunk vectors
using **cosine similarity** — how closely two vectors point in the same direction.

The **top 5 most similar chunks** are returned — these are the parts of the PDF
most likely to contain the answer for that parameter.

```
Query vector for "SS"
        │
        ▼
Compare against 41 chunk vectors
        │
        ▼
Rank by cosine similarity score
        │
        ▼
Return top 5 chunks  ← sent to LLM as context
```

> **Why custom queries instead of just "What is SS?"**
> The PDF writes it as `S_S` (subscript notation). A short vague query
> does not match that chunk well in vector space. The richer query includes
> keywords like `S_S`, `spectral`, `Table 1604` that actually appear near
> the value in the PDF — so the right chunk scores high similarity.

---

### Step 5: LLM Extraction — Structured Output

The 5 retrieved chunks are joined and sent to **GPT-4o-mini** via OpenAI API.

**Prompt sent to the LLM:**

```
System:
  You are a structural engineering document analyst. Use only the context provided.

Human:
  Context: <top 5 chunks joined together>
  Extract the value of the parameter: "SS"
```

The LLM response is enforced via a **Pydantic schema** using LangChains `with_structured_output()`:

```python
class ParameterResult(BaseModel):
    parameter:      str           # e.g. "SS"
    value:          Optional[str] # e.g. "0.284"
    unit:           Optional[str] # e.g. null
    source_snippet: Optional[str] # quote from the PDF
```

LangChain forces GPT-4o-mini to return exactly this structure — no free-form text, no JSON parsing errors.

---

## Tech Stack

| Layer          | Tool                                     | Why                                            |
|----------------|------------------------------------------|------------------------------------------------|
| PDF extraction | PyMuPDF + pytesseract                    | Handles native text and scanned pages          |
| Chunking       | LangChain RecursiveCharacterTextSplitter | Smart overlap-aware splits                     |
| Embeddings     | HuggingFace all-MiniLM-L6-v2 (local)    | Free, fast, runs on M3 8GB                     |
| Vector Store   | ChromaDB (langchain-chroma)              | Local persistent store, zero setup             |
| Retrieval      | Cosine similarity via PARAM_QUERIES      | Rich queries handle subscript PDF notation     |
| LLM            | gpt-4o-mini via OpenAI API               | Fast, cheap, structured output via Pydantic    |
| Output         | JSON (ParameterResult schema)            | value + unit + source snippet                  |

## Installation

```bash
pip install pymupdf pytesseract pillow \
            langchain langchain-text-splitters \
            langchain-chroma langchain-huggingface \
            langchain-openai sentence-transformers \
            openai
```

## Run

```bash
python extract_pdf_params.py
```
