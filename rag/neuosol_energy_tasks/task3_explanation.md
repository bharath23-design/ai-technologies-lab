# Task 3 — AutoCAD Text Mismatch Detection Across Viewports

## Objective

Detect text mismatches between the **Key Plan** and **Title Block** viewports on each page of a CAD engineering drawing PDF, generate a highlighted output PDF with red bounding boxes around mismatched regions, and produce a structured alert report.

**Input:**  any CAD-exported PDF (e.g. `data/Test Drawing 2.pdf`)
**Output:** `data/<name> - Mismatch Highlighted.pdf` + `<name> - Mismatch Highlighted.json`
**Script:** `task3.py`

---

## Why OCR-only

Earlier versions of the pipeline used PyMuPDF native vector text extraction as the primary path and fell back to Tesseract OCR only for scanned pages. That worked for untouched CAD exports, but it broke the moment a user edited the PDF with an online editor (Smallpdf, iLovePDF, Adobe Acrobat, etc.):

- Edits stored as **real PDF annotation objects** showed up in `page.annots()` but were invisible to `get_text("blocks")`.
- Edits **flattened into the content stream** were readable but appeared as duplicate tokens layered over the original values.
- Edits drawn as **image overlays** were completely invisible to any text-layer reader.

Rather than maintaining three different reconciliation strategies for the three cases, the pipeline now rasterises every page at 300 DPI and runs Tesseract on the image. OCR sees whatever a human would see on the rendered page, regardless of how the editor chose to store the edit.

Tradeoff: slower than native extraction, but correct on all three edit types.

---

## Pipeline

### Quick Overview

```
PDF
 ↓
Convert to Image (300 DPI)
 ↓
OCR (Tesseract)
 ↓
Extract text + positions
 ↓
Find ROOF AREA / ARRAY blocks
 ↓
Split → Left (Key Plan) | Right (Title Block)
 ↓
Extract values
 ↓
Compare values
 ↓
If mismatch:
    → Draw red box
    → Add "MISMATCH"
 ↓
Save new PDF + JSON report
```

### Detailed Step-by-Step

```
CAD Drawing PDF
     │
     ▼
[Step 1] Text Extraction  —  Tesseract OCR (always)
     │   page.get_pixmap(dpi=300) renders EVERYTHING visible on the
     │   page (native text + annotations + image overlays + flattened
     │   edits), then pytesseract --psm 11 extracts words + bboxes.
     │   All blocks tagged source="ocr".
     ▼
[Step 2] Value-First Viewport Detection
     │   Find all ROOF AREA / ARRAY value blocks on the page
     │   Sort by x-position → split at largest x-gap
     │   Left cluster  = Key Plan
     │   Right cluster = Title Block
     │   (Positional fallback if no value blocks found)
     ▼
[Step 3] Three-Layer Value Extraction  (per viewport)
     │
     ├─ Layer 1: Regex              (strict, exact match)
     │           ROOF\s+AREA\s+(\d+[A-Z]?)  /  ARRAY\s+(\d+[A-Z]?)
     │
     ├─ Layer 2: Fuzzy token scan   (only if regex misses)
     │           handles OCR noise: "R0OF AREA1", "ARR AY 2"
     │
     └─ Layer 3: Ollama llama3.2:1b (only if layers 1+2 both fail)
                 local LLM, no internet, last resort
     ▼
[Step 4] Normalised Comparison
     │   Uppercase + collapse whitespace → compare
     │   Detect: missing / mismatch / consistent
     ▼
[Step 5] Highlight Output PDF
     │   Red semi-transparent boxes over mismatched blocks
     ▼
[Step 6] Alert Report + JSON Output
         extraction_method + detection_mode logged per page
```

---

## Step 1 — Text Block Extraction (OCR only)

**Functions:** `_ocr_blocks()`, `extract_blocks()`

```python
def extract_blocks(page: fitz.Page) -> tuple[list[dict], str]:
    # Always OCR. page.get_pixmap() renders annotations, image overlays,
    # and flattened edits alongside the native text layer, so Tesseract
    # sees whatever is visually on the page — regardless of how an
    # online PDF editor stored the edit.
    return _ocr_blocks(page), "ocr"
```

`_ocr_blocks()` does:

```python
pix  = page.get_pixmap(dpi=300)
img  = Image.open(io.BytesIO(pix.tobytes("png")))
data = pytesseract.image_to_data(
    img,
    config="--psm 11",                     # sparse text — best for CAD
    output_type=pytesseract.Output.DICT,
)
```

- **300 DPI** — the sweet spot for small CAD labels. Lower DPIs lose thin strokes; higher DPIs cost seconds per page with no accuracy gain.
- **`--psm 11`** sparse-text mode — Tesseract does not assume paragraphs or columns, which matches how text is scattered around a CAD drawing.
- **Confidence < 30 dropped** — removes obvious noise from anti-aliased vector edges.
- **Words grouped into lines** by `(block_num, par_num, line_num)`.
- **Pixel coords → fractional page coords** — so downstream classification works for any paper size.

Every returned block looks like:

```python
{
    "text":   "ARRAY 2",
    "x0":     0.078, "y0": 0.913,   # fractional — 0..1
    "x1":     0.120, "y1": 0.922,
    "source": "ocr",
}
```

---

## Step 2 — Value-First Viewport Detection

**Function:** `classify_blocks()`

Instead of searching for `"KEY PLAN"` / `"TITLE"` labels (which can appear anywhere in notes/body text and cause wrong matches), the classifier **finds the actual value blocks first** then uses their x-positions to determine which viewport each belongs to.

```
All blocks on page
    ↓
Filter: blocks matching ROOF AREA <N> or ARRAY <N>  (exclude "ARRAY LAYOUT")
    ↓
Sort by x-centre
    ↓
Find largest x-gap between consecutive value blocks → split there
    ↓
Left cluster  = Key Plan    (wherever it sits on the page)
Right cluster = Title Block (wherever it sits on the page)
    ↓
Expand ±0.08 padding → collect all nearby blocks in each region
```

Real example from Page 1:

```
Value blocks found:
  "ROOF AREA 1"  x_centre=0.089  ← left  → Key Plan
  "ARRAY 1"      x_centre=0.091  ← left  → Key Plan
  "ROOF AREA 1"  x_centre=0.846  ← right → Title Block
  "ARRAY 12"     x_centre=0.930  ← right → Title Block  (edited via online editor)

Largest x-gap: between 0.091 and 0.846 → split here
```

Detection modes logged per page:

| Mode       | Meaning                                                      |
|------------|--------------------------------------------------------------|
| `VALUE`    | Both regions found by value-block x-split (most reliable)    |
| `FALLBACK` | No value blocks found — broad positional quadrants used     |

---

## Step 3 — Three-Layer Value Extraction

**Function:** `parse_viewport()`

Each viewport's blocks go through three layers in sequence. A later layer is only called if the previous one did not find both values.

### Layer 1 — Regex (always tried first)

```python
_ROOF_RE  = re.compile(r"ROOF\s+AREA\s+(\d+[A-Z]?)", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\bARRAY\s+(\d+[A-Z]?)\b",   re.IGNORECASE)
```

`\b` word boundaries prevent `"ARRAY LAYOUT"` from matching — only `"ARRAY 1"`, `"ARRAY 12"`, `"ARRAY 2A"` etc. are captured.

### Layer 2 — Fuzzy Token Scan (OCR noise fallback)

Scans sliding windows of adjacent words to handle artefacts like `"R0OF AREA1"`, `"ARR AY 2"`:

```python
if re.match(r"R[O0]{1,2}F$", w):          # ROOF or R0OF
    if re.match(r"AR[E3][A4]$", next_w):  # AREA or AR3A
        roof = f"ROOF AREA {num_w}"
```

This layer is important now that OCR is the only extraction path — it recovers values Tesseract splits or mis-segments.

### Layer 3 — Ollama Local LLM (last resort)

Only invoked when both Layer 1 and Layer 2 return `None` for a value.

```python
# Calls http://localhost:11434/api/generate  (no internet needed)
# Model: llama3.2:1b  (1.3 GB, installed locally)
# Prompt: short text snippet → returns {"roof_area": "...", "array_value": "..."}
```

**Setup (one-time):**
```bash
ollama pull llama3.2:1b   # download model once
ollama serve               # auto-starts on Mac
```

If Ollama is not running, `_ollama_parse()` catches the exception and returns `(None, None)` silently — the pipeline never crashes.

**When Layer 3 activates:**
```
OCR returned:  "RF AREA ONE | ARR. 2"
Layer 1 regex: no match (ONE is a word, not a digit)
Layer 2 fuzzy: no match (RF ≠ ROOF pattern)
Layer 3 Ollama: reads context → {"roof_area": "ROOF AREA 1", "array_value": "ARRAY 2"}
```

---

## Step 4 — Normalised Comparison

**Function:** `compare_viewports()`

```
Key Plan    "ROOF AREA  1 " → strip → "ROOF AREA 1"
Title Block "Roof Area 1"   → upper → "ROOF AREA 1"
                                    → MATCH ✓
```

Four conditions checked per field:

| Condition            | Output                                                        |
|----------------------|---------------------------------------------------------------|
| Both missing         | `<field>: MISSING in both viewports`                          |
| Key Plan missing     | `<field>: MISSING in Key Plan (Title Block='...')`            |
| Title Block missing  | `<field>: MISSING in Title Block (Key Plan='...')`            |
| Values differ        | `<field> MISMATCH -> Key Plan='...' vs Title Block='...'`     |

---

## Step 5 — Highlight PDF Output

**Function:** `highlight_region()`

For each mismatched page, PyMuPDF draws a **red semi-transparent rectangle** (`fill_opacity=0.22`) over the union bounding box of the Key Plan cluster and the Title Block cluster, with a `"MISMATCH"` label above each box.

```
Page 2 Key Plan cluster (OCR coords → fractional → PDF points):
  "ROOF AREA 1"  frac (0.078, 0.913, 0.120, 0.922)
  "ARRAY 2"      frac (0.079, 0.923, 0.103, 0.932)
  Union + 6pt pad → highlight drawn in PDF points
```

---

## Output Files

| File                                    | Description                                               |
|-----------------------------------------|-----------------------------------------------------------|
| `<name> - Mismatch Highlighted.pdf`     | Original PDF with red highlight boxes on mismatched pages |
| `<name> - Mismatch Highlighted.json`    | Per-page structured results                               |

### JSON structure

```json
[
  {
    "page": 2,
    "sheet_id": "",
    "key_plan":    { "roof_area": "ROOF AREA 1", "array_value": "ARRAY 2" },
    "title_block": { "roof_area": "ROOF AREA 1", "array_value": "ARRAY 1" },
    "mismatches":  ["Array Value MISMATCH -> Key Plan='ARRAY 2'  vs  Title Block='ARRAY 1'"],
    "has_mismatch": true,
    "extraction_method": "ocr",
    "detection_mode": "value"
  }
]
```

Note: `sheet_id` is currently empty under OCR because the sheet ID regex (`SM[\.\-]\d{3,}`) is strict and Tesseract sometimes mis-reads the dot/dash separator. Loosen the regex to `SM[\.\-_,\s]?\d{3,}` if you want that field populated.

---

## Multi-PDF Support

```bash
python task3.py                                   # default PDF
python task3.py drawing.pdf                       # single PDF
python task3.py a.pdf b.pdf c.pdf                 # multiple PDFs
python task3.py data/*.pdf                        # glob (shell expands)
python task3.py drawing.pdf --output result.pdf   # custom output name
```

Each PDF gets its own `<name> - Mismatch Highlighted.pdf` and `.json` automatically. A grand summary is printed when processing multiple files.

---

## Tech Stack

| Component        | Library / Tool                   | Used      | Purpose                                          |
|------------------|----------------------------------|-----------|--------------------------------------------------|
| PDF rendering    | PyMuPDF (`fitz`)                 | Always    | Rasterise each page to a 300 DPI image           |
| PDF writing      | PyMuPDF (`fitz`)                 | Always    | Draw highlight rectangles on output PDF          |
| OCR              | `pytesseract` + Tesseract binary | Always    | Word-level OCR on the rendered page              |
| Image handling   | Pillow (`PIL`)                   | Always    | In-memory image buffer for OCR input             |
| Pattern match    | `re` (stdlib)                    | Always    | Layer 1 regex + Layer 2 fuzzy extraction         |
| Local LLM        | Ollama `llama3.2:1b` (1.3 GB)    | If needed | Last-resort extraction when regex + fuzzy fail   |
| Data structures  | `dataclasses`                    | Always    | `ViewportData`, `PageResult`                     |
| Output           | `json` (stdlib)                  | Always    | Per-page structured JSON report                  |

No external API keys. No internet. No quota limits.

---

## Requirements

```bash
# Python packages (in project .venv)
pip install pymupdf pytesseract pillow

# Tesseract binary
#   macOS:  brew install tesseract
#   Ubuntu: sudo apt-get install tesseract-ocr

# Optional — only needed if Layer 3 Ollama fallback should run
ollama pull llama3.2:1b
ollama serve
```

---

## Known Limitations and Tuning Points

| Limitation                                    | Workaround                                              |
|-----------------------------------------------|---------------------------------------------------------|
| ~3–5 seconds per page (OCR is slow)           | Run in parallel per page, or cache results per PDF hash |
| `sheet_id` empty (strict `SM.\d{3}` regex)    | Loosen to `SM[\.\-_,\s]?\d{3,}`                         |
| Tesseract occasionally merges adjacent tokens | Layer 2 fuzzy scan recovers most cases                  |
| 300 DPI hard-coded in `_ocr_blocks()`         | Bump to 400 DPI for noisy scans, 200 DPI for speed      |
| `--psm 11` assumes sparse text                | Switch to `--psm 6` for dense table-like title blocks   |

---

## Alternative Approaches (for reference)

### PaddleOCR (higher accuracy, larger install)

```python
from paddleocr import PaddleOCR
ocr    = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr(image_path, cls=True)
```

| Property  | Value                                           |
|-----------|-------------------------------------------------|
| Accuracy  | 93–98% on engineering drawings                  |
| Cost      | Free (~200 MB model, local)                     |
| Speed     | ~1–3s/page GPU, ~5–10s CPU                      |
| Best for  | Scanned / rotated-text PDFs                     |
| Weakness  | Larger install footprint than Tesseract         |

### Vision LLM (Gemini / GPT-4o) as cloud fallback

```python
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[Part.from_bytes(png_bytes, "image/png"),
              Part.from_text(PROMPT)],
)
```

| Property  | Value                                                  |
|-----------|--------------------------------------------------------|
| Accuracy  | 95–99%                                                 |
| Cost      | Free tier (limited) / paid above quota                 |
| Best for  | Handwritten annotations, unusual layouts               |
| Weakness  | Quota limits, internet required, latency               |

### LayoutLMv3 Fine-tuned (enterprise)

```
PDF → image → LayoutLMv3 (Microsoft)
    → jointly models text + 2D spatial layout
    → fine-tuned token classification: KEY_PLAN_ROOF_AREA, TITLE_BLOCK_ARRAY
    → structured extraction → comparison
```

| Property  | Value                                                         |
|-----------|---------------------------------------------------------------|
| Accuracy  | 99%+ after fine-tuning                                        |
| Cost      | Free inference; needs GPU + 100–500 labelled training samples |
| Best for  | Large-scale, diverse drawing libraries                        |
| Weakness  | Training data required; GPU for inference                    |

---

## Decision Guide

```
Has the PDF been edited with an online PDF editor?
  YES → OCR only        (current solution — reads all edit types)
  NO  → OCR still works (slower than native, but accurate and simple)

Did regex + fuzzy both fail to parse a value?
  YES → Layer 3 Ollama llama3.2:1b  (current solution, auto)

OCR too slow?
  → Drop DPI to 200, or parallelise per page
  → Or re-introduce native extraction as a fast path for unedited PDFs

Processing thousands of drawings with varied layouts?
  → LayoutLMv3 fine-tuned
```