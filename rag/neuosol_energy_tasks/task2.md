# Task 2 — Array Layout PDF Visual Question Answering

## Objective

Given the solar array layout PDF, extract the diagram image and use GPT-4o Vision
to count every instance of each legend item present in the layout.

**Input PDF:** `data/Array Layout.pdf`
**Output:** `data/array_layout_counts.json`

---

## Legend Items to Count

| Legend       | Visual Description          |
|--------------|-----------------------------|
| FULL RAIL    | Green horizontal line       |
| CUT RAIL     | Red horizontal line         |
| ATTACHMENT   | Circle with hatch symbol    |
| SPLICE       | Small square symbol         |
| MODULE       | Blue rectangle              |

---

## Pipeline Design

```
data/Array Layout.pdf
     │
     ▼
Extract page as image (PyMuPDF → full page render at 200 DPI)
     │
     ▼
Encode image to base64 PNG
     │
     ▼
GPT-4o Vision API
  ┌────────────────────────────────────────────────────────┐
  │ "Count every FULL RAIL, CUT RAIL, ATTACHMENT,          │
  │  SPLICE and MODULE in the diagram."                    │
  └────────────────────────────────────────────────────────┘
     │
     ▼
Pydantic structured output (LangChain with_structured_output)
     │
     ▼
JSON output with count per legend item
```

---

## Expected Output

```json
{
  "full_rail_count":   12,
  "cut_rail_count":    4,
  "attachment_count":  80,
  "splice_count":      8,
  "module_count":      40,
  "notes": "Counts based on visual inspection of the array layout diagram."
}
```

---

## Tech Stack

| Layer          | Tool                                        | Why                                          |
|----------------|---------------------------------------------|----------------------------------------------|
| PDF rendering  | PyMuPDF (`fitz`)                            | Render full page as high-res image           |
| Image encoding | `base64` (stdlib)                           | Required for llama-cpp vision input          |
| Vision LLM     | `Qwen2-VL-2B-Instruct` (Q4_K_M GGUF)       | Local, fits in M3 8GB, general vision        |
| Runtime        | `llama-cpp-python` with Metal (Apple M3)    | GPU-accelerated inference on Apple Silicon   |
| Vision bridge  | `Qwen2VLChatHandler` + mmproj               | Connects image encoder to language model     |
| Output         | JSON (parsed from model response)           | Counts per legend type                       |

---

## Model Files

Download from [`bartowski/Qwen2-VL-2B-Instruct-GGUF`](https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF)
and place in `neuosol_energy_tasks/models/`:

| File | Size | Purpose |
|------|------|---------|
| `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` | ~1.5 GB | Main language model |
| `mmproj-Qwen2-VL-2B-Instruct-f32.gguf` | ~300 MB | Vision projector (image encoder) |

---

## Installation

```bash
# Install llama-cpp-python with Apple Metal GPU support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

pip install pymupdf pillow
```
