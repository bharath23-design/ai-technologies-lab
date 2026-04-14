"""
Task 3 – AutoCAD Text Mismatch Detection Across Viewports
==========================================================

Primary:  PyMuPDF native vector text extraction (100% accurate for AutoCAD exports).
Fallback: Tesseract OCR — triggered automatically when a page has fewer than
          MIN_NATIVE_WORDS embedded words (scanned / image-based PDFs).

Pipeline per page:
  1. Extract text blocks (native → OCR fallback if sparse)
  2. Classify blocks into Key Plan / Title Block by fractional position
  3. Parse "ROOF AREA X" and "ARRAY X" patterns from each viewport
  4. Compare — flag mismatches
  5. Highlight mismatched regions in output PDF
  6. Print structured alert report

Usage:
    python task3.py                                    # default PDF
    python task3.py drawing.pdf                        # single PDF
    python task3.py a.pdf b.pdf c.pdf                  # multiple PDFs
    python task3.py *.pdf                              # glob (shell expands)
    python task3.py drawing.pdf --output result.pdf   # custom output (single PDF only)

Requirements (all in project .venv):
    pymupdf, pytesseract, pillow
    tesseract binary: brew install tesseract
"""

from __future__ import annotations
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# Tesseract OCR — optional; only imported when the fallback is actually needed
try:
    import pytesseract
    from PIL import Image
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

# If native extraction yields fewer than this many words on the whole page,
# the page is treated as image-based and OCR is used instead.
MIN_NATIVE_WORDS = 20


# Highlight style
HIGHLIGHT_COLOR = (1.0, 0.2, 0.2)
FILL_OPACITY    = 0.22
BORDER_WIDTH    = 2.5

# Broad positional fallback thresholds — used only when NO value blocks found
# (Key Plan bottom-left, Title Block bottom-right convention)
_FB_KP_X_MAX = 0.30
_FB_KP_Y_MIN = 0.75
_FB_TB_X_MIN = 0.75
_FB_TB_Y_MIN = 0.65


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ViewportData:
    roof_area:   Optional[str] = None
    array_value: Optional[str] = None
    blocks:      list[dict]    = field(default_factory=list)   # raw blocks for bbox


@dataclass
class PageResult:
    page_number:       int
    sheet_id:          str = ""
    key_plan:          ViewportData = field(default_factory=ViewportData)
    title_block:       ViewportData = field(default_factory=ViewportData)
    mismatches:        list[str]    = field(default_factory=list)
    extraction_method: str = "native"   # "native" or "ocr"
    detection_mode:    str = "anchor"   # "anchor", "partial", or "fallback"

    @property
    def has_mismatch(self) -> bool:
        return bool(self.mismatches)


# ---------------------------------------------------------------------------
# Step 1: Extract text blocks with fractional bounding boxes
#         Primary: PyMuPDF native  |  Fallback: Tesseract OCR
# ---------------------------------------------------------------------------

def _native_blocks(page: fitz.Page) -> list[dict]:
    """Extract embedded vector text blocks with fractional bounding boxes."""
    pw, ph = page.rect.width, page.rect.height
    result = []
    for b in page.get_text("blocks"):   # (x0,y0,x1,y1, text, block_no, block_type)
        text = b[4].strip().replace("\n", " ")
        if text:
            result.append({
                "text": text,
                "x0": b[0] / pw, "y0": b[1] / ph,
                "x1": b[2] / pw, "y1": b[3] / ph,
                "source": "native",
            })
    return result


def _ocr_blocks(page: fitz.Page) -> list[dict]:
    """
    Render page to 300 DPI image and extract words via Tesseract OCR.
    Returns blocks in the same fractional-coordinate format as _native_blocks().
    """
    if not _TESSERACT_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR fallback required but pytesseract/Pillow not installed.\n"
            "Run: pip install pytesseract pillow\n"
            "And: brew install tesseract"
        )

    # Render page to image
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    iw, ih = img.size

    # Word-level OCR with bounding boxes
    data = pytesseract.image_to_data(
        img,
        config="--psm 11",          # sparse text — best for CAD drawings
        output_type=pytesseract.Output.DICT,
    )

    # Group words into lines by (block_num, par_num, line_num)
    lines: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word or int(data["conf"][i]) < 30:   # skip low-confidence noise
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in lines:
            lines[key] = {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h}
        else:
            lines[key]["x0"] = min(lines[key]["x0"], x)
            lines[key]["y0"] = min(lines[key]["y0"], y)
            lines[key]["x1"] = max(lines[key]["x1"], x + w)
            lines[key]["y1"] = max(lines[key]["y1"], y + h)
        lines[key]["words"].append(word)

    # Convert pixel coords → fractional page coords
    pw, ph = page.rect.width, page.rect.height
    # Scale factor: image pixels → PDF points
    sx, sy = pw / iw, ph / ih

    result = []
    for line in lines.values():
        text = " ".join(line["words"])
        result.append({
            "text": text,
            "x0": line["x0"] * sx / pw,
            "y0": line["y0"] * sy / ph,
            "x1": line["x1"] * sx / pw,
            "y1": line["y1"] * sy / ph,
            "source": "ocr",
        })
    return result


def extract_blocks(page: fitz.Page) -> tuple[list[dict], str]:
    """
    Extract text blocks from a page via Tesseract OCR only.

    We always rasterise the page and OCR it, regardless of whether the
    PDF has an embedded text layer. This is the only reliable way to
    pick up edits made with online PDF editors — whether the editor
    stored the edit as a real annotation, flattened it into the content
    stream, or drew it as an image overlay, OCR sees what a human sees.
    """
    return _ocr_blocks(page), "ocr"


# ---------------------------------------------------------------------------
# Step 2: Value-first viewport detection + block classification
#
# Strategy:
#   Find every block that contains a ROOF AREA or ARRAY value.
#   Sort them by x-position — leftmost cluster = Key Plan,
#   rightmost cluster = Title Block.
#   Expand a tight region around each cluster to collect all nearby blocks.
#   Fall back to positional thresholds only if no value blocks found at all.
# ---------------------------------------------------------------------------

# Quick pre-scan patterns (same as the extraction regexes)
_VALUE_RE = re.compile(
    r"(ROOF\s+AREA\s+\d+[A-Z]?|\bARRAY\s+\d+[A-Z]?\b)",
    re.IGNORECASE,
)

# Padding added around each value-block cluster when building the region
_CLUSTER_PAD = 0.08


def _value_blocks(blocks: list[dict]) -> list[dict]:
    """Return blocks that contain a ROOF AREA or ARRAY value."""
    return [b for b in blocks if _VALUE_RE.search(b["text"])
            and not re.search(r"ARRAY\s+LAYOUT", b["text"], re.IGNORECASE)]


def _cluster_region(cluster: list[dict]) -> tuple:
    """Return the tight bounding box (+ padding) around a list of blocks."""
    x0 = min(b["x0"] for b in cluster) - _CLUSTER_PAD
    y0 = min(b["y0"] for b in cluster) - _CLUSTER_PAD
    x1 = max(b["x1"] for b in cluster) + _CLUSTER_PAD
    y1 = max(b["y1"] for b in cluster) + _CLUSTER_PAD
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def _blocks_in_region(blocks: list[dict], region: tuple) -> list[dict]:
    """Return blocks whose centre falls inside the given fractional region."""
    rx0, ry0, rx1, ry1 = region
    result = []
    for b in blocks:
        cx = (b["x0"] + b["x1"]) / 2
        cy = (b["y0"] + b["y1"]) / 2
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            result.append(b)
    return result


def classify_blocks(blocks: list[dict]) -> tuple[list[dict], list[dict], str]:
    """
    Detect Key Plan and Title Block regions, return (kp_blocks, tb_blocks, mode).

    Detection modes:
      'value'    — both regions found by locating ROOF AREA / ARRAY value blocks
                   and splitting by x-position (most reliable)
      'fallback' — no value blocks found; broad positional thresholds used
    """
    vblocks = _value_blocks(blocks)

    if len(vblocks) >= 2:
        # Split value blocks into two x-position clusters:
        # sort by x-centre, then split at the midpoint gap
        vblocks_sorted = sorted(vblocks, key=lambda b: (b["x0"] + b["x1"]) / 2)
        # Find the largest x-gap between consecutive value blocks → split there
        gaps = [
            (vblocks_sorted[i + 1]["x0"] - vblocks_sorted[i]["x1"], i)
            for i in range(len(vblocks_sorted) - 1)
        ]
        split_idx = max(gaps, key=lambda t: t[0])[1] + 1
        left_vals  = vblocks_sorted[:split_idx]   # Key Plan  (smaller x)
        right_vals = vblocks_sorted[split_idx:]   # Title Block (larger x)

        kp_region = _cluster_region(left_vals)
        tb_region = _cluster_region(right_vals)
        mode = "value"

    elif len(vblocks) == 1:
        # Only one instance — determine which viewport it belongs to by x-position
        v = vblocks[0]
        cx = (v["x0"] + v["x1"]) / 2
        if cx < 0.5:
            kp_region = _cluster_region(vblocks)
            tb_region = (_FB_TB_X_MIN, _FB_TB_Y_MIN, 1.0, 1.0)
        else:
            kp_region = (0.0, _FB_KP_Y_MIN, _FB_KP_X_MAX, 1.0)
            tb_region = _cluster_region(vblocks)
        mode = "value"

    else:
        # No value blocks found — broad positional fallback
        kp_region = (0.0,          _FB_KP_Y_MIN, _FB_KP_X_MAX, 1.0)
        tb_region = (_FB_TB_X_MIN, _FB_TB_Y_MIN, 1.0,          1.0)
        mode = "fallback"

    kp_blocks = _blocks_in_region(blocks, kp_region)
    tb_blocks = _blocks_in_region(blocks, tb_region)

    return kp_blocks, tb_blocks, mode


# ---------------------------------------------------------------------------
# Step 3: Parse Roof Area and Array Value from a list of text blocks
#         Layer 1 — Regex (fast, exact)
#         Layer 2 — Fuzzy token scan (handles OCR spacing errors)
#         Layer 3 — Ollama local LLM (fallback when both layers fail)
# ---------------------------------------------------------------------------

# --- Regex patterns (strict) ------------------------------------------------
_ROOF_RE  = re.compile(r"ROOF\s+AREA\s+(\d+[A-Z]?)", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\bARRAY\s+(\d+[A-Z]?)\b",   re.IGNORECASE)

# --- Ollama config -----------------------------------------------------------
# llama3.2:1b  — 1.3 GB, fastest, sufficient for 2-field JSON extraction
# llama3.2     — 2.0 GB, slightly more accurate, already installed
OLLAMA_MODEL   = "llama3.2:1b"
OLLAMA_TIMEOUT = 15               # seconds


def _regex_parse(text: str) -> tuple[Optional[str], Optional[str]]:
    """Layer 1: strict regex on a single text string."""
    roof  = None
    array = None

    m = _ROOF_RE.search(text)
    if m:
        roof = f"ROOF AREA {m.group(1).upper()}"

    m = _ARRAY_RE.search(text)
    if m:
        candidate = m.group(1).upper()
        if candidate[0].isdigit():           # must start with a digit, not "LAYOUT"
            array = f"ARRAY {candidate}"

    return roof, array


def _fuzzy_token_parse(blocks: list[dict]) -> tuple[Optional[str], Optional[str]]:
    """
    Layer 2: fuzzy token scan.
    Handles OCR artefacts like "R0OF AREA 1", "ARR AY 2", "ROOF  AREA1",
    by scanning sliding windows of 2–3 adjacent words across all blocks.
    """
    # Collect all words in order (preserve original for number extraction)
    words = []
    for b in blocks:
        words.extend(b["text"].upper().split())

    roof  = None
    array = None

    for i, w in enumerate(words):
        # Match ROOF + AREA + <number> in positions i, i+1, i+2
        if re.match(r"R[O0]{1,2}F$", w):                          # ROOF / R0OF
            if i + 2 < len(words):
                area_tok = words[i + 1]
                num_tok  = words[i + 2]
                if re.match(r"AR[E3][A4]$", area_tok) and re.match(r"\d+[A-Z]?$", num_tok):
                    roof = f"ROOF AREA {num_tok}"

        # Match ARRAY + <number> — but not ARRAY LAYOUT
        if re.match(r"ARR[A4]Y$", w):                              # ARRAY / ARR4Y
            if i + 1 < len(words):
                num_tok = words[i + 1]
                if re.match(r"^\d+[A-Z]?$", num_tok):
                    array = f"ARRAY {num_tok}"

    return roof, array


def _ollama_parse(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Layer 3: ask a local Ollama LLM to extract the values when regex fails.
    Returns (roof_area, array_value) or (None, None) if Ollama is unavailable.
    """
    import urllib.request
    import urllib.error

    prompt = (
        "You are reading text extracted from a CAD engineering drawing.\n"
        "Extract ONLY these two values from the text below:\n"
        "1. Roof Area  (e.g. ROOF AREA 1, ROOF AREA 2)\n"
        "2. Array Value (e.g. ARRAY 1, ARRAY 2)\n\n"
        "Rules:\n"
        "- Reply with ONLY a JSON object, no explanation.\n"
        "- Copy values exactly as they appear.\n"
        "- Use null if not found.\n"
        'Format: {"roof_area": "<value or null>", "array_value": "<value or null>"}\n\n'
        f"Text:\n{raw_text}"
    )

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read())
            raw  = body.get("response", "").strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw)
            raw  = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            roof  = data.get("roof_area")
            array = data.get("array_value")
            # Normalise returned values
            if roof  and not re.search(r"\d", str(roof)):   roof  = None
            if array and not re.search(r"\d", str(array)):  array = None
            return roof, array
    except Exception:
        return None, None        # Ollama not running or model not available


def parse_viewport(blocks: list[dict]) -> ViewportData:
    """
    Three-layer extraction:
      1. Regex on each block text   (fast, exact)
      2. Fuzzy token scan across all blocks  (handles OCR noise)
      3. Ollama local LLM on concatenated raw text  (last resort)
    """
    roof_area   = None
    array_value = None

    # --- Layer 1: regex per block
    for b in blocks:
        r, a = _regex_parse(b["text"])
        if r and roof_area   is None: roof_area   = r
        if a and array_value is None: array_value = a
        if roof_area and array_value:
            break

    # --- Layer 2: fuzzy token scan (only if layer 1 missed something)
    if roof_area is None or array_value is None:
        r, a = _fuzzy_token_parse(blocks)
        if r and roof_area   is None: roof_area   = r
        if a and array_value is None: array_value = a

    # --- Layer 3: Ollama LLM (only if both previous layers failed)
    if roof_area is None or array_value is None:
        raw_text = " | ".join(b["text"] for b in blocks)
        r, a = _ollama_parse(raw_text)
        if r and roof_area   is None: roof_area   = r
        if a and array_value is None: array_value = a

    return ViewportData(roof_area=roof_area, array_value=array_value, blocks=blocks)


# ---------------------------------------------------------------------------
# Step 4: Normalise & compare
# ---------------------------------------------------------------------------

def _norm(v: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", v.strip().upper()) if v else None


def compare_viewports(kp: ViewportData, tb: ViewportData) -> list[str]:
    issues: list[str] = []

    def _check(label: str, kv: Optional[str], tv: Optional[str]) -> None:
        k, t = _norm(kv), _norm(tv)
        if k is None and t is None:
            issues.append(f"{label}: MISSING in both viewports")
        elif k is None:
            issues.append(f"{label}: MISSING in Key Plan (Title Block='{t}')")
        elif t is None:
            issues.append(f"{label}: MISSING in Title Block (Key Plan='{k}')")
        elif k != t:
            issues.append(f"{label} MISMATCH -> Key Plan='{k}'  vs  Title Block='{t}'")

    _check("Roof Area",   kp.roof_area,   tb.roof_area)
    _check("Array Value", kp.array_value, tb.array_value)
    return issues


# ---------------------------------------------------------------------------
# Step 5: Highlight mismatched regions in the PDF
# ---------------------------------------------------------------------------

def _blocks_union_rect(blocks: list[dict], page: fitz.Page) -> Optional[fitz.Rect]:
    """Return the union rectangle (in PDF pts) of a list of classified blocks."""
    if not blocks:
        return None
    pw, ph = page.rect.width, page.rect.height
    x0 = min(b["x0"] for b in blocks) * pw
    y0 = min(b["y0"] for b in blocks) * ph
    x1 = max(b["x1"] for b in blocks) * pw
    y1 = max(b["y1"] for b in blocks) * ph
    # Add a small padding
    pad = 6
    return fitz.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def highlight_region(page: fitz.Page, rect: fitz.Rect, label: str) -> None:
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(
        color=HIGHLIGHT_COLOR,
        fill=HIGHLIGHT_COLOR,
        fill_opacity=FILL_OPACITY,
        width=BORDER_WIDTH,
    )
    shape.commit()
    lbl_rect = fitz.Rect(rect.x0, max(rect.y0 - 16, 0), rect.x1, rect.y0)
    page.insert_textbox(lbl_rect, label, fontsize=6, color=HIGHLIGHT_COLOR, align=0)


# ---------------------------------------------------------------------------
# Step 6: Alert output
# ---------------------------------------------------------------------------

def print_alert_report(results: list[PageResult]) -> None:
    sep        = "=" * 70
    mismatched = [r for r in results if r.has_mismatch]

    print(sep)
    print("MISMATCH DETECTION REPORT")
    print(sep)

    for r in results:
        status = "MISMATCH DETECTED" if r.has_mismatch else "OK"
        tags   = f"[{r.extraction_method.upper()}][{r.detection_mode.upper()}]"
        print(f"Page {r.page_number} | Sheet: {r.sheet_id or 'N/A'} | {tags} | {status}")
        kp, tb = r.key_plan, r.title_block
        print(f"  Key Plan    -> Roof Area: {str(kp.roof_area):<20}  Array: {kp.array_value}")
        print(f"  Title Block -> Roof Area: {str(tb.roof_area):<20}  Array: {tb.array_value}")
        for m in r.mismatches:
            print(f"  [MISMATCH] {m}")

    print(sep)
    print(f"SUMMARY: {len(mismatched)} mismatch(es) found across {len(results)} page(s)")
    if mismatched:
        for r in mismatched:
            print(f"  ALERT Page {r.page_number} ({r.sheet_id}): {'; '.join(r.mismatches)}")
    else:
        print("  All pages consistent. No action required.")
    print(sep)


def build_alert_text(results: list[PageResult]) -> str:
    mismatched = [r for r in results if r.has_mismatch]
    if not mismatched:
        return "STATUS: OK - No mismatches detected. All viewports are consistent."

    lines = ["STATUS: ALERT - Mismatches detected!\n"]
    for r in mismatched:
        lines.append(f"Page {r.page_number} | Sheet: {r.sheet_id}")
        lines.append(f"  Key Plan    -> Roof Area: {r.key_plan.roof_area}  |  Array: {r.key_plan.array_value}")
        lines.append(f"  Title Block -> Roof Area: {r.title_block.roof_area}  |  Array: {r.title_block.array_value}")
        for m in r.mismatches:
            lines.append(f"  *** {m}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: str, output_pdf_path: str) -> list[PageResult]:
    doc     = fitz.open(pdf_path)
    total   = doc.page_count
    results: list[PageResult] = []

    for idx in range(total):
        page     = doc[idx]
        page_num = idx + 1

        # Extract + classify text blocks (native or OCR fallback)
        blocks, method               = extract_blocks(page)
        kp_blocks, tb_blocks, d_mode = classify_blocks(blocks)

        # Parse viewport values
        kp_data = parse_viewport(kp_blocks)
        tb_data = parse_viewport(tb_blocks)

        # Compare
        mismatches = compare_viewports(kp_data, tb_data)

        # Sheet ID
        sheet_id = ""
        for b in tb_blocks:
            m = re.search(r"SM[\.\-]\d{3,}", b["text"], re.IGNORECASE)
            if m:
                sheet_id = m.group(0).upper()
                break

        results.append(PageResult(
            page_number=page_num,
            sheet_id=sheet_id,
            key_plan=kp_data,
            title_block=tb_data,
            mismatches=mismatches,
            extraction_method=method,
            detection_mode=d_mode,
        ))

    # Write highlighted PDF
    for result in results:
        if result.has_mismatch:
            page  = doc[result.page_number - 1]
            label = "MISMATCH"

            kp_rect = _blocks_union_rect(result.key_plan.blocks, page)
            tb_rect = _blocks_union_rect(result.title_block.blocks, page)

            if kp_rect:
                highlight_region(page, kp_rect, label)
            if tb_rect:
                highlight_region(page, tb_rect, label)

    doc.save(output_pdf_path)
    doc.close()

    # Save JSON report
    json_path = Path(output_pdf_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(
            [
                {
                    "page": r.page_number,
                    "sheet_id": r.sheet_id,
                    "key_plan":    {"roof_area": r.key_plan.roof_area,    "array_value": r.key_plan.array_value},
                    "title_block": {"roof_area": r.title_block.roof_area, "array_value": r.title_block.array_value},
                    "mismatches":         r.mismatches,
                    "has_mismatch":        r.has_mismatch,
                    "extraction_method":   r.extraction_method,
                    "detection_mode":      r.detection_mode,
                }
                for r in results
            ],
            f, indent=2,
        )

    return results


# ---------------------------------------------------------------------------
# Entry point  — supports any number of PDF inputs
# ---------------------------------------------------------------------------

def _output_path(pdf_path: str) -> str:
    """Derive output path by appending ' - Mismatch Highlighted' before the extension."""
    p = Path(pdf_path)
    return str(p.parent / f"{p.stem} - Mismatch Highlighted{p.suffix}")


if __name__ == "__main__":
    DEFAULT_PDF = str(Path(__file__).parent / "data" / "Test Drawing 2.pdf")

    # Collect PDF inputs from argv (skip --output flag and its value)
    args      = sys.argv[1:]
    skip_next = False
    pdf_inputs: list[str] = []
    custom_output: Optional[str] = None

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--output":
            custom_output = args[i + 1] if i + 1 < len(args) else None
            skip_next = True
        else:
            pdf_inputs.append(arg)

    if not pdf_inputs:
        pdf_inputs = [DEFAULT_PDF]

    all_results: dict[str, list[PageResult]] = {}

    for pdf_in in pdf_inputs:
        if not Path(pdf_in).exists():
            print(f"[SKIP] File not found: {pdf_in}")
            continue

        pdf_out = custom_output if (custom_output and len(pdf_inputs) == 1) else _output_path(pdf_in)

        if len(pdf_inputs) > 1:
            print(f"\n{'─'*70}")
            print(f"FILE: {pdf_in}")
            print(f"{'─'*70}")

        results = run_pipeline(pdf_in, pdf_out)
        all_results[pdf_in] = results

        print_alert_report(results)
        print()
        print(build_alert_text(results))
        print()
        print(f"Highlighted PDF : {pdf_out}")
        print(f"JSON report     : {Path(pdf_out).with_suffix('.json')}")

    # Grand summary when processing multiple files
    if len(all_results) > 1:
        total_mismatches = sum(
            sum(1 for r in results if r.has_mismatch)
            for results in all_results.values()
        )
        print(f"\n{'='*70}")
        print(f"GRAND SUMMARY: {len(all_results)} PDF(s) processed  |  {total_mismatches} mismatch page(s) total")
        print(f"{'='*70}")