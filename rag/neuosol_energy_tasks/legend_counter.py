#!/usr/bin/env python3
"""
Array Layout Legend Counter v5 — Gemini VLM Only

Strategy:
  1. Render PDF at high DPI
  2. Detect number of tables via pixel-level color analysis (green rail bands)
  3. Count components using Gemini VLM (multi-pass, take median)
"""

import io, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env_txt(path: Path) -> dict:
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line:
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    return cfg

_env = _load_env_txt(Path(__file__).parent.parent / "env.txt")

GEMINI_API_KEY = _env["gemini_api_key"]
PDF_PATH       = str(Path(__file__).parent / "data" / "Array Layout 2.pdf")
OUTPUT_PATH    = str(Path(__file__).parent / "data" / "array_layout_counts2.json")

GEMINI_MODEL = "gemini-2.5-pro"
RENDER_DPI   = 300
MAX_RETRIES  = 5
BASE_WAIT    = 60
NUM_PASSES   = 3

client = genai.Client(api_key=GEMINI_API_KEY)

COMPONENT_KEYS = ["modules", "full_rails", "cut_rails", "attachments", "splices"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_pdf_page(pdf_path: str, page: int = 0, dpi: int = 300) -> Image.Image:
    doc = fitz.open(pdf_path)
    pix = doc[page].get_pixmap(dpi=dpi)
    doc.close()
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    print(f"  Rendered {img.size[0]}x{img.size[1]} @ {dpi} DPI")
    return img


def _call_gemini(contents, config, label="request"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                wait = int(float(match.group(1))) + 5 if match else BASE_WAIT * attempt
                print(f"\n  [Rate limit on {label}] Waiting {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for {label}")


def _extract_text(response) -> str:
    try:
        if response.text is not None:
            return response.text.strip()
    except Exception:
        pass
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise RuntimeError(f"Empty Gemini response: {response}")
    parts = getattr(candidates[0].content, "parts", None) or []
    texts, thinking_texts = [], []
    for p in parts:
        t = getattr(p, "text", None)
        if not t:
            continue
        if getattr(p, "thought", False):
            thinking_texts.append(t)
        else:
            texts.append(t)
    result = texts or thinking_texts
    if not result:
        raise RuntimeError(f"No text in Gemini response parts: {parts}")
    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Step 0 — Detect layout (number of tables) via pixel color analysis
# ---------------------------------------------------------------------------

def detect_tables_from_image(image: Image.Image):
    """
    Find horizontal green-rail bands. Returns (num_tables, diagram_band_centers).
    diagram_band_centers is a list of y-positions [upper_1, lower_1, upper_2, lower_2, ...]
    after stripping legend bands.
    """
    arr = np.array(image).astype(np.int32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_mask = (g > 80) & (g > r * 1.4) & (g > b * 1.4)
    green_per_row = green_mask.sum(axis=1)
    threshold = image.width * 0.01

    band_centers = []
    in_band = False
    band_start = 0
    for y, count in enumerate(green_per_row):
        if count > threshold and not in_band:
            in_band = True
            band_start = y
        elif count <= threshold and in_band:
            in_band = False
            band_centers.append((band_start + y) // 2)
    if in_band:
        band_centers.append((band_start + len(green_per_row)) // 2)

    if len(band_centers) < 2:
        return max(1, len(band_centers)), band_centers

    gaps = [band_centers[i + 1] - band_centers[i] for i in range(len(band_centers) - 1)]
    max_gap_idx = gaps.index(max(gaps))
    median_gap = sorted(gaps)[len(gaps) // 2]

    if gaps[max_gap_idx] > median_gap * 2:
        diagram_bands = band_centers[max_gap_idx + 1:]
        print(f"  Skipped {max_gap_idx + 1} legend band(s), "
              f"{len(diagram_bands)} diagram rail bands remain")
    else:
        diagram_bands = band_centers

    return max(1, len(diagram_bands) // 2), diagram_bands


def detect_layout(image: Image.Image):
    """Returns (num_tables, band_centers). Falls back to (4, []) on failure."""
    try:
        n, bands = detect_tables_from_image(image)
        print(f"  Image analysis: {len(bands)} rail bands → {n} tables")
        return n, bands
    except Exception as e:
        print(f"  Image analysis failed ({e}), defaulting to 4 tables")
        return 4, []


# ---------------------------------------------------------------------------
# Step 1 — Three focused counting prompts (templates)
# ---------------------------------------------------------------------------

# Visual identifiers from the diagram legend — injected into every prompt
_LEGEND = """\
COMPONENT VISUAL IDENTIFIERS (from diagram legend):
| Component   | Visual Identifier   |
|-------------|---------------------|
| Modules     | Blue Rectangles     |
| Full Rails  | Green Line Segments |
| Cut Rails   | Red Line Segments   |
| Attachments | Blue Circles        |
| Splices     | Black Rectangles    |"""


# ── Call 1: rails + splices on the full image ────────────────────────────────

RAILS_SPLICES_PROMPT = ("""\
You are an expert engineering diagram analyst. Count ONLY rails and splices
in this solar panel array layout diagram.

""" + _LEGEND + """

DIAGRAM STRUCTURE:
There are exactly {num_tables} tables stacked vertically. Each table has TWO
completely independent horizontal rail rows: UPPER and LOWER.
That is {num_rail_rows} rail rows total. Do NOT count the legend/key area.

CRITICAL: upper and lower rail rows are SEPARATE — each has its own rails and splices.

COLOR RULES — be very precise about this:
  - Full Rails  : segments that are CLEARLY GREEN (hue ~120°, not yellowish, not teal).
  - Cut Rails   : segments that are CLEARLY RED (hue ~0°/360°, not orange, not pink).
  - If you are unsure whether a segment is green or red, look at the legend color key
    at the bottom or top of the diagram and compare carefully.

SPLICES — what they look like:
  - Small BLACK (or very dark) filled rectangles that appear at the JOINT between
    two rail segments (where one rail ends and another begins).
  - They sit directly ON the rail line at the junction point, overlapping the seam.
  - There is one splice at EVERY joint between two rail segments.
  - Do NOT skip junctions — scan carefully left to right on every rail row.

COUNTING METHOD — one rail row at a time:
  For EACH of the {num_rail_rows} rows (table_1_upper, table_1_lower, ...):
    1. Scan left to right.
    2. Count every GREEN Line Segment → full_rails (must be clearly green, not red).
    3. Count every RED Line Segment  → cut_rails (must be clearly red, not green).
    4. Count every BLACK Rectangle at rail junctions → splices.

Reply with ONLY a JSON object:
{{{{
  "per_rail_row": {{{{ {per_rail_row_keys} }}}},
  "full_rails": <int>, "cut_rails": <int>, "splices": <int>
}}}}
Each total MUST equal the sum across all per_rail_row entries.""")


# ── Call 2: modules — receives a CROPPED image of ONE table ─────────────────

MODULE_CROP_PROMPT = ("""\
You are counting solar modules in this CROPPED VIEW of a single table.

""" + _LEGEND + """

This image shows exactly ONE table. The two Green Line Segments (rails) run
horizontally near the top and bottom of the crop.

WHAT A MODULE LOOKS LIKE:
- A HOLLOW rectangle — only the border/outline is drawn in blue, the inside
  is empty (no fill). Think of it as a blue rectangular frame.
- They are all roughly the same size and sit in a row between the two rails.
- They are NOT filled with any color — just the blue outline.

WHAT IS NOT A MODULE:
- The outer border/frame of the entire diagram — do NOT count this.
- The rails themselves (horizontal green lines).
- Any gap or empty column position between modules.
- Partial rectangles that are cut off at the image edge.

COUNTING METHOD:
1. Locate the upper rail (green line near top) and lower rail (green line near bottom).
2. Look ONLY in the space between those two rails.
3. Count every hollow blue rectangular outline you see, left to right.
4. If a position between the rails has NO blue rectangle outline → skip it (it is a gap).

Reply with ONLY a JSON object:
{"modules": <int>}""")


# ── Call 3: attachments — receives a CROPPED image of ONE rail row ───────────

ATTACHMENT_CROP_PROMPT = ("""\
You are counting attachment hardware in this CROPPED VIEW of a single rail row.

""" + _LEGEND + """

This image shows exactly ONE horizontal rail row.

WHAT AN ATTACHMENT LOOKS LIKE:
- A BLUE CIRCLE with lines drawn INSIDE the circle (hatched or cross-hatched).
  The interior of the circle has diagonal lines, an X pattern, or parallel
  lines through it — it is NOT a plain solid or empty circle.
- The circle sits directly ON the rail line.
- They appear at regular intervals along the rail AND at both ends.

WHAT IS NOT AN ATTACHMENT:
- Plain circles with no interior lines.
- Square or rectangular shapes.
- Module rectangles above or below the rail.

COUNTING METHOD:
1. Scan the rail from LEFT to RIGHT.
2. Count every blue circle that has interior lines/hatching sitting on the rail.
3. Make sure to include the circle at the very LEFT end of the rail.
4. Make sure to include the circle at the very RIGHT end of the rail.
5. Count every one in between.

Reply with ONLY a JSON object:
{"attachments": <int>}""")


# ── Call: modules + attachments together on the full image ──────────────────

MODULES_ATTACHMENTS_PROMPT = ("""\
You are counting components in this solar panel array layout diagram.

""" + _LEGEND + """

DIAGRAM STRUCTURE:
There are exactly {num_tables} tables stacked vertically. Each table has TWO
horizontal rail rows (UPPER and LOWER). Do NOT count anything in the legend/key area.

COUNT MODULES:
- Hollow blue rectangular outlines (frames) that sit between the two rails of each table.
- The inside of each rectangle is EMPTY — only the blue border/outline is visible.
- They are all roughly the same size, lined up in a row between the rails.
- Do NOT count the outer diagram border or the rails themselves.
- Count total modules across ALL {num_tables} tables.

COUNT ATTACHMENTS:
- Blue circles with interior hatching (diagonal lines / X pattern) sitting ON the rails.
- They appear at regular intervals along every rail row AND at both ends of each rail.
- Count total attachments across ALL {num_rail_rows} rail rows.

COUNTING METHOD:
1. For modules: scan each table top-to-bottom, count hollow blue rectangles between the rails.
2. For attachments: scan each rail row left-to-right, count every hatched blue circle on the rail.

Reply with ONLY a JSON object:
{{"modules": <int>, "attachments": <int>}}""")




# ---------------------------------------------------------------------------
# Counting functions — 3 parallel specialized calls
# ---------------------------------------------------------------------------

def _gemini_json(prompt: str, image: Image.Image, label: str) -> dict:
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )
    response = _call_gemini(contents=[prompt, image], config=config, label=label)
    return json.loads(_extract_text(response))


def _crop(image: Image.Image, y1: int, y2: int) -> Image.Image:
    return image.crop((0, max(0, y1), image.width, min(image.height, y2)))


# ---------------------------------------------------------------------------
# Pixel-based counting (deterministic, no API calls)
# ---------------------------------------------------------------------------

def _blue_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (b > 80) & (b > r * 1.3) & (b > g * 1.3)


def count_modules_pixel(image: Image.Image, band_centers: list) -> list:
    """
    Count modules per table by detecting vertical blue divider lines.
    Each module is bordered by a blue vertical line; modules = dividers - 1.
    """
    arr = np.array(image).astype(np.int32)
    blue = _blue_mask(arr)
    counts = []
    margin = 25  # skip left/right outer diagram border so it isn't counted as a divider
    for t in range(len(band_centers) // 2):
        upper_y, lower_y = band_centers[2 * t], band_centers[2 * t + 1]
        between = blue[upper_y:lower_y, margin:-margin]
        blue_per_col = between.sum(axis=0)
        height = lower_y - upper_y
        threshold = height * 0.2          # column must be ≥20% blue to be a divider
        dividers, in_run = 0, False
        for cnt in blue_per_col:
            if cnt > threshold and not in_run:
                dividers += 1; in_run = True
            elif cnt <= threshold:
                in_run = False
        counts.append(max(0, dividers - 1))  # dividers - 1 = modules
    return counts


def count_attachments_pixel(image: Image.Image, band_centers: list) -> list:
    """
    Count attachment circles per rail row.
    Circles are wide blue clusters (≥10px); narrow spikes are module border lines.
    """
    arr = np.array(image).astype(np.int32)
    blue = _blue_mask(arr)
    pad, min_width, col_threshold = 30, 7, 2
    counts = []
    for rail_y in band_centers:
        strip = blue[max(0, rail_y - pad):rail_y + pad, :]
        blue_per_col = strip.sum(axis=0)
        clusters, in_cluster, w = 0, False, 0
        for cnt in blue_per_col:
            if cnt > col_threshold:
                if not in_cluster: in_cluster = True; w = 1
                else: w += 1
            else:
                if in_cluster:
                    if w >= min_width: clusters += 1
                    in_cluster = False; w = 0
        if in_cluster and w >= min_width:
            clusters += 1
        counts.append(clusters)
    return counts


def count_modules_attachments_vlm(image: Image.Image, num_tables: int) -> dict:
    """
    Run 5 parallel Gemini calls for modules + attachments on the full image.
    Each call returns {"modules": int, "attachments": int}.
    Returns the median of each across the 5 passes.
    """
    nr = num_tables * 2
    prompt = MODULES_ATTACHMENTS_PROMPT.format(
        num_tables=num_tables,
        num_rail_rows=nr,
    )
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    def _single_pass(i: int) -> dict:
        response = _call_gemini(
            contents=[prompt, image],
            config=config,
            label=f"modules-attachments-pass-{i + 1}",
        )
        return json.loads(_extract_text(response))

    results = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_single_pass, i): i for i in range(NUM_PASSES)}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  [modules_attachments VLM pass failed] {e}")

    if not results:
        return {"modules_vlm": 0, "attachments_vlm": 0}

    mod_vals  = sorted(r.get("modules", 0)     for r in results)
    att_vals  = sorted(r.get("attachments", 0) for r in results)
    med_mod   = mod_vals[len(mod_vals) // 2]
    med_att   = att_vals[len(att_vals) // 2]

    print(f"  [modules VLM]      passes={[r.get('modules', 0) for r in results]}  median={med_mod}")
    print(f"  [attachments VLM]  passes={[r.get('attachments', 0) for r in results]}  median={med_att}")

    return {"modules_vlm": med_mod, "attachments_vlm": med_att}


def count_all_parallel(image: Image.Image, num_tables: int,
                       band_centers: list) -> dict:
    """
    Counting strategy:
      - Modules     : pixel-based (blue vertical dividers)  + 5-pass VLM
      - Attachments : pixel-based (wide blue clusters)      + 5-pass VLM
      - Rails/splices: 1 Gemini VLM call on the full image
    """
    nr = num_tables * 2
    per_rail_row_keys = ", ".join(
        f'"table_{t}_{side}": <int>'
        for t in range(1, num_tables + 1)
        for side in ("upper", "lower")
    )

    # ── Pixel-based counts (instant, deterministic) ─────────────────────────
    if band_centers:
        mod_per_table = count_modules_pixel(image, band_centers)
        att_per_row   = count_attachments_pixel(image, band_centers)
    else:
        mod_per_table = [0] * num_tables
        att_per_row   = [0] * nr

    modules_pixel     = sum(mod_per_table)
    attachments_pixel = sum(att_per_row)
    print(f"  [modules pixel]      per_table={mod_per_table}  total={modules_pixel}")
    print(f"  [attachments pixel]  per_rail_row={att_per_row}  total={attachments_pixel}")

    # ── VLM calls in parallel: rails/splices AND 5-pass modules/attachments ─
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_rs = ex.submit(
            _gemini_json,
            RAILS_SPLICES_PROMPT.format(
                num_tables=num_tables,
                num_rail_rows=nr,
                per_rail_row_keys=per_rail_row_keys,
            ),
            image,
            "count-rails-splices",
        )
        fut_ma = ex.submit(count_modules_attachments_vlm, image, num_tables)
        rs = fut_rs.result()
        ma = fut_ma.result()

    print(f"  [rails_splices VLM]  {json.dumps({k: rs.get(k,0) for k in ('full_rails','cut_rails','splices')})}")
    if "per_rail_row" in rs:
        print(f"  [rails breakdown]    {rs['per_rail_row']}")

    return {
        "modules":     {"pixel": modules_pixel,     "vlm": ma["modules_vlm"]},
        "full_rails":  rs.get("full_rails", 0),
        "cut_rails":   rs.get("cut_rails", 0),
        "attachments": {"pixel": attachments_pixel, "vlm": ma["attachments_vlm"]},
        "splices":     rs.get("splices", 0),
    }



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pdf_path: str, output_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Legend Counter v5 ({GEMINI_MODEL} — VLM Only)")
    print(f"{'='*60}\n")

    # Step 1: Render
    print("[1/4] Rendering PDF at high DPI...")
    image = render_pdf_page(pdf_path, dpi=RENDER_DPI)

    # Step 2: Detect layout
    print("\n[2/3] Detecting diagram layout...")
    try:
        num_tables, band_centers = detect_layout(image)
        print(f"  Detected {num_tables} tables ({num_tables * 2} rail rows total)")
    except Exception as e:
        num_tables, band_centers = 4, []
        print(f"  Layout detection failed ({e}), defaulting to {num_tables} tables")

    # Step 3: Count components
    print("\n[3/3] Counting components...")
    initial = count_all_parallel(image, num_tables, band_centers)
    print(f"  Combined: {json.dumps(initial)}")

    final = initial

    # Print final results
    print(f"\n  {'='*40}")
    print(f"  FINAL RESULTS")
    print(f"  {'='*40}")
    for k in COMPONENT_KEYS:
        label = k.upper().replace('_', ' ')
        v = final[k]
        if isinstance(v, dict):
            print(f"  {label:15s}: pixel={v['pixel']}  vlm={v['vlm']}")
        else:
            print(f"  {label:15s}: {v}")

    with open(output_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to: {output_path}")
    return final


if __name__ == "__main__":
    out = run_pipeline(pdf_path=PDF_PATH, output_path=OUTPUT_PATH)
    print("\nFinal:", json.dumps(out, indent=2))