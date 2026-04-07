# Task 2 — Array Layout Component Counter

## Objective

Count every instance of each legend component in a solar array layout PDF using
a hybrid pixel-analysis + VLM approach.

**Input:**  `data/Array Layout.pdf`
**Output:** `data/array_layout_counts.json`
**Script:** `legend_counter_gemini_v0.py`

---

## Final Results

| Component   | Visual Identifier    | Count |
|-------------|----------------------|-------|
| Modules     | Blue Rectangles      | 44    |
| Full Rails  | Green Line Segments  | 16    |
| Cut Rails   | Red Line Segments    | 8     |
| Attachments | Blue Circles         | 108   |
| Splices     | Black Rectangles     | 16    |

---

## Pipeline

```
data/Array Layout.pdf
     │
     ▼
[Step 1] Render PDF at 300 DPI → PIL Image (3509 × 2480 px)
     │
     ▼
[Step 2] Pixel-based layout detection
         - Scan each pixel row for green-dominant pixels
         - Group into horizontal "rail bands"
         - Discard legend bands (largest inter-band gap heuristic)
         - num_tables = remaining bands ÷ 2
         - Output: num_tables=4, band_centers=[678,890,1105,...]
     │
     ▼
[Step 3] Three parallel counting methods
     │
     ├── Pixel: Modules
     │     - For each table: crop between upper_y and lower_y
     │     - Count vertical blue column-runs (≥20% of table height)
     │     - modules per table = dividers − 1
     │
     ├── Pixel: Attachments
     │     - For each rail row: scan ±30px strip around rail_y
     │     - Count blue column-clusters with width ≥10px
     │       (narrow spikes = module borders, wide clusters = circles)
     │
     └── VLM: Full Rails / Cut Rails / Splices
           - 1 Gemini 2.5 Pro call on the full image
           - Per-rail-row JSON response with sum constraint
     │
     ▼
Merge results → JSON output
```

---

## Why Hybrid (Pixel + VLM)?

VLM counting on a full 3509×2480 image is unreliable for small repeated items:

| Component   | Method    | Why                                                          |
|-------------|-----------|--------------------------------------------------------------|
| Modules     | Pixel     | Blue vertical dividers are machine-precise; VLM miscounted  |
| Attachments | Pixel     | Blue circle clusters are clearly wider than border lines     |
| Full Rails  | VLM       | Color segmentation confused green rails with other elements  |
| Cut Rails   | VLM       | Small count, VLM handles it reliably with focused prompt     |
| Splices     | VLM       | Black rectangles; pixel detection would need extra tuning    |

---

## Step 2 — Layout Detection (Dynamic Table Count)

The number of tables is **never hardcoded**. It is derived from the image itself:

```
1. Convert image to numpy array
2. Mark green pixels: G > 80 AND G > R×1.4 AND G > B×1.4
3. Sum green pixels per row → green_per_row
4. Threshold: row is a "rail band" if green_per_row > image_width × 1%
5. Collect band center y-positions
6. Find largest inter-band gap → everything before it is legend noise
7. num_tables = remaining_bands ÷ 2
```

Example for this PDF:
```
All bands:     [155, 173, 678, 890, 1105, 1317, 1532, 1744, 1959, 2171]
Gaps:          [18,  505, 212, 215, 212,  215,  212,  215,  212       ]
Largest gap:   505px at index 1  (legend → diagram boundary)
Diagram bands: [678, 890, 1105, 1317, 1532, 1744, 1959, 2171]
num_tables:    8 ÷ 2 = 4
```

---

## Step 3a — Module Counting (Pixel)

Modules are hollow blue rectangles sharing vertical borders. Counting vertical
blue divider lines and subtracting 1 gives the exact module count:

```
Between upper_y=678 and lower_y=890 (table 1):
  Blue column runs at x: [799, 997, 1199, 1400, 1601, 1802, 2003, 2204, 2405, 2607, 2808, 3009]
  12 vertical dividers → 12 − 1 = 11 modules per table
  4 tables × 11 = 44 total
```

---

## Step 3b — Attachment Counting (Pixel)

Attachments (blue hatched circles) and module border lines both appear as blue
vertical clusters in a rail strip, but at very different widths:

```
Cluster widths in table 1 upper rail strip (±30px around y=678):
  Narrow (6–7px):  module border lines  → skip (width < 10px)
  Wide  (48–49px): attachment circles   → count (width ≥ 10px)

Result per rail row: [14, 13, 14, 13, 14, 13, 14, 13]
Total: 4×14 + 4×13 = 56 + 52 = 108
```

---

## Step 3c — Rails / Splices (VLM)

A single Gemini 2.5 Pro call on the full image with a structured prompt:

- Prompt specifies exactly `num_tables` tables and `num_tables × 2` rail rows
- Forces a **per-rail-row JSON breakdown** to prevent the model from scanning
  only one row per table
- Sum constraint in the prompt ensures totals match the breakdown

```json
{
  "per_rail_row": {
    "table_1_upper": 3, "table_1_lower": 3,
    ...
  },
  "full_rails": 16,
  "cut_rails": 8,
  "splices": 16
}
```

---

## Dynamic vs Hardcoded

| Property              | How it is determined             |
|-----------------------|----------------------------------|
| Number of tables      | Pixel scan (green band detection)|
| Rail row y-positions  | Pixel scan (band_centers)        |
| Module count per table| Pixel scan (vertical dividers)   |
| Attachment count      | Pixel scan (wide cluster filter) |
| Prompt table counts   | Injected from detected values    |
| Fallback (scan fails) | Default to 4 tables              |

The code works for any solar array layout PDF with the same visual conventions
(green rails, blue module rectangles, blue circle attachments) regardless of
how many tables the layout contains.

---

## Limitations

| Limitation | Impact |
|------------|--------|
| Green rail detection assumes the rail color is consistently green-dominant | Different color schemes would require tuning the color thresholds |
| Blue vertical divider detection assumes modules share full-height borders | Layouts with partial borders or gaps in dividers would undercount |
| Wide-cluster attachment filter (≥10px) tuned at 300 DPI | At different render DPIs, the min_width threshold should scale proportionally |
| VLM rails/splices call can fail under high API load | Retried up to 5 times with exponential backoff |

---

## Tech Stack

| Component       | Library / Tool             | Purpose                              |
|-----------------|----------------------------|--------------------------------------|
| PDF rendering   | PyMuPDF (`fitz`)           | Render page to PIL image at 300 DPI  |
| Pixel analysis  | NumPy + PIL                | Layout detection, module & attachment counting |
| Vision LLM      | Gemini 2.5 Pro (Google)    | Count full rails, cut rails, splices |
| Parallelism     | `concurrent.futures`       | Rails VLM call runs alongside pixel counts |
| Output          | JSON                       | `data/array_layout_counts.json`      |
