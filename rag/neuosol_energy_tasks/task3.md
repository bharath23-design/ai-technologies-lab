# Task – AutoCAD Text Mismatch Detection Across Viewports

## Description
Analyze the provided CAD drawing PDF and detect any AutoCAD text mismatches between the viewports within the same sheet.

---

## Goal

```
                ┌──────────────┐
                │   PDF Input  │
                └──────┬───────┘
                       ↓
        ┌──────────────────────────┐
        │ Text + Layout Extraction │
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Viewport Detection (CV)  │
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Text Normalization       │
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Embeddings + Matching    │
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Mismatch Detection       │
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Highlight PDF Output     │
        └──────────────────────────┘
```

---

## Mismatch Detection Logic

### Key Fields to Validate
- **Roof Area**
- **Array Value**

---

### Comparison Strategy

1. Extract values from:
   - **Key Plan Viewport**
     - Roof Area
     - Array Value
   - **Array Layout Viewport**
     - Roof Area
     - Array Value

2. Perform comparison:
   - Match **Roof Area (Key Plan vs Array Layout)**
   - Match **Array Value (Key Plan vs Array Layout)**

---

### Mismatch Conditions

Trigger mismatch if:
- Roof Area values are **not equal**
- Array values are **not equal**
- Array Layout values are **missing or inconsistent**

---

### Output Actions

If mismatch detected:
- 🚨 Send **alert/notification**
- 🔴 **Highlight the mismatched regions** in the PDF
- 📍 Mark bounding boxes around:
  - Key Plan values
  - Array Layout values

---

## Summary
The system extracts structured text from different viewports, compares critical engineering values (Roof Area & Array Value), detects inconsistencies, and generates a highlighted PDF with alerts for quick validation.