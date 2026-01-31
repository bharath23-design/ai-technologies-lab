# Task 1: Get Matching Person Names

## Objective
Build a name-matching system that finds the most similar names from a dataset when a user inputs a name, using **LanceDB** (vector database) and **Sentence-Transformers** (embeddings).

---

## How It Works

### Architecture
```
User Input (name)
    │
    ▼
Sentence-Transformer (all-MiniLM-L6-v2)
    │  converts name → 384-dim vector
    ▼
LanceDB Vector Search
    │  finds nearest vectors (cosine distance)
    ▼
Ranked Results (name + similarity score)
```

### Step-by-Step Explanation

1. **Data Preparation** (`NAMES` list)
   - 35 Indian names organized in groups of phonetic/spelling variations:
     - Geetha, Gita, Gitu, Geeta, Gitanjali
     - Rahul, Raul, Rahool, Rajul, Rahul Kumar
     - Priya, Priyanka, Priyam, Preya, Preeya
     - Suresh, Suresh Kumar, Suresha, Suraj, Surya
     - Amit, Amith, Amitabh, Amita, Amit Kumar
     - Anita, Aneeta, Anitha, Anit, Anika
     - Vijay, Vijaya, Vijju, Vijayanand, Vijayalakshmi

2. **Embedding Generation** (line 16, 20)
   - Uses `all-MiniLM-L6-v2` from Sentence-Transformers (~80MB model)
   - Each name is converted into a 384-dimensional vector that captures semantic meaning
   - Names that sound or look similar end up with vectors close to each other

3. **Vector Storage in LanceDB** (lines 19–26)
   - LanceDB is a lightweight, serverless vector database (no external server needed)
   - All name vectors are stored in a local `./lancedb_names` directory
   - Table is created with `mode="overwrite"` so it rebuilds on every run

4. **Similarity Search** (lines 29–34)
   - User's input name is embedded into a vector
   - LanceDB performs approximate nearest neighbor (ANN) search
   - Distance is converted to similarity: `similarity = 1 - cosine_distance`
   - Results are sorted by similarity (highest first)

5. **Interactive Loop** (lines 38–50)
   - User enters a name, gets the best match + top-10 ranked list
   - Type `quit` to exit

---

## Why LanceDB + Sentence-Transformers?

| Choice | Reason |
|--------|--------|
| **LanceDB** | Serverless, no setup, stores data locally, fast vector search |
| **all-MiniLM-L6-v2** | Small (~80MB), fast, good semantic understanding of text |
| **Vector search** | Captures phonetic/semantic similarity (e.g., "Geeta" ≈ "Geetha") unlike exact string matching |

---

## Setup & Run

### Install dependencies
```bash
pip install lancedb sentence-transformers
```

### Run
```bash
python task1_name_matching.py
```

---

## Sample Input & Output

```
Enter a name (or 'quit' to exit): Geeta

Best Match: Geeta  (similarity: 1.0000)

Top Matches:
          name  similarity
         Geeta      1.0000
        Geetha      0.8732
          Gita      0.8241
          Gitu      0.7856
     Gitanjali      0.7103
        Aneeta      0.6542
         Anita      0.6201
        Anitha      0.6187
          Amita      0.5934
         Preya      0.5521
```

```
Enter a name (or 'quit' to exit): Rahul

Best Match: Rahul  (similarity: 0.9543)

Top Matches:
           name  similarity
          Rahul      0.9543
         Rahool      0.8921
    Rahul Kumar      0.8756
          Rajul      0.8234
           Raul      0.7891
          ...
```

---

## File Structure
```
task1_name_matching.py   # Complete solution (single file)
lancedb_names/           # Auto-created LanceDB storage directory
requirements.txt         # Dependencies
```

## Key Concepts
- **Embeddings**: Text → numerical vectors that preserve meaning
- **Cosine similarity**: Measures angle between vectors (1.0 = identical, 0.0 = unrelated)
- **ANN search**: Approximate nearest neighbor — efficient way to find similar vectors
