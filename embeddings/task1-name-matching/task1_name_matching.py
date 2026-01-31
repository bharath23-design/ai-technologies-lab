import lancedb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 1. Dataset: 30+ names with phonetic/spelling variations ---
NAMES = [
    "Geetha", "Gita", "Gitu", "Geeta", "Gitanjali",
    "Rahul", "Raul", "Rahool", "Rajul", "Rahul Kumar",
    "Priya", "Priyanka", "Priyam", "Preya", "Preeya",
    "Suresh", "Suresh Kumar", "Suresha", "Suraj", "Surya",
    "Amit", "Amith", "Amitabh", "Amita", "Amit Kumar",
    "Anita", "Aneeta", "Anitha", "Anit", "Anika",
    "Vijay", "Vijaya", "Vijju", "Vijayanand", "Vijayalakshmi",
]

# --- 2. Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 3. Create LanceDB table ---
DB_PATH = Path(__file__).parent / "lancedb_names"
db = lancedb.connect(str(DB_PATH))
embeddings = model.encode(NAMES)

table = db.create_table(
    "names",
    data=[{"name": n, "vector": e} for n, e in zip(NAMES, embeddings)],
    mode="overwrite",
)

# --- 4. Query function ---
def find_matching_names(query: str, top_k: int = 10):
    query_vec = model.encode([query])[0]
    results = table.search(query_vec).limit(top_k).to_pandas()
    results["similarity"] = 1 - results["_distance"]
    results = results.sort_values("similarity", ascending=False)
    return results[["name", "similarity"]]


# --- 5. Interactive loop ---
if __name__ == "__main__":
    print("=== Name Matching System (LanceDB + Sentence-Transformers) ===\n")
    while True:
        query = input("Enter a name (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break

        matches = find_matching_names(query)
        best = matches.iloc[0]
        print(f"\nBest Match: {best['name']}  (similarity: {best['similarity']:.4f})")
        print(f"\nTop Matches:")
        print(matches.to_string(index=False))
        print()
