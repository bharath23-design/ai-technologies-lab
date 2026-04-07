import json
import io
import shutil
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env_txt(path: Path) -> dict:
    """Parse a plain key=value txt file and return a dict."""
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config

_env = _load_env_txt(Path(__file__).parent.parent / "env.txt")

OPENAI_API_KEY = _env.get("openai_api_key", "")
HF_TOKEN       = _env.get("hg_token", "")   # used for HuggingFace Inference API (task 2)

LLM_MODEL      = "gpt-4o-mini"           # cheap + fast; swap to "gpt-4o" for higher accuracy
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR     = str(Path(__file__).parent / "chroma_store")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 5

# Default parameters used when none are supplied via CLI
DEFAULT_PARAMETERS = [
    "ground snow load",
    "roof snow load",
    "wind speed",
    "seismic SS",    # seismic spectral response — short period
    "seismic S1",    # seismic spectral response — 1 second
    "seismic SDS",   # design spectral acceleration — short period
    "seismic SD1",   # design spectral acceleration — 1 second
]


# ---------------------------------------------------------------------------
# Step 1: PDF -> Raw Text  (native text + OCR fallback for scanned pages)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()

        if len(text) < 50:
            # Scanned page — render at 200 dpi and OCR
            pix     = page.get_pixmap(dpi=200)
            img     = Image.open(io.BytesIO(pix.tobytes("png")))
            text    = pytesseract.image_to_string(img).strip()
            print(f"  Page {page_num + 1}: OCR applied (scanned/image page)")

        if text:
            pages_text.append(text)

    doc.close()
    return "\n\n".join(pages_text)


# ---------------------------------------------------------------------------
# Step 2: Chunk text
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Step 3: Embed chunks and store in ChromaDB
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list[str]) -> Chroma:
    # Wipe any existing store so old PDF embeddings don't bleed in
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


# ---------------------------------------------------------------------------
# Step 4: Retrieve top-k relevant chunks for a query
# ---------------------------------------------------------------------------

def retrieve_chunks(vector_store: Chroma, query: str, top_k: int = TOP_K) -> list[str]:
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


# ---------------------------------------------------------------------------
# Step 5: Structured output schema (Pydantic)
# ---------------------------------------------------------------------------

class ParameterResult(BaseModel):
    parameter:      str           = Field(description="Name of the extracted parameter")
    value:          Optional[str] = Field(description="Extracted value, or null if not found")
    unit:           Optional[str] = Field(description="Unit of measurement if present, else null")
    source_snippet: Optional[str] = Field(description="Short quote from context supporting the answer")


# ---------------------------------------------------------------------------
# Step 6: LLM — extract a single parameter using LangChain structured output
# ---------------------------------------------------------------------------

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a structural engineering document analyst specializing in "
            "building design reports. You are an expert at extracting design parameters "
            "such as snow loads, wind speeds, and seismic values (SS, S1, SDS, SD1) "
            "from structural reports.\n\n"
            "CRITICAL RULES:\n"
            "- Return the EXACT value as it appears in the document. Do NOT round, "
            "calculate, convert, or infer values.\n"
            "- Copy the number character-for-character from the source text.\n"
            "- If the document says 25.6, return 25.6 — not 26 or ~26.\n"
            "- Use only the context provided. If the value is not explicitly stated, "
            "return null."
        ),
    ),
    (
        "human",
        "Context:\n{context}\n\nExtract the EXACT value of the parameter: \"{parameter}\". "
        "Do not round or approximate — copy the number exactly as written in the document.",
    ),
])


def extract_parameter(parameter: str, context_chunks: list[str]) -> dict:
    context = "\n\n---\n\n".join(context_chunks)
    llm     = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    chain   = _PROMPT | llm.with_structured_output(ParameterResult, method="function_calling")
    result: ParameterResult = chain.invoke({"context": context, "parameter": parameter})
    return result.model_dump()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    parameters: list[str],
    output_path: str = "extracted_params.json",
    rebuild_store: bool = True,
) -> dict:

    print(f"\n{'='*50}")
    print(f"RAG PDF Parameter Extraction")
    print(f"PDF        : {pdf_path}")
    print(f"{'='*50}\n")

    # Step 1 — Extract text
    print("[1/4] Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"      {len(raw_text)} characters extracted\n")

    print(f"Parameters : {parameters}\n")

    # Step 2 — Chunk
    print("[2/4] Chunking text...")
    chunks = chunk_text(raw_text)
    print(f"      {len(chunks)} chunks created\n")

    # Step 3 — Vector store
    if rebuild_store:
        print("[3/4] Building vector store (ChromaDB)...")
        vector_store = build_vector_store(chunks)
    else:
        print("[3/4] Loading existing vector store...")
        vector_store = load_vector_store()
    print("      Vector store ready\n")

    # Step 4 — Extract each parameter
    print("[4/4] Querying LLM for each parameter...\n")
    results = {}

    for param in parameters:
        print(f"  -> '{param}'")
        top_chunks = retrieve_chunks(vector_store, query=f"What is {param}?")
        result     = extract_parameter(param, top_chunks)
        results[param] = result
        print(f"     value : {result.get('value')}")
        print(f"     unit  : {result.get('unit')}\n")

    # Save output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point  ← edit these three variables, then run: python extract_pdf_params.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- configure here ----
    PDF_PATH    = str(Path(__file__).parent / "data" / "Structural report 3.pdf")  # <-- change to your PDF
    PARAMETERS  = DEFAULT_PARAMETERS   # <-- or replace with a custom list, e.g. ["live load", "wind speed"]
    OUTPUT_PATH = str(Path(__file__).parent / "data" / "extracted_params3.json")
    # ------------------------

    output = run_pipeline(
        pdf_path=PDF_PATH,
        parameters=PARAMETERS,
        output_path=OUTPUT_PATH,
    )

    print("\nFinal Output:")
    print(json.dumps(output, indent=2))