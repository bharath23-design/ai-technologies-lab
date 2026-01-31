"""Step 3: FastAPI server with chatbot UI that uses the fine-tuned model."""
import torch
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = Path(__file__).parent / "finetuned_model"
TEMPLATE_DIR = Path(__file__).parent / "templates"
SYSTEM_PROMPT = "You are a helpful recipe assistant. Given a list of ingredients, suggest a recipe."

# --- Load model once at startup ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
model.eval()
print("Model loaded.")

app = FastAPI(title="Recipe Chatbot")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


class Query(BaseModel):
    ingredients: str


def generate_recipe(ingredients: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ingredients: {ingredients}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, temperature=0.3,
            do_sample=True, top_p=0.85, repetition_penalty=1.3,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


# --- API endpoint (JSON) ---
@app.post("/api/recipe")
async def get_recipe(query: Query):
    recipe = generate_recipe(query.ingredients)
    return JSONResponse({"ingredients": query.ingredients, "recipe": recipe})


# --- Chatbot Web UI ---
@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
