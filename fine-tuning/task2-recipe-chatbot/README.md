# Task 2: Local LLM Integration & Recipe Chatbot

## Objective
Set up a local AI model, fine-tune it on recipe data, expose it via a FastAPI API, and build a web chatbot that suggests recipes based on user-provided ingredients.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Web Browser                      │
│         Chatbot UI (http://localhost:8000)        │
│         Served from templates/index.html          │
└────────────────────┬─────────────────────────────┘
                     │ POST /api/recipe
                     ▼
┌──────────────────────────────────────────────────┐
│              FastAPI Server (app.py)              │
│  - Serves index.html via Jinja2 templates        │
│  - Loads base model + LoRA adapter               │
│  - Tokenizes input with chat template            │
│  - Generates recipe via model.generate()         │
│  - Returns JSON response                         │
└────────────────────┬─────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐    ┌────────────────────┐
│ Qwen2.5-0.5B  │    │  LoRA Adapter      │
│ (Base Model)  │◄───│  (finetuned_model/)│
│    ~1GB       │    │  Fine-tuned on     │
│               │    │  20 recipes        │
└───────────────┘    └────────────────────┘
```

---

## How It Works — Step by Step

### Step 1: Data Preparation (`prepare_data.py`)

Creates `recipe_dataset.json` containing 20 recipe entries. Each entry has:
- **ingredients**: comma-separated list (e.g., `"egg, onion"`)
- **recipe**: a short recipe with name, steps, and serving suggestion

Example:
```json
{
  "ingredients": "egg, onion",
  "recipe": "Egg Onion Scramble: Heat oil, sauté diced onions until golden. Crack 2 eggs, season with salt and pepper, scramble together. Serve hot with toast."
}
```

### Step 2: Fine-Tuning (`finetune.py`)

| Component | Detail |
|-----------|--------|
| **Base model** | `Qwen/Qwen2.5-0.5B-Instruct` — 0.5B params, ~1GB download |
| **Method** | LoRA (Low-Rank Adaptation) via HuggingFace PEFT |
| **LoRA targets** | `q_proj`, `k_proj`, `v_proj`, `o_proj` (all attention layers) |
| **LoRA rank** | r=16, alpha=32, dropout=0.05 |
| **Training** | 30 epochs, batch size 2, gradient accumulation 2, lr=3e-4 |
| **Output** | Saved to `finetuned_model/` (~10MB adapter, not full model copy) |

**Why these LoRA settings?**
- **r=16, alpha=32**: Higher rank and alpha give the adapter more capacity to learn recipe-specific patterns
- **All 4 attention layers**: Training `q/k/v/o_proj` instead of just `q/v` improves the model's ability to retain fine-tuned knowledge
- **30 epochs**: More training passes help the small dataset (20 recipes) get properly memorized
- **lr=3e-4**: Slightly higher learning rate for faster convergence on a small dataset

**Why LoRA?**
- Only trains a small fraction of parameters (the adapter weights)
- Needs far less memory than full fine-tuning
- Adapter is tiny (~10MB) vs full model (~1GB)
- Works on CPU/MPS (Mac M3 compatible, no CUDA required)

**How training data is formatted:**
Each recipe is converted to a chat conversation using the model's chat template:
```
<system> You are a helpful recipe assistant...
<user> Ingredients: egg, onion
<assistant> Egg Onion Scramble: Heat oil, sauté diced onions...
```

### Step 3: FastAPI Server + Chatbot UI (`app.py` + `templates/index.html`)

**Server startup:**
1. Loads the base Qwen2.5-0.5B-Instruct model
2. Loads the LoRA adapter on top via `PeftModel.from_pretrained()`
3. Sets model to eval mode (no gradient computation)
4. Sets up Jinja2 template engine pointing to `templates/` directory

**API endpoint — `POST /api/recipe`:**
1. Receives `{"ingredients": "egg, onion"}`
2. Formats as a chat message with system prompt
3. Tokenizes using `apply_chat_template`
4. Runs `model.generate()` with:
   - `max_new_tokens=200` — limits response length
   - `temperature=0.3` — low temperature for more deterministic, recipe-faithful output
   - `top_p=0.85` — nucleus sampling
   - `repetition_penalty=1.3` — avoids repeated text
5. Decodes and returns the generated recipe as JSON

**Web UI — `GET /` → `templates/index.html`:**
- Separate HTML file served via Jinja2 templates (clean separation of concerns)
- Gradient-themed header with recipe icon
- Chat bubble UI with animated message transitions
- "Chef Bot" labels on bot responses
- Animated typing indicator (bouncing dots) while waiting for response
- Quick suggestion buttons for common ingredient combos
- Responsive design that works on desktop and mobile
- Custom scrollbar styling
- Hover animations on buttons

---

## Why Qwen2.5-0.5B-Instruct?

| Factor | Value |
|--------|-------|
| **Size** | ~1GB — fits easily on 8GB RAM Mac |
| **Parameters** | 0.5B — fast inference on CPU/MPS |
| **Instruction-tuned** | Already understands chat/instruction format |
| **Disk usage** | ~1GB model + ~10MB adapter = minimal on 256GB SSD |
| **Fine-tuning** | Trainable with LoRA on M3 / any standard laptop |

---

## Setup & Run

### Install dependencies
```bash
cd task2_recipe_chatbot
pip install -r requirements.txt
```

### Dependencies
| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | Load & run HuggingFace models |
| `peft` | LoRA fine-tuning |
| `datasets` | Dataset utilities |
| `accelerate` | Training acceleration |
| `fastapi` | Web API framework |
| `uvicorn` | ASGI server |
| `jinja2` | Template engine for serving HTML |
| `python-multipart` | Form data parsing |

### Run (3 commands, in order)
```bash
python prepare_data.py     # Creates recipe_dataset.json
python finetune.py         # Downloads model (~1GB), trains LoRA adapter
python app.py              # Starts server at http://localhost:8000
```

---

## Sample Input & Output

### Via Web UI
1. Open `http://localhost:8000` in browser
2. Type `egg, onion` or click a suggestion button
3. Bot responds: *"Egg Onion Scramble: Heat oil, sauté diced onions until golden. Crack 2 eggs, season with salt and pepper, scramble together. Serve hot with toast."*

### Via API (curl)
```bash
curl -X POST http://localhost:8000/api/recipe \
  -H "Content-Type: application/json" \
  -d '{"ingredients": "egg, onion"}'
```
Response:
```json
{
  "ingredients": "egg, onion",
  "recipe": "Egg Onion Scramble: Heat oil, sauté diced onions until golden. Crack 2 eggs, season with salt and pepper, scramble together. Serve hot with toast."
}
```

### More Examples

| Input | Expected Output |
|-------|-----------------|
| `egg, onion` | Egg Onion Scramble recipe |
| `chicken, garlic, soy sauce` | Garlic Soy Chicken recipe |
| `pasta, tomato, cheese` | Tomato Cheese Pasta recipe |
| `banana, milk, honey` | Banana Smoothie recipe |
| `rice, egg, soy sauce` | Egg Fried Rice recipe |

---

## File Structure
```
task2_recipe_chatbot/
├── requirements.txt           # Python dependencies
├── prepare_data.py            # Step 1: Generate recipe dataset
├── recipe_dataset.json        # Generated training data (20 recipes)
├── finetune.py                # Step 2: LoRA fine-tuning script
├── finetuned_model/           # Generated LoRA adapter weights
├── app.py                     # Step 3: FastAPI server (backend)
├── templates/
│   └── index.html             # Chatbot web UI (frontend)
└── README.md                  # This file
```

## Key Concepts

- **LLM (Large Language Model)**: Neural network trained on text data that can generate human-like responses
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning that trains small adapter layers instead of the full model
- **PEFT**: HuggingFace library for parameter-efficient fine-tuning methods
- **Chat template**: Standardized format for structuring system/user/assistant messages
- **FastAPI**: Modern Python web framework for building APIs with automatic OpenAPI docs
- **Jinja2 Templates**: Server-side template engine used to serve the HTML chatbot UI separately from the Python backend
