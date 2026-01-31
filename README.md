# AI Technologies Lab

A collection of AI projects covering embeddings, fine-tuning, AI agents, MCP servers, RAG, and more.

## Prerequisites

- Python 3.12+
- pip

## Repository Structure

```
ai-technologies-lab/
│
├── embeddings/                        # Vector embeddings & similarity search
│   └── task1-name-matching/           # Name matching using LanceDB + Sentence-Transformers
│
├── fine-tuning/                       # Model fine-tuning projects
│   └── task2-recipe-chatbot/          # Recipe chatbot with Qwen2.5 + LoRA + FastAPI
│
├── ai-agents/                         # AI agent implementations
│
├── mcp-servers/                       # Model Context Protocol servers
│
├── rag/                               # Retrieval-Augmented Generation
│
├── prompt-engineering/                # Prompt engineering techniques
│
├── langchain/                         # LangChain projects
│
└── README.md
```

## Projects

### Embeddings

| Project | Description | Stack |
|---------|-------------|-------|
| [task1-name-matching](embeddings/task1-name-matching/) | Finds similar names from a dataset using vector similarity search | LanceDB, Sentence-Transformers |

### Fine-Tuning

| Project | Description | Stack |
|---------|-------------|-------|
| [task2-recipe-chatbot](fine-tuning/task2-recipe-chatbot/) | Recipe suggestion chatbot fine-tuned on custom recipe data | Qwen2.5-0.5B, LoRA/PEFT, FastAPI |

---

## Quick Start

Each project has its own `README.md` with setup instructions. General pattern:

```bash
cd <project-folder>
pip install -r requirements.txt
python <main-script>.py
```

### Task 1: Name Matching
```bash
cd embeddings/task1-name-matching
pip install -r requirements.txt
python task1_name_matching.py
```

### Task 2: Recipe Chatbot
```bash
cd fine-tuning/task2-recipe-chatbot
pip install -r requirements.txt
python prepare_data.py
python finetune.py
python app.py
# Open http://localhost:8000
```
