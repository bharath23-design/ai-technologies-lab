# NLP Project Lifecycle — Complete Reference Guide

> A concise, production-oriented reference for any NLP project from idea to deployment.

---

## Table of Contents

1. [Problem Definition & Task Selection](#1-problem-definition--task-selection)
2. [Data Collection & Annotation](#2-data-collection--annotation)
3. [Text Preprocessing & Cleaning](#3-text-preprocessing--cleaning)
4. [Exploratory Text Analysis](#4-exploratory-text-analysis)
5. [Text Representation & Embeddings](#5-text-representation--embeddings)
6. [Baseline Models & Quick Wins](#6-baseline-models--quick-wins)
7. [Model Selection & Architecture](#7-model-selection--architecture)
8. [Training & Fine-Tuning](#8-training--fine-tuning)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Evaluation & Validation](#10-evaluation--validation)
11. [Advanced NLP Techniques](#11-advanced-nlp-techniques)
12. [Production Deployment](#12-production-deployment)
13. [Monitoring & Maintenance](#13-monitoring--maintenance)

---

## 1. Problem Definition & Task Selection

### Core NLP Task Taxonomy

| Task | Description | Example |
|---|---|---|
| **Text Classification** | Assign labels to text | Sentiment analysis, spam detection, intent classification |
| **Named Entity Recognition (NER)** | Extract entities from text | Person, org, location, date extraction |
| **Question Answering (QA)** | Answer questions from context | Extractive QA, open-domain QA |
| **Text Summarization** | Condense text | Extractive vs abstractive summaries |
| **Machine Translation** | Translate between languages | English → French, multilingual |
| **Text Generation** | Generate coherent text | Chatbots, content generation, code generation |
| **Semantic Similarity** | Measure text similarity | Duplicate detection, paraphrase identification |
| **Information Extraction** | Extract structured data from text | Relation extraction, event extraction |
| **Topic Modeling** | Discover themes in corpus | Document clustering, content tagging |
| **Text-to-SQL** | Convert natural language to SQL | Database querying via natural language |
| **Coreference Resolution** | Link pronouns to entities | "She" → "Dr. Smith" |
| **Dialogue Systems** | Multi-turn conversation | Task-oriented bots, open-domain chat |

### Decision Checklist

- Define the **business objective** — what decision does this NLP system support?
- Is an LLM API call sufficient, or do you need a custom model?
- Set **success metrics** upfront: F1, BLEU, ROUGE, accuracy, latency, cost-per-query
- What **languages** are required? Monolingual vs multilingual
- What's the **latency budget**? Real-time (<100ms) vs batch (minutes/hours)
- What's the **data availability**? Labeled vs unlabeled vs zero-shot
- Ask: _Is ML/NLP actually needed?_ Regex, keyword matching, or rule-based systems might suffice
- Consider **privacy**: does data contain PII? On-premise vs cloud requirements?

### When NOT to Use a Custom Model

```
Use an LLM API (GPT-4, Claude, Gemini) when:
  - Task is general-purpose (summarization, Q&A, classification)
  - You have <1K labeled examples
  - Latency is not critical (<2s acceptable)
  - You don't need full control over model behavior

Train a custom model when:
  - You need <50ms latency
  - Domain-specific jargon (medical, legal, financial)
  - Privacy: data cannot leave your infrastructure
  - Cost: high volume makes API calls expensive
  - You need fine-grained control over outputs
```

---

## 2. Data Collection & Annotation

### Data Sources

| Source Type | Examples |
|---|---|
| Public datasets | HuggingFace Datasets, Kaggle, GLUE, SuperGLUE, SQuAD, CoNLL |
| Web scraping | BeautifulSoup, Scrapy, Playwright (respect robots.txt) |
| APIs | Twitter/X API, Reddit API, news APIs, CommonCrawl |
| Internal data | Support tickets, emails, documents, chat logs, CRM notes |
| Synthetic data | LLM-generated examples, back-translation, paraphrasing |
| Crowd-sourced | Amazon MTurk, Prolific, Surge AI, Scale AI |

### Annotation Tools

| Tool | Strengths |
|---|---|
| **Label Studio** | Open-source, multi-task (NER, classification, etc.) |
| **Prodigy** | Active learning built-in, spaCy integration |
| **Doccano** | Simple, web-based, open-source |
| **Argilla** | HuggingFace-integrated, for LLM feedback loops |
| **Labelbox** | Enterprise, team collaboration |
| **Scale AI** | Managed annotation workforce |

### Annotation Best Practices

- Write **clear annotation guidelines** with examples and edge cases
- Measure **inter-annotator agreement** (Cohen's Kappa > 0.7 is acceptable)
- Start with a **pilot round** (50–100 samples) to calibrate annotators
- Use **adjudication** for disagreements — expert resolves conflicts
- For NER: define entity boundaries precisely (include "Dr." in person name?)
- For classification: ensure labels are **mutually exclusive** (or define multi-label)
- Budget: **~60–80% of NLP project time** goes to data work

### Synthetic Data Generation with LLMs

```python
from openai import OpenAI

client = OpenAI()

def generate_training_examples(task_description, num_examples=50):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Generate {num_examples} diverse training examples for:
            Task: {task_description}
            Format: JSON array with 'text' and 'label' fields.
            Ensure variety in length, style, and difficulty."""
        }],
        temperature=0.9
    )
    return response.choices[0].message.content

# Example: generate sentiment analysis data
examples = generate_training_examples("sentiment analysis of product reviews")
```

> Use LLM-generated data for **bootstrapping** and **augmentation**, not as sole training source. Always validate with human review.

---

## 3. Text Preprocessing & Cleaning

### Preprocessing Pipeline

```python
import re
import unicodedata

def clean_text(text):
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # 2. Lowercase (task-dependent — skip for NER)
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 4. Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

### Preprocessing Decisions by Task

| Step | Classification | NER | Generation | Search/RAG |
|---|---|---|---|---|
| Lowercasing | Yes | No (case matters) | No | Depends |
| Remove punctuation | Maybe | No | No | No |
| Remove stop words | Maybe | No | No | Sometimes |
| Stemming/Lemmatization | Rarely (use embeddings instead) | No | No | Sometimes |
| Remove URLs/emails | Yes | Task-dependent | No | No |
| Expand contractions | Optional | Optional | No | Optional |
| Remove numbers | Rarely | No | No | No |

### Tokenization

| Tokenizer | Type | Used By |
|---|---|---|
| **WordPiece** | Subword | BERT, DistilBERT |
| **BPE (Byte-Pair Encoding)** | Subword | GPT-2, GPT-4, RoBERTa |
| **SentencePiece** | Subword | T5, Llama, XLNet |
| **Tiktoken** | BPE variant | OpenAI models |
| **spaCy tokenizer** | Rule-based + statistical | spaCy pipelines |

```python
# HuggingFace tokenizer (handles everything)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(
    "Hello world!",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
# Keys: input_ids, attention_mask, token_type_ids

# Check tokenization
tokens = tokenizer.tokenize("unbelievable")
# → ['un', '##believ', '##able']
```

### Handling Long Documents

| Strategy | Max Length | When |
|---|---|---|
| **Truncation** | 512 tokens | Simple, when start of text is most important |
| **Sliding window** | Unlimited | Full document needed, overlap chunks |
| **Hierarchical** | Unlimited | Summarize chunks, then combine |
| **Longformer / BigBird** | 4096 tokens | Long document classification |
| **Chunking + RAG** | Unlimited | QA over large documents |

```python
# Sliding window approach
def chunk_text(text, tokenizer, max_length=512, overlap=64):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
    return chunks
```

---

## 4. Exploratory Text Analysis

### Quick EDA Checklist

- [ ] Corpus size: number of documents, total tokens
- [ ] Text length distribution (min, max, mean, percentiles)
- [ ] Label distribution (imbalanced?)
- [ ] Language detection (unexpected languages?)
- [ ] Vocabulary size and frequency distribution
- [ ] Missing/empty text fields
- [ ] Duplicate documents
- [ ] Noise: HTML, boilerplate, encoding issues

### EDA Code

```python
import pandas as pd
from collections import Counter

df = pd.read_csv("data.csv")

# Basic stats
print(f"Documents: {len(df)}")
print(f"Avg length: {df['text'].str.len().mean():.0f} chars")
print(f"Avg words: {df['text'].str.split().str.len().mean():.0f}")

# Label distribution
print(df["label"].value_counts(normalize=True))

# Word frequency
all_words = " ".join(df["text"]).lower().split()
word_freq = Counter(all_words).most_common(50)

# Detect empty/short texts
short_texts = df[df["text"].str.split().str.len() < 3]
print(f"Very short texts (<3 words): {len(short_texts)}")

# Duplicates
duplicates = df["text"].duplicated().sum()
print(f"Duplicate texts: {duplicates}")
```

### Language Detection

```python
from langdetect import detect_langs

def detect_language(text):
    try:
        return detect_langs(text)[0]
    except:
        return None

df["language"] = df["text"].apply(detect_language)
print(df["language"].value_counts())
```

### Text Visualization

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Word cloud per class
for label in df["label"].unique():
    text = " ".join(df[df["label"] == label]["text"])
    wc = WordCloud(width=800, height=400, max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()
```

**Tools:** `wordcloud`, `scattertext`, `pyLDAvis`, `BERTopic` (for topic exploration), `matplotlib`, `plotly`

---

## 5. Text Representation & Embeddings

### Evolution of Text Representations

```
Bag of Words → TF-IDF → Word2Vec → GloVe → ELMo → BERT → Sentence Transformers → LLM Embeddings
```

### Methods Comparison

| Method | Type | Dim | Context-Aware | Best For |
|---|---|---|---|---|
| **Bag of Words** | Sparse | Vocab size | No | Quick baseline |
| **TF-IDF** | Sparse | Vocab size | No | Information retrieval, baseline |
| **Word2Vec** | Dense | 100–300 | No | Word-level similarity |
| **GloVe** | Dense | 50–300 | No | Pre-trained word vectors |
| **FastText** | Dense | 100–300 | No | OOV words (subword-based) |
| **BERT [CLS]** | Dense | 768 | Yes | Classification (fine-tuned) |
| **Sentence-Transformers** | Dense | 384–1024 | Yes | Semantic search, similarity |
| **OpenAI Embeddings** | Dense | 1536–3072 | Yes | RAG, search (API-based) |
| **E5 / GTE / BGE** | Dense | 384–1024 | Yes | Open-source semantic search |
| **ColBERT** | Multi-vector | 128 per token | Yes | Late-interaction retrieval |

### TF-IDF Baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=2,               # Ignore very rare terms
    max_df=0.95,            # Ignore very common terms
    sublinear_tf=True       # Apply log normalization
)
X = tfidf.fit_transform(texts)
```

### Sentence Transformers (Recommended for Embeddings)

```python
from sentence_transformers import SentenceTransformer

# Best open-source embedding models (2025–2026)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# Alternatives: "sentence-transformers/all-MiniLM-L6-v2" (fast)
#               "intfloat/e5-large-v2" (high quality)
#               "Alibaba-NLP/gte-large-en-v1.5" (balanced)

embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
# Shape: (num_texts, 1024)

# Semantic similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
```

### Choosing an Embedding Model

```
Need speed + low cost?        → all-MiniLM-L6-v2 (384d, very fast)
Need quality (open-source)?   → bge-large-en-v1.5, gte-large-en-v1.5
Need multilingual?            → multilingual-e5-large, paraphrase-multilingual
Need API simplicity?          → OpenAI text-embedding-3-large, Cohere embed-v4
Need custom domain?           → Fine-tune Sentence-Transformers on your data
```

### Fine-Tuning Embeddings

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Prepare training pairs
train_examples = [
    InputExample(texts=["query", "relevant doc"], label=1.0),
    InputExample(texts=["query", "irrelevant doc"], label=0.0),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-embeddings"
)
```

---

## 6. Baseline Models & Quick Wins

### The Baseline Ladder

```
Level 0: Rule-based / regex       → Instant, interpretable
Level 1: TF-IDF + Logistic Reg    → 5 min, surprisingly strong
Level 2: Zero-shot LLM (API)      → 10 min, no training data needed
Level 3: Fine-tuned transformer   → Hours, best quality
```

### Level 0: Rule-Based

```python
import re

def rule_based_sentiment(text):
    positive_words = {"great", "love", "excellent", "amazing", "good", "best"}
    negative_words = {"bad", "terrible", "worst", "hate", "awful", "poor"}

    words = set(text.lower().split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "neutral"
```

### Level 1: TF-IDF + Classical ML

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, C=1.0))
])

scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
print(f"TF-IDF + LR: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

# Try other classifiers
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# MultinomialNB is surprisingly good for text
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf", MultinomialNB(alpha=0.1))
])
```

> TF-IDF + Logistic Regression is a **shockingly strong baseline**. Always try it first — it beats transformers on small datasets and is 100x faster.

### Level 2: Zero-Shot Classification

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = classifier(
    "The new iPhone camera is incredible but the battery is disappointing",
    candidate_labels=["positive", "negative", "mixed"],
)
print(result["labels"][0], result["scores"][0])
# → mixed 0.87

# Or use an LLM API
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=100,
    messages=[{
        "role": "user",
        "content": f"Classify the sentiment of this review as positive, negative, or mixed. Reply with just the label.\n\nReview: {text}"
    }]
)
```

### Level 3: spaCy Quick Training

```python
# spaCy CLI — train a text classifier in minutes
# 1. Convert data to spaCy format
# 2. Generate config
# python -m spacy init config config.cfg --lang en --pipeline textcat
# 3. Train
# python -m spacy train config.cfg --output ./model --paths.train train.spacy --paths.dev dev.spacy
```

### Baseline Strategy

```
1. Rule-based / regex           → Quick sanity check (5 min)
2. TF-IDF + Logistic Regression → Strong baseline (10 min)
3. Zero-shot LLM                → No-training baseline (10 min)
4. Fine-tuned transformer       → Only if baselines aren't enough
```

---

## 7. Model Selection & Architecture

### Models by Task

**Text Classification:**

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| **TF-IDF + LogReg** | Tiny | Very fast | Good | Always start here |
| **DistilBERT** | 66M | Fast | Good | 60% faster than BERT, 97% quality |
| **BERT-base** | 110M | Moderate | Good | Solid general baseline |
| **RoBERTa-base** | 125M | Moderate | Better | Better pre-training than BERT |
| **DeBERTa-v3-base** | 183M | Slower | Best (encoder) | SOTA for encoder-based classification |
| **ModernBERT** | 149M/395M | Fast | Excellent | 2024 encoder, 8K context, fast |
| **SetFit** | Any | Fast training | Very good | Few-shot, no prompts needed |

**Named Entity Recognition (NER):**

| Model | Notes |
|---|---|
| **spaCy NER** | Fast, production-ready, rule + ML hybrid |
| **BERT/RoBERTa + token classification** | Standard fine-tuned NER |
| **GLiNER** | Zero-shot NER — no training needed |
| **UniversalNER** | Generalist NER via instruction tuning |
| **LLM + structured output** | Flexible, handles novel entity types |

**Text Generation / Conversational:**

| Model | Size | Notes |
|---|---|---|
| **Llama 3.1 / 3.2** | 1B–405B | Best open-source family |
| **Mistral / Mixtral** | 7B–8x22B | Efficient, strong coding |
| **Qwen 2.5** | 0.5B–72B | Strong multilingual |
| **Phi-3 / Phi-4** | 3.8B–14B | Small but capable |
| **Gemma 2** | 2B–27B | Google's open models |
| **GPT-4o / Claude** | API | Best quality, highest cost |

**Summarization:**

| Model | Type | Notes |
|---|---|---|
| **BART-large-CNN** | Encoder-decoder | Fine-tuned on CNN/DailyMail |
| **Pegasus** | Encoder-decoder | Pre-trained for summarization |
| **T5 / Flan-T5** | Encoder-decoder | Versatile seq2seq |
| **LED (Longformer Encoder-Decoder)** | Encoder-decoder | Long documents (16K tokens) |
| **LLM (GPT-4, Claude)** | Decoder | Best quality, most flexible |

**Semantic Search & Retrieval:**

| Model | Dim | Notes |
|---|---|---|
| **all-MiniLM-L6-v2** | 384 | Fast, good quality |
| **bge-large-en-v1.5** | 1024 | Top open-source |
| **gte-large-en-v1.5** | 1024 | Strong alternative |
| **E5-mistral-7b-instruct** | 4096 | LLM-based, highest quality |
| **ColBERT v2** | 128/token | Late interaction, best for retrieval |
| **OpenAI text-embedding-3-large** | 3072 | API, very strong |

### Architecture Decision Tree

```
Task: Classification?
├── <1K samples + general domain → Zero-shot LLM or SetFit
├── 1K–10K samples → Fine-tune DeBERTa-v3 or ModernBERT
├── 10K+ samples → Fine-tune RoBERTa/DeBERTa, or TF-IDF + LR
└── Need speed → DistilBERT or TF-IDF + LR

Task: NER?
├── Standard entities → spaCy (pre-trained) or GLiNER (zero-shot)
├── Custom entities + data → Fine-tune BERT/RoBERTa token classifier
└── Complex/novel entities → LLM with structured output

Task: Generation / Chat?
├── Best quality → GPT-4o / Claude API
├── Open-source, large → Llama 3.1 70B
├── Open-source, medium → Mistral 7B / Qwen 2.5 7B
├── On-device → Llama 3.2 1B–3B / Phi-3 mini
└── Custom domain → Fine-tune (LoRA) on domain data

Task: Search / Retrieval?
├── API → OpenAI embeddings
├── Open-source quality → bge-large / gte-large
├── Speed → all-MiniLM-L6-v2
└── Best retrieval → ColBERT v2
```

---

## 8. Training & Fine-Tuning

### Fine-Tuning with HuggingFace Transformers

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Load data and model
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base", num_labels=2
)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

# Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    logging_steps=100,
    report_to="wandb",  # or "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Fine-Tuning NER

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    "roberta-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Important: align labels with subword tokens
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore subword continuations
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized
```

### Few-Shot with SetFit

```python
from setfit import SetFitModel, Trainer, sample_dataset

# SetFit: fine-tune with as few as 8 examples per class
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")

# Sample 8 examples per class
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dataset["test"],
)
trainer.train()

# Predict
preds = model.predict(["This movie was amazing!", "Terrible waste of time."])
```

> SetFit achieves near-BERT performance with **8–64 labeled examples** per class. No prompts, no large LMs needed.

### LoRA Fine-Tuning for Text Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch

# QLoRA setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Train with TRL's SFTTrainer
sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)
trainer.train()
```

### Training Tips

| Tip | Details |
|---|---|
| **LR for fine-tuning** | 1e-5 to 5e-5 for encoders; 1e-4 to 3e-4 for LoRA |
| **Epochs** | 2–5 for fine-tuning (more = overfitting risk) |
| **Warmup** | 5–10% of total steps |
| **Max length** | Set to 95th percentile of your text lengths |
| **Class imbalance** | Use weighted loss, oversampling, or focal loss |
| **Mixed precision** | Always use `fp16=True` or `bf16=True` |
| **Gradient checkpointing** | Saves ~30% memory: `model.gradient_checkpointing_enable()` |

---

## 9. Hyperparameter Tuning

### Key Hyperparameters by Priority

```
1. Learning rate        → Most impactful. Try: 1e-5, 2e-5, 3e-5, 5e-5
2. Batch size           → 16, 32 (larger needs LR scaling)
3. Number of epochs     → 2–5 (use early stopping)
4. Max sequence length  → Affects memory and speed
5. Weight decay         → 0.01–0.1
6. Warmup ratio         → 0.05–0.1
7. Dropout              → 0.1–0.3 (for small datasets)
```

### Optuna Integration with HuggingFace

```python
import optuna

def objective(trial):
    training_args = TrainingArguments(
        output_dir=f"./trial_{trial.number}",
        learning_rate=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        num_train_epochs=trial.suggest_int("epochs", 2, 5),
        weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.15),
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model_init=lambda: AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", num_labels=num_labels
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    result = trainer.train()
    return trainer.evaluate()["eval_f1"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Practical Strategy

```
1. Start with published defaults (LR=2e-5, batch=16, epochs=3)
2. Quick LR sweep: [1e-5, 2e-5, 3e-5, 5e-5] with fixed other params
3. If needed: Optuna search over narrow ranges (10–20 trials)
4. Always use early stopping — don't waste compute
5. Report results with confidence intervals (3 seeds minimum)
```

---

## 10. Evaluation & Validation

### Metrics by Task

**Classification:**

| Metric | When to Use |
|---|---|
| **Accuracy** | Balanced classes only |
| **F1 (Macro)** | Imbalanced, care about all classes equally |
| **F1 (Weighted)** | Imbalanced, care proportional to class frequency |
| **F1 (Micro)** | Multi-label classification |
| **AUC-ROC** | Binary, threshold-independent evaluation |
| **MCC** | Best single metric for imbalanced binary |
| **Precision@K** | When top-K predictions matter (search, tagging) |

**NER:**

| Metric | Notes |
|---|---|
| **Entity-level F1** | Strict: exact span + type match |
| **Token-level F1** | Partial credit per token |
| **seqeval** | Standard NER evaluation library |

```python
from seqeval.metrics import classification_report
print(classification_report(true_labels, pred_labels))
```

**Generation / Summarization:**

| Metric | What It Measures |
|---|---|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | N-gram overlap with reference |
| **BLEU** | N-gram precision (translation) |
| **BERTScore** | Semantic similarity using BERT embeddings |
| **METEOR** | Improved BLEU with synonyms + stemming |
| **Perplexity** | Language model fluency (lower = better) |
| **Human evaluation** | Gold standard for generation quality |
| **LLM-as-judge** | Automated evaluation using GPT-4/Claude |

```python
# ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
scores = scorer.score("reference summary text", "generated summary text")

# BERTScore
from bert_score import score
P, R, F1 = score(candidates, references, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")
```

### LLM-as-Judge Evaluation

```python
def llm_judge(generated_text, reference_text):
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Rate the quality of this generated text on a scale of 1-5.
            Consider: accuracy, fluency, completeness, and relevance.

            Reference: {reference_text}
            Generated: {generated_text}

            Score (1-5):
            Brief justification:"""
        }]
    )
    return response.choices[0].message.content
```

### Validation Checklist

- [ ] Cross-validation (5-fold or 3-seed runs)
- [ ] Train/val/test split — **never tune on test set**
- [ ] Confusion matrix analysis — which classes are confused?
- [ ] **Error analysis**: manually inspect 50–100 worst predictions
- [ ] Test on **out-of-domain** data (different time period, source, style)
- [ ] Test on **adversarial** examples (negation, sarcasm, typos)
- [ ] Measure **latency** and **throughput** alongside quality
- [ ] Check performance across **subgroups** (fairness: gender, language, etc.)
- [ ] **Calibration**: are confidence scores reliable?

### Error Analysis Template

```python
# Get predictions with confidence
probs = model.predict_proba(test_texts)
preds = probs.argmax(axis=1)
confidence = probs.max(axis=1)

# Find errors
errors = pd.DataFrame({
    "text": test_texts,
    "true_label": true_labels,
    "pred_label": preds,
    "confidence": confidence,
})
errors = errors[errors["true_label"] != errors["pred_label"]]

# High-confidence errors (model is confidently wrong)
high_conf_errors = errors[errors["confidence"] > 0.9].sort_values("confidence", ascending=False)

# Low-confidence correct predictions (model is uncertain)
correct = pd.DataFrame({...})
uncertain_correct = correct[correct["confidence"] < 0.6]
```

---

## 11. Advanced NLP Techniques

### Retrieval-Augmented Generation (RAG)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(documents)

# 2. Create vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 3. Retrieve + Generate
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke("What is the return policy?")

# 4. Pass to LLM
context = "\n".join([doc.page_content for doc in relevant_docs])
prompt = f"Based on this context:\n{context}\n\nAnswer: What is the return policy?"
```

### RAG Optimization

| Technique | Description |
|---|---|
| **Hybrid search** | Combine dense (embedding) + sparse (BM25) retrieval |
| **Re-ranking** | Use a cross-encoder to re-rank retrieved chunks |
| **Chunking strategy** | Sentence-level, paragraph-level, or semantic chunking |
| **Query expansion** | Rewrite query using LLM for better retrieval |
| **HyDE** | Generate hypothetical document, then retrieve similar |
| **Parent document retrieval** | Retrieve small chunks, return full parent documents |
| **Metadata filtering** | Pre-filter by date, source, category before semantic search |

### Named Entity Recognition (Advanced)

```python
# Zero-shot NER with GLiNER
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

text = "Apple CEO Tim Cook announced the new iPhone at a San Francisco event."
labels = ["person", "organization", "product", "location"]

entities = model.predict_entities(text, labels)
for entity in entities:
    print(f"{entity['text']} → {entity['label']} ({entity['score']:.2f})")
# Tim Cook → person (0.98)
# Apple → organization (0.95)
# iPhone → product (0.91)
# San Francisco → location (0.97)
```

### Topic Modeling with BERTopic

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    min_topic_size=10,
    nr_topics="auto",
)

topics, probs = topic_model.fit_transform(documents)
topic_model.get_topic_info()
topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=10)
```

### Text-to-SQL

```python
# Using an LLM with schema context
schema = """
Tables:
- users (id, name, email, created_at)
- orders (id, user_id, product_id, amount, created_at)
- products (id, name, category, price)
"""

prompt = f"""Given this database schema:
{schema}

Convert this question to SQL:
"What are the top 5 customers by total order amount in the last 30 days?"

Return only the SQL query."""
```

### Structured Output Extraction

```python
from pydantic import BaseModel
from anthropic import Anthropic

class Invoice(BaseModel):
    vendor: str
    date: str
    total: float
    items: list[dict]

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": f"""Extract invoice data from this text into JSON matching this schema:
        {Invoice.model_json_schema()}

        Text: {invoice_text}"""
    }]
)
```

### Multilingual NLP

| Task | Model | Languages |
|---|---|---|
| Classification | `xlm-roberta-large` | 100+ languages |
| NER | `Davlan/xlm-roberta-large-ner-hrl` | 10 high-resource languages |
| Embeddings | `intfloat/multilingual-e5-large` | 100+ languages |
| Translation | `facebook/nllb-200-distilled-600M` | 200 languages |
| Zero-shot | `joeddav/xlm-roberta-large-xnli` | 15+ languages |

---

## 12. Production Deployment

### Serving Patterns

| Pattern | Latency | Use Case |
|---|---|---|
| **REST API** (FastAPI) | ms–seconds | Real-time classification, NER |
| **Batch inference** | minutes–hours | Document processing, bulk scoring |
| **Streaming** | sub-second | Chat, real-time translation |
| **Edge / On-device** | ms | Mobile apps, privacy-sensitive |
| **LLM API proxy** | seconds | Thin wrapper over GPT-4/Claude |

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("text-classification", model="./fine-tuned-model", device=0)

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    result = classifier(request.text, truncation=True, max_length=512)[0]
    return PredictionResponse(
        label=result["label"],
        confidence=round(result["score"], 4)
    )
```

### Optimizing Inference

| Technique | Speedup | Effort |
|---|---|---|
| **ONNX Runtime** | 2–4x | Low |
| **Quantization (INT8)** | 2–3x | Low |
| **Distillation** | 2–6x | Medium |
| **torch.compile** | 1.2–2x | Trivial |
| **Batching** | 2–10x throughput | Low |
| **TensorRT** | 3–6x | Medium |
| **vLLM** (for LLMs) | 5–20x throughput | Low |

### ONNX Export for NLP

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Export to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "./fine-tuned-model", export=True
)
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

# Save
model.save_pretrained("./onnx-model")
tokenizer.save_pretrained("./onnx-model")

# Inference (2-4x faster)
inputs = tokenizer("Hello world", return_tensors="np")
outputs = model(**inputs)
```

### LLM Serving

| Tool | Best For |
|---|---|
| **vLLM** | High-throughput LLM serving with PagedAttention |
| **TGI** | HuggingFace models, easy deployment |
| **Ollama** | Local LLM serving, simple setup |
| **llama.cpp** | CPU inference, GGUF quantized models |
| **TensorRT-LLM** | Maximum NVIDIA GPU performance |

```bash
# vLLM — serve any HuggingFace model
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --max-model-len 4096

# Ollama — local serving
ollama run llama3.1
```

### Production Checklist

- [ ] Model versioning and experiment tracking (MLflow, W&B)
- [ ] Input validation (max length, encoding, language check)
- [ ] Rate limiting and authentication
- [ ] Logging: inputs, outputs, latency, token counts
- [ ] Caching frequent queries (Redis, in-memory)
- [ ] Fallback strategy (if model fails → rule-based or cached response)
- [ ] Load testing (Locust, k6) — measure p50, p95, p99 latency
- [ ] A/B testing framework
- [ ] Rollback plan
- [ ] Cost monitoring (API calls, GPU hours, token usage)
- [ ] PII detection and redaction in logs

### Key Tools

| Category | Tools |
|---|---|
| Experiment tracking | W&B, MLflow, Neptune, Comet |
| Pipeline orchestration | Airflow, Prefect, Dagster |
| Model serving | vLLM, TGI, BentoML, Triton, Ray Serve |
| NLP frameworks | HuggingFace, spaCy, LangChain, LlamaIndex |
| Vector databases | Pinecone, Weaviate, Milvus, Qdrant, ChromaDB, FAISS |
| Containerization | Docker, Kubernetes |

---

## 13. Monitoring & Maintenance

### What to Monitor

| What | How | Tool |
|---|---|---|
| **Model quality** | Track metrics on live data | Evidently, Arize, WhyLabs |
| **Input drift** | Text length, vocabulary, language distribution | Evidently, custom |
| **Output drift** | Prediction distribution changes | Histogram monitoring |
| **Embedding drift** | Cosine distance between batches | Custom + vector DB |
| **Latency (p50, p95, p99)** | Response time distribution | Prometheus + Grafana |
| **Error rate** | Failed predictions, timeouts | Application logging |
| **Token usage** | API cost tracking | LLM observability tools |
| **Data quality** | Empty inputs, encoding issues, injection attempts | Input validation |
| **Hallucination rate** | Factual accuracy of generated text | Human review + automated |

### NLP-Specific Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset

report = Report(metrics=[TextOverviewPreset()])
report.run(reference_data=train_df, current_data=live_df)
report.save_html("nlp_drift_report.html")
```

### Monitoring Text-Specific Issues

| Issue | Detection | Action |
|---|---|---|
| **New vocabulary / jargon** | OOV rate increase | Retrain tokenizer or model |
| **Language distribution shift** | Language detection stats | Add multilingual support |
| **Text length shift** | Length distribution change | Adjust truncation / chunking |
| **Topic drift** | Topic model on live data | Retrain or add new topics |
| **Adversarial inputs** | Pattern detection, input validation | Add guardrails |
| **Prompt injection** (LLMs) | Input scanning | Input sanitization, guardrails |

### LLM-Specific Monitoring

| What | Tool |
|---|---|
| Token usage & cost | LangSmith, Helicone, Portkey |
| Prompt/completion logging | LangSmith, Arize Phoenix |
| Hallucination detection | Vectara HHEM, custom validation |
| Guardrails | Guardrails AI, NeMo Guardrails, Llama Guard |

### Retraining Triggers

```
1. Scheduled       → Retrain monthly with new labeled data
2. Performance     → Retrain when F1 drops below threshold
3. Drift-triggered → Retrain when input distribution shifts
4. Vocabulary      → Retrain when OOV rate exceeds threshold
5. Domain change   → Retrain when new topics/entities emerge
```

### Common Failure Modes

| Issue | Symptom | Fix |
|---|---|---|
| Domain drift | Accuracy drops on new data | Retrain on recent data |
| Label noise | Inconsistent predictions | Audit and clean labels |
| Truncation issues | Long documents misclassified | Adjust max_length or use chunking |
| Tokenizer mismatch | Garbled outputs, poor performance | Ensure consistent tokenizer |
| Hallucination (LLM) | Fabricated facts | RAG, grounding, fact-checking |
| Prompt injection | Unexpected behavior | Input sanitization, guardrails |
| Class imbalance shift | Minority class performance drops | Rebalance training data |

---

## Quick Reference — NLP Project Kickstart

```
1.  Define problem, task type, and success metrics
2.  Collect & annotate data (or find existing dataset)
3.  EDA: text length distribution, label balance, language detection, duplicates
4.  Preprocess: clean, tokenize, handle long documents
5.  Baselines:
    a. Rule-based / regex         → sanity check
    b. TF-IDF + Logistic Reg     → strong baseline
    c. Zero-shot LLM             → no-training baseline
6.  Choose architecture:
    a. Classification → DeBERTa-v3 / ModernBERT
    b. NER → spaCy / GLiNER / fine-tuned BERT
    c. Generation → Llama 3.1 / Mistral + LoRA
    d. Search → Sentence-Transformers / ColBERT
7.  Fine-tune:
    a. Few-shot? → SetFit or zero-shot LLM
    b. Full data? → HuggingFace Trainer + mixed precision
    c. Large LLM? → QLoRA + TRL
8.  Tune: Optuna for LR, batch size, epochs
9.  Evaluate: F1/ROUGE/BERTScore + error analysis + adversarial tests
10. Optimize: ONNX / quantization / vLLM
11. Deploy: FastAPI / vLLM + Docker + K8s
12. Monitor: Evidently for drift, LangSmith for LLM observability
13. Retrain on schedule or drift trigger
```

---

## Sources & Further Reading

- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [spaCy Documentation](https://spacy.io/usage)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [MTEB Leaderboard — Embedding Model Rankings](https://huggingface.co/spaces/mteb/leaderboard)
- [SetFit: Few-Shot Classification](https://huggingface.co/docs/setfit)
- [GLiNER: Zero-Shot NER](https://github.com/urchade/GLiNER)
- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [PEFT / LoRA Guide](https://huggingface.co/docs/peft)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [Optimum — ONNX for HuggingFace](https://huggingface.co/docs/optimum)
- [ModernBERT: A Modern Encoder](https://huggingface.co/blog/modernbert)
- [Evidently AI — ML Monitoring](https://www.evidentlyai.com/)
- [LangSmith — LLM Observability](https://docs.smith.langchain.com/)
- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [NLP Best Practices (Microsoft)](https://github.com/microsoft/nlp-recipes)
