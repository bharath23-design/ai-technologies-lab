# Deep Learning Project Lifecycle — Complete Reference Guide

> A concise, production-oriented reference for any DL project from idea to deployment.

---

## Table of Contents

1. [Problem Definition & Feasibility](#1-problem-definition--feasibility)
2. [Data Collection & Labeling](#2-data-collection--labeling)
3. [Data Preprocessing & Augmentation](#3-data-preprocessing--augmentation)
4. [Architecture Selection](#4-architecture-selection)
5. [Transfer Learning & Fine-Tuning](#5-transfer-learning--fine-tuning)
6. [Training Best Practices](#6-training-best-practices)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Regularization & Generalization](#8-regularization--generalization)
9. [Model Explainability & Debugging](#9-model-explainability--debugging)
10. [Evaluation & Validation](#10-evaluation--validation)
11. [Model Optimization & Compression](#11-model-optimization--compression)
12. [Production Deployment](#12-production-deployment)
13. [Monitoring & Maintenance](#13-monitoring--maintenance)

---

## 1. Problem Definition & Feasibility

- **Is DL needed?** If a simple ML model (XGBoost, logistic regression) works, prefer it
- Define task type: classification, detection, segmentation, generation, sequence-to-sequence, etc.
- Define success metrics: accuracy, mAP, BLEU, FID, perplexity, latency, etc.
- Estimate **data requirements** — DL is data-hungry:

| Task | Typical Data Needed |
|---|---|
| Image classification (transfer learning) | 100–1K per class |
| Image classification (from scratch) | 10K–100K+ per class |
| Object detection | 1K–10K annotated images |
| NLP classification (fine-tuned LLM) | 100–1K examples |
| NLP from scratch | 100K–1M+ sentences |

- Estimate **compute budget** — GPU hours, cloud costs
- Check for **pretrained models** first (HuggingFace, timm, torchvision)

---

## 2. Data Collection & Labeling

### Sources

| Source | Examples |
|---|---|
| Public datasets | ImageNet, COCO, SQuAD, LAION, CommonCrawl |
| Platform datasets | HuggingFace Datasets, Kaggle, TensorFlow Datasets |
| Synthetic data | Diffusion models, GANs, simulation engines |
| Manual labeling | Labelbox, Label Studio, Prodigy, CVAT, Roboflow |
| Weak supervision | Snorkel, programmatic labeling functions |

### Best Practices

- Version datasets (DVC, Delta Lake, HuggingFace Datasets)
- Document annotation guidelines — inter-annotator agreement matters
- Check label quality: random audit 5–10% of labels regularly
- For imbalanced data: oversample minority, use weighted loss, or focal loss
- **~80% of DL project time** is data work — invest here

---

## 3. Data Preprocessing & Augmentation

### Image Preprocessing

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Advanced Augmentation

| Technique | Library | Use Case |
|---|---|---|
| **Albumentations** | `albumentations` | CV: fast, diverse transforms, bbox-aware |
| **RandAugment** | `torchvision` | Automated augmentation policy |
| **CutMix / MixUp** | `timm`, custom | Regularization via sample mixing |
| **Mosaic** | YOLO pipelines | Object detection |
| **Back-translation** | `nlpaug` | NLP data augmentation |
| **Synonym replacement** | `nlpaug` | NLP text augmentation |
| **SpecAugment** | `torchaudio` | Audio/speech augmentation |

### Text Preprocessing

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

### DataLoader Best Practices

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Only for training
    num_workers=4,          # Parallel data loading (set to num CPU cores)
    pin_memory=True,        # Faster GPU transfer
    persistent_workers=True, # Avoid worker restart overhead
    prefetch_factor=2       # Prefetch batches
)
```

---

## 4. Architecture Selection

### Vision

| Task | Architecture | Notes |
|---|---|---|
| Image classification | ViT, ConvNeXt v2, EfficientNet v2 | ViT dominates with enough data |
| Object detection | YOLOv8/v10, RT-DETR, DINO | YOLO for speed, DETR for accuracy |
| Segmentation | SAM 2, Mask2Former, SegFormer | SAM 2 is foundation model for segmentation |
| Image generation | Stable Diffusion 3, FLUX | Diffusion-based |
| Video understanding | VideoMAE v2, InternVideo | Transformer-based |

### NLP

| Task | Architecture | Notes |
|---|---|---|
| Text classification | BERT, RoBERTa, DeBERTa v3 | Fine-tune pretrained encoder |
| Text generation | GPT-4, Llama 3, Mistral, Qwen 2.5 | Decoder-only transformers |
| Seq2seq | T5, BART, mT5 | Encoder-decoder |
| Embeddings | Sentence-Transformers, E5, GTE | For RAG, search, clustering |
| Code | StarCoder 2, DeepSeek Coder, Codestral | Code generation/understanding |

### Multimodal

| Task | Architecture |
|---|---|
| Vision-Language | LLaVA, Qwen-VL, GPT-4V, InternVL |
| Audio | Whisper, SeamlessM4T |
| Any-to-any | GPT-4o, Gemini |

### Tabular (DL)

| Model | Notes |
|---|---|
| TabNet | Attention-based, interpretable |
| FT-Transformer | Transformer for tabular, competitive with GBMs |
| RealMLP | Simple MLP with modern tricks, strong baseline |
| TabPFN | Foundation model, zero-shot |

### Architecture Search Shortcut

```
Start with pretrained model → Fine-tune → Only build custom if needed
```

---

## 5. Transfer Learning & Fine-Tuning

### Strategy Ladder

```
Level 0: Feature extraction  → Freeze all, train new head only
Level 1: Fine-tune top layers → Freeze early layers, tune last N layers
Level 2: Full fine-tuning     → Unfreeze all, lower learning rate
Level 3: PEFT (LoRA/QLoRA)   → Freeze all, add small trainable adapters
```

### When to Use What

| Scenario | Strategy |
|---|---|
| Small data, similar domain | Feature extraction (Level 0) |
| Small data, different domain | PEFT / LoRA (Level 3) |
| Medium data, similar domain | Fine-tune top layers (Level 1) |
| Large data | Full fine-tuning (Level 2) |
| Very large model (>7B params) | LoRA / QLoRA (Level 3) |

### Vision Transfer Learning (timm)

```python
import timm

# Load pretrained model with custom head
model = timm.create_model(
    "convnext_base.fb_in22k_ft_in1k",
    pretrained=True,
    num_classes=10  # Your number of classes
)

# Freeze backbone, train head only
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

# Or: gradual unfreezing (unfreeze last N layers progressively)
```

### NLP Transfer Learning (HuggingFace)

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # Lower LR for fine-tuning
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                   # Mixed precision
)

trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### LoRA / QLoRA (PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,    # or CAUSAL_LM, SEQ_2_SEQ_LM
    r=8,                            # Rank (start low, increase if needed)
    lora_alpha=16,                  # Scaling factor (usually 2x r)
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 0.6M || all params: 184M || trainable%: 0.33%
```

**LoRA Key Hyperparameters:**

| Param | Default | Notes |
|---|---|---|
| `r` (rank) | 8 | Higher = more capacity but more params. Start with 4–16 |
| `lora_alpha` | 16 | Scaling. Common: `alpha = 2 * r` |
| `lora_dropout` | 0.05–0.1 | Regularization for small datasets |
| `target_modules` | attention layers | `q_proj, v_proj` minimum; add `k_proj, o_proj` for more capacity |

### QLoRA (4-bit + LoRA)

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Saves extra memory
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
# Then apply LoRA on top
model = get_peft_model(model, lora_config)
```

> QLoRA enables fine-tuning a **65B model on a single 48GB GPU**. Trains ~1% of params, saves 50–70% of compute cost.

### PEFT Methods Comparison

| Method | Params Trained | Memory | Quality | Best For |
|---|---|---|---|---|
| **Full fine-tuning** | 100% | Very high | Best (if enough data) | Large datasets + budget |
| **LoRA** | 0.1–1% | Low | Near full FT | Most tasks |
| **QLoRA** | 0.1–1% (4-bit base) | Very low | Near LoRA | Limited GPU memory |
| **Prefix Tuning** | <0.1% | Very low | Good | Generation tasks |
| **Prompt Tuning** | <0.01% | Minimal | Moderate | Quick adaptation |
| **Adapters** | 1–5% | Low | Good | Multi-task setups |

---

## 6. Training Best Practices

### PyTorch Training Loop (Modern)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = GradScaler()  # For mixed precision

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch[0].cuda(), batch[1].cuda()

        with autocast(device_type="cuda"):  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
```

### Mixed Precision Training

```python
# Automatic Mixed Precision (AMP) — up to 3x speedup on modern GPUs
with torch.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

# Gotcha: prefer BCEWithLogitsLoss over BCELoss (numerical stability)
# Gotcha: loss scaling is essential — always use GradScaler
```

### torch.compile (PyTorch 2.x)

```python
# Fuse operations, optimize graph — free speedup
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
```

### Learning Rate Schedules

| Schedule | When |
|---|---|
| **Cosine Annealing** | Default best choice for most tasks |
| **Cosine with Warm Restarts** | Long training, cyclical exploration |
| **Linear Warmup + Cosine Decay** | Fine-tuning transformers |
| **OneCycleLR** | Fast convergence, super-convergence |
| **ReduceLROnPlateau** | When you don't know training length |

```python
# Warmup + cosine (most common for transformers)
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
```

### Optimizer Cheat Sheet

| Optimizer | When |
|---|---|
| **AdamW** | Default for transformers and most DL |
| **SGD + Momentum** | CNNs, when you want to tune carefully |
| **Lion** | Memory-efficient alternative to AdamW |
| **8-bit Adam** (`bitsandbytes`) | Low memory fine-tuning |
| **Adafactor** | Very large models, memory constrained |
| **LAMB/LARS** | Large batch distributed training |

### Gradient Accumulation (Simulate Larger Batches)

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    with autocast(device_type="cuda"):
        loss = model(batch) / accumulation_steps
    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Distributed Training

| Method | Scale | Tool |
|---|---|---|
| **DataParallel** | Single node, multi-GPU (avoid) | `nn.DataParallel` |
| **DistributedDataParallel** | Single/multi-node | `nn.DistributedDataParallel` |
| **FSDP** | Very large models | `torch.distributed.fsdp` |
| **DeepSpeed ZeRO** | Largest models (100B+) | DeepSpeed |
| **Accelerate** | Simple multi-GPU wrapper | HuggingFace `accelerate` |

```python
# Simplest multi-GPU with HuggingFace Accelerate
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

> Always prefer `DistributedDataParallel` over `DataParallel`. Use one process per GPU.

### Training Tips

- **Batch size rule:** When increasing batch size by `n`, increase LR by `sqrt(n)`
- **Gradient clipping:** `max_norm=1.0` prevents exploding gradients
- **Early stopping:** Monitor val loss, patience of 3–5 epochs
- **Seed everything** for reproducibility:

```python
import torch, random, numpy as np
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 7. Hyperparameter Tuning

### Key Hyperparameters by Priority

```
1. Learning rate           → Most impactful. Try: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
2. Batch size              → Larger = more stable but needs LR scaling
3. Weight decay            → 0.01–0.1 for AdamW
4. Number of epochs        → Use early stopping instead of fixing
5. Architecture-specific   → Dropout, number of layers, hidden size
6. Augmentation strength   → More data → less augmentation needed
```

### Tools

| Tool | Strengths |
|---|---|
| **Optuna** | Bayesian search, pruning, dashboards |
| **Ray Tune** | Distributed tuning, many algorithms |
| **W&B Sweeps** | Integrated with experiment tracking |
| **Keras Tuner** | TensorFlow/Keras native |

### Practical Strategy

```
1. Start with published hyperparams from similar papers/models
2. LR range test (find order of magnitude)
3. Random search over narrow ranges (20–50 trials)
4. Optuna/Bayesian for final refinement
5. Always use early stopping — don't waste compute
```

---

## 8. Regularization & Generalization

| Technique | How | When |
|---|---|---|
| **Dropout** | Random neuron deactivation | FC layers (0.1–0.5) |
| **DropPath** | Random layer skip | Vision Transformers |
| **Weight Decay** | L2 penalty on weights | Always with AdamW (0.01–0.1) |
| **Data Augmentation** | Transform training samples | Always |
| **Label Smoothing** | Soften one-hot labels | Classification (0.1) |
| **MixUp / CutMix** | Blend training samples | Image classification |
| **Stochastic Depth** | Randomly drop layers | Deep networks |
| **Early Stopping** | Stop when val loss plateaus | Always |
| **Gradient Clipping** | Cap gradient norms | RNNs, transformers |
| **Batch/Layer/Group Norm** | Normalize activations | Architecture-dependent |
| **Knowledge Distillation** | Learn from larger teacher | Model compression |

### Normalization Layers

| Layer | When |
|---|---|
| **BatchNorm** | CNNs, large batches |
| **LayerNorm** | Transformers, RNNs, small batches |
| **GroupNorm** | Small batch CNNs, detection |
| **RMSNorm** | Modern transformers (Llama, etc.) |
| **InstanceNorm** | Style transfer, image generation |

---

## 9. Model Explainability & Debugging

### Explainability Tools

| Method | Type | Best For |
|---|---|---|
| **GradCAM / GradCAM++** | Gradient-based | CNN attention visualization |
| **Attention Maps** | Built-in | Transformer visualization |
| **SHAP DeepExplainer** | SHAP for DL | Any neural network |
| **Integrated Gradients** | Attribution | Feature importance |
| **Captum** (PyTorch) | Library | Unified interpretability API |
| **LIME** | Perturbation | Model-agnostic local explanations |

```python
# GradCAM with torchcam
from torchcam.methods import GradCAM
cam_extractor = GradCAM(model, target_layer="layer4")

with torch.no_grad():
    output = model(input_tensor)
cam = cam_extractor(output.argmax().item(), output)

# Captum (PyTorch official)
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
attributions = ig.attribute(input_tensor, target=predicted_class)
```

### Debugging Checklist

- [ ] **Overfit on 1 batch first** — if it can't memorize 1 batch, architecture/loss is broken
- [ ] **Check loss curve** — should decrease smoothly. Spikes = LR too high or data issues
- [ ] **Gradient norms** — log them. Exploding/vanishing = architecture or LR problem
- [ ] **Activation distributions** — dead ReLUs? All zeros = bad initialization or LR
- [ ] **Learning rate finder** — sweep LR from 1e-7 to 10, find steepest descent point
- [ ] **Data sanity check** — visualize random batch with labels. Labels correct?
- [ ] **Input range** — normalized correctly? Channels in right order (RGB vs BGR)?
- [ ] **Class weights** — imbalanced data without compensation?
- [ ] **Validation leakage** — same sample in train and val?

---

## 10. Evaluation & Validation

### Metrics by Task

**Classification:**

| Metric | When |
|---|---|
| Accuracy | Balanced classes |
| F1 / Macro-F1 | Imbalanced classes |
| AUC-ROC | Ranking, threshold-independent |
| Top-5 Accuracy | Large-scale (ImageNet-style) |
| MCC | Best single balanced metric |

**Object Detection:**

| Metric | Notes |
|---|---|
| mAP@0.5 | Standard COCO metric |
| mAP@[0.5:0.95] | Stricter, primary COCO metric |
| FPS | Inference speed |

**Segmentation:**

| Metric | Notes |
|---|---|
| mIoU | Mean Intersection over Union |
| Dice Score | Medical imaging standard |
| Pixel Accuracy | Simple but misleading for imbalanced |

**NLP:**

| Metric | Task |
|---|---|
| BLEU / ROUGE | Translation, summarization |
| Perplexity | Language modeling |
| Exact Match / F1 | QA tasks |
| BERTScore | Semantic similarity of generated text |

**Generation:**

| Metric | Notes |
|---|---|
| FID | Image quality vs dataset |
| CLIP Score | Image-text alignment |
| Human evaluation | Gold standard |

### Validation Best Practices

- [ ] Use a **held-out test set** never seen during training or tuning
- [ ] K-fold CV is expensive for DL — use single val split with different seeds instead
- [ ] Report confidence intervals (3–5 runs with different seeds)
- [ ] **Error analysis**: manually inspect worst predictions
- [ ] Test on **distribution shifts** (different lighting, domains, demographics)
- [ ] Profile **inference latency** and **memory usage** alongside accuracy

---

## 11. Model Optimization & Compression

### Techniques Overview

| Technique | Size Reduction | Speed Gain | Quality Loss | Effort |
|---|---|---|---|---|
| **Quantization (PTQ)** | 2–4x | 2–4x | Minimal | Low |
| **Quantization (QAT)** | 2–4x | 2–4x | Very low | Medium |
| **Pruning (structured)** | 2–10x | 2–5x | Low–Medium | Medium |
| **Knowledge Distillation** | Custom | Custom | Low | High |
| **ONNX Export** | — | 1.5–3x | None | Low |
| **TensorRT** | — | 2–6x | Minimal | Medium |
| **torch.compile** | — | 1.2–2x | None | Trivial |

### Quantization

```python
# Post-Training Quantization (PyTorch)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX + quantization
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QUInt8)
```

| Precision | Bits | Memory | Use Case |
|---|---|---|---|
| FP32 | 32 | Baseline | Training |
| FP16 / BF16 | 16 | 2x smaller | Training (AMP), inference |
| INT8 | 8 | 4x smaller | Inference (PTQ/QAT) |
| INT4 / NF4 | 4 | 8x smaller | LLM inference (QLoRA, GPTQ, AWQ) |

### ONNX Export

```python
import torch

dummy_input = torch.randn(1, 3, 224, 224).cuda()
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# Inference with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
result = session.run(None, {"input": input_array})
```

### TensorRT

```python
# Convert ONNX to TensorRT
# CLI:  trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

# Or via torch-tensorrt
import torch_tensorrt
optimized = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 224, 224))])
```

### Knowledge Distillation

```python
# Teacher-student training
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### Pruning

```python
import torch.nn.utils.prune as prune

# Unstructured pruning (30% of weights)
prune.l1_unstructured(model.layer, name="weight", amount=0.3)

# Structured pruning (remove entire channels)
prune.ln_structured(model.conv, name="weight", amount=0.4, n=2, dim=0)
```

### NVIDIA Model Optimizer (2025)

Unified library for quantization, pruning, distillation, speculative decoding:
- Supports PyTorch, HuggingFace, ONNX inputs
- Outputs optimized for TensorRT, TensorRT-LLM, vLLM
- Supports FP4/FP8/INT8/INT4 quantization

### Optimization Decision Tree

```
Need faster inference?
├── Free speedup    → torch.compile, ONNX Runtime
├── 2-4x speedup   → Quantization (INT8), TensorRT
├── Smaller model   → Pruning + fine-tune, knowledge distillation
├── Edge/mobile     → ONNX + quantization, TFLite, CoreML
└── LLM inference   → vLLM, TensorRT-LLM, GPTQ/AWQ quantization
```

---

## 12. Production Deployment

### Serving Options

| Pattern | Latency | Use Case | Tools |
|---|---|---|---|
| **REST API** | ms | Real-time | FastAPI, Flask |
| **gRPC** | sub-ms | High-throughput internal | Triton, TF Serving |
| **Batch** | hours | Nightly scoring | Spark, Ray |
| **Streaming** | sub-second | Event-driven | Kafka + model server |
| **Edge** | us–ms | Mobile, IoT | ONNX, TFLite, CoreML |
| **Serverless** | cold start | Low traffic | AWS Lambda, GCP Functions |

### Model Serving Frameworks

| Framework | Strengths |
|---|---|
| **Triton Inference Server** | Multi-framework, dynamic batching, GPU optimized |
| **TorchServe** | PyTorch native, easy setup |
| **BentoML** | Python-first, easy containerization |
| **Ray Serve** | Scalable, composable pipelines |
| **vLLM** | LLM serving with PagedAttention |
| **TGI** | HuggingFace LLM serving |
| **Ollama** | Local LLM serving, simple CLI |

### FastAPI + ONNX Example

```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

@app.post("/predict")
async def predict(data: dict):
    input_array = preprocess(data)
    result = session.run(None, {"input": input_array})
    return {"prediction": postprocess(result)}
```

### Production Checklist

- [ ] Model versioning (MLflow, W&B, DVC)
- [ ] Input validation and preprocessing pipeline
- [ ] Dynamic batching for GPU efficiency
- [ ] Health check endpoint
- [ ] Graceful error handling and fallback
- [ ] Load testing (Locust, k6)
- [ ] A/B testing or canary deployment
- [ ] Logging: inputs, outputs, latency, errors
- [ ] Rollback strategy
- [ ] GPU monitoring (nvidia-smi, DCGM)

### Key Tools

| Category | Tools |
|---|---|
| Experiment tracking | W&B, MLflow, Neptune, Comet |
| Pipeline orchestration | Airflow, Prefect, Dagster, Kubeflow |
| Containerization | Docker, Kubernetes, KServe |
| GPU management | NVIDIA Triton, DCGM, Ray |
| LLM serving | vLLM, TGI, Ollama, TensorRT-LLM |

---

## 13. Monitoring & Maintenance

### What to Monitor

| What | How | Tool |
|---|---|---|
| **Model accuracy** | Track metrics on live data | Evidently, NannyML, Arize |
| **Data drift** | Compare input distributions | Evidently, WhyLabs |
| **Concept drift** | Performance degradation over time | NannyML |
| **GPU utilization** | Avoid idle/OOM | Prometheus + Grafana, nvidia-smi |
| **Latency (p50, p95, p99)** | Response time distribution | Prometheus + Grafana |
| **Throughput** | Requests per second | Load balancer metrics |
| **Error rate** | Failed predictions | Application logging |
| **Data quality** | Schema, null values, ranges | Great Expectations |

### Retraining Strategy

```
1. Scheduled       → Retrain weekly/monthly with fresh data
2. Performance     → Retrain when metric drops below threshold
3. Drift-triggered → Retrain when data distribution shifts
4. Continuous      → Online learning / incremental updates
```

### Common Failure Modes

| Issue | Symptom | Fix |
|---|---|---|
| Data drift | Gradual accuracy drop | Retrain on recent data |
| Concept drift | Sudden accuracy drop | Retrain + update labels |
| Adversarial inputs | Unexpected wrong outputs | Input validation, adversarial training |
| GPU OOM in prod | Crashes under load | Optimize batch size, quantize |
| Stale features | Feature pipeline breaks | Data pipeline monitoring |

---

## Quick Reference — DL Project Kickstart

```
1.  Define problem, metrics, and compute budget
2.  Collect & label data (or find pretrained model)
3.  EDA: inspect samples, check label quality, class balance
4.  Preprocess: normalize, tokenize, augment
5.  Pick architecture:
    a. Check HuggingFace / timm for pretrained models
    b. Start with transfer learning (fine-tune or LoRA)
6.  Train:
    a. Mixed precision (AMP) — always
    b. AdamW + cosine schedule + warmup
    c. Gradient clipping, early stopping
    d. Overfit 1 batch first to verify setup
7.  Tune: Optuna / Ray Tune for LR, batch size, augmentation
8.  Explain: GradCAM (vision), attention maps (NLP), SHAP
9.  Evaluate: held-out test + error analysis + distribution shift
10. Optimize: ONNX → quantize → TensorRT (if latency matters)
11. Deploy: Triton / BentoML / vLLM + Docker + K8s
12. Monitor: Evidently for drift, Prometheus for infra
13. Retrain on schedule or drift trigger
```

---

## Sources & Further Reading

- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [PyTorch Mixed Precision (AMP)](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [DeepSpeed Training](https://www.deepspeed.ai/training/)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
- [LoRA Conceptual Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [Efficient Fine-Tuning with LoRA (Databricks)](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
- [Fine-Tuning LLMs with PEFT and LoRA (Mercity)](https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [Top 5 AI Model Optimization Techniques (NVIDIA)](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference)
- [Model Compression Survey 2025 (Frontiers)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full)
- [Fine-Tuning Landscape 2025](https://medium.com/@pradeepdas/the-fine-tuning-landscape-in-2025-a-comprehensive-analysis-d650d24bed97)
- [MLOps Best Practices 2025 (Shakudo)](https://www.shakudo.io/blog/mlops-best-practices-enterprise-2025)
- [AI Tech Stack 2026 (Kellton)](https://www.kellton.com/kellton-tech-blog/ai-tech-stack-2026)
- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
