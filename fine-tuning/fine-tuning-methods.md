# Fine-Tuning Methods for Large Language Models

## Overview

Fine-tuning adapts a pre-trained model to a specific task or domain. Methods range from updating all parameters (expensive) to updating a tiny fraction (efficient).

---

## 1. Full Fine-Tuning

Updates **all** model parameters on your dataset.

- **Pros**: Maximum performance, full model adaptation
- **Cons**: Requires large GPU memory (equal to model size), slow, risk of catastrophic forgetting
- **When to use**: Large dataset, sufficient compute, need maximum quality
- **Memory**: ~4x model size (model + optimizer + gradients)

```
Model: 7B params → ~28GB+ GPU RAM needed
```

---

## 2. LoRA (Low-Rank Adaptation)

Freezes the base model and injects small trainable **low-rank matrices** into attention layers.

- **Pros**: Trains <1% of parameters, tiny adapter files (~5-50MB), fast, multiple adapters per base model
- **Cons**: Slightly lower performance than full fine-tuning on complex tasks
- **When to use**: Limited GPU memory, need quick domain adaptation
- **Key params**: `r` (rank), `alpha` (scaling), `target_modules` (which layers)

```python
# Example
LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
```

**Used in this repo**: `task2-recipe-chatbot`

---

## 3. QLoRA (Quantized LoRA)

Combines **4-bit quantization** of the base model with LoRA adapters.

- **Pros**: Fine-tune a 7B model on a single 8GB GPU, almost no quality loss vs LoRA
- **Cons**: Requires `bitsandbytes` library (CUDA only, no Mac MPS support)
- **When to use**: Very limited GPU memory, want to fine-tune larger models cheaply

```python
# Example
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
# + LoRA on top
```

---

## 4. Prefix Tuning

Prepends **trainable virtual tokens** (prefix vectors) to every transformer layer's input.

- **Pros**: Very few trainable parameters, no modification to model architecture
- **Cons**: Can reduce effective context length, less flexible than LoRA
- **When to use**: Text generation tasks where you want soft prompts at every layer

```
Input: [PREFIX_1][PREFIX_2]...[PREFIX_k] + actual tokens
       (trainable)                        (frozen)
```

---

## 5. Prompt Tuning

Adds **trainable soft prompt embeddings** only at the input layer (not every layer like prefix tuning).

- **Pros**: Simplest PEFT method, extremely few parameters (~0.01%), multiple tasks via different prompts
- **Cons**: Lower performance than LoRA/prefix tuning, needs larger base models (10B+) to work well
- **When to use**: Very large models, many tasks sharing one base model

```
Input: [SOFT_1][SOFT_2]...[SOFT_k] + "Translate this to French: ..."
       (trainable)                   (frozen)
```

---

## 6. Adapter Tuning

Inserts small **bottleneck layers** (adapter modules) between existing transformer layers.

- **Pros**: Modular, each task gets its own adapter, base model stays frozen
- **Cons**: Adds inference latency (extra layers), more parameters than LoRA
- **When to use**: Multi-task setups where you swap adapters per task

```
Transformer Block:
  Attention → [ADAPTER] → FFN → [ADAPTER] → Output
               (trainable)       (trainable)
```

---

## 7. RLHF (Reinforcement Learning from Human Feedback)

Fine-tunes a model using **human preference rankings** via a reward model + PPO.

- **Pros**: Aligns model with human values, reduces harmful/incorrect outputs
- **Cons**: Complex pipeline (SFT → reward model → PPO), expensive, unstable training
- **When to use**: Chat/assistant models that need alignment with human preferences
- **Steps**:
  1. Supervised Fine-Tuning (SFT) on instruction data
  2. Train a reward model on human preference pairs
  3. Optimize policy with PPO against the reward model

---

## 8. DPO (Direct Preference Optimization)

Simplifies RLHF by **skipping the reward model** — directly optimizes on preference pairs.

- **Pros**: Simpler than RLHF (no reward model, no PPO), stable training, comparable results
- **Cons**: Needs good preference data (chosen vs rejected pairs)
- **When to use**: Alignment without the complexity of RLHF

```
Training data format:
  prompt → chosen_response (good)
  prompt → rejected_response (bad)
```

---

## 9. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Learns **rescaling vectors** that multiply key, value, and FFN activations.

- **Pros**: Even fewer parameters than LoRA, fast inference (no extra layers)
- **Cons**: Less expressive than LoRA on complex tasks
- **When to use**: Extreme parameter efficiency needed

---

## 10. Supervised Fine-Tuning (SFT)

Standard fine-tuning on **instruction-response pairs** using cross-entropy loss.

- **Pros**: Straightforward, well-understood, works with any PEFT method on top
- **Cons**: Quality depends heavily on dataset quality
- **When to use**: First step before RLHF/DPO, or standalone for domain adaptation

```json
{"instruction": "Summarize this article", "input": "...", "output": "..."}
```

---

## Comparison Table

| Method | Trainable Params | GPU Memory | Quality | Complexity |
|--------|-----------------|------------|---------|------------|
| Full Fine-Tuning | 100% | Very High | Best | Low |
| LoRA | ~0.1-1% | Low | Very Good | Low |
| QLoRA | ~0.1-1% | Very Low | Very Good | Medium |
| Prefix Tuning | ~0.1% | Low | Good | Medium |
| Prompt Tuning | ~0.01% | Very Low | Moderate | Low |
| Adapter Tuning | ~1-5% | Medium | Very Good | Medium |
| RLHF | Varies | Very High | Best (aligned) | Very High |
| DPO | Varies | High | Very Good (aligned) | Medium |
| IA3 | ~0.01% | Very Low | Good | Low |
| SFT | Varies | Varies | Good | Low |

---

## Choosing the Right Method

```
Need alignment/safety?
  ├─ Yes → DPO (simpler) or RLHF (maximum control)
  └─ No → Domain/task adaptation?
           ├─ Limited GPU? → QLoRA (CUDA) or LoRA (any hardware)
           ├─ Multiple tasks, one model? → Adapter Tuning or Prompt Tuning
           ├─ Maximum quality, big budget? → Full Fine-Tuning
           └─ Minimal params? → IA3 or Prompt Tuning
```
