"""Step 2: Fine-tune Qwen2.5-0.5B-Instruct on recipe data using LoRA."""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "finetuned_model"
DATA_PATH = Path(__file__).parent / "recipe_dataset.json"

SYSTEM_PROMPT = "You are a helpful recipe assistant. Given a list of ingredients, suggest a recipe."

# --- Dataset ---
class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.samples = []
        for item in data:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Ingredients: {item['ingredients']}"},
                {"role": "assistant", "content": item["recipe"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
            input_ids = enc["input_ids"]
            self.samples.append({
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(input_ids),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True
    )

    # LoRA config — lightweight fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    recipes = json.loads(DATA_PATH.read_text())
    dataset = RecipeDataset(recipes, tokenizer)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=30,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    print("Starting fine-tuning...")
    trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()