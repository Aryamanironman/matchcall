import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL_NAME = "gpt2"
OUTPUT_DIR = "models/gpt2_lora"
DATA_PATH = "data/commentator_train.jsonl"

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, target_modules=["c_attn", "c_proj", "c_fc"], lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    num_train_epochs=4,
    fp16=False,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # use processing_class to avoid warning
)
trainer.train()
trainer.save_model(OUTPUT_DIR)