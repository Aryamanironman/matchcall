"""Train the grounded GPT-2 LoRA commentator (completion-only loss).

The first attempt trained on the whole sequence, so most of the loss went to
predicting the deterministic history tokens; the adapter learned the markup
but not which event maps to which kind of line. Here the prompt tokens are
masked (-100) and the loss is computed only on
``[INTENSITY] <level> [COMMENTARY] <line>``.

Usage:
  python scripts/build_commentary_corpus.py
  python scripts/train_commentator_lora.py --data data/commentator_train_v2.jsonl \
      --out models/gpt2_lora_v3 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
MARKER = "[INTENSITY]"


class CommentaryDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line)["text"]
                     for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text = self.rows[i]
        idx = text.find(MARKER)
        if idx < 0:
            idx = text.find("[COMMENTARY]")
        idx = max(0, idx)
        prompt = text[:idx]
        completion = text[idx:]
        p = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c = self.tok(completion, add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > self.max_length:
            c = c[:max(8, self.max_length - len(p))]
        input_ids = p + c
        labels = [-100] * len(p) + c
        return {"input_ids": input_ids, "labels": labels}


def make_collator(pad_id: int):
    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids, labels, att = [], [], []
        for x in batch:
            k = n - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad_id] * k)
            labels.append(x["labels"] + [-100] * k)
            att.append([1] * len(x["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(att),
                "labels": torch.tensor(labels)}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "commentator_train_v2.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "models" / "gpt2_lora_v3"))
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=90)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float32, device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

    ds = CommentaryDataset(Path(args.data), tok, args.max_length)
    args_train = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=False,
        logging_steps=20,
        save_steps=2000,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=ds,
                      data_collator=make_collator(tok.eos_token_id))
    print(f"training {len(ds)} examples (completion-only loss) -> {out}")
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print("saved adapter ->", out)
    print("final:", trainer.state.log_history[-1])


if __name__ == "__main__":
    main()
