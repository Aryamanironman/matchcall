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
                          TrainerCallback, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
MARKER = "[INTENSITY]"


class ClearMLCallback(TrainerCallback):
    """Stream Trainer scalars into the ClearML task as they are logged."""

    def __init__(self, task):
        self.logger = task.get_logger()

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                self.logger.report_scalar(
                    "train", key, value=float(value),
                    iteration=state.global_step)


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
    ap.add_argument("--fp16", action="store_true",
                    help="mixed precision on CUDA (much faster on a small GPU)")
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--clearml", action="store_true",
                    help="log the run, curves and adapter to ClearML")
    ap.add_argument("--task-name",
                    default="GPT-2 LoRA v4 — SoccerReplay-1988 commentary")
    args = ap.parse_args()

    task = None
    callbacks = []
    if args.clearml:
        try:
            from clearml import Task, TaskTypes
            task = Task.init(
                project_name="Football Commentator", task_name=args.task_name,
                task_type=TaskTypes.training, output_uri=True,
                reuse_last_task_id=False,
                tags=["aryaman", "gpt2", "lora", "commentary",
                      Path(args.out).name.replace("gpt2_lora_", ""),
                      "soccerreplay-1988"])
            task.connect({
                "base_model": args.base,
                "method": f"LoRA r={args.rank}, completion-only loss",
                "epochs": args.epochs,
                "batch_size": args.batch,
                "grad_accum": args.grad_accum,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "max_words": 55,
                "dataset": "Homie0609/SoccerReplay-1988 + synthetic grounding",
            })
            callbacks.append(ClearMLCallback(task))
            print("ClearML task:", task.get_output_log_web_page(), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"ClearML disabled ({type(exc).__name__}: {exc})", flush=True)
            task = None

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
        fp16=args.fp16,
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=ds,
                      data_collator=make_collator(tok.eos_token_id),
                      callbacks=callbacks)
    print(f"training {len(ds)} examples (completion-only loss) -> {out}")
    if task is not None:
        task.get_logger().report_text(
            f"corpus: {len(ds)} examples from {args.data}")
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print("saved adapter ->", out)
    print("final:", trainer.state.log_history[-1])

    if task is not None:
        try:
            from clearml import OutputModel
            uploads = {
                "adapter_config": out / "adapter_config.json",
                "adapter_model": out / "adapter_model.safetensors",
                "tokenizer": out / "tokenizer.json",
                "trainer_state": out / "trainer_state.json",
                "training_args": out / "training_args.bin",
                "corpus": Path(args.data),
            }
            for name, path in uploads.items():
                if path.exists():
                    task.upload_artifact(name, artifact_object=str(path))
            adapter = out / "adapter_model.safetensors"
            if adapter.exists():
                run_tag = Path(args.out).name.replace("gpt2_lora_", "")
                om = OutputModel(task=task,
                                 name=f"aryaman-gpt2-lora-commentary-{run_tag}")
                om.update_weights(weights_filename=str(adapter),
                                  auto_delete_file=False)
            task.mark_completed()
            url = task.get_output_log_web_page()
            (out / "clearml_task.json").write_text(json.dumps({
                "id": task.id, "url": url, "name": args.task_name,
            }, indent=2), encoding="utf-8")
            print("ClearML task:", url, flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"ClearML finalize warning: {type(exc).__name__}: {exc}",
                  flush=True)
            try:
                task.mark_completed()
            except Exception:  # noqa: BLE001
                pass
        finally:
            task.close()


if __name__ == "__main__":
    main()
