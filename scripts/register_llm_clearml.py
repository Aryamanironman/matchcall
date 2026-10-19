"""Register the GPT-2 LoRA v3 commentary run in ClearML.

The adapter was trained before this task existed, so this replays its recorded
loss history as ClearML scalars, uploads the adapter/tokenizer/logs as
artifacts, registers the adapter as an output model, and writes the task URL
to ``models/gpt2_lora_v3/clearml_task.json`` for the demo and report.

    .venv\\Scripts\\python scripts\\register_llm_clearml.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import training_status as ts
from app.config import settings

PROJECT = "Football Commentator"
TASK_NAME = "GPT-2 LoRA v3 — grounded commentary (completion-only)"
MODEL_NAME = "aryaman-gpt2-lora-commentary"


def main() -> None:
    llm = ts.llm_status()
    demo = ts.demo_status()
    total = demo.get("clips") or len(demo.get("moments") or []) or 10
    accepted = int(demo.get("gpt2_lines") or 0)
    fallback = int(demo.get("template_lines") or 0)
    ev = {
        "eval": "ARG-FRA 2022 second half — 10 selected moments",
        "total": total,
        "gpt2_accepted": accepted,
        "template_fallback": fallback,
        "validator_rejections": total - accepted,
        "accept_rate": round(accepted / max(1, total), 3),
        "final_loss": llm.get("final_loss"),
        "steps": llm.get("steps"),
        "note": ("Completion-only LoRA rank 16 on a grounded corpus. Lines that "
                 "invent teams, scores, cards or players are rejected by the "
                 "grounding validator and replaced with templates."),
    }
    eval_path = settings.out_root / "llm_eval.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print("wrote", eval_path, flush=True)

    from clearml import OutputModel, Task, TaskTypes

    task = Task.init(
        project_name=PROJECT, task_name=TASK_NAME,
        task_type=TaskTypes.training, output_uri=True,
        reuse_last_task_id=False,
        tags=["aryaman", "gpt2", "lora", "commentary", "v3"],
    )
    task.connect({
        "base_model": "gpt2",
        "method": "LoRA r=16, completion-only loss",
        "epochs": llm.get("epochs"),
        "batch_size": 4,
        "steps": llm.get("steps"),
        "corpus_examples": llm.get("corpus", {}).get("examples"),
        "max_words": 20,
        "validator": "grounding (team/score/player/markup)",
    })
    task.set_user_properties(
        adapter=str(llm.get("adapter", {}).get("path")),
        checkpoint=llm.get("trainer_state"),
        eval="ARG-FRA 10 moments",
    )

    logger = task.get_logger()
    for h in llm.get("loss_history") or []:
        step = h.get("step")
        if h.get("loss") is not None:
            logger.report_scalar("train", "loss", value=h["loss"],
                                 iteration=step)
        if h.get("lr") is not None:
            logger.report_scalar("train", "learning_rate", value=h["lr"],
                                 iteration=step)
    logger.report_scalar("eval", "gpt2_accept_rate",
                         value=ev["accept_rate"], iteration=0)
    logger.report_scalar("eval", "template_fallback",
                         value=fallback, iteration=0)

    uploads = {
        "llm_eval": eval_path,
        "trainer_state": Path(llm.get("trainer_state", "")),
        "adapter_config": ts.LLM_DIR / "adapter_config.json",
        "README": ts.LLM_DIR / "README.md",
        "training_args": ts.LLM_DIR / "training_args.bin",
        "corpus_sample": settings.data_dir / "commentator_train.jsonl",
    }
    for name, path in uploads.items():
        try:
            if path and Path(path).exists():
                task.upload_artifact(name, artifact_object=str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"upload warning {name}: {exc}", flush=True)

    adapter = Path(llm.get("adapter", {}).get("path", ""))
    if adapter.exists():
        try:
            om = OutputModel(task=task, name=MODEL_NAME)
            om.update_weights(weights_filename=str(adapter),
                              auto_delete_file=False)
            try:
                om.update_design(
                    config=(ts.LLM_DIR / "adapter_config.json").read_text())
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"model registry warning: {exc}", flush=True)

    task.mark_completed()
    url = task.get_output_log_web_page()
    print("ClearML task:", url, flush=True)
    (ts.LLM_DIR / "clearml_task.json").write_text(json.dumps({
        "id": task.id, "url": url, "name": TASK_NAME,
        "registered": True,
    }, indent=2), encoding="utf-8")
    task.close()


if __name__ == "__main__":
    main()
