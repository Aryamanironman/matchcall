"""Cleanup v2: delete junk models (dead URIs / dupes), re-register LoRA properly."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clearml import Task, OutputModel
from clearml import TaskTypes

s = Task._get_default_session()
PROJECT = "ef18bd575abc408591532be2ac8f39b8"

DELETE = [
    "76ed36a2cf81491e926e3b33db48624d",  # 'Imported by...' junk from failed run
    "3eb8b856658741bdaafad8c11ec01356",  # duplicate resnet (failed run)
    "0f3bf58489ce49869fcfefb45d9c48b5",  # duplicate yolov8 (first attempt)
    "792f149a5352478b8191beb7a679fab7",  # lora: TensorFlow label + dead temp URI
]
for mid in DELETE:
    r = s.send_request(service="models", action="delete",
                       json={"model": mid, "force": True}).json()
    meta = r.get("meta", {})
    print(f"delete {mid[:8]}... -> {meta.get('result_code')} {meta.get('result_msg', '')}")

# re-register the LoRA adapter: explicit framework + real upload to file server
task = Task.init(
    project_name="Football Commentator",
    task_name="GPT-2 LoRA commentary adapter — registration",
    task_type=getattr(TaskTypes, "training", None) or TaskTypes.custom,
    reuse_last_task_id=False,
)
lora_dir = ROOT / "models" / "gpt2_lora"
cfg = json.loads((lora_dir / "adapter_config.json").read_text())
props = {f"peft/{k}": str(cfg[k]) for k in
         ("r", "lora_alpha", "lora_dropout", "target_modules",
          "base_model_name_or_path", "bias", "task_type") if k in cfg}
props["note"] = ("training script/curves not persisted; adapter registered "
                 "for provenance and lineage (base: gpt2)")
task.set_user_properties(**props)

with tempfile.TemporaryDirectory() as td:
    base = Path(td) / "gpt2_lora_adapter"
    zpath = shutil.make_archive(str(base), "zip", lora_dir)
    model = OutputModel(task=task, name="gpt2-lora-commentary", framework="PyTorch")
    model.update_weights(weights_filename=zpath,
                         upload_uri="https://files.clear.ml",
                         auto_delete_file=False)

task.mark_completed()
task.close()

print("\nLoRA re-registered.")
