"""Fine-tune YOLOv8n on the football-players dataset (Roboflow CC-BY mirror)."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def find_data_yaml(root: Path) -> Path | None:
    cands = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    for c in cands:
        if "train" in c.read_text(encoding="utf-8", errors="ignore"):
            return c
    return cands[0] if cands else None


def fix_yaml_paths(yaml_path: Path) -> None:
    """Roboflow exports carry absolute machine paths; rewrite to local splits."""
    base = yaml_path.parent
    prefix = "images/" if (base / "images").exists() else ""

    def resolve(split: str) -> str:
        cands = ([f"{prefix}{split}"] if prefix else []) + [split]
        alts = {"val": ["valid", "test"], "test": ["val", "valid"]}.get(split, [])
        for alt in alts:
            cands += [f"{prefix}{alt}"] if prefix else []
            cands += [alt]
        for c in cands:
            if (base / c).is_dir():
                return str(base / c).replace("\\", "/")
        return str(base / split).replace("\\", "/")

    lines = []
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(train|val|test):", line)
        if m:
            lines.append(f"{m.group(1)}: {resolve(m.group(1))}")
        elif line.startswith("path:"):
            continue  # drop absolute machine root; splits above are absolute
        else:
            lines.append(line)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(data_root: Path, out_dir: Path, epochs: int = 40, imgsz: int = 640,
          batch: int = 16):
    from ultralytics import YOLO

    yaml_path = find_data_yaml(data_root)
    if not yaml_path:
        raise FileNotFoundError(f"no data.yaml under {data_root}")
    yaml_path = yaml_path.resolve()  # split paths are written relative to yaml dir
    fix_yaml_paths(yaml_path)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=batch,
        project=str(out_dir.resolve()), name="run1", exist_ok=True,
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        workers=2, patience=10, cache=False,
    )
    print("trained ->", out_dir / "run1" / "weights" / "best.pt")
    return results


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/roboflow-mirror/datasets/Football-players-detection-1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models/yolov8_football")
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    train(root, out, epochs=epochs)
