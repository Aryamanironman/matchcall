"""ResNet18 event classifier: inference over analysis windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from app.config import EVENT_CLASSES

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def load_model(weights: Path, device: str | None = None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(weights, map_location=dev, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ckpt.get("classes", EVENT_CLASSES)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model, dev, ckpt.get("classes", EVENT_CLASSES)


@torch.inference_mode()
def classify_windows(proxy_dir: Path, windows: list, t0: float, weights: Path,
                     batch: int = 64) -> None:
    """Adds cls_probs / net_label / net_conf to each Window (in place)."""
    if not windows:
        return
    model, dev, classes = load_model(weights)
    step = 1.0 / 4.0  # analysis fps (settings.analysis_fps)
    from app.config import settings as s
    step = 1.0 / s.analysis_fps

    # gather all frame paths in window order
    frame_lists: list[list[Path]] = []
    for w in windows:
        s_i = int(round((w.t0 - t0) / step))
        e_i = int(round((w.t1 - t0) / step))
        ps = [proxy_dir / f"{i + 1:06d}.jpg" for i in range(s_i, e_i)]
        frame_lists.append([p for p in ps if p.exists()])

    tensors: list = []
    owners: list[int] = []
    for wi, ps in enumerate(frame_lists):
        for p in ps:
            try:
                img = Image.open(p).convert("RGB")
            except OSError:
                continue
            tensors.append(_TF(img))
            owners.append(wi)

    probs = [np.zeros(len(classes), dtype=np.float32) for _ in windows]
    counts = [0] * len(windows)
    for b in range(0, len(tensors), batch):
        xb = torch.stack(tensors[b:b + batch]).to(dev)
        out = torch.softmax(model(xb), dim=1).cpu().numpy()
        for k, row in enumerate(out):
            wi = owners[b + k]
            probs[wi] += row
            counts[wi] += 1

    for wi, w in enumerate(windows):
        if counts[wi] == 0:
            w.cls_probs = {}
            continue
        mean_p = probs[wi] / counts[wi]
        w.cls_probs = {c: round(float(mean_p[j]), 4) for j, c in enumerate(classes)}
        top = int(mean_p.argmax())
        w.net_label = classes[top]
        w.net_conf = float(mean_p[top])
