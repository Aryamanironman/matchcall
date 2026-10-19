"""Train ResNet18 event classifier on auto-labeled windows (weak supervision).

Every proxy frame inherits its window's label; standard supervised training
with train/val split by time (no leakage across the boundary).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from app.config import settings, EVENT_CLASSES

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FrameDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], train: bool):
        self.items = items
        norm = transforms.Normalize(MEAN, STD)
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(), norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def build_items(proxy_dir: Path, windows: list[dict], t0: float) -> list[tuple[str, int]]:
    step = 1.0 / settings.analysis_fps
    cls_idx = {c: i for i, c in enumerate(EVENT_CLASSES)}
    items = []
    for k, w in enumerate(windows):
        s = int(round((w["t0"] - t0) / step))
        e = int(round((w["t1"] - t0) / step))
        for fi in range(s, e):
            p = proxy_dir / f"{fi + 1:06d}.jpg"
            if p.exists():
                items.append((str(p), cls_idx[w["label"]]))
    return items


def train(proxy_dir: Path, windows: list[dict], t0: float, out: Path,
          epochs: int = 8, batch: int = 64, lr: float = 3e-4, val_frac: float = 0.15,
          device_str: str | None = None, on_epoch=None):
    items = build_items(proxy_dir, windows, t0)
    if len(items) < 32:
        raise RuntimeError(f"only {len(items)} labeled frames — not enough to train")
    rng = random.Random(7)
    rng.shuffle(items)
    n_val = max(8, int(len(items) * val_frac))
    val_items, train_items = items[:n_val], items[n_val:]

    dev = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    tl = DataLoader(FrameDataset(train_items, True), batch_size=batch, shuffle=True,
                    num_workers=2, pin_memory=(dev.type == "cuda"), drop_last=True)
    vl = DataLoader(FrameDataset(val_items, False), batch_size=batch, shuffle=False,
                    num_workers=2, pin_memory=(dev.type == "cuda"))

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(EVENT_CLASSES))
    model.to(dev)

    # class weights for imbalance
    counts = np.bincount([lab for _, lab in train_items], minlength=len(EVENT_CLASSES))
    wts = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=dev)
    wts = wts / wts.mean()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=wts)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, history = 0.0, []
    for ep in range(epochs):
        model.train()
        tot = cor = loss_sum = 0
        for xb, yb in tl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            cor += (logits.argmax(1) == yb).sum().item()
            tot += len(xb)
        sched.step()
        train_acc = cor / max(tot, 1)

        model.eval()
        vcor = vt = 0
        with torch.inference_mode():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                pred = model(xb).argmax(1)
                vcor += (pred == yb).sum().item()
                vt += len(xb)
        val_acc = vcor / max(vt, 1)
        history.append({"epoch": ep + 1, "train_loss": loss_sum / max(tot, 1),
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {ep+1}/{epochs} loss={loss_sum/max(tot,1):.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if on_epoch:
            on_epoch(ep + 1, epochs, val_acc)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": EVENT_CLASSES, "val_acc": val_acc}, out)
    print(f"best val_acc={best_acc:.3f} -> {out}")
    return history


if __name__ == "__main__":
    import sys
    d = json.loads(Path(sys.argv[3]).read_text())
    train(Path(sys.argv[1]), d["windows"], 0.0, Path(sys.argv[2]))
