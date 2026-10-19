"""YOLOv8 football detector wrapper (players / referees / ball)."""
from __future__ import annotations

from pathlib import Path

from app.config import settings, YOLO_WEIGHTS, YOLO_FALLBACK


class FootballDetector:
    """Batched detector. Trained weights preferred, COCO fallback labelled clearly."""

    def __init__(self, weights: Path | None = None, conf: float = 0.25):
        from ultralytics import YOLO

        self.trained = False
        w = weights or YOLO_WEIGHTS
        if not w.exists():
            w = YOLO_FALLBACK
        self.weights_used = w
        if w.name.startswith("yolov8n.pt"):
            # COCO fallback: person->player, sports ball->ball; no referee class
            self.names = {0: "player", 32: "ball"}
            self.coco_fallback = True
        else:
            self.coco_fallback = False
        self.conf = conf
        self.model = YOLO(str(w))
        if not self.coco_fallback:
            self.names = {i: n.lower() for i, n in self.model.names.items()}

    def infer(self, images: list, chunk: int = 64, on_chunk=None) -> list[dict]:
        """images: list of BGR numpy arrays or paths. Returns per-image detection dicts.

        Runs in chunks so huge videos stream through with per-chunk progress
        (on_chunk(done, total) fires after each chunk).
        """
        if not images:
            return []
        out_all: list[dict] = []
        total = len(images)
        for i in range(0, total, chunk):
            results = self.model.predict(
                images[i:i + chunk], conf=self.conf, imgsz=480, device=settings.device,
                verbose=False, stream=False,
            )
            for r in results:
                dets = {"player": [], "referee": [], "ball": None}
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        name = self.names.get(cls_id, "")
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        score = float(box.conf.item())
                        if name in ("player", "football-players", "person"):
                            dets["player"].append((x1, y1, x2, y2, score))
                        elif name in ("referee", "refree"):
                            dets["referee"].append((x1, y1, x2, y2, score))
                        elif name == "ball":
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            if dets["ball"] is None or score > dets["ball"][2]:
                                dets["ball"] = (cx, cy, score)
                out_all.append(dets)
            if on_chunk:
                on_chunk(min(i + chunk, total), total)
        return out_all
