"""FootballVision detector + ByteTrack + team/possession enrichment.

Wraps the detector shipped by github.com/mserra0/FootballVision
(``models/yolov5su/best.pt``; classes ball, goalkeeper, player, referee) and
its ByteTrack / KMeans team-assignment ideas, while keeping this project's
per-frame feature contract.

The FootballVision repo's own ``main.py`` expects a ``yolov5lu`` checkpoint
that is not published; ``yolov5su/best.pt`` is the checkpoint that actually
ships, so that is what we use here.

Per analysed frame we return the same keys as ``FootballDetector.infer``
plus tracking:

    {"player": [(x1,y1,x2,y2,score), ...],
     "referee": [...],
     "ball": (cx, cy, score) | None,
     "players_tracked": {track_id: (bbox, team_id, side)},
     "possession_side": "home" | "away" | None}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.config import settings

# Prefer Aryaman's own fine-tuned YOLOv5su once it exists; fall back to the
# checkpoint shipped by the FootballVision repo until then.
_ARYAMAN_WEIGHTS = (settings.models_dir / "aryaman_yolo" /
                    "aryaman_yolov5lu_football" / "weights" / "best.pt")
DEFAULT_WEIGHTS = (_ARYAMAN_WEIGHTS if _ARYAMAN_WEIGHTS.exists() else
                   settings.models_dir / "footballvision" / "yolov5su_best.pt")


def _player_color(image: np.ndarray, bbox) -> np.ndarray | None:
    """Jersey colour of the top half of a player box (FootballVision method).

    Two-cluster KMeans; the cluster occupying the four crop corners is
    background, the other is the shirt. Returns None for unusable crops.
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    top = image[y1:y1 + max(2, (y2 - y1) // 2), x1:x2]
    if top.shape[0] < 4 or top.shape[1] < 4:
        return None
    from sklearn.cluster import KMeans

    pixels = top.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 8:
        return None
    km = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
    labels = km.fit_predict(pixels)
    clustered = labels.reshape(top.shape[0], top.shape[1])
    corners = [clustered[0, 0], clustered[0, -1],
               clustered[-1, 0], clustered[-1, -1]]
    background = max(set(corners), key=corners.count)
    return km.cluster_centers_[1 - background].astype(np.float32)


def _smooth_ball(raw: list, max_gap: int = 16, width: int = 1920) -> list:
    """FootballVision-style ball interpolation, bounded and outlier-aware.

    The upstream tracker keeps raw low-confidence ball detections and linearly
    interpolates the bbox offline (``trackers/tracker.py: ball_interpolation``),
    with no gap limit. This does the same on centres, but only across gaps of
    at most ``max_gap`` frames, rejects single-frame teleports, and marks each
    predicted position so the overlay can draw it differently.

    raw: per-frame ``(cx, cy, score)`` or None
    ->  per-frame ``(cx, cy, score, interpolated)`` or None
    """
    pts = [None if b is None else (float(b[0]), float(b[1]), float(b[2]))
           for b in raw]
    n = len(pts)
    if n == 0:
        return []

    # A detection far from the midpoint of its neighbours is probably a
    # spectator/line false positive at low confidence: drop it.
    jump = max(48.0, 0.18 * max(1, width))
    for i in range(1, n - 1):
        p, c, q = pts[i - 1], pts[i], pts[i + 1]
        if p and c and q:
            mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
            if ((c[0] - mx) ** 2 + (c[1] - my) ** 2) ** 0.5 > jump \
                    and c[2] < 0.4:
                pts[i] = None

    out = [None] * n
    for i, p in enumerate(pts):
        if p is not None:
            out[i] = (p[0], p[1], p[2], False)

    # internal gaps: linear interpolation, like the repo but bounded
    i = 0
    while i < n:
        if pts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and pts[j] is None:
            j += 1
        prev = pts[i - 1] if i > 0 else None
        nxt = pts[j] if j < n else None
        if prev is not None and nxt is not None and (j - i) <= max_gap:
            span = j - (i - 1)
            for k in range(i, j):
                a = (k - (i - 1)) / span
                out[k] = (prev[0] + (nxt[0] - prev[0]) * a,
                          prev[1] + (nxt[1] - prev[1]) * a,
                          min(prev[2], nxt[2]) * 0.85, True)
        i = j

    # leading / trailing runs: carry the nearest detection for a short span
    first = next((k for k, p in enumerate(pts) if p is not None), None)
    last = next((k for k in range(n - 1, -1, -1) if pts[k] is not None),
                None)
    if first is not None:
        for k in range(max(0, first - max_gap), first):
            out[k] = (pts[first][0], pts[first][1], pts[first][2] * 0.8, True)
    if last is not None:
        for k in range(last + 1, min(n, last + 1 + max_gap)):
            out[k] = (pts[last][0], pts[last][1], pts[last][2] * 0.8, True)
    return out


class FootballVisionTracker:
    """YOLOv5 detector + ByteTrack + team colour clustering + possession."""

    def __init__(self, weights: Path | None = None, conf: float = 0.15,
                 imgsz: int = 640, device: str | None = None):
        from types import SimpleNamespace
        from ultralytics import YOLO
        from ultralytics.trackers.byte_tracker import BYTETracker

        w = Path(weights or DEFAULT_WEIGHTS)
        if not w.exists():
            raise FileNotFoundError(
                f"FootballVision weights missing: {w}. Copy "
                "FootballVision/models/yolov5su/best.pt there first.")
        self.weights_used = w
        self.model = YOLO(str(w))
        self.names = {int(i): str(n).lower() for i, n in self.model.names.items()}
        self._inv_names = {n: i for i, n in self.names.items()}
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or settings.device
        # ultralytics ships ByteTrack; supervision >=0.31 no longer exports it
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=max(0.1, conf),
            track_low_thresh=0.1,
            new_track_thresh=max(0.1, conf),
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        ))
        self.home_attacks_right = os.getenv("HOME_ATTACKS_RIGHT", "1") == "1"
        self.trained = True
        self.coco_fallback = False

    # -- internals -----------------------------------------------------------

    def _fit_teams(self, track_colors: dict[int, np.ndarray],
                   track_cx: dict[int, float], width: int):
        """Return {track_id: team_id}, {team_id: "home"|"away"}.

        Colour clusters are mapped to home/away by their mean pitch x and the
        configured attacking direction. Tracks without a colour sample fall
        back to their own x-position at query time.
        """
        if len(track_colors) < 4:
            return {}, {}
        from sklearn.cluster import KMeans
        tids = list(track_colors)
        arr = np.stack([track_colors[t] for t in tids])
        # a 10 s goal clip may expose only a handful of tracks; clustering two
        # jerseys from four players is still better than the x-position
        # fallback, which labels everyone on one half as one team
        km = KMeans(n_clusters=min(2, len(tids)), init="k-means++", n_init=4,
                    random_state=0).fit(arr)
        team_of = {t: int(km.predict(track_colors[t].reshape(1, -1))[0])
                   for t in tids}
        means: dict[int, list[float]] = {}
        for t, team in team_of.items():
            means.setdefault(team, []).append(track_cx.get(t, 0.5))
        team_side = {}
        if len(means) >= 2:
            # Compare the teams to each other, not to frame centre: in a goal
            # clip both sides can sit in the same half, which used to send
            # every cluster to "away".
            right_team = max(means, key=lambda t: float(np.mean(means[t])))
            for team in means:
                is_right = team == right_team
                team_side[team] = ("home" if is_right == self.home_attacks_right
                                   else "away")
        else:
            for team, xs in means.items():
                right = float(np.mean(xs)) >= 0.5
                team_side[team] = ("home" if right == self.home_attacks_right
                                   else "away")
        return team_of, team_side

    # -- inference -----------------------------------------------------------

    def infer(self, images: list, chunk: int = 32, on_chunk=None) -> list[dict]:
        """Run detector + tracker over BGR images (in order).

        Two logical passes: detection/tracking first (collecting one jersey
        colour sample per track), then team assignment and possession over
        the retained lightweight per-frame rows. No images are retained.
        """
        raw: list[dict] = []
        track_colors: dict[int, np.ndarray] = {}
        track_cx: dict[int, float] = {}
        width = 1
        total = len(images)

        for start in range(0, total, chunk):
            batch = images[start:start + chunk]
            results = self.model.predict(
                batch, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False, stream=False,
            )
            for offset, r in enumerate(results):
                image = batch[offset]
                width = image.shape[1]
                boxes = r.boxes
                if boxes is not None:
                    # ultralytics' ByteTrack calls .numpy() on tensors; use CPU
                    boxes = boxes.cpu()
                if boxes is None or len(boxes) == 0:
                    # still advance the tracker's frame counter
                    self.tracker.update(boxes, img=image)
                    raw.append({"players": {}, "referees": [], "ball": None})
                    continue

                # goalkeepers are players for tracking/possession purposes
                cls_t = boxes.cls
                for j in range(len(cls_t)):
                    if self.names.get(int(cls_t[j])) == "goalkeeper":
                        cls_t[j] = self._inv_names.get("player", 2)

                players: dict[int, tuple] = {}
                referees: list[tuple] = []
                tracks = self.tracker.update(boxes, img=image)
                if tracks is not None and len(tracks):
                    for row in tracks:
                        x1, y1, x2, y2, tid, score, cid, _idx = row.tolist()
                        tid, cid = int(tid), int(cid)
                        name = self.names.get(cid, "")
                        if name == "player":
                            players[tid] = (x1, y1, x2, y2)
                            if tid not in track_colors and len(track_colors) < 500:
                                c = _player_color(image, (x1, y1, x2, y2))
                                if c is not None:
                                    track_colors[tid] = c
                                    track_cx[tid] = ((x1 + x2) / 2.0) / max(1, width)
                        elif name == "referee":
                            referees.append((x1, y1, x2, y2, float(score)))

                # ball: untracked raw detection, highest confidence
                ball = None
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy()
                for j in range(len(classes)):
                    if self.names.get(int(classes[j])) != "ball":
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy[j]]
                    score = float(confs[j])
                    if ball is None or score > ball[2]:
                        ball = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, score)

                raw.append({"players": players, "referees": referees, "ball": ball})
            if on_chunk:
                on_chunk(min(start + chunk, total), total)

        team_of, team_side = self._fit_teams(track_colors, track_cx, width)

        # FootballVision-style ball path: raw detections, bounded interpolation
        # across short gaps, predicted frames flagged for the overlay.
        ball_path = _smooth_ball([r["ball"] for r in raw], width=width)

        out: list[dict] = []
        for idx, row in enumerate(raw):
            players = row["players"]
            ball = ball_path[idx] if idx < len(ball_path) else None
            tracked: dict[int, tuple] = {}
            best = None
            for tid, bbox in players.items():
                team = team_of.get(tid)
                if team is not None:
                    side = team_side.get(team, "home")
                else:
                    right = ((bbox[0] + bbox[2]) / 2.0) / max(1, width) >= 0.5
                    side = "home" if right == self.home_attacks_right else "away"
                tracked[tid] = (bbox, team, side)
                if ball is not None:
                    bx, by = ball[0], ball[1]
                    # FootballVision possession: nearest bottom corner, 70 px
                    d1 = ((bbox[0] - bx) ** 2 + (bbox[3] - by) ** 2) ** 0.5
                    d2 = ((bbox[2] - bx) ** 2 + (bbox[3] - by) ** 2) ** 0.5
                    dist = min(d1, d2)
                    if best is None or dist < best[0]:
                        best = (dist, side)
            possession = best[1] if (best is not None and best[0] <= 70.0) else None
            out.append({
                "player": [(x1, y1, x2, y2, 1.0)
                           for (x1, y1, x2, y2) in players.values()],
                "referee": row["referees"],
                "ball": (ball[0], ball[1], ball[2]) if ball is not None else None,
                "ball_interpolated": bool(ball[3]) if ball is not None else False,
                "players_tracked": tracked,
                "possession_side": possession,
            })
        return out


def make_detector():
    """Return the FootballVision tracker when explicitly configured.

    The whole-match action-spotting pass is much faster with the project
    YOLOv8 model, so the default stays legacy; set ``FC_DETECTOR=footballvision``
    to use the FootballVision checkpoint for the full scan as well. The
    highlights script always uses ``FootballVisionTracker`` on its 10 s clips.
    """
    mode = os.getenv("FC_DETECTOR", "legacy").strip().lower()
    if mode in ("footballvision", "fv", "auto"):
        try:
            return FootballVisionTracker()
        except Exception as e:  # noqa: BLE001
            print(f"  [fv] detector unavailable ({type(e).__name__}: {e}) — "
                  "falling back to the project YOLOv8 model")
    from app.vision.detector import FootballDetector
    return FootballDetector()
