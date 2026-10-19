"""Snapshot the ClearML project (tasks + models) for the demo and report.

Writes ``output/clearml_snapshot.json``; both the web demo and the report read
that file, so neither needs the ClearML SDK or network at serve time.

    .venv\\Scripts\\python scripts\\clearml_snapshot.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import training_status as ts

PROJECT = ts.PROJECT


def main() -> None:
    from clearml import Model, Task

    snap = {
        "generated": time.time(),
        "project": PROJECT,
        "project_url": ts.PROJECT_URL,
        "tasks": [],
        "models": [],
    }
    try:
        tasks = Task.get_tasks(project_name=PROJECT)
    except Exception as exc:  # noqa: BLE001
        print(f"task query failed: {exc}", flush=True)
        tasks = []
    for t in tasks:
        try:
            snap["tasks"].append({
                "id": t.id,
                "name": t.name,
                "type": str(t.task_type or ""),
                "status": str(t.status or ""),
                "url": t.get_output_log_web_page(),
            })
        except Exception:  # noqa: BLE001
            continue
    try:
        models = Model.query_models(project_name=PROJECT,
                                    only_published=False, max_results=100)
    except Exception as exc:  # noqa: BLE001
        print(f"model query failed: {exc}", flush=True)
        models = []
    for m in models:
        try:
            snap["models"].append({
                "id": m.id, "name": m.name,
                "url": m.get_web_page() if hasattr(m, "get_web_page") else "",
                "framework": getattr(m, "framework", ""),
            })
        except Exception:  # noqa: BLE001
            continue

    ts.SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    ts.SNAPSHOT.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"wrote {ts.SNAPSHOT}: {len(snap['tasks'])} tasks, "
          f"{len(snap['models'])} models", flush=True)


if __name__ == "__main__":
    main()
