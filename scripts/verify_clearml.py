"""Verify the ClearML upload: list tasks + models in the project."""
import sys
from clearml import Task

s = Task._get_default_session()
PROJECT = "ef18bd575abc408591532be2ac8f39b8"

r = s.send_request(
    service="tasks", action="get_all_ex",
    json={"project": [PROJECT], "page_size": 30,
          "only_fields": ["name", "status"],
          "order_by": ["-created"]},
)
tasks = r.json().get("data", {}).get("tasks", [])
print(f"=== {len(tasks)} tasks (including archived) ===")
for t in tasks:
    print(f"  [{t['status']:>9}] {t['name']}")

r = s.send_request(
    service="models", action="get_all_ex",
    json={"project": [PROJECT], "page_size": 20,
          "only_fields": ["name", "framework", "uri"],
          "order_by": ["-created"]},
)
models = r.json().get("data", {}).get("models", [])
print(f"\n=== {len(models)} models registered ===")
for m in models:
    print(f"  {m['name']}  | {m.get('framework')} | {m.get('uri', '')[:65]}")
