"""One-off cleanup: archive stale models from failed attempts, fix LoRA framework."""
from clearml import Task

s = Task._get_default_session()
PROJECT = "ef18bd575abc408591532be2ac8f39b8"

r = s.send_request(
    service="models", action="get_all_ex",
    json={"project": [PROJECT], "page_size": 20,
          "only_fields": ["name", "framework", "uri", "created"],
          "order_by": ["-created"]},
)
models = r.json().get("data", {}).get("models", [])

keep, archive_ids, fix_framework = set(), [], []
seen = {}
for m in models:
    name = m["name"]
    if name.startswith("Imported by"):
        archive_ids.append(m["id"])          # auto-import junk from failed run
    elif name in seen:                        # duplicate -> archive the older one
        archive_ids.append(m["id"])
    else:
        seen[name] = m["id"]
        if name == "gpt2-lora-commentary" and m.get("framework") != "PyTorch":
            fix_framework.append(m["id"])

if archive_ids:
    s.send_request(service="models", action="archive", json={"models": archive_ids})
    print(f"archived {len(archive_ids)} stale models")
for mid in fix_framework:
    s.send_request(service="models", action="update",
                   json={"model": mid, "framework": "PyTorch"})
    print(f"fixed framework on {mid}")

r = s.send_request(
    service="models", action="get_all_ex",
    json={"project": [PROJECT], "page_size": 20,
          "only_fields": ["name", "framework", "uri"],
          "order_by": ["-created"]},
)
models = r.json().get("data", {}).get("models", [])
print(f"\n=== {len(models)} active models ===")
for m in models:
    print(f"  {m['name']}  | {m.get('framework')} | {m.get('uri', '')[:60]}")
