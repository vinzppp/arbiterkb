"""Compare pod probe prompts with the Mac G1 probe, field by field and byte for byte."""
import json
import sys
from pathlib import Path

mac_dir, pod_dir = Path(sys.argv[1]), Path(sys.argv[2])
report = {}
for mac in sorted(mac_dir.glob("*.json")):
    pod = pod_dir / mac.name
    if not pod.exists():
        report[mac.name] = "MISSING on pod"
        continue
    a, b = json.loads(mac.read_text()), json.loads(pod.read_text())
    r = {"file_bytes_equal": mac.read_bytes() == pod.read_bytes()}
    for k in ("task", "episode", "outcome", "commands", "objects", "stock_memi_system_ok"):
        r[k] = a.get(k) == b.get(k)
    arms = {}
    for arm in sorted(set(a["prompts"]) | set(b["prompts"])):
        pa, pb = a["prompts"].get(arm, []), b["prompts"].get(arm, [])
        diffs = []
        for i in range(max(len(pa), len(pb))):
            sa = json.dumps(pa[i], sort_keys=True) if i < len(pa) else None
            sb = json.dumps(pb[i], sort_keys=True) if i < len(pb) else None
            if sa != sb:
                if sa is None or sb is None:
                    diffs.append({"call": i, "where": "missing"})
                else:
                    j = next((j for j, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
                    diffs.append({"call": i, "at": j, "mac": sa[max(0, j - 60):j + 80], "pod": sb[max(0, j - 60):j + 80]})
        arms[arm] = {"calls": [len(pa), len(pb)], "equal": not diffs, "diffs": diffs[:2]}
    r["arms"] = arms
    report[mac.name] = r
print(json.dumps(report, indent=1, ensure_ascii=False))
