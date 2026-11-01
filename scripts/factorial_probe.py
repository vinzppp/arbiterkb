"""One sanitized authorized endpoint probe for the frozen factorial model."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def cost(usage: dict) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return ((prompt - cached) * .10 + cached * .01 + output * .50) / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--url", default="https://openrouter.ai/api/v1/chat/completions")
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY absent")
    payload = {"model": a.model, "messages": [{"role": "user", "content": "Return {} only."}],
               "response_format": {"type": "json_object"}, "max_completion_tokens": 32,
               "reasoning_effort": "low"}
    req = urllib.request.Request(a.url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=45) as response:
        parsed = json.loads(response.read())
    usage = parsed.get("usage") or {}
    returned = str(parsed.get("model") or "")
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
           "requested_model": a.model, "returned_model": returned,
           "passed": returned == a.model, "latency_s": round(time.monotonic() - started, 3),
           "usage": usage, "cost_usd": cost(usage), "credential_source": "macOS Keychain environment injection",
           "credential_persisted": False, "response_content_persisted": False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(row, indent=2))
    print(json.dumps({k: row[k] for k in ("passed", "requested_model", "returned_model", "cost_usd")}))
    return 0 if row["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
