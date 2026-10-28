"""OpenRouter chat client (DEVELOPMENT §5.2): disk cache, hard budget cap, usage accounting. Stdlib only.

The key comes from the git-ignored .env (OPENROUTER_API_KEY=...) or the environment and is never printed,
logged or cached. Cache key = sha256(model, provider, messages, schema, temperature, seed, max_tokens);
a hit costs 0 and is counted. ARBITER_MAX_USD (default 2.00) caps the spend of one LLM object: the call
that would start above the cap raises BudgetExceeded instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
PROVIDER = {"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True}


class BudgetExceeded(RuntimeError):
    pass


def load_env(path: Path = ROOT / ".env") -> dict:
    """KEY=VALUE lines; blank lines and # comments ignored; quotes stripped."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def post(body: dict, key: str, timeout: float = 60.0) -> dict:
    """One HTTPS request; 2 retries with backoff on 429 / 5xx / network errors."""
    data = json.dumps(body).encode()
    for attempt in range(3):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == 2:
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 * 2 ** attempt)
    raise AssertionError("unreachable")


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = ROOT / "results" / "llm_cache",
                 max_usd: float | None = None, transport=post, provider: dict | None = PROVIDER):
        env = {**load_env(), **os.environ}
        self.key = env.get("OPENROUTER_API_KEY", "")
        self.max_usd = float(max_usd if max_usd is not None else env.get("ARBITER_MAX_USD", 2.0))
        self.model, self.cache_dir, self.transport, self.provider = model, Path(cache_dir), transport, provider
        self.usage = {"calls": 0, "hits": 0, "misses": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "images": 0, "usd": 0.0, "seconds": 0.0}

    def body(self, messages: list, schema: dict | None, temperature: float, seed: int, max_tokens: int) -> dict:
        b = {"model": self.model, "messages": messages, "temperature": temperature, "seed": seed,
             "max_tokens": max_tokens, "usage": {"include": True}}
        if self.provider:
            b["provider"] = self.provider
        if schema:
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "strict": True, "schema": schema}}
        return b

    def chat(self, messages: list, schema: dict | None = None, temperature: float = 0.0, seed: int = 0,
             max_tokens: int = 600) -> str:
        """The assistant's text. Cached; counted; refuses to start a call once the budget is spent."""
        body = self.body(messages, schema, temperature, seed, max_tokens)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        self.usage["calls"] += 1
        self.usage["images"] += count_images(messages)
        if path.exists():
            self.usage["hits"] += 1
            return json.loads(path.read_text())["response"]["choices"][0]["message"]["content"]
        if self.usage["usd"] >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.usage['usd']:.4f} >= cap ${self.max_usd:.2f} (ARBITER_MAX_USD)")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY missing: put it in the git-ignored .env")
        t0 = time.time()
        resp = self.transport(body, self.key)
        u = resp.get("usage") or {}
        self.usage.update(misses=self.usage["misses"] + 1, seconds=self.usage["seconds"] + time.time() - t0,
                          prompt_tokens=self.usage["prompt_tokens"] + int(u.get("prompt_tokens", 0)),
                          completion_tokens=self.usage["completion_tokens"] + int(u.get("completion_tokens", 0)),
                          usd=self.usage["usd"] + float(u.get("cost", 0.0)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": body, "response": resp}))
        return resp["choices"][0]["message"]["content"]


def count_images(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")
