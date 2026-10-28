"""OpenRouter chat client (DEVELOPMENT §5.2): disk cache, hard budget cap, usage accounting. Stdlib only.

The key comes from the git-ignored .env (OPENROUTER_API_KEY=...) or the environment and is never printed,
logged or cached. Cache key = sha256(model, provider, messages, schema, temperature, seed, max_tokens, reasoning).
A hit spends nothing but reports the original call's tokens, seconds and cost (as cached_usd), so a run
replayed from the cache gives the same numbers (review G_M4). ARBITER_MAX_USD (default 2.00) caps the spend
of one LLM object, which threads may share: a call is refused (BudgetExceeded) when the spend so far plus
the calls in flight and itself, each priced at the dearest call seen, would pass the cap. Before the first
call returns that price is unknown, so up to one call per thread may start.
Reasoning models (the default Qwen3.6 thinks before answering) spend completion tokens on reasoning; pass
`reasoning` (OpenRouter's {"effort": "low|medium|high"} or {"enabled": false}) and leave room in max_tokens.
An answer cut off before any content returns "" (counted as `empty`), never None.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
PROVIDER = {"order": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True}


USAGE = ("calls", "hits", "misses", "prompt_tokens", "completion_tokens", "reasoning_tokens", "images", "empty")


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
                 max_usd: float | None = None, transport=post, provider: dict | None = PROVIDER,
                 reasoning: dict | None = None):
        env = {**load_env(), **os.environ}
        self.key = env.get("OPENROUTER_API_KEY", "")
        self.max_usd = float(max_usd if max_usd is not None else env.get("ARBITER_MAX_USD", 2.0))
        self.model, self.cache_dir, self.transport, self.provider = model, Path(cache_dir), transport, provider
        self.reasoning = reasoning
        self.usage = dict.fromkeys(USAGE, 0) | {"usd": 0.0, "cached_usd": 0.0, "seconds": 0.0}
        self._lock, self._inflight, self._price = threading.Lock(), 0, 0.0

    def body(self, messages: list, schema: dict | None, temperature: float, seed: int, max_tokens: int,
             reasoning: dict | None = None) -> dict:
        b = {"model": self.model, "messages": messages, "temperature": temperature, "seed": seed,
             "max_tokens": max_tokens, "usage": {"include": True}}
        if self.provider:
            b["provider"] = self.provider
        if reasoning or self.reasoning:
            b["reasoning"] = reasoning or self.reasoning
        if schema:
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "strict": True, "schema": schema}}
        return b

    def chat(self, messages: list, schema: dict | None = None, temperature: float = 0.0, seed: int = 0,
             max_tokens: int = 600, reasoning: dict | None = None) -> str:
        """The assistant's text. Cached; counted; refuses a call the budget cannot cover."""
        return self.call(messages, schema, temperature, seed, max_tokens, reasoning)[0]

    def call(self, messages: list, schema: dict | None = None, temperature: float = 0.0, seed: int = 0,
             max_tokens: int = 600, reasoning: dict | None = None) -> tuple[str, dict]:
        """(text, this call's usage). Thread-safe; `reasoning` overrides the client's for this call."""
        body = self.body(messages, schema, temperature, seed, max_tokens, reasoning)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            u = self._usage(cached["response"], cached.get("seconds", 0.0), hit=True, images=count_images(messages))
            return self._add(u, cached["response"])
        with self._lock:
            if self.usage["usd"] >= self.max_usd or \
                    self.usage["usd"] + (self._inflight + 1) * self._price > self.max_usd:
                raise BudgetExceeded(f"spent ${self.usage['usd']:.4f}, {self._inflight} calls in flight at up to "
                                     f"${self._price:.4f}: cap ${self.max_usd:.2f} (ARBITER_MAX_USD)")
            self._inflight += 1
        try:
            if not self.key:
                raise RuntimeError("OPENROUTER_API_KEY missing: put it in the git-ignored .env")
            t0 = time.time()
            resp = self.transport(body, self.key)
            dt = time.time() - t0
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request": body, "response": resp, "seconds": dt}))
        finally:
            with self._lock:
                self._inflight -= 1
        return self._add(self._usage(resp, dt, hit=False, images=count_images(messages)), resp)

    @staticmethod
    def _usage(resp: dict, seconds: float, hit: bool, images: int) -> dict:
        """One call's usage; tokens, seconds and cost are the original call's even on a hit."""
        u = resp.get("usage") or {}
        cost = float(u.get("cost", 0.0))
        return {"calls": 1, "hits": int(hit), "misses": int(not hit), "images": images,
                "prompt_tokens": int(u.get("prompt_tokens", 0)), "completion_tokens": int(u.get("completion_tokens", 0)),
                "reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                "usd": 0.0 if hit else cost, "cached_usd": cost if hit else 0.0, "seconds": float(seconds),
                "empty": int(not resp["choices"][0]["message"].get("content"))}

    def _add(self, u: dict, resp: dict) -> tuple[str, dict]:
        with self._lock:
            for k, v in u.items():
                self.usage[k] += v
            if not u["hits"]:
                self._price = max(self._price, u["usd"])
        return resp["choices"][0]["message"].get("content") or "", u


def count_images(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")
