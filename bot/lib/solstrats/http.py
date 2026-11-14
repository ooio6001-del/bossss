"""Throttled, cached HTTP client with per-provider rate limits, exponential backoff + jitter.

Keys are read from env vars only and are never logged (see `redact`)."""
from __future__ import annotations
import hashlib, json, logging, os, random, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import requests

log = logging.getLogger("solstrats.http")

@dataclass
class Limit:
    per_min: float          # sustained requests per minute
    burst: int = 1

class TokenBucket:
    def __init__(self, per_min: float, burst: int = 1, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate = per_min / 60.0
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.clock, self.sleep = clock, sleep
        self.t = clock()
        self.lock = threading.Lock()
        self.penalty_until = 0.0

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now

    def acquire(self):
        while True:
            with self.lock:
                self._refill()
                now = self.clock()
                if now < self.penalty_until:
                    wait = self.penalty_until - now
                elif self.tokens >= 1:
                    self.tokens -= 1
                    return
                else:
                    wait = (1 - self.tokens) / self.rate
            self.sleep(max(wait, 0.001))

    def penalize(self, seconds: float):
        with self.lock:
            self.penalty_until = max(self.penalty_until, self.clock() + seconds)

# Conservative defaults (below published free-tier limits)
LIMITS = {
    "dexscreener_fast": Limit(240, 5),   # /latest/dex/*, /token-pairs: 300/min published
    "dexscreener_slow": Limit(50, 2),    # /token-profiles, /token-boosts: 60/min published
    "geckoterminal": Limit(6, 1),        # 30/min published, but measured ~7-10/min sustained from this box; stay under it
    "birdeye": Limit(50, 2),             # free tier ~1 rps; keep low
    "helius": Limit(240, 3),             # free plan: RPC/DAS 5 req/s, Enhanced 1 req/s
    "solana_rpc": Limit(80, 3),          # public mainnet RPC
}

class HttpError(Exception):
    def __init__(self, status: int, msg: str = ""):
        super().__init__(f"HTTP {status} {msg}")
        self.status = status

def redact(s: str) -> str:
    for k in ("BIRDEYE_API_KEY", "HELIUS_API_KEY"):
        v = os.environ.get(k)
        if v and v in s:
            s = s.replace(v, "***")
    return s

class Client:
    RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}

    def __init__(self, cache_dir: Optional[str] = None, max_retries: int = 6, base_delay: float = 2.0,
                 max_delay: float = 60.0, timeout: float = 20.0, session: Optional[requests.Session] = None,
                 sleep: Callable[[float], None] = time.sleep, limits: Optional[dict] = None):
        self.s = session or requests.Session()
        self.s.headers.setdefault("accept", "application/json")
        self.s.headers.setdefault("user-agent", "solstrats/1.0 (research)")
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.max_retries, self.base_delay, self.max_delay, self.timeout = max_retries, base_delay, max_delay, timeout
        self.sleep = sleep
        lim = limits or LIMITS
        self.buckets = {k: TokenBucket(v.per_min, v.burst, sleep=sleep) for k, v in lim.items()}
        self.stats = {"requests": 0, "retries": 0, "429": 0, "errors": 0, "cache_hits": 0}

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, hashlib.sha1(key.encode()).hexdigest() + ".json")

    def backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        d = min(self.max_delay, self.base_delay * (2 ** attempt))
        jit = random.uniform(d / 2, d)   # "equal jitter"
        if retry_after is not None:      # never retry sooner than the server asked, nor sooner than our own backoff
            return min(self.max_delay, max(retry_after, jit)) + random.uniform(0, 0.5)
        return jit

    def request(self, provider: str, method: str, url: str, *, params=None, json_body=None, headers=None,
                cache_ttl: float = 0, cache_key_extra: str = "") -> Any:
        key = f"{method} {url} {json.dumps(params, sort_keys=True)} {json.dumps(json_body, sort_keys=True)} {cache_key_extra}"
        cp = self._cache_path(key) if cache_ttl > 0 else None
        if cp and os.path.exists(cp) and (time.time() - os.path.getmtime(cp)) < cache_ttl:
            try:
                with open(cp) as f:
                    self.stats["cache_hits"] += 1
                    return json.load(f)
            except (OSError, ValueError):
                pass
        bucket = self.buckets.get(provider)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if bucket:
                bucket.acquire()
            self.stats["requests"] += 1
            try:
                r = self.s.request(method, url, params=params, json=json_body, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                self.stats["retries"] += 1
                self.sleep(self.backoff(attempt))
                continue
            if r.status_code in self.RETRY_STATUS:
                if r.status_code == 429:
                    self.stats["429"] += 1
                ra = r.headers.get("retry-after")
                try:
                    ra_f = float(ra) if ra is not None else None
                except ValueError:
                    ra_f = None
                wait = self.backoff(attempt, ra_f)
                if bucket and r.status_code == 429:
                    bucket.penalize(wait)
                self.stats["retries"] += 1
                last_exc = HttpError(r.status_code)
                log.warning("%s %s -> %s, retry in %.1fs", provider, redact(url), r.status_code, wait)
                self.sleep(wait)
                continue
            if r.status_code >= 400:
                self.stats["errors"] += 1
                raise HttpError(r.status_code, redact(r.text[:200]))
            try:
                data = r.json()
            except ValueError as e:
                self.stats["errors"] += 1
                raise HttpError(r.status_code, "invalid json") from e
            if cp:
                try:
                    tmp = cp + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(data, f)
                    os.replace(tmp, cp)
                except OSError:
                    pass
            return data
        self.stats["errors"] += 1
        raise last_exc if last_exc else HttpError(0, "exhausted")

    def get(self, provider: str, url: str, **kw):
        return self.request(provider, "GET", url, **kw)

    def post(self, provider: str, url: str, **kw):
        return self.request(provider, "POST", url, **kw)
