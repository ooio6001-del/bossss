import json, os
import pytest, requests
from solstrats.http import Client, TokenBucket, HttpError, redact, Limit

class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status; self._b = body if body is not None else {}; self.headers = headers or {}; self.text = "" if isinstance(self._b, Exception) else json.dumps(self._b)
    def json(self):
        if isinstance(self._b, Exception): raise self._b
        return self._b

class FakeSession:
    def __init__(self, seq):
        self.seq = list(seq); self.calls = 0; self.headers = {}
    def request(self, *a, **k):
        self.calls += 1
        r = self.seq.pop(0)
        if isinstance(r, Exception): raise r
        return r

def mk(seq, **kw):
    sleeps = []
    c = Client(session=FakeSession(seq), sleep=sleeps.append, limits={"p": Limit(6000, 10)}, **kw)
    return c, sleeps

def test_retry_on_429_then_success():
    c, sl = mk([FakeResp(429), FakeResp(503), FakeResp(200, {"ok": 1})])
    assert c.get("p", "http://x") == {"ok": 1}
    assert c.stats["429"] == 1 and c.stats["retries"] == 2 and len(sl) >= 2

def test_retry_after_header_respected():
    c, sl = mk([FakeResp(429, headers={"retry-after": "7"}), FakeResp(200, {"a": 1})])
    c.get("p", "http://x")
    assert any(7 <= s <= 7.6 for s in sl)

def test_retry_after_zero_still_backs_off():
    c, sl = mk([FakeResp(429, headers={"retry-after": "0"}), FakeResp(200, {"a": 1})])
    c.get("p", "http://x")
    assert sl and sl[0] >= c.base_delay / 2

def test_exhausted_raises():
    c, _ = mk([FakeResp(429)] * 4, max_retries=3)
    with pytest.raises(HttpError):
        c.get("p", "http://x")

def test_connection_error_retried():
    c, _ = mk([requests.ConnectionError("boom"), FakeResp(200, {"x": 2})])
    assert c.get("p", "http://x") == {"x": 2}

def test_4xx_not_retried():
    c, _ = mk([FakeResp(404, {"e": 1}), FakeResp(200)])
    with pytest.raises(HttpError) as ei:
        c.get("p", "http://x")
    assert ei.value.status == 404 and c.s.calls == 1

def test_invalid_json():
    c, _ = mk([FakeResp(200, ValueError("bad"))])
    with pytest.raises(HttpError):
        c.get("p", "http://x")

def test_backoff_is_jittered_and_capped():
    c = Client(max_delay=10, base_delay=1)
    vals = [c.backoff(a) for a in range(12)]
    assert all(0 < v <= 10 for v in vals)
    assert len(set(round(v, 6) for v in vals)) > 3

def test_cache(tmp_path):
    c, _ = mk([FakeResp(200, {"v": 1})], cache_dir=str(tmp_path))
    assert c.get("p", "http://x", cache_ttl=60) == {"v": 1}
    assert c.get("p", "http://x", cache_ttl=60) == {"v": 1}   # served from cache; session would raise if called
    assert c.stats["cache_hits"] == 1

def test_token_bucket_throttles():
    t = [0.0]
    def clock(): return t[0]
    def sleep(s): t[0] += s
    b = TokenBucket(60, burst=1, clock=clock, sleep=sleep)   # 1 req/s
    for _ in range(5): b.acquire()
    assert 3.9 <= t[0] <= 4.1

def test_token_bucket_penalty():
    t = [0.0]
    b = TokenBucket(6000, burst=5, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    b.penalize(10); b.acquire()
    assert t[0] >= 10

def test_redact(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "secret123")
    assert "secret123" not in redact("https://h/?api-key=secret123")
