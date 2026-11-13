from solstrats.providers import Providers, norm_ds_pair
from solstrats.http import HttpError

class StubClient:
    def __init__(self, routes):
        self.routes = routes; self.stats = {}; self.calls = []
    def get(self, provider, url, **kw):
        self.calls.append((provider, url, kw))
        for k, v in self.routes.items():
            if k in url:
                if isinstance(v, Exception): raise v
                return v(url, kw) if callable(v) else v
        raise HttpError(404)
    def post(self, provider, url, **kw):
        self.calls.append((provider, url, kw))
        r = self.routes.get(provider)
        if isinstance(r, Exception): raise r
        return r

PAIR = {"chainId": "solana", "dexId": "raydium", "pairAddress": "P1", "baseToken": {"address": "M1", "symbol": "AAA"},
        "quoteToken": {"symbol": "SOL"}, "priceUsd": "0.5", "marketCap": 12345, "liquidity": {"usd": 9000},
        "volume": {"h1": 100}, "priceChange": {"h1": 5}, "txns": {"h1": {"buys": 3, "sells": 1}}, "pairCreatedAt": 1}

def test_norm_pair_handles_missing_fields():
    n = norm_ds_pair({"baseToken": {"address": "x"}})
    assert n["mint"] == "x" and n["liq"] == 0.0 and n["price"] is None

def test_new_pairs_falls_back_when_gecko_fails():
    c = StubClient({"geckoterminal": HttpError(500), "token-profiles": [{"chainId": "solana", "tokenAddress": "M1"}], "tokens/v1": [PAIR]})
    pv = Providers(c, env={})
    out = pv.new_pairs()
    assert out and out[0]["mint"] == "M1" and pv.failures.get("geckoterminal") == 1

def test_birdeye_skipped_without_key():
    c = StubClient({"tokens/v1": [PAIR]})
    pv = Providers(c, env={})
    pv.prices(["M1", "M2"])
    assert not any(p == "birdeye" for p, *_ in c.calls)

def test_birdeye_used_with_key_and_key_only_in_header():
    c = StubClient({"tokens/v1": [], "multi_price": {"data": {"M2": {"value": 2.0}}}})
    pv = Providers(c, env={"BIRDEYE_API_KEY": " k-123 "})
    out = pv.prices(["M2"])
    assert out == {"M2": 2.0}
    be = [x for x in c.calls if x[0] == "birdeye"][0]
    assert "k-123" not in be[1] and be[2]["headers"]["X-API-KEY"] == "k-123"

def test_rpc_falls_back_from_helius_to_public():
    acct = {"value": {"data": {"parsed": {"info": {"mintAuthority": None, "freezeAuthority": None, "supply": "1000", "decimals": 0}}}}}
    c = StubClient({"helius": HttpError(503), "solana_rpc": {"result": acct}})
    pv = Providers(c, env={"HELIUS_API_KEY": "h"})
    r = pv.mint_safety("M1")
    assert r["mint_auth"] is False and r["supply"] == 1000 and pv.failures["helius"] >= 1

def test_status_never_contains_keys():
    pv = Providers(StubClient({}), env={"HELIUS_API_KEY": "hk", "BIRDEYE_API_KEY": "bk"})
    s = str(pv.status())
    assert "hk" not in s and "bk" not in s
