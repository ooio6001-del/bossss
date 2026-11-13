"""Data providers with graceful fallback. DexScreener + GeckoTerminal need no key.
Birdeye (BIRDEYE_API_KEY) and Helius (HELIUS_API_KEY) are used only when the env var is set.
Keys stay server-side: they are only placed in request headers / URLs and never logged or exported."""
from __future__ import annotations
import logging, os, time
from typing import Any, Dict, List, Optional
from .http import Client, HttpError

log = logging.getLogger("solstrats.providers")
DS = "https://api.dexscreener.com"
GT = "https://api.geckoterminal.com/api/v2"
BE = "https://public-api.birdeye.so"
PUBLIC_RPC = "https://api.mainnet-beta.solana.com"

def _f(x, default=None):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default

def norm_ds_pair(p: dict) -> dict:
    """Normalize a DexScreener pair into our live-token schema."""
    return {
        "mint": (p.get("baseToken") or {}).get("address"),
        "symbol": (p.get("baseToken") or {}).get("symbol"),
        "name": (p.get("baseToken") or {}).get("name"),
        "pool": p.get("pairAddress"), "dex": p.get("dexId"),
        "quote": (p.get("quoteToken") or {}).get("symbol"),
        "price": _f(p.get("priceUsd")),
        "mcap": _f(p.get("marketCap"), _f(p.get("fdv"))),
        "liq": _f((p.get("liquidity") or {}).get("usd"), 0.0),
        "vol_m5": _f((p.get("volume") or {}).get("m5"), 0.0), "vol_h1": _f((p.get("volume") or {}).get("h1"), 0.0),
        "vol_h24": _f((p.get("volume") or {}).get("h24"), 0.0),
        "chg_m5": _f((p.get("priceChange") or {}).get("m5"), 0.0), "chg_h1": _f((p.get("priceChange") or {}).get("h1"), 0.0),
        "chg_h6": _f((p.get("priceChange") or {}).get("h6"), 0.0), "chg_h24": _f((p.get("priceChange") or {}).get("h24"), 0.0),
        "buys_m5": ((p.get("txns") or {}).get("m5") or {}).get("buys", 0), "sells_m5": ((p.get("txns") or {}).get("m5") or {}).get("sells", 0),
        "buys_h1": ((p.get("txns") or {}).get("h1") or {}).get("buys", 0), "sells_h1": ((p.get("txns") or {}).get("h1") or {}).get("sells", 0),
        "created_ms": p.get("pairCreatedAt"), "url": p.get("url"), "source": "dexscreener",
    }

class Providers:
    def __init__(self, client: Optional[Client] = None, env: Optional[Dict[str, str]] = None):
        self.c = client or Client(cache_dir=None)
        env = env if env is not None else os.environ
        self.birdeye_key = (env.get("BIRDEYE_API_KEY") or "").strip() or None
        self.helius_key = (env.get("HELIUS_API_KEY") or "").strip() or None
        self.failures: Dict[str, int] = {}

    def status(self) -> dict:
        return {"dexscreener": True, "geckoterminal": True, "birdeye": bool(self.birdeye_key),
                "helius": bool(self.helius_key), "public_rpc": True, "failures": dict(self.failures)}

    def _fail(self, name: str, e: Exception):
        self.failures[name] = self.failures.get(name, 0) + 1
        log.warning("provider %s failed: %s", name, type(e).__name__)

    # ---------- discovery ----------
    def new_pairs(self, limit: int = 60) -> List[dict]:
        """Newest Solana pairs. GeckoTerminal new_pools -> DexScreener profiles -> Birdeye new listings."""
        out: Dict[str, dict] = {}
        try:
            d = self.c.get("geckoterminal", f"{GT}/networks/solana/new_pools", params={"page": 1, "include": "base_token"})
            toks = {t["id"]: t["attributes"] for t in d.get("included", [])}
            for p in d.get("data", []):
                a = p["attributes"]; bt = p["relationships"]["base_token"]["data"]["id"]; t = toks.get(bt, {})
                m = t.get("address")
                if not m: continue
                out[m] = {"mint": m, "symbol": t.get("symbol"), "pool": a.get("address"),
                          "dex": p["relationships"].get("dex", {}).get("data", {}).get("id"),
                          "price": _f(a.get("base_token_price_usd")), "mcap": _f(a.get("market_cap_usd"), _f(a.get("fdv_usd"))),
                          "liq": _f(a.get("reserve_in_usd"), 0.0), "vol_h1": _f((a.get("volume_usd") or {}).get("h1"), 0.0),
                          "vol_m5": _f((a.get("volume_usd") or {}).get("m5"), 0.0),
                          "chg_m5": _f((a.get("price_change_percentage") or {}).get("m5"), 0.0),
                          "chg_h1": _f((a.get("price_change_percentage") or {}).get("h1"), 0.0),
                          "buys_h1": ((a.get("transactions") or {}).get("h1") or {}).get("buys", 0),
                          "sells_h1": ((a.get("transactions") or {}).get("h1") or {}).get("sells", 0),
                          "created": a.get("pool_created_at"), "source": "geckoterminal"}
        except Exception as e:  # noqa: BLE001 - provider failure must not kill the loop
            self._fail("geckoterminal", e)
        try:
            prof = self.c.get("dexscreener_slow", f"{DS}/token-profiles/latest/v1")
            mints = [p["tokenAddress"] for p in prof if p.get("chainId") == "solana" and p.get("tokenAddress")][:30]
            for t in self.tokens(mints):
                out.setdefault(t["mint"], t)
        except Exception as e:  # noqa: BLE001
            self._fail("dexscreener", e)
        if self.birdeye_key:
            try:
                d = self.c.get("birdeye", f"{BE}/defi/v2/tokens/new_listing", params={"limit": 20},
                               headers={"X-API-KEY": self.birdeye_key, "x-chain": "solana"})
                for t in (d.get("data") or {}).get("items", []):
                    m = t.get("address")
                    if m and m not in out:
                        out[m] = {"mint": m, "symbol": t.get("symbol"), "liq": _f(t.get("liquidity"), 0.0),
                                  "created": t.get("liquidityAddedAt"), "source": "birdeye"}
            except Exception as e:  # noqa: BLE001
                self._fail("birdeye", e)
        return list(out.values())[:limit]

    def tokens(self, mints: List[str]) -> List[dict]:
        """Best (highest-liquidity) Solana pair per mint via DexScreener, in batches of 30."""
        res = []
        for i in range(0, len(mints), 30):
            chunk = [m for m in mints[i:i + 30] if m]
            if not chunk: continue
            try:
                d = self.c.get("dexscreener_fast", f"{DS}/tokens/v1/solana/{','.join(chunk)}")
            except Exception as e:  # noqa: BLE001
                self._fail("dexscreener", e); continue
            best: Dict[str, dict] = {}
            for p in d if isinstance(d, list) else (d.get("pairs") or []):
                if p.get("chainId") != "solana": continue
                n = norm_ds_pair(p)
                if n["mint"] in chunk and (n["mint"] not in best or n["liq"] > best[n["mint"]]["liq"]):
                    best[n["mint"]] = n
            res.extend(best.values())
        return res

    def prices(self, mints: List[str]) -> Dict[str, float]:
        """USD prices. DexScreener first; Birdeye multi_price as fallback for any missing."""
        out = {t["mint"]: t["price"] for t in self.tokens(mints) if t.get("price")}
        missing = [m for m in mints if m not in out]
        if missing and self.birdeye_key:
            try:
                d = self.c.get("birdeye", f"{BE}/defi/multi_price", params={"list_address": ",".join(missing[:100])},
                               headers={"X-API-KEY": self.birdeye_key, "x-chain": "solana"})
                for m, v in (d.get("data") or {}).items():
                    if v and v.get("value"): out[m] = float(v["value"])
            except Exception as e:  # noqa: BLE001
                self._fail("birdeye", e)
        return out

    def ohlcv_hour(self, pool: str, limit: int = 200) -> List[list]:
        """Recent hourly candles from GeckoTerminal: [ts, o, h, l, c, vol_usd] ascending."""
        d = self.c.get("geckoterminal", f"{GT}/networks/solana/pools/{pool}/ohlcv/hour",
                       params={"aggregate": 1, "limit": limit, "currency": "usd", "token": "base"}, cache_ttl=600)
        return sorted(d["data"]["attributes"]["ohlcv_list"])

    # ---------- on-chain safety checks ----------
    def _rpc(self, method: str, params: list) -> Any:
        urls = []
        if self.helius_key:
            urls.append(("helius", f"https://mainnet.helius-rpc.com/?api-key={self.helius_key}"))
        urls.append(("solana_rpc", PUBLIC_RPC))
        last = None
        for prov, url in urls:
            try:
                d = self.c.post(prov, url, json_body={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                cache_ttl=3600, cache_key_extra=prov)
                if "error" in d:
                    raise HttpError(int((d["error"] or {}).get("code", 0)) if isinstance(d["error"], dict) else 0, "rpc error")
                return d.get("result")
            except Exception as e:  # noqa: BLE001
                self._fail(prov, e); last = e
        raise last if last else HttpError(0, "no rpc")

    def mint_safety(self, mint: str) -> dict:
        """mint/freeze authority + top-holder concentration (raw, includes LP vaults)."""
        rec: Dict[str, Any] = {"mint": mint}
        try:
            ai = self._rpc("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
            info = ai["value"]["data"]["parsed"]["info"]
            rec.update(mint_auth=info.get("mintAuthority") is not None, freeze_auth=info.get("freezeAuthority") is not None,
                       supply=int(info["supply"]) / 10 ** int(info["decimals"]))
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"account:{type(e).__name__}"
            return rec
        try:
            la = self._rpc("getTokenLargestAccounts", [mint])
            amts = sorted((float(a.get("uiAmount") or 0) for a in la["value"]), reverse=True)
            # drop the single largest (typically the AMM vault / bonding curve) for an LP-adjusted figure
            sup = rec["supply"] or 1
            rec["top10_raw"] = sum(amts[:10]) / sup
            rec["top10_ex_lp"] = sum(amts[1:11]) / sup
        except Exception as e:  # noqa: BLE001
            rec["holders_error"] = type(e).__name__
        return rec
