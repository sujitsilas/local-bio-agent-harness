"""Knowledge-layer plugin base (§6.2, §13.1).

The ONE runtime egress. Every request is checked against the knowledge-source
allowlist and cached to SQLite. Callers send minimal queries (gene symbols, signature
names, search terms) — never expression matrices or raw data.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

from bioagent.config import Config

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_cache (
    url TEXT PRIMARY KEY, body TEXT, fetched_at REAL
);
"""


class NetworkPolicyError(RuntimeError):
    pass


class KnowledgeClient:
    """Shared, allowlisted, cached HTTP client for all knowledge plugins."""

    def __init__(self, config: Config, ttl_s: float = 7 * 24 * 3600):
        self.config = config
        self.ttl_s = ttl_s
        allow = yaml.safe_load(config.path("network_allowlist").read_text())
        self._hosts = set(allow.get("knowledge", {}).get("hosts", []))
        self._client = httpx.Client(timeout=30.0, headers={"User-Agent": "bioagent/0.1"})
        # NCBI allows 3 req/s anonymously, 10 with an API key. Space requests to stay under.
        self._api_key = os.getenv("NCBI_API_KEY")
        self._min_interval = 0.12 if self._api_key else 0.35
        self._last_request = 0.0
        self._rate_lock = threading.Lock()
        db = config.path("state_db")
        db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db), check_same_thread=False)
        self._conn.execute(_CACHE_SCHEMA)
        self._conn.commit()

    def _check_host(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if host not in self._hosts:
            raise NetworkPolicyError(
                f"host {host!r} not in knowledge allowlist (§13.1); "
                f"allowed: {sorted(self._hosts)}"
            )

    def _throttle(self) -> None:
        with self._rate_lock:
            wait = self._min_interval - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

    def get_json(self, url: str, params: dict | None = None) -> dict | list:
        params = dict(params or {})
        if self._api_key and "ncbi.nlm.nih.gov" in url:
            params["api_key"] = self._api_key
        full = str(httpx.URL(url, params=params))
        self._check_host(full)
        cache_key = full.split("&api_key=")[0]  # don't leak the key into the cache key
        cached = self._conn.execute(
            "SELECT body, fetched_at FROM knowledge_cache WHERE url = ?", (cache_key,)
        ).fetchone()
        if cached and (time.time() - cached[1]) < self.ttl_s:
            return json.loads(cached[0])

        body = self._get_with_retry(full)
        self._conn.execute(
            "INSERT OR REPLACE INTO knowledge_cache(url, body, fetched_at) VALUES (?,?,?)",
            (cache_key, body, time.time()),
        )
        self._conn.commit()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"_text": body}

    def _get_with_retry(self, url: str, tries: int = 4) -> str:
        """Rate-limited GET with exponential backoff on 429/5xx (transient rate limits)."""
        for attempt in range(tries):
            self._throttle()
            resp = self._client.get(url)
            if resp.status_code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s
                continue
            resp.raise_for_status()
            return resp.text
        resp.raise_for_status()  # exhausted retries
        return resp.text
