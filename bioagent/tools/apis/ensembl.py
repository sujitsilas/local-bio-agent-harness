"""Ensembl knowledge plugin (§6.2). Gene symbol -> id / cross-references. Symbols only."""
from __future__ import annotations

from bioagent.tools.apis.base import KnowledgeClient

_REST = "https://rest.ensembl.org"


class EnsemblPlugin:
    name = "ensembl"
    description = "Look up gene metadata by symbol (Ensembl); sends gene symbols only."

    def __init__(self, client: KnowledgeClient):
        self.client = client

    def lookup_symbol(self, symbol: str, species: str = "homo_sapiens") -> dict:
        data = self.client.get_json(
            f"{_REST}/lookup/symbol/{species}/{symbol}",
            {"content-type": "application/json"},
        )
        if not isinstance(data, dict):
            return {}
        return {
            "symbol": symbol,
            "id": data.get("id", ""),
            "description": data.get("description", ""),
            "biotype": data.get("biotype", ""),
        }
