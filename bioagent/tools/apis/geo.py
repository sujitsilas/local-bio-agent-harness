"""GEO knowledge plugin (§6.2). Search dataset metadata by accession/term. Terms only."""
from __future__ import annotations

from bioagent.tools.apis.base import KnowledgeClient

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class GEOPlugin:
    name = "geo"
    description = "Search GEO (Gene Expression Omnibus) dataset metadata by term or accession."

    def __init__(self, client: KnowledgeClient):
        self.client = client

    def search(self, term: str, *, retmax: int = 5) -> list[dict]:
        ids = self.client.get_json(
            f"{_EUTILS}/esearch.fcgi",
            {"db": "gds", "term": term, "retmax": retmax, "retmode": "json"},
        )
        uids = ids.get("esearchresult", {}).get("idlist", []) if isinstance(ids, dict) else []
        if not uids:
            return []
        summ = self.client.get_json(
            f"{_EUTILS}/esummary.fcgi", {"db": "gds", "id": ",".join(uids), "retmode": "json"}
        )
        result = summ.get("result", {}) if isinstance(summ, dict) else {}
        return [
            {
                "accession": result[u].get("accession", ""),
                "title": result[u].get("title", ""),
                "n_samples": result[u].get("n_samples", ""),
                "taxon": result[u].get("taxon", ""),
            }
            for u in uids
            if u in result
        ]
