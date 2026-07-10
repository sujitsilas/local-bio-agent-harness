"""PubMed knowledge plugin (§6.2, §8.3). Minimal queries only — search terms, never data."""
from __future__ import annotations

from bioagent.tools.apis.base import KnowledgeClient

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedPlugin:
    name = "pubmed"
    description = "Search PubMed for literature grounding implications; returns PMIDs + summaries."

    def __init__(self, client: KnowledgeClient):
        self.client = client

    def search(self, term: str, *, retmax: int = 5) -> list[dict]:
        ids = self.client.get_json(
            f"{_EUTILS}/esearch.fcgi",
            {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
        )
        pmids = ids.get("esearchresult", {}).get("idlist", []) if isinstance(ids, dict) else []
        if not pmids:
            return []
        summ = self.client.get_json(
            f"{_EUTILS}/esummary.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        )
        result = summ.get("result", {}) if isinstance(summ, dict) else {}
        return [
            {
                "pmid": pid,
                "title": result[pid].get("title", ""),
                "journal": result[pid].get("fulljournalname", ""),
                "year": result[pid].get("pubdate", "")[:4],
                "citation": f"PMID:{pid}",
            }
            for pid in pmids
            if pid in result
        ]
