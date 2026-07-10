"""UniProt knowledge plugin (§6.2). Gene/protein function lookup. Symbols only."""
from __future__ import annotations

from bioagent.tools.apis.base import KnowledgeClient

_REST = "https://rest.uniprot.org"


class UniProtPlugin:
    name = "uniprot"
    description = "Look up protein function by gene symbol (UniProt); sends gene symbols only."

    def __init__(self, client: KnowledgeClient):
        self.client = client

    def function(self, gene: str, organism_id: int = 9606) -> dict:
        data = self.client.get_json(
            f"{_REST}/uniprotkb/search",
            {
                "query": f"gene:{gene} AND organism_id:{organism_id} AND reviewed:true",
                "fields": "accession,protein_name,cc_function",
                "format": "json",
                "size": 1,
            },
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            return {"gene": gene, "found": False}
        r = results[0]
        comments = r.get("comments", [])
        func = next(
            (
                c["texts"][0]["value"]
                for c in comments
                if c.get("commentType") == "FUNCTION" and c.get("texts")
            ),
            "",
        )
        return {
            "gene": gene,
            "found": True,
            "accession": r.get("primaryAccession", ""),
            "function": func,
            "citation": f"UniProt:{r.get('primaryAccession', '')}",
        }
