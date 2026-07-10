"""Prior / marker knowledge base (§7.1).

A STRUCTURED store (SQLite + YAML seed files), not prose in a prompt. Holds cell-type
marker sets / gene signatures per tissue context and expected-composition priors per
tissue. The schema is extensible so the user adds signature sets over time.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from bioagent.models import CompositionPrior, SignatureSet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signature_sets (
    id TEXT PRIMARY KEY, name TEXT, tissue_context TEXT, source_ref TEXT,
    genes TEXT, direction TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS composition_priors (
    id TEXT PRIMARY KEY, tissue_type TEXT, expected_compartments TEXT, source_ref TEXT
);
"""


class PriorsKB:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- seeding ------------------------------------------------------------- #
    def seed(self, seeds_dir: Path) -> None:
        """Load only the tissue composition priors.

        There is NO pre-loaded gene-signature dictionary: the agent calls cell types de
        novo from DE markers + literature, and any signatures in the store are ones it
        discovered itself (id prefix `agent_`). We purge any curated/pre-loaded signatures
        so that invariant holds even on an existing store.
        """
        self.conn.execute("DELETE FROM signature_sets WHERE id NOT LIKE 'agent_%'")
        self.conn.commit()
        comp_file = seeds_dir / "composition_priors.yaml"
        if comp_file.exists():
            for c in yaml.safe_load(comp_file.read_text())["composition_priors"]:
                self.add_composition_prior(CompositionPrior(**c))

    # -- writes -------------------------------------------------------------- #
    def add_signature(self, s: SignatureSet) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO signature_sets VALUES (?,?,?,?,?,?,?)",
            (s.id, s.name, s.tissue_context, s.source_ref, "\t".join(s.genes), s.direction, s.notes),
        )
        self.conn.commit()

    def add_composition_prior(self, c: CompositionPrior) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO composition_priors VALUES (?,?,?,?)",
            (c.id, c.tissue_type, "\t".join(c.expected_compartments), c.source_ref),
        )
        self.conn.commit()

    # -- reads --------------------------------------------------------------- #
    def signatures(self, tissue_context: str | None = None) -> list[SignatureSet]:
        q = "SELECT id,name,tissue_context,source_ref,genes,direction,notes FROM signature_sets"
        rows = self.conn.execute(q).fetchall()
        out = [
            SignatureSet(
                id=r[0], name=r[1], tissue_context=r[2], source_ref=r[3],
                genes=r[4].split("\t") if r[4] else [], direction=r[5], notes=r[6],
            )
            for r in rows
        ]
        if tissue_context:
            out = [s for s in out if s.tissue_context in (tissue_context, "any")]
        return out

    def composition_prior(self, tissue_type: str) -> CompositionPrior | None:
        r = self.conn.execute(
            "SELECT id,tissue_type,expected_compartments,source_ref FROM composition_priors "
            "WHERE tissue_type = ?",
            (tissue_type,),
        ).fetchone()
        if not r:
            return None
        return CompositionPrior(
            id=r[0], tissue_type=r[1],
            expected_compartments=r[2].split("\t") if r[2] else [], source_ref=r[3],
        )

    def all_tissue_types(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT tissue_type FROM composition_priors").fetchall()]

    def close(self) -> None:
        self.conn.close()
