"""The KB holds NO pre-loaded gene-signature dictionary — the agent calls cell types de
novo. Only tissue composition priors are seeded, and curated signatures are purged."""
from __future__ import annotations

from pathlib import Path

from bioagent.bio.priors_kb import PriorsKB
from bioagent.models import SignatureSet

SEEDS = Path(__file__).resolve().parent.parent / "bioagent" / "seeds"


def test_seed_loads_no_predefined_gene_signatures(tmp_path):
    kb = PriorsKB(tmp_path / "kb.sqlite")
    kb.seed(SEEDS)
    assert kb.signatures() == []                       # no pre-loaded dictionary
    assert "solid_tumor" in kb.all_tissue_types()      # composition priors still seeded


def test_seed_purges_curated_keeps_agent_discovered(tmp_path):
    kb = PriorsKB(tmp_path / "kb.sqlite")
    kb.add_signature(SignatureSet(id="curated_x", name="Curated X", tissue_context="any",
                                  genes=["A", "B"]))
    kb.add_signature(SignatureSet(id="agent_y", name="Agent Y", tissue_context="tissue",
                                  genes=["C", "D"]))
    kb.seed(SEEDS)
    assert {s.id for s in kb.signatures()} == {"agent_y"}
