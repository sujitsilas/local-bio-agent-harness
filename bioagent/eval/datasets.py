"""Benchmark datasets with trusted, held-out cell-type labels.

Public datasets are fetched via scanpy inside the Run kernel (they download from scanpy's
CDN — provisioning egress). The agent clusters and calls types INDEPENDENTLY; the reference
column is used only for scoring, never shown to the agent.

  * pbmc68k   — sc.datasets.pbmc68k_reduced(); `bulk_labels` are FACS-sorted populations
                (as close to ground truth as scRNA gets).
  * pbmc3k    — sc.datasets.pbmc3k_processed(); `louvain` holds curated cell-type names.
  * <path>    — any local .h5ad; you supply the reference obs column.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSpec:
    name: str
    load_code: str           # kernel code that binds `adata`
    ref_col: str             # obs column with trusted labels (scoring only)
    embedding: str | None    # obsm key to cluster on (None -> compute PCA)
    normalized: bool         # X already log-normalized


_PUBLIC = {
    "pbmc68k": DatasetSpec(
        name="pbmc68k",
        load_code="import scanpy as sc\nadata = sc.datasets.pbmc68k_reduced()",
        ref_col="bulk_labels", embedding="X_pca", normalized=True),
    "pbmc3k": DatasetSpec(
        name="pbmc3k",
        load_code="import scanpy as sc\nadata = sc.datasets.pbmc3k_processed()",
        ref_col="louvain", embedding="X_pca", normalized=True),
}


def resolve(dataset: str, ref_col: str | None = None, embedding: str | None = None,
            normalized: bool | None = None) -> DatasetSpec:
    """A named public dataset, or a local .h5ad path (ref_col required)."""
    if dataset in _PUBLIC:
        spec = _PUBLIC[dataset]
        if ref_col:
            spec = DatasetSpec(spec.name, spec.load_code, ref_col, spec.embedding, spec.normalized)
        return spec
    path = Path(dataset)
    if not path.exists():
        raise ValueError(f"unknown dataset {dataset!r} (not a public name or existing .h5ad)")
    if not ref_col:
        raise ValueError(f"--reference-col is required for a local dataset ({dataset})")
    return DatasetSpec(name=path.stem, load_code=f"import scanpy as sc\nadata = sc.read_h5ad({str(path)!r})",
                       ref_col=ref_col, embedding=embedding, normalized=bool(normalized))


def available() -> list[str]:
    return list(_PUBLIC)
