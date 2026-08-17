"""Generate a synthetic labelled dataset for end-to-end testing.

The design mirrors the situation ROCker-style filters exist for: positives and
confounders share a highly conserved domain in the middle of the protein and
diverge sharply outside it. A single global bitscore cutoff cannot separate them;
position-specific cutoffs can.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.fasta import CODON_TABLE, write_fasta  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
_REVERSE = {}
for _codon, _aa in CODON_TABLE.items():
    if _aa != "*":
        _REVERSE.setdefault(_aa, []).append(_codon)


def mutate_protein(seq: str, rate: float, rng: np.random.Generator) -> str:
    arr = list(seq)
    hits = rng.random(len(arr)) < rate
    for i in np.flatnonzero(hits):
        arr[i] = AA[rng.integers(0, len(AA))]
    return "".join(arr)


def reverse_translate(protein: str, rng: np.random.Generator) -> str:
    out = []
    for aa in protein:
        codons = _REVERSE.get(aa, _REVERSE["A"])
        out.append(codons[rng.integers(0, len(codons))])
    return "".join(out)


def make_family(
    ancestor: str,
    conserved: slice,
    n: int,
    divergence: float,
    conserved_divergence: float,
    prefix: str,
    rng: np.random.Generator,
) -> dict:
    """Mutate an ancestor n times, protecting a conserved window."""
    family = {}
    for i in range(n):
        variable = mutate_protein(ancestor, divergence, rng)
        core = mutate_protein(ancestor[conserved], conserved_divergence, rng)
        seq = variable[: conserved.start] + core + variable[conserved.stop :]
        family[f"{prefix}_{i:03d}"] = seq
    return family


def build(outdir: str, seed: int = 7, n_pos: int = 12, n_neg: int = 12, length: int = 320):
    rng = np.random.default_rng(seed)
    ancestor = "".join(AA[i] for i in rng.integers(0, len(AA), size=length))
    conserved = slice(120, 200)

    positives = make_family(
        ancestor, conserved, n_pos,
        divergence=0.18, conserved_divergence=0.03, prefix="target", rng=rng,
    )

    # The confounder lineage diverges hard outside the conserved core but keeps
    # the core nearly intact -- the shared, non-discriminative domain.
    outgroup_ancestor = mutate_protein(ancestor, 0.55, rng)
    outgroup_ancestor = (
        outgroup_ancestor[: conserved.start]
        + mutate_protein(ancestor[conserved], 0.08, rng)
        + outgroup_ancestor[conserved.stop :]
    )
    negatives = make_family(
        outgroup_ancestor, conserved, n_neg,
        divergence=0.15, conserved_divergence=0.05, prefix="confounder", rng=rng,
    )

    pos_dir = os.path.join(outdir, "positives")
    neg_dir = os.path.join(outdir, "negatives")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    write_fasta(
        os.path.join(pos_dir, "targets.fna"),
        {k: reverse_translate(v, rng) for k, v in positives.items()},
    )
    write_fasta(
        os.path.join(neg_dir, "confounders.fna"),
        {k: reverse_translate(v, rng) for k, v in negatives.items()},
    )
    return pos_dir, neg_dir, positives, negatives


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "test_data"
    p, n, pos, neg = build(target)
    print(f"positives: {p} ({len(pos)} sequences)")
    print(f"negatives: {n} ({len(neg)} sequences)")
