"""Input discovery: turn two labelled directories into a validated project.

This module is where the "trust the user's labelling" premise lives. ROCkOut has
to download genomes and intersect read coordinates with gene boundaries to decide
what is positive; here, provenance *is* the label.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .fasta import (
    FASTA_AA_EXTS,
    FASTA_NT_EXTS,
    MSA_EXTS,
    looks_like_protein,
    read_fasta,
    translate,
)

POSITIVE = "Positive"
NEGATIVE = "Negative"
BACKGROUND = "Background"


@dataclass
class Inputs:
    """Everything the builder needs, already validated."""

    positive_nt: Dict[str, str] = field(default_factory=dict)
    negative_nt: Dict[str, str] = field(default_factory=dict)
    background_nt: Dict[str, str] = field(default_factory=dict)
    positive_aa: Dict[str, str] = field(default_factory=dict)
    positive_msa: Optional[Dict[str, str]] = None
    aa_source: str = "translated"  # or "user-supplied"
    msa_source: str = "computed"  # or "user-supplied"
    sources: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def n_positive(self) -> int:
        return len(self.positive_nt)

    @property
    def n_negative(self) -> int:
        return len(self.negative_nt)

    @property
    def n_background(self) -> int:
        return len(self.background_nt)


def _files_with_ext(directory: str, exts) -> List[str]:
    found: List[str] = []
    for ext in exts:
        found.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        found.extend(glob.glob(os.path.join(directory, f"*{ext}.gz")))
    return sorted(set(found))


def _load_dir(
    directory: str, exts, label: str, on_duplicate: str = "error"
) -> tuple[Dict[str, str], List[str]]:
    paths = _files_with_ext(directory, exts)
    records: Dict[str, str] = {}
    for path in paths:
        for name, seq in read_fasta(path):
            if not seq:
                continue
            if name in records:
                if on_duplicate == "skip":
                    # Background contigs are only ever read sources, never DIAMOND
                    # subjects, so the same genome appearing twice is harmless.
                    continue
                raise ValueError(
                    f"duplicate sequence ID {name!r} found in {label} inputs "
                    f"(second occurrence in {path}). IDs must be unique across the "
                    "whole project because they become DIAMOND subject names."
                )
            if ";" in name:
                raise ValueError(
                    f"sequence ID {name!r} in {path} contains ';', which separates "
                    "fields in simulated read names. Rename the sequence."
                )
            records[name] = seq.upper()
    return records, paths


def load_inputs(
    positive_dir: str, negative_dir: str, background_dir: str | None = None
) -> Inputs:
    for d, what in ((positive_dir, "positive"), (negative_dir, "negative")):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{what} directory not found: {d}")

    inputs = Inputs()
    sources: Dict[str, List[str]] = {}

    pos_nt, pos_nt_files = _load_dir(positive_dir, FASTA_NT_EXTS, "positive")
    neg_nt, neg_nt_files = _load_dir(negative_dir, FASTA_NT_EXTS, "negative")
    sources["positive_nucleotide"] = pos_nt_files
    sources["negative_nucleotide"] = neg_nt_files

    # A .fa/.fasta file holding proteins is a common user mistake; catch it early
    # rather than producing a model trained on nonsense reads.
    for name, seq in list(pos_nt.items()) + list(neg_nt.items()):
        if looks_like_protein(seq):
            raise ValueError(
                f"sequence {name!r} looks like protein but was read as nucleotide. "
                "Put protein sequences in a .faa file; read simulation needs "
                "nucleotide input."
            )

    if not pos_nt:
        raise ValueError(
            f"no nucleotide FASTA found in {positive_dir} "
            f"(looked for {', '.join(FASTA_NT_EXTS)})"
        )
    if not neg_nt:
        raise ValueError(
            f"no nucleotide FASTA found in {negative_dir} "
            f"(looked for {', '.join(FASTA_NT_EXTS)}). Negatives are required: "
            "without confounders every threshold collapses to 'accept everything'."
        )

    overlap = set(pos_nt) & set(neg_nt)
    if overlap:
        raise ValueError(
            "the same sequence ID appears in both positive and negative inputs: "
            + ", ".join(sorted(overlap)[:5])
            + ("..." if len(overlap) > 5 else "")
        )

    inputs.positive_nt = pos_nt
    inputs.negative_nt = neg_nt

    # Optional protein override.
    pos_aa, pos_aa_files = _load_dir(positive_dir, FASTA_AA_EXTS, "positive protein")
    if pos_aa:
        missing = set(pos_nt) - set(pos_aa)
        extra = set(pos_aa) - set(pos_nt)
        if missing or extra:
            raise ValueError(
                "supplied positive protein FASTA must have exactly the same IDs as "
                "the positive nucleotide FASTA (reads are labelled by source ID).\n"
                f"  missing from proteins: {sorted(missing)[:5]}\n"
                f"  not in nucleotides:    {sorted(extra)[:5]}"
            )
        inputs.positive_aa = {k: pos_aa[k].upper() for k in pos_nt}
        inputs.aa_source = "user-supplied"
        sources["positive_protein"] = pos_aa_files
    else:
        inputs.positive_aa = {k: translate(v) for k, v in pos_nt.items()}
        inputs.aa_source = "translated"

    for name, seq in inputs.positive_aa.items():
        if not seq:
            raise ValueError(f"positive {name!r} translated to an empty protein")

    # Optional pre-built MSA override.
    msa, msa_files = _load_dir(positive_dir, MSA_EXTS, "positive MSA")
    if msa:
        lengths = {len(s) for s in msa.values()}
        if len(lengths) != 1:
            raise ValueError(
                "supplied MSA is not aligned: sequences have differing lengths "
                f"({sorted(lengths)[:5]})"
            )
        if set(msa) != set(inputs.positive_aa):
            raise ValueError(
                "supplied MSA must contain exactly the positive protein IDs; "
                "ROCkOut's filter drops any alignment whose target is absent "
                "from the MSA."
            )
        inputs.positive_msa = {k: v.upper() for k, v in msa.items()}
        inputs.msa_source = "user-supplied"
        sources["positive_msa"] = msa_files

    # Optional genomic background. Reads simulated from these contigs are
    # confounders. Unlike the negatives directory, background may legitimately
    # contain copies of the target gene (a positive genome is a valid background),
    # so those reads are screened out after alignment rather than trusted here.
    if background_dir:
        if not os.path.isdir(background_dir):
            raise FileNotFoundError(f"background directory not found: {background_dir}")
        bg, bg_files = _load_dir(
            background_dir, FASTA_NT_EXTS, "background", on_duplicate="skip"
        )
        if not bg:
            raise ValueError(
                f"no nucleotide FASTA found in {background_dir} "
                f"(looked for {', '.join(FASTA_NT_EXTS)})"
            )
        clash = (set(bg) & set(pos_nt)) | (set(bg) & set(neg_nt))
        if clash:
            bg = {f"background__{k}": v for k, v in bg.items()}
        inputs.background_nt = bg
        sources["background"] = bg_files

    inputs.sources = sources
    return inputs
