"""Multiple alignment of the positive proteins, and the gap-offset table.

The MSA defines the x-axis of two of the three classifiers, so its column count
is load-bearing: ROCkOut's filter indexes its cutoff matrix by MA column *directly*
(`filter_matrix[readlen_index, ma_pos]`), not by value lookup.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict

import numpy as np

from .fasta import read_fasta_dict, write_fasta


def _muscle_is_v5(exe: str) -> bool:
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True)
    blob = (proc.stdout + proc.stderr).lower()
    if "muscle 5" in blob or "muscle v5" in blob:
        return True
    # v5 exits non-zero on -version in some builds; fall back to help text.
    proc = subprocess.run([exe, "-h"], capture_output=True, text=True)
    return "-align" in (proc.stdout + proc.stderr)


def align_proteins(
    proteins: Dict[str, str], workdir: str, aligner: str = "auto", threads: int = 1, log=print
) -> Dict[str, str]:
    """Run MUSCLE or MAFFT; return {id: gapped_sequence}."""
    os.makedirs(workdir, exist_ok=True)
    in_path = os.path.join(workdir, "positive_proteins_unaligned.fasta")
    out_path = os.path.join(workdir, "positive_proteins_aligned.fasta")
    write_fasta(in_path, proteins)

    if len(proteins) == 1:
        # Nothing to align; the single sequence is its own alignment.
        log("  only one positive protein -- using it directly as the alignment")
        return dict(proteins)

    chosen = aligner
    if aligner == "auto":
        for candidate in ("muscle", "mafft"):
            if shutil.which(candidate):
                chosen = candidate
                break
        else:
            raise FileNotFoundError(
                "no aligner found on PATH (looked for muscle, mafft). Install one, "
                "or supply a pre-built alignment as a .afa file in the positives "
                "directory."
            )

    exe = shutil.which(chosen)
    if exe is None:
        raise FileNotFoundError(f"--aligner {chosen} requested but {chosen!r} is not on PATH")

    if chosen == "muscle":
        if _muscle_is_v5(exe):
            cmd = [exe, "-align", in_path, "-output", out_path, "-threads", str(threads)]
        else:
            cmd = [exe, "-in", in_path, "-out", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    else:
        cmd = [exe, "--auto", "--anysymbol", "--thread", str(threads), in_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            with open(out_path, "w") as fh:
                fh.write(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{chosen} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )

    msa = read_fasta_dict(out_path)
    msa = {k: v.upper() for k, v in msa.items()}
    validate_msa(msa, proteins)
    log(f"  aligned {len(msa)} proteins with {chosen} -> {len(next(iter(msa.values())))} columns")
    return msa


def validate_msa(msa: Dict[str, str], proteins: Dict[str, str]) -> None:
    if set(msa) != set(proteins):
        missing = set(proteins) - set(msa)
        raise ValueError(f"aligner dropped sequences: {sorted(missing)[:5]}")
    lengths = {len(s) for s in msa.values()}
    if len(lengths) != 1:
        raise ValueError(f"alignment rows have differing lengths: {sorted(lengths)[:5]}")
    for name, gapped in msa.items():
        if gapped.replace("-", "").replace(".", "") != proteins[name].replace("*", "X"):
            # Aligners occasionally case-fold or substitute; only the length is
            # structurally required, so warn via exception only on length drift.
            if len(gapped.replace("-", "")) != len(proteins[name]):
                raise ValueError(
                    f"alignment for {name!r} has {len(gapped.replace('-', ''))} residues "
                    f"but the input protein has {len(proteins[name])}"
                )


def compute_offsets(msa: Dict[str, str]) -> tuple[Dict[str, np.ndarray], int]:
    """For each protein, offsets[i] = number of gaps before ungapped residue i.

    MA column of residue i is therefore `offsets[i] + i`. This mirrors
    `rocker_filter.get_offsets` exactly so training and filtering agree.
    """
    offsets: Dict[str, np.ndarray] = {}
    width = 0
    for name, gapped in msa.items():
        width = len(gapped)
        arr = np.empty(len(gapped), dtype=np.int32)
        n = 0
        gaps = 0
        for ch in gapped:
            if ch == "-" or ch == ".":
                gaps += 1
            else:
                arr[n] = gaps
                n += 1
        offsets[name] = arr[:n].copy()
    return offsets, width
