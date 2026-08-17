"""Translated alignment via DIAMOND, and best-hit reduction.

The column set here is not negotiable: ROCkOut's filter reads columns
0,1,2,3,6,7,8,9,11,12,13 out of a 14-column table, so any model we build must be
fitted on the same fields.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .simulate import parse_defline

OUTFMT = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen",
]

SENSITIVITY_FLAGS = {
    "fast": ["--fast"],
    "default": [],
    "sensitive": ["--sensitive"],
    "more-sensitive": ["--more-sensitive"],
    "very-sensitive": ["--very-sensitive"],
    "ultra-sensitive": ["--ultra-sensitive"],
}


@dataclass
class Alignments:
    """Best-hit alignments for one read length, as parallel arrays."""

    read_id: np.ndarray
    target: np.ndarray
    pct_id: np.ndarray
    aln_len: np.ndarray
    sstart: np.ndarray
    send: np.ndarray
    bitscore: np.ndarray
    qlen: np.ndarray
    pct_aln: np.ndarray
    label: np.ndarray
    source: np.ndarray

    def __len__(self) -> int:
        return self.read_id.size

    def subset(self, mask: np.ndarray) -> "Alignments":
        return Alignments(
            **{f: getattr(self, f)[mask] for f in self.__dataclass_fields__}
        )


def require_diamond(path: Optional[str] = None) -> str:
    exe = path or shutil.which("diamond")
    if exe is None:
        raise FileNotFoundError(
            "DIAMOND not found on PATH. Install it (conda install -c bioconda diamond) "
            "or pass --diamond /path/to/diamond. Translated alignment is required: "
            "the model's axes are bitscore and percent identity from blastx."
        )
    return exe


def make_db(diamond: str, protein_fasta: str, db_path: str, threads: int = 1) -> str:
    cmd = [diamond, "makedb", "--in", protein_fasta, "-d", db_path, "--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"diamond makedb failed:\n{proc.stderr[-2000:]}")
    produced = db_path if db_path.endswith(".dmnd") else db_path + ".dmnd"
    if not os.path.exists(produced):
        raise RuntimeError(f"diamond makedb reported success but {produced} is missing")
    return produced


def blastx(
    diamond: str,
    reads_fasta: str,
    db_path: str,
    out_path: str,
    threads: int = 1,
    sensitivity: str = "sensitive",
    max_target_seqs: int = 0,
    evalue: float = 0.001,
) -> str:
    # ROCkOut aligns with `--sensitive --max-target-seqs 0` at DIAMOND's default
    # e-value. Matching keeps the confounder set comparable, and is far faster than
    # a permissive e-value once deep genomic background is in play.
    cmd = [
        diamond, "blastx",
        "-q", reads_fasta,
        "-d", db_path,
        "-o", out_path,
        "--outfmt", "6", *OUTFMT,
        "--threads", str(threads),
        "--max-target-seqs", str(max_target_seqs),
        "--evalue", str(evalue),
        "--quiet",
        *SENSITIVITY_FLAGS.get(sensitivity, []),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"diamond blastx failed:\n{proc.stderr[-2000:]}")
    return out_path


def find_target_regions(
    diamond: str,
    contigs_fasta: str,
    db_path: str,
    min_identity: float = 90.0,
    threads: int = 1,
    evalue: float = 1e-5,
    workdir: str = ".",
) -> dict:
    """Locate copies of the positive genes inside background contigs.

    This is the coordinate-based equivalent of what ROCkOut gets from genome
    annotation: it needs to know which stretches of a background genome *are* the
    target, so reads drawn from them are labelled positive instead of being
    trained against as confounders.

    Only high-identity hits count. Distant paralogues sit far below the threshold
    and must stay confounders -- they are the whole reason the filter exists.

    -> {contig: [(start, end), ...]} with 0-based, half-open, merged intervals.
    """
    out_path = os.path.join(workdir, "background_target_regions.tsv")
    cmd = [
        diamond, "blastx",
        "-q", contigs_fasta,
        "-d", db_path,
        "-o", out_path,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qstart", "qend",
        "--threads", str(threads),
        "--max-target-seqs", "0",
        "--evalue", str(evalue),
        "--sensitive",
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"diamond blastx on background contigs failed:\n{proc.stderr[-2000:]}")

    per_contig: dict = {}
    with open(out_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if float(f[2]) < min_identity:
                continue
            qs, qe = int(f[4]), int(f[5])
            # BLAST coordinates are 1-based inclusive, and reverse-strand hits
            # report qstart > qend. Normalise to 0-based half-open.
            lo, hi = (qs, qe) if qs <= qe else (qe, qs)
            per_contig.setdefault(f[0], []).append((lo - 1, hi))

    return {name: _merge_intervals(iv) for name, iv in per_contig.items()}


def _merge_intervals(intervals: list) -> list:
    intervals = sorted(intervals)
    merged: list = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def load_alignments(path: str, valid_targets: set, rng: np.random.Generator) -> Alignments:
    """Parse a DIAMOND table, keep best hit per read, derive pct_aln and labels."""
    read_id, target = [], []
    pct_id, aln_len, sstart, send, bitscore, qlen = [], [], [], [], [], []

    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if f[1] not in valid_targets:
                continue
            read_id.append(f[0])
            target.append(f[1])
            pct_id.append(float(f[2]))
            aln_len.append(int(f[3]))
            sstart.append(int(f[8]))
            send.append(int(f[9]))
            bitscore.append(float(f[11]))
            qlen.append(int(f[12]))

    if not read_id:
        raise ValueError(
            f"no usable alignments in {path}. Either DIAMOND found nothing (check "
            "that positives and negatives are actually homologous) or every subject "
            "name was absent from the alignment."
        )

    read_id = np.array(read_id)
    aln = Alignments(
        read_id=read_id,
        target=np.array(target),
        pct_id=np.array(pct_id, dtype=np.float64),
        aln_len=np.array(aln_len, dtype=np.int32),
        sstart=np.array(sstart, dtype=np.int32),
        send=np.array(send, dtype=np.int32),
        bitscore=np.array(bitscore, dtype=np.float64),
        qlen=np.array(qlen, dtype=np.int32),
        pct_aln=np.zeros(read_id.size, dtype=np.float64),
        label=np.empty(read_id.size, dtype=object),
        source=np.empty(read_id.size, dtype=object),
    )

    # Identical to rocker_filter.load_reads: alignment length over translated
    # query length.
    aln.pct_aln = np.round(100.0 * aln.aln_len / (aln.qlen / 3.0), 2)

    labels, sources = [], []
    for name in aln.read_id:
        src, _, label = parse_defline(name)
        labels.append(label)
        sources.append(src)
    aln.label = np.array(labels)
    aln.source = np.array(sources)

    return best_hits(aln, rng)


def best_hits(aln: Alignments, rng: np.random.Generator) -> Alignments:
    """Keep the max-bitscore alignment per read, breaking ties at random."""
    order = np.lexsort((-aln.bitscore, aln.read_id))
    sorted_reads = aln.read_id[order]
    sorted_bits = aln.bitscore[order]

    # Group boundaries in the read-sorted array.
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_reads[1:] != sorted_reads[:-1]))
    )
    ends = np.concatenate((starts[1:], [sorted_reads.size]))

    chosen = np.empty(starts.size, dtype=np.int64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        top = sorted_bits[s]
        n_tied = int(np.searchsorted(-sorted_bits[s:e], -top, side="right"))
        chosen[i] = s if n_tied <= 1 else s + int(rng.integers(0, n_tied))

    keep = order[chosen]
    keep.sort()
    return aln.subset(keep)
