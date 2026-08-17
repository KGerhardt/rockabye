"""FASTA parsing/writing and nucleotide translation.

Deliberately dependency-free: the whole point of trusting user labels is that we
never have to touch a network or a heavyweight parser.
"""

from __future__ import annotations

import gzip
import io
import os
from typing import Dict, Iterator, Tuple

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_COMPLEMENT = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")

FASTA_NT_EXTS = (".fna", ".fa", ".fasta", ".fas", ".ffn")
FASTA_AA_EXTS = (".faa", ".fasta.aa", ".pep")
MSA_EXTS = (".afa", ".aln", ".msa", ".mfa")


def _open_text(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (defline_id, sequence). ID is the defline up to first whitespace.

    DIAMOND truncates subject names at the first whitespace, and ROCkOut's filter
    matches MSA deflines against those names the same way, so we normalise here.
    """
    name = None
    chunks: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0] if len(line) > 1 else ""
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def read_fasta_dict(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, seq in read_fasta(path):
        if name in out:
            raise ValueError(f"duplicate sequence ID {name!r} in {path}")
        out[name] = seq
    return out


def write_fasta(path: str, records: Dict[str, str], wrap: int = 60) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            if wrap and wrap > 0:
                for i in range(0, len(seq), wrap):
                    fh.write(seq[i : i + wrap] + "\n")
            else:
                fh.write(seq + "\n")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def translate(seq: str, stop_char: str = "X", trim_trailing_stop: bool = True) -> str:
    """Translate a CDS in frame 0. Internal stops become `stop_char`."""
    seq = seq.upper().replace("U", "T")
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        aa.append(CODON_TABLE.get(seq[i : i + 3], "X"))
    if trim_trailing_stop and aa and aa[-1] == "*":
        aa.pop()
    return "".join(c if c != "*" else stop_char for c in aa)


def looks_like_protein(seq: str, sample: int = 300) -> bool:
    """Heuristic: nucleotide sequences are ~entirely ACGTUN."""
    s = seq[:sample].upper()
    if not s:
        return False
    nt = sum(1 for c in s if c in "ACGTUN-")
    return (nt / len(s)) < 0.85
