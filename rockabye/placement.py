"""Phylogenetic reference packages and read placement, from any set of sequences.

This is ROCkOut's `place` machinery (modules/pplacer/) made standalone: the same
tools with the same parameters, but every step is checked. ROCkOut ran each one
through os.system, so a missing `taxit` left an empty reference_package/ behind
and `place` later died on `os.listdir(...)[0]` -- 7 of 14 reference packages
across the ROCkOut projects on hand were empty that way.

    refpkg:  sequences -> MUSCLE 5 -> FastTree -> taxit create
    place:   reads -> mafft --addfragments --adjustdirection -> pplacer -> guppy to_csv -> TSV

Placement departs from ROCkOut in one place, the mafft mode; see MAFFT_MODES.

Neither step depends on a rockabye model.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .fasta import FASTA_AA_EXTS, FASTA_NT_EXTS, MSA_EXTS, looks_like_protein, read_fasta, read_fasta_dict, write_fasta
from .msa import _muscle_is_v5

# Newick metacharacters. FastTree and pplacer either reject these in a name or
# silently split on them, after which taxit and pplacer no longer agree on which
# leaf is which.
_NEWICK_RESERVED = set("():;,[]'")

_FASTA_EXTS = FASTA_NT_EXTS + FASTA_AA_EXTS + MSA_EXTS

# File names inside a refpkg build directory.
SOURCE_DIR = "source"
PACKAGE_DIR = "reference_package"
COMBINED = "combined_sequences.fasta"
ALIGNMENT = "combined_sequences_multiple_alignment.fasta"
TREE_LOG = "combined_sequences_fasttree_log.txt"
TREE = "combined_sequences_fasttree_tree.txt"


def require_tool(name: str, override: Optional[str] = None, hint: str = "") -> str:
    exe = shutil.which(override or name)
    if exe is None:
        raise FileNotFoundError(
            f"{override or name!r} not found on PATH.{(' ' + hint) if hint else ''}"
        )
    return exe


def _run(cmd: List[str], what: str, stdout_path: Optional[str] = None) -> None:
    if stdout_path is None:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    else:
        with open(stdout_path, "w") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )


def fasta_files(paths: Sequence[str]) -> List[str]:
    """Expand a mix of files and directories into FASTA file paths, sorted per directory."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(
                os.path.join(p, f) for f in os.listdir(p)
                if f.lower().removesuffix(".gz").endswith(_FASTA_EXTS)
            )
            if not found:
                raise ValueError(f"no FASTA files in directory {p}")
            out.extend(found)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return out


def check_names(names, source: str) -> None:
    for name in names:
        if not name:
            raise ValueError(f"empty sequence name in {source}")
        bad = sorted(_NEWICK_RESERVED.intersection(name))
        if bad:
            raise ValueError(
                f"sequence name {name!r} in {source} contains {''.join(bad)!r}, which "
                "FastTree and pplacer treat as tree syntax. Rename it."
            )


def combine_sequences(paths: Sequence[str]) -> Dict[str, str]:
    combined: Dict[str, str] = {}
    origin: Dict[str, str] = {}
    for path in paths:
        for name, seq in read_fasta(path):
            if name in combined:
                raise ValueError(
                    f"sequence ID {name!r} appears in both {origin[name]} and {path}; "
                    "IDs must be unique across all inputs"
                )
            combined[name] = seq.replace("-", "").replace(".", "").upper()
            origin[name] = path
    check_names(combined, "the inputs")
    return combined


def detect_molecule(seqs: Dict[str, str]) -> str:
    kinds = {looks_like_protein(s) for s in seqs.values() if s}
    if len(kinds) != 1:
        raise ValueError("inputs mix nucleotide and protein sequences; use one or the other")
    return "aa" if kinds.pop() else "nt"


def align_muscle5(in_path: str, out_path: str, threads: int, muscle: Optional[str] = None) -> None:
    exe = require_tool("muscle", muscle, "Install MUSCLE 5 (conda install -c bioconda muscle).")
    if not _muscle_is_v5(exe):
        raise RuntimeError(
            f"{exe} is not MUSCLE 5. MUSCLE 3's -in/-out interface is not supported; "
            "install MUSCLE 5 or supply your own alignment with --alignment."
        )
    _run([exe, "-align", in_path, "-output", out_path, "-threads", str(threads)], "MUSCLE")


def fasttree_command(exe: str, molecule: str, log_path: str, tree_path: str, aln_path: str) -> List[str]:
    # Nucleotide: ROCkOut's exact invocation. Protein: FastTree's default model
    # (JTT+CAT); ROCkOut never built protein packages ("Make proteins work?").
    model = ["-nt", "-gtr"] if molecule == "nt" else []
    return [exe, *model, "-log", log_path, "-out", tree_path, aln_path]


@dataclass
class RefpkgResult:
    package: str
    alignment: str
    n_sequences: int
    n_columns: int
    molecule: str


def build_refpkg(
    inputs: Sequence[str],
    outdir: str,
    name: str = "reference",
    locus: str = "placeholder",
    alignment: Optional[str] = None,
    threads: int = 1,
    force: bool = False,
    muscle: Optional[str] = None,
    fasttree: Optional[str] = None,
    taxit: Optional[str] = None,
    log=print,
) -> RefpkgResult:
    """Align, tree and package a set of reference sequences.

    `alignment`, when given, is used as-is in place of the MUSCLE step and
    `inputs` may be empty.
    """
    # Resolve every tool before any work: a tree can take a long time to build.
    # Bioconda installs `fasttree`; the upstream binary is `FastTree`.
    if fasttree is None and shutil.which("fasttree") is None:
        fasttree = "FastTree"
    fasttree_exe = require_tool(
        "fasttree", fasttree, "Install FastTree (conda install -c bioconda fasttree)."
    )
    taxit_exe = require_tool("taxit", taxit, "Install taxtastic (pip install taxtastic).")

    outdir = os.path.abspath(outdir)
    package = os.path.join(outdir, PACKAGE_DIR, f"{name}.refpkg")
    if os.path.exists(package):
        if not force:
            raise ValueError(
                f"{package} already exists. Pass --force to replace it, or choose "
                "another output directory or --name."
            )
        shutil.rmtree(package)
    source = os.path.join(outdir, SOURCE_DIR)
    os.makedirs(source, exist_ok=True)
    os.makedirs(os.path.dirname(package), exist_ok=True)

    aln_path = os.path.join(source, ALIGNMENT)
    if alignment is not None:
        msa = read_fasta_dict(alignment)
        check_names(msa, alignment)
        widths = {len(s) for s in msa.values()}
        if len(widths) != 1:
            raise ValueError(f"{alignment} is not an alignment: rows have lengths {sorted(widths)[:5]}")
        msa = {k: v.upper() for k, v in msa.items()}
        write_fasta(aln_path, msa)
        log(f"using supplied alignment of {len(msa)} sequences x {widths.pop()} columns")
    else:
        seqs = combine_sequences(fasta_files(inputs))
        if len(seqs) < 3:
            raise ValueError(
                f"{len(seqs)} sequence(s) supplied; a tree to place onto needs at least 3"
            )
        combined = os.path.join(source, COMBINED)
        write_fasta(combined, seqs)
        log(f"aligning {len(seqs)} sequences with MUSCLE 5")
        align_muscle5(combined, aln_path, threads, muscle)
        msa = read_fasta_dict(aln_path)
        if set(msa) != set(seqs):
            raise RuntimeError(f"MUSCLE dropped sequences: {sorted(set(seqs) - set(msa))[:5]}")

    if len(msa) < 3:
        raise ValueError(f"{len(msa)} sequence(s) in the alignment; a tree needs at least 3")
    molecule = detect_molecule({k: v.replace("-", "") for k, v in msa.items()})
    n_columns = len(next(iter(msa.values())))

    log(f"building {'GTR' if molecule == 'nt' else 'JTT+CAT'} tree with FastTree")
    log_path = os.path.join(source, TREE_LOG)
    tree_path = os.path.join(source, TREE)
    _run(fasttree_command(fasttree_exe, molecule, log_path, tree_path, aln_path), "FastTree")

    log("packaging with taxit")
    _run(
        [taxit_exe, "create", "-l", locus, "-P", package,
         "--aln-fasta", aln_path, "--tree-stats", log_path, "--tree-file", tree_path],
        "taxit create",
    )
    contents = os.path.join(package, "CONTENTS.json")
    if not os.path.isfile(contents):
        raise RuntimeError(f"taxit reported success but wrote no {contents}")

    return RefpkgResult(package, aln_path, len(msa), n_columns, molecule)


def refpkg_alignment(package: str) -> str:
    """The reference alignment a taxit package points at."""
    contents = os.path.join(package, "CONTENTS.json")
    if not os.path.isfile(contents):
        raise FileNotFoundError(
            f"{package} is not a reference package (no CONTENTS.json). Build one "
            "with `rockabye refpkg`."
        )
    with open(contents) as fh:
        files = json.load(fh).get("files", {})
    if "aln_fasta" not in files:
        raise ValueError(f"{contents} lists no aln_fasta")
    return os.path.join(package, files["aln_fasta"])


def guppy_csv_to_tsv(csv_path: str, tsv_path: str) -> int:
    """Rewrite guppy's CSV as TSV; returns the number of placement rows.

    ROCkOut replaced every comma with a tab, which splits any quoted field that
    contains one. Parse it as CSV instead.
    """
    rows = 0
    with open(csv_path, newline="") as src, open(tsv_path, "w", newline="") as dst:
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        for i, row in enumerate(csv.reader(src)):
            writer.writerow(row)
            rows += i > 0
    return rows


# How reads are added to the reference alignment. ROCkOut used plain `--add`,
# which aligns each read as if it were full length and in the references'
# orientation. On 1,000 real nirK reads (121-sequence GTR+CAT package) that put
# 32% on their source gene's leaf edge and left 63% at pplacer's pendant-length
# ceiling, i.e. fitting nowhere. `--addfragments --adjustdirection` put 77% on
# the source edge and 7% at the ceiling, for ~7x the mafft time.
MAFFT_MODES = {
    "fragments": ["--adjustdirection", "--addfragments"],
    "add": ["--add"],
}


def _restore_read_names(raw_path: str, out_path: str, read_names) -> int:
    """Undo mafft --adjustdirection's `_R_` prefix on reversed reads; returns how many."""
    reversed_ = 0
    with open(raw_path) as src, open(out_path, "w") as dst:
        for line in src:
            if line.startswith(">_R_"):
                name = line[1:].split()[0]
                if name not in read_names and name[3:] in read_names:
                    line = ">" + line[4:]
                    reversed_ += 1
            dst.write(line)
    return reversed_


@dataclass
class PlacementResult:
    reads: str
    alignment: str
    jplace: str
    tsv: str
    n_reads: int
    n_rows: int
    n_reversed: int


def place_reads(
    package: str,
    reads: Sequence[str],
    outdir: str,
    threads: int = 1,
    keep_at_most: int = 1,
    mafft_mode: str = "fragments",
    force: bool = False,
    mafft: Optional[str] = None,
    pplacer: Optional[str] = None,
    guppy: Optional[str] = None,
    log=print,
) -> List[PlacementResult]:
    """Place each reads FASTA onto `package`; one alignment, jplace and TSV per file."""
    if mafft_mode not in MAFFT_MODES:
        raise ValueError(f"mafft mode must be one of {sorted(MAFFT_MODES)}, not {mafft_mode!r}")
    mafft_exe = require_tool("mafft", mafft, "Install MAFFT (conda install -c bioconda mafft).")
    pplacer_exe = require_tool("pplacer", pplacer, "Install pplacer (conda install -c bioconda pplacer).")
    guppy_exe = require_tool(
        "guppy", guppy, "guppy ships with pplacer (conda install -c bioconda pplacer)."
    )

    package = os.path.abspath(package)
    ref_aln = refpkg_alignment(package)
    ref = read_fasta_dict(ref_aln)
    ref_molecule = detect_molecule({k: v.replace("-", "") for k, v in ref.items()})

    files = fasta_files(reads)
    outdir = os.path.abspath(outdir)
    dirs = {k: os.path.join(outdir, k) for k in ("read_multiple_alignments", "pplacer_jplace", "pplacer_tsv")}

    plans = []
    for path in files:
        base = os.path.basename(path)
        for suffix in (".gz", ".filtered.fasta", *_FASTA_EXTS):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
        plans.append((path, base))
    bases = [b for _, b in plans]
    if len(set(bases)) != len(bases):
        raise ValueError(f"read files share an output name: {sorted(b for b in bases if bases.count(b) > 1)[:5]}")
    if not force:
        for _, base in plans:
            existing = os.path.join(dirs["pplacer_jplace"], f"{base}.jplace")
            if os.path.exists(existing):
                raise ValueError(f"{existing} already exists. Pass --force to overwrite.")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    results = []
    for path, base in plans:
        seqs = read_fasta_dict(path)
        if not seqs:
            log(f"  {base}: no reads, skipping")
            continue
        clash = sorted(set(seqs) & set(ref))
        if clash:
            raise ValueError(
                f"{path} reuses reference sequence IDs {clash[:5]}; mafft --add would "
                "make them indistinguishable from the references"
            )
        # Read names never enter the tree, so Newick metacharacters are harmless
        # there; ROCkOut's own read names are ';'-delimited.
        if detect_molecule(seqs) != ref_molecule:
            raise ValueError(
                f"{path} is {detect_molecule(seqs)} but the reference package is "
                f"{ref_molecule}; reads must match the package's alphabet"
            )

        aln_out = os.path.join(dirs["read_multiple_alignments"], f"{base}_multiple_alignment.fasta")
        jplace = os.path.join(dirs["pplacer_jplace"], f"{base}.jplace")
        tsv = os.path.join(dirs["pplacer_tsv"], f"{base}.jplace.tsv")

        log(f"  {base}: adding {len(seqs)} reads to the reference alignment")
        raw_aln = aln_out + ".tmp"
        # Proteins have no reverse strand.
        mode_flags = [f for f in MAFFT_MODES[mafft_mode] if ref_molecule == "nt" or f != "--adjustdirection"]
        _run(
            [mafft_exe, "--quiet", "--thread", str(threads), *mode_flags,
             path, "--reorder", ref_aln],
            f"mafft ({mafft_mode})", stdout_path=raw_aln,
        )
        n_reversed = _restore_read_names(raw_aln, aln_out, set(seqs))
        os.remove(raw_aln)
        if "--adjustdirection" in mode_flags:
            log(f"  {base}: {n_reversed} of {len(seqs)} reads were reverse-complemented to match the references")
        log(f"  {base}: placing with pplacer")
        _run(
            [pplacer_exe, "--discard-nonoverlapped", "-c", package, "-o", jplace,
             "-j", str(threads), "--keep-at-most", str(keep_at_most), aln_out],
            "pplacer",
        )
        csv_tmp = tsv + ".csv.tmp"
        _run([guppy_exe, "to_csv", jplace], "guppy to_csv", stdout_path=csv_tmp)
        n_rows = guppy_csv_to_tsv(csv_tmp, tsv)
        os.remove(csv_tmp)
        results.append(PlacementResult(path, aln_out, jplace, tsv, len(seqs), n_rows, n_reversed))
    return results
