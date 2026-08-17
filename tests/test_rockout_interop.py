"""Load a rockabye model with ROCkOut's own filter code and classify fresh reads.

This is the only test that actually proves compatibility: it imports the real
`modules/rocker_filter.py` from a ROCkOut checkout and runs its loading and
classification path unmodified.

Usage:
    python3 tests/test_rockout_interop.py <rocker_filter.py> <project_dir> <diamond>
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.fasta import read_fasta_dict  # noqa: E402
from rockabye.simulate import SimConfig, simulate_all  # noqa: E402
from rockabye.project import Inputs  # noqa: E402


def _stub_plotly() -> None:
    """ROCkOut imports plotly at module scope purely for its figures."""
    for name in ("plotly", "plotly.express", "plotly.graph_objects"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = lambda attr: (lambda *a, **k: None)  # type: ignore
            sys.modules[name] = mod


def load_rockout_filter(path: str):
    _stub_plotly()
    spec = importlib.util.spec_from_file_location("rocker_filter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate_holdout(pos_dir: str, neg_dir: str, out_fasta: str, read_length: int = 150):
    """Fresh reads with a different seed, so nothing is memorised from training."""
    def load(directory):
        records = {}
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".fna", ".fa", ".fasta")):
                records.update(read_fasta_dict(os.path.join(directory, fname)))
        return records

    inputs = Inputs(positive_nt=load(pos_dir), negative_nt=load(neg_dir))
    cfg = SimConfig(read_lengths=(read_length,), coverage=8.0, seed=99991)
    workdir = os.path.dirname(os.path.abspath(out_fasta))
    produced = simulate_all(inputs, cfg, os.path.join(workdir, "holdout_sim"), log=lambda *a: None)
    shutil.copyfile(produced[read_length], out_fasta)
    return out_fasta


def main() -> int:
    filter_src, project, diamond = sys.argv[1], sys.argv[2], sys.argv[3]
    pos_dir, neg_dir = sys.argv[4], sys.argv[5]
    workdir = sys.argv[6]
    os.makedirs(workdir, exist_ok=True)

    reads_fa = os.path.join(workdir, "holdout_reads.fasta")
    simulate_holdout(pos_dir, neg_dir, reads_fa)

    db = os.path.join(project, "final_outputs", "database", "positive_proteins_diamond_db.dmnd")
    tsv = os.path.join(workdir, "holdout_alignments.tsv")
    cmd = [
        diamond, "blastx", "-q", reads_fa, "-d", db, "-o", tsv,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "mismatch",
        "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "qlen", "slen",
        "--sensitive", "--max-target-seqs", "25", "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"diamond failed:\n{proc.stderr}")
        return 1

    rf = load_rockout_filter(filter_src)

    # Instantiate exactly as rockout_main.py does, then disable only the plotting.
    filt = rf.rocker_filterer(project_directory=project, filter_dir=workdir)
    filt.plot_results = lambda *a, **k: None

    filt.find_filters()
    assert filt.filter_file, "ROCkOut did not find bitscore_vs_MA_pos.txt"
    assert filt.id_filter_file, "ROCkOut did not find pct_id_vs_MA_pos.txt"
    assert filt.idaln_file, "ROCkOut did not find pct_id_vs_pct_aln.txt"
    print("OK    ROCkOut located all three cutoff tables")

    filt.find_ma()
    assert filt.ma_file, "ROCkOut did not find the multiple alignment"
    print(f"OK    ROCkOut loaded the alignment ({len(filt.offsets)} proteins)")

    filt.load_filters()
    print(
        f"OK    ROCkOut built cutoff matrices: "
        f"bitscore {filt.filter_matrix.shape}, "
        f"pct_id/pos {filt.idpos_filtmat.shape}, "
        f"pct_id/aln {filt.idaln_filtmat.shape} "
        f"(read lengths {filt.min_readlen}-{filt.max_readlen})"
    )

    df = filt.load_reads(tsv)
    df = filt.besthit_reads(df)
    passing, failing = filt.filter_reads(df, "interop")

    def labels(frame):
        return np.array([n.split(";")[-1] == "Positive" for n in frame["read"]])

    passed_truth = labels(passing)
    failed_truth = labels(failing)

    tp = int(np.sum(passed_truth))
    fp = int(np.sum(~passed_truth))
    fn = int(np.sum(failed_truth))
    tn = int(np.sum(~failed_truth))
    total = tp + fp + fn + tn
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    print(f"\nROCkOut's filter classified {total:,} held-out alignments")
    print(f"  TP {tp:,}   FP {fp:,}   FN {fn:,}   TN {tn:,}")
    print(f"  F1 {f1:.4f}   accuracy {(tp + tn) / max(1, total):.4f}")
    print(f"  FPR {fp / max(1, fp + tn) * 100:.2f}%   FNR {fn / max(1, fn + tp) * 100:.2f}%")

    if f1 < 0.80:
        print("\nFAIL  model loaded but discriminates poorly")
        return 1
    print("\nPASS  model is loadable and discriminative under ROCkOut's own filter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
