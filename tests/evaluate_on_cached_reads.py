"""Score any model against a ROCkOut project's cached, ground-truth-labelled reads.

Unlike compare_to_rockout.py (which feeds ROCkOut's reads through rockabye's
*fitting* code), this evaluates finished models end to end: each model brings its
own alignment and its own cutoff curves, and both are asked to classify the same
labelled alignments. Because the cached reads were aligned against the same
positive proteins, the alignments are reusable verbatim; only the multiple
alignment used to map a read to a position differs between models.

Usage:
    python3 tests/evaluate_on_cached_reads.py <rockout_project> <model_dir> [<model_dir> ...]
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.binning import Axes  # noqa: E402
from rockabye.fasta import read_fasta_dict  # noqa: E402
from rockabye.msa import compute_offsets  # noqa: E402
from rockabye.refine import CutoffSet, apply_model, filter_midpoints  # noqa: E402

from compare_to_rockout import load_rockout_reads, load_reference_curves  # noqa: E402


def load_model(model_root: str):
    model_dir = os.path.join(model_root, "final_outputs", "model")
    msa = read_fasta_dict(os.path.join(model_dir, "complete_multiple_alignment_aa.fasta"))
    offsets, width = compute_offsets(msa)
    bs, idma, idaln, aln_grid = load_reference_curves(model_dir)
    return msa, offsets, width, bs, idma, idaln, aln_grid


def evaluate(project: str, model_root: str, label: str) -> dict:
    msa, offsets, width, bs, idma, idaln, aln_grid = load_model(model_root)

    read_files = sorted(
        glob.glob(
            os.path.join(project, "final_outputs", "reads", "complete_reads_read_length_*.txt")
        ),
        key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1)),
    )

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    skipped = 0
    for path in read_files:
        rl = int(re.search(r"_(\d+)\.txt$", path).group(1))
        if rl not in bs:
            continue
        # Only alignments whose target this model knows can be scored; the rest
        # are exactly what ROCkOut's filter drops.
        aln, _ = load_rockout_reads(path, set(msa))
        total_in_file = sum(1 for _ in open(path)) - 1
        skipped += total_in_file - len(aln)

        axes = Axes(
            bitscore=np.array([0.0]),
            pct_id=np.array([0.0]),
            pct_aln=aln_grid,
            ma_width=width,
        )
        mids = filter_midpoints(aln, offsets)
        rows = np.arange(len(aln))
        called = apply_model(aln, rows, mids, CutoffSet(bs[rl], idma[rl], idaln[rl]), axes)
        truth = aln.label == "Positive"
        counts["tp"] += int(np.sum(called & truth))
        counts["fp"] += int(np.sum(called & ~truth))
        counts["fn"] += int(np.sum(~called & truth))
        counts["tn"] += int(np.sum(~called & ~truth))

    d = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    f1 = 2 * counts["tp"] / d if d else 0.0
    total = sum(counts.values())
    fpr = counts["fp"] / max(1, counts["fp"] + counts["tn"])
    fnr = counts["fn"] / max(1, counts["fn"] + counts["tp"])
    print(
        f"  {label:28s} MA {len(msa):3d}x{width:<4d} "
        f"F1 {f1:.4f}  acc {(counts['tp'] + counts['tn']) / max(1, total):.4f}  "
        f"FPR {fpr * 100:5.2f}%  FNR {fnr * 100:5.2f}%  "
        f"TP {counts['tp']:,} FP {counts['fp']:,} FN {counts['fn']:,} TN {counts['tn']:,}"
        + (f"  (skipped {skipped:,} unknown-target)" if skipped else "")
    )
    return counts


def main() -> int:
    project = sys.argv[1]
    models = sys.argv[2:]
    print(f"\nscoring on cached labelled reads from {os.path.basename(os.path.normpath(project))}")
    evaluate(project, project, "ROCkOut (own model)")
    for m in models:
        evaluate(project, m, os.path.basename(os.path.normpath(m)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
