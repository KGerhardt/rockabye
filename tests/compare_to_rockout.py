"""Compare rockabye's thresholding against a real ROCkOut project's own output.

A completed ROCkOut project caches everything needed to remove simulation and
alignment noise from the comparison:

    final_outputs/reads/complete_reads_read_length_*.txt   the exact fitted data,
                                                           including its own bin
                                                           assignments
    final_outputs/model/complete_multiple_alignment_aa.fasta   the exact alignment
    final_outputs/model/*.txt                               the reference curves

So we feed ROCkOut's own reads through rockabye's binning and thresholding and ask
how close the resulting curves are. Any difference is attributable to the
algorithm, not to a different random draw of reads.

Usage:
    python3 tests/compare_to_rockout.py <project_dir> [--cv]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.align import Alignments  # noqa: E402
from rockabye.binning import MASpans, build_axes, nearest_index  # noqa: E402
from rockabye.fasta import read_fasta_dict  # noqa: E402
from rockabye.msa import compute_offsets  # noqa: E402
from rockabye.refine import (  # noqa: E402
    apply_model,
    fit_cutoffs,
    filter_midpoints,
    score,
    CutoffSet,
)

COLS = [
    "read_id", "target", "pct_id", "aln_len", "alignment_min_pos",
    "alignment_max_pos", "bitscore", "query_length", "pct_overlap", "pct_aln",
    "classifier", "annotations", "origin_genome", "bitscore_bin", "id_bin",
    "aln_indices", "MA_median_mapping_pos",
]


def load_rockout_reads(path: str, valid_targets: set) -> tuple[Alignments, dict]:
    """Parse a cached complete_reads table into rockabye's Alignments form."""
    # Column sets vary between ROCkOut vintages (some include e_value), so index
    # by name from the actual header rather than by position.
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in COLS if c not in idx]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}\nheader: {header}")

        cols: Dict[str, list] = {name: [] for name in COLS}
        width = len(header)
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != width:
                continue
            if f[idx["target"]] not in valid_targets:
                continue
            for name in COLS:
                cols[name].append(f[idx[name]])

    n = len(cols["read_id"])
    if n == 0:
        raise ValueError(f"no usable rows in {path}")

    aln = Alignments(
        read_id=np.array(cols["read_id"]),
        target=np.array(cols["target"]),
        pct_id=np.array(cols["pct_id"], dtype=np.float64),
        aln_len=np.array(cols["aln_len"], dtype=np.int32),
        sstart=np.array(cols["alignment_min_pos"], dtype=np.int32),
        send=np.array(cols["alignment_max_pos"], dtype=np.int32),
        bitscore=np.array(cols["bitscore"], dtype=np.float64),
        qlen=np.array(cols["query_length"], dtype=np.int32),
        pct_aln=np.array(cols["pct_aln"], dtype=np.float64),
        # refiner.py:603 -- only "Positive" is a target. "Negative" (the same gene
        # in a confounder genome) and "Non_Target" (genomic chaff from either) are
        # both confounders.
        label=np.where(np.array(cols["classifier"]) == "Positive", "Positive", "Negative"),
        source=np.array(cols["origin_genome"]),
    )
    cached = {
        "bitscore_bin": np.array(cols["bitscore_bin"], dtype=np.float64),
        "id_bin": np.array(cols["id_bin"], dtype=np.float64),
        "aln_indices": np.array(cols["aln_indices"], dtype=np.int32),
        "ma_median": np.array(cols["MA_median_mapping_pos"], dtype=np.float64),
    }
    return aln, cached


def load_reference_curves(model_dir: str) -> tuple[dict, dict, dict, np.ndarray]:
    def read_curve(fname):
        per_rl: Dict[int, list] = {}
        with open(os.path.join(model_dir, fname)) as fh:
            fh.readline()
            for line in fh:
                if not line.strip():
                    continue
                a, b, c = line.rstrip("\n").split("\t")
                per_rl.setdefault(int(a), []).append((float(b), float(c)))
        out = {}
        grid = None
        for rl, rows in per_rl.items():
            rows.sort()
            grid = np.array([x for x, _ in rows])
            out[rl] = np.array([y for _, y in rows])
        return out, grid

    bs, _ = read_curve("bitscore_vs_MA_pos.txt")
    idma, _ = read_curve("pct_id_vs_MA_pos.txt")
    idaln, aln_grid = read_curve("pct_id_vs_pct_aln.txt")
    return bs, idma, idaln, aln_grid


def pct(a, b) -> str:
    return f"{100.0 * a / max(1, b):6.2f}%"


def summarise_curve(name, mine, theirs, sentinel_aware=False):
    mine = np.asarray(mine, dtype=np.float64)
    theirs = np.asarray(theirs, dtype=np.float64)
    if mine.shape != theirs.shape:
        print(f"    {name:22s} SHAPE MISMATCH mine={mine.shape} theirs={theirs.shape}")
        return None

    exact = int(np.sum(np.isclose(mine, theirs, atol=1e-6)))
    diff = mine - theirs
    real = np.ones(mine.size, dtype=bool)
    if sentinel_aware:
        real = (mine < 100.0) & (theirs < 100.0)
        sent_agree = int(np.sum((mine >= 100.0) == (theirs >= 100.0)))
    finite = diff[real]
    med = float(np.median(np.abs(finite))) if finite.size else 0.0
    p95 = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.0
    corr = (
        float(np.corrcoef(mine[real], theirs[real])[0, 1])
        if finite.size > 2 and np.std(mine[real]) > 0 and np.std(theirs[real]) > 0
        else float("nan")
    )
    extra = ""
    if sentinel_aware:
        extra = f"  sentinel agree {pct(sent_agree, mine.size)}"
    print(
        f"    {name:22s} exact {pct(exact, mine.size)}  "
        f"median|d| {med:7.3f}  p95|d| {p95:7.3f}  r {corr:6.3f}{extra}"
    )
    return med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--cv", action="store_true", help="also fit with 5-fold CV")
    ap.add_argument("--compat", default="rockout", choices=("rockout", "hardened"))
    ap.add_argument("--bias", default="balanced")
    args = ap.parse_args()

    model_dir = os.path.join(args.project, "final_outputs", "model")
    msa = read_fasta_dict(os.path.join(model_dir, "complete_multiple_alignment_aa.fasta"))
    offsets, ma_width = compute_offsets(msa)
    valid = set(msa)

    read_files = sorted(
        glob.glob(
            os.path.join(args.project, "final_outputs", "reads", "complete_reads_read_length_*.txt")
        ),
        key=lambda p: int(re.search(r"_(\d+)\.txt$", p).group(1)),
    )
    if not read_files:
        print("no cached read tables in this project")
        return 2

    datasets: Dict[int, Alignments] = {}
    cached: Dict[int, dict] = {}
    for path in read_files:
        rl = int(re.search(r"_(\d+)\.txt$", path).group(1))
        datasets[rl], cached[rl] = load_rockout_reads(path, valid)

    total = sum(len(a) for a in datasets.values())
    print(f"\n=== {os.path.basename(os.path.normpath(args.project))} ===")
    print(
        f"  MSA {len(msa)} sequences x {ma_width} columns; "
        f"{total:,} cached alignments across read lengths {sorted(datasets)}"
    )

    axes = build_axes(datasets, ma_width, compat=args.compat)
    print(
        f"  reconstructed axes: bitscore {axes.bitscore.size} bins "
        f"[{axes.bitscore[0]:.1f}, {axes.bitscore[-1]:.1f}], "
        f"pct_id {axes.pct_id.size} bins [{axes.pct_id[0]:.2f}, {axes.pct_id[-1]:.2f}], "
        f"pct_aln {axes.pct_aln.size} bins [{axes.pct_aln[0]:.2f}, {axes.pct_aln[-1]:.2f}]"
    )

    # --- A: do our axes reproduce ROCkOut's own cached bin assignments? ---
    print("\n  [A] binning agreement with ROCkOut's cached bins")
    for rl in sorted(datasets):
        aln, c = datasets[rl], cached[rl]
        my_bs = axes.bitscore[nearest_index(axes.bitscore, aln.bitscore)]
        my_id = axes.pct_id[nearest_index(axes.pct_id, aln.pct_id)]
        my_ax = nearest_index(axes.pct_aln, np.minimum(aln.pct_aln, 100.0))
        n = len(aln)
        print(
            f"    rl {rl:4d}  bitscore_bin {pct(int(np.sum(np.isclose(my_bs, c['bitscore_bin']))), n)}"
            f"  id_bin {pct(int(np.sum(np.isclose(my_id, c['id_bin'], atol=0.011))), n)}"
            f"  aln_index {pct(int(np.sum(my_ax == c['aln_indices'])), n)}"
        )

    # --- B: does our MA mapping match theirs? ---
    print("\n  [B] multiple-alignment position mapping")
    for rl in sorted(datasets):
        aln, c = datasets[rl], cached[rl]
        spans = MASpans(aln, offsets)
        medians = np.empty(len(aln))
        for i in range(len(aln)):
            s, e = spans.indptr[i], spans.indptr[i + 1]
            medians[i] = np.median(spans.data[s:e]) if e > s else -1
        agree = int(np.sum(np.isclose(medians, c["ma_median"])))
        print(f"    rl {rl:4d}  median MA position matches {pct(agree, len(aln))}")

    # --- C: curve agreement ---
    ref_bs, ref_idma, ref_idaln, ref_aln_grid = load_reference_curves(model_dir)
    if ref_aln_grid is not None and ref_aln_grid.size == axes.pct_aln.size:
        max_grid_diff = float(np.max(np.abs(ref_aln_grid - axes.pct_aln)))
        print(f"\n  pct_aln x-grid max difference from reference: {max_grid_diff:.4f}")
    elif ref_aln_grid is not None:
        print(
            f"\n  pct_aln x-grid size differs: mine {axes.pct_aln.size}, "
            f"reference {ref_aln_grid.size}"
        )

    print("\n  [C] cutoff curves, fitted on ROCkOut's own reads (single fit, all data)")
    fitted: Dict[int, CutoffSet] = {}
    for rl in sorted(datasets):
        aln = datasets[rl]
        spans = MASpans(aln, offsets)
        rows = np.arange(len(aln))
        cut = fit_cutoffs(aln, spans, rows, axes, args.bias, args.compat)
        fitted[rl] = cut
        print(f"    read length {rl}:")
        if rl in ref_bs:
            summarise_curve("bitscore vs MA pos", cut.bitscore_vs_ma, ref_bs[rl])
            summarise_curve("pct_id vs MA pos", cut.pct_id_vs_ma, ref_idma[rl])
            summarise_curve(
                "pct_id vs pct_aln", cut.pct_id_vs_aln, ref_idaln[rl], sentinel_aware=True
            )
        else:
            print(f"      no reference curve for read length {rl}")

    # --- D: do the two curve sets make the same calls on the same reads? ---
    print("\n  [D] classification agreement on ROCkOut's own reads")
    tot_agree = tot_n = 0
    my_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    their_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for rl in sorted(datasets):
        if rl not in ref_bs:
            continue
        aln = datasets[rl]
        mids = filter_midpoints(aln, offsets)
        rows = np.arange(len(aln))
        ref_cut = CutoffSet(ref_bs[rl], ref_idma[rl], ref_idaln[rl])
        mine = apply_model(aln, rows, mids, fitted[rl], axes)
        theirs = apply_model(aln, rows, mids, ref_cut, axes)
        agree = int(np.sum(mine == theirs))
        tot_agree += agree
        tot_n += len(aln)
        for k, v in score(aln, rows, mine).items():
            my_counts[k] += v
        for k, v in score(aln, rows, theirs).items():
            their_counts[k] += v
        print(f"    rl {rl:4d}  same verdict on {pct(agree, len(aln))} of {len(aln):,} reads")

    def f1(c):
        d = 2 * c["tp"] + c["fp"] + c["fn"]
        return 2 * c["tp"] / d if d else 0.0

    print(f"\n  overall verdict agreement: {pct(tot_agree, tot_n)} of {tot_n:,} reads")
    print(f"    rockabye curves F1 {f1(my_counts):.4f}  {my_counts}")
    print(f"    ROCkOut  curves F1 {f1(their_counts):.4f}  {their_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
