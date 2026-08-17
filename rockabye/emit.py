"""Write a ROCkOut-compatible model directory.

`rocker_filter.py` reads exactly six files and nothing else, so this module is the
entire compatibility surface. Two invariants are enforced here because violating
either produces a model that loads cleanly and then behaves wrongly:

1. `position_in_MA` must be the contiguous integers 0..width-1. The filter uses
   the value as a column index (`filter_matrix[readlen_index, ma_pos]`), so any
   gap or offset silently misaligns every threshold.
2. Every read length must repeat the full x grid. The filter zero-fills its
   matrix and interpolates across read-length rows only, never across columns, so
   an x value present at one read length but missing at another leaves a cutoff of
   0.0 -- which passes everything.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from .align import make_db
from .binning import Axes
from .fasta import write_fasta
from .refine import RefinedModel

MODEL_FILES = (
    "bitscore_vs_MA_pos.txt",
    "pct_id_vs_MA_pos.txt",
    "pct_id_vs_pct_aln.txt",
    "complete_multiple_alignment_aa.fasta",
)


def _write_curve(path: str, header: tuple, rows) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for read_length, x, y in rows:
            fh.write(f"{read_length}\t{x}\t{y}\n")


def _check_full_grid(model: RefinedModel, axes: Axes) -> None:
    for read_length, cut in model.cutoffs.items():
        if cut.bitscore_vs_ma.size != axes.ma_width:
            raise ValueError(
                f"read length {read_length}: bitscore curve has "
                f"{cut.bitscore_vs_ma.size} points but the alignment has "
                f"{axes.ma_width} columns"
            )
        if cut.pct_id_vs_ma.size != axes.ma_width:
            raise ValueError(
                f"read length {read_length}: identity/position curve has "
                f"{cut.pct_id_vs_ma.size} points but the alignment has "
                f"{axes.ma_width} columns"
            )
        if cut.pct_id_vs_aln.size != axes.pct_aln.size:
            raise ValueError(
                f"read length {read_length}: identity/alignment curve has "
                f"{cut.pct_id_vs_aln.size} points but the axis has "
                f"{axes.pct_aln.size} bins"
            )


def write_model(
    outdir: str,
    model: RefinedModel,
    proteins: Dict[str, str],
    msa: Dict[str, str],
    diamond: str,
    threads: int = 1,
    metadata: dict | None = None,
    log=print,
) -> str:
    axes = model.axes
    _check_full_grid(model, axes)

    final = os.path.join(outdir, "final_outputs")
    model_dir = os.path.join(final, "model")
    db_dir = os.path.join(final, "database")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    protein_path = os.path.join(db_dir, "positive_proteins_aa.fasta")
    write_fasta(protein_path, proteins)
    make_db(
        diamond,
        protein_path,
        os.path.join(db_dir, "positive_proteins_diamond_db"),
        threads=threads,
    )

    write_fasta(os.path.join(model_dir, "complete_multiple_alignment_aa.fasta"), msa)

    read_lengths = sorted(model.cutoffs)
    positions = np.arange(axes.ma_width, dtype=np.int32)

    _write_curve(
        os.path.join(model_dir, "bitscore_vs_MA_pos.txt"),
        ("read_length", "position_in_MA", "bitscore"),
        (
            (rl, int(p), float(model.cutoffs[rl].bitscore_vs_ma[p]))
            for rl in read_lengths
            for p in positions
        ),
    )
    _write_curve(
        os.path.join(model_dir, "pct_id_vs_MA_pos.txt"),
        ("read_length", "position_in_MA", "percent_id"),
        (
            (rl, int(p), float(model.cutoffs[rl].pct_id_vs_ma[p]))
            for rl in read_lengths
            for p in positions
        ),
    )
    _write_curve(
        os.path.join(model_dir, "pct_id_vs_pct_aln.txt"),
        ("read_length", "percent_aln", "percent_id"),
        (
            (rl, float(axes.pct_aln[i]), float(model.cutoffs[rl].pct_id_vs_aln[i]))
            for rl in read_lengths
            for i in range(axes.pct_aln.size)
        ),
    )

    _write_reports(final, model, metadata or {}, read_lengths, axes)
    log(f"  wrote model to {final}")
    return final


def _write_reports(
    final: str, model: RefinedModel, metadata: dict, read_lengths, axes: Axes
) -> None:
    perf_path = os.path.join(final, "cross_validation_performance.tsv")
    with open(perf_path, "w") as fh:
        fh.write(
            "read_length\tsplit\tn_train_alignments\tn_test_alignments\t"
            "tp\tfp\tfn\ttn\tf1\taccuracy\tfpr\tfnr\n"
        )
        for r in model.reports:
            fh.write(
                f"{r.read_length}\t{r.split}\t{r.n_train}\t{r.n_test}\t"
                f"{r.tp}\t{r.fp}\t{r.fn}\t{r.tn}\t"
                f"{r.f1:.6f}\t{r.accuracy:.6f}\t{r.fpr:.6f}\t{r.fnr:.6f}\n"
            )

    by_rl = {}
    for r in model.reports:
        by_rl.setdefault(r.read_length, []).append(r.f1)

    manifest = {
        "generator": "rockabye",
        "compatible_with": "ROCkOut rockout_main.py filter",
        "read_lengths": list(read_lengths),
        "multiple_alignment_columns": int(axes.ma_width),
        "percent_alignment_bins": int(axes.pct_aln.size),
        "mean_cv_f1_by_read_length": {
            str(rl): round(float(np.mean(v)), 6) for rl, v in sorted(by_rl.items())
        },
        **metadata,
    }
    with open(os.path.join(final, "rockabye_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    with open(os.path.join(final, "readme.txt"), "w") as fh:
        fh.write(
            "This directory was produced by rockabye and is laid out for ROCkOut's\n"
            "filter step. Use it as the -d argument:\n\n"
            "    python3 rockout_main.py align -d <this project> -f <filter dir> ...\n"
            "    python3 rockout_main.py filter -d <this project> -f <filter dir>\n\n"
            "model/    three cutoff tables plus the positive protein alignment\n"
            "database/ positive proteins and their DIAMOND database\n\n"
            "cross_validation_performance.tsv holds held-out performance for each of\n"
            "the five partitions per read length. The shipped cutoffs are the\n"
            "F1-weighted average across those partitions, so these numbers describe\n"
            "the ensemble members, not the shipped model itself.\n"
        )
