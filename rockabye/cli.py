"""Command line interface."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict

import numpy as np

from . import __version__
from .align import (
    Alignments,
    blastx,
    find_target_regions,
    load_alignments,
    make_db,
    require_diamond,
)
from .binning import build_axes
from .emit import MODEL_FILES, write_model
from .fasta import write_fasta
from .msa import align_proteins, compute_offsets, validate_msa
from .project import load_inputs
from .refine import refine
from .simulate import (
    DEFAULT_READ_LENGTHS,
    SimConfig,
    get_simulator,
    simulate_all,
)
from .youden import BIAS_CHOICES


def _log(verbose: bool):
    start = time.time()

    def log(msg: str = ""):
        if verbose:
            print(f"[{time.time() - start:7.1f}s] {msg}", flush=True)

    return log


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rockabye",
        description=(
            "Build a ROCkOut-compatible read filtering model from labelled "
            "positive and negative gene sequences."
        ),
    )
    p.add_argument("--version", action="version", version=f"rockabye {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a model from labelled directories")
    b.add_argument("-p", "--positives", required=True, help="directory of positive gene FASTA")
    b.add_argument("-n", "--negatives", required=True, help="directory of confounder gene FASTA")
    b.add_argument("-o", "--output", required=True, help="project directory to create")
    b.add_argument(
        "--background", default=None,
        help=(
            "optional directory of genome/contig FASTA. Reads simulated from these "
            "become confounders. Real ROCkOut models draw most of their "
            "discriminative power from genomic background rather than from curated "
            "confounder genes, so supplying this markedly reduces false positives"
        ),
    )
    b.add_argument(
        "--background-coverage", type=float, default=10.0,
        help=(
            "fold coverage for background contigs. 10x matches ROCkOut; the run is "
            "expensive but a model is built once (default: %(default)s)"
        ),
    )
    b.add_argument(
        "--background-target-identity", type=float, default=90.0,
        help=(
            "percent identity at which a region of a background contig counts as a "
            "copy of a positive gene. Reads overlapping such a region are labelled "
            "positive, exactly as ROCkOut does from genome annotation. Lower it if "
            "your background genomes are divergent from your positive sequences "
            "(default: %(default)s)"
        ),
    )
    b.add_argument("-t", "--threads", type=int, default=1)
    b.add_argument(
        "--read-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_READ_LENGTHS),
        help="nominal read lengths to simulate (default: %(default)s)",
    )
    b.add_argument("--coverage", type=float, default=20.0)
    b.add_argument("--snp-rate", type=float, default=0.01)
    b.add_argument(
        "--insertion-rate", type=float, default=None,
        help="default: --snp-rate / 19, matching ROCkOut",
    )
    b.add_argument(
        "--deletion-rate", type=float, default=None,
        help="default: --snp-rate / 19, matching ROCkOut",
    )
    b.add_argument(
        "--length-jitter", type=float, default=0.10,
        help="read length varies by +/- this fraction",
    )
    b.add_argument("--aligner", choices=("auto", "muscle", "mafft"), default="auto")
    b.add_argument("--diamond", default=None, help="path to the diamond binary")
    b.add_argument(
        "--sensitivity",
        choices=("fast", "default", "sensitive", "more-sensitive", "very-sensitive", "ultra-sensitive"),
        default="sensitive",
        help="DIAMOND sensitivity mode (ROCkOut uses --sensitive)",
    )
    b.add_argument(
        "--evalue", type=float, default=0.001,
        help="DIAMOND maximum e-value (default matches ROCkOut: %(default)s)",
    )
    b.add_argument("--splits", type=int, default=5, help="cross-validation partitions")
    b.add_argument(
        "--train-fraction", type=float, default=0.75,
        help=(
            "fraction of source sequences used for training in each partition. "
            "0.75 matches ROCkOut: its default 'sequence_outgroups' splitter "
            "hardcodes 0.75 and discards the 0.4 the refiner passes it "
            "(default: %(default)s)"
        ),
    )
    b.add_argument(
        "--cutoff-bias", choices=sorted(BIAS_CHOICES), default="balanced",
        help="how to break ties among equally-good thresholds",
    )
    b.add_argument(
        "--compat", choices=("rockout", "hardened"), default="hardened",
        help=(
            "hardened (default) anchors the percent-alignment axis at zero, closing "
            "a lookup wrap in ROCkOut's filter that lets very short alignments be "
            "judged by the most permissive cutoff; rockout reproduces ROCkOut's "
            "binning bit-for-bit instead"
        ),
    )
    b.add_argument("--seed", type=int, default=1337)
    b.add_argument("--keep-intermediates", action="store_true")
    b.add_argument("-q", "--quiet", action="store_true")

    v = sub.add_parser(
        "validate", help="check that a model directory satisfies ROCkOut's expectations"
    )
    v.add_argument("model_dir", help="project directory containing final_outputs/")

    return p


def cmd_build(args) -> int:
    log = _log(not args.quiet)

    # Fail before doing any expensive work.
    diamond = require_diamond(args.diamond)
    log(f"using DIAMOND at {diamond}")
    sim_cfg = SimConfig(
        read_lengths=tuple(args.read_lengths),
        coverage=args.coverage,
        length_jitter=args.length_jitter,
        snp_rate=args.snp_rate,
        insertion_rate=args.insertion_rate,
        deletion_rate=args.deletion_rate,
        seed=args.seed,
        background_coverage=args.background_coverage,
    )
    # Fail now rather than after minutes of alignment work.
    get_simulator(sim_cfg)

    outdir = os.path.abspath(args.output)
    work = os.path.join(outdir, "intermediates")
    os.makedirs(work, exist_ok=True)

    log("reading labelled inputs")
    inputs = load_inputs(args.positives, args.negatives, args.background)
    log(
        f"  {inputs.n_positive} positive and {inputs.n_negative} confounder sequences; "
        f"proteins {inputs.aa_source}"
    )
    if inputs.n_background:
        log(f"  {inputs.n_background} background contigs at {args.background_coverage}x")
    else:
        log(
            "  no --background supplied; confounders come only from the negatives "
            "directory. See the README on genomic background and false positives."
        )
    if inputs.n_positive < args.splits:
        log(
            f"  note: only {inputs.n_positive} positive sequences for {args.splits} "
            "splits; held-out partitions will be very small"
        )

    log("aligning positive proteins")
    if inputs.positive_msa is not None:
        msa = inputs.positive_msa
        validate_msa(msa, inputs.positive_aa)
        log(f"  using supplied alignment ({len(next(iter(msa.values())))} columns)")
    else:
        msa = align_proteins(inputs.positive_aa, work, args.aligner, args.threads, log)
    offsets, ma_width = compute_offsets(msa)

    log("building DIAMOND database of positive proteins")
    protein_path = os.path.join(work, "positive_proteins_aa.fasta")
    write_fasta(protein_path, inputs.positive_aa)
    db = make_db(diamond, protein_path, os.path.join(work, "positives"), args.threads)

    target_regions = None
    if inputs.background_nt:
        log("locating positive-gene copies inside the background contigs")
        bg_path = os.path.join(work, "background_contigs.fasta")
        write_fasta(bg_path, inputs.background_nt)
        target_regions = find_target_regions(
            diamond, bg_path, db,
            min_identity=args.background_target_identity,
            threads=args.threads,
            workdir=work,
        )
        n_regions = sum(len(v) for v in target_regions.values())
        log(
            f"  found {n_regions} target region(s) across "
            f"{len(target_regions)} contig(s); reads overlapping them will be "
            "labelled positive rather than trained against"
        )
        if not target_regions:
            log(
                "  none found -- if your background genomes do contain the "
                "target, lower --background-target-identity"
            )

    log("simulating reads")
    read_files = simulate_all(
        inputs, sim_cfg, os.path.join(work, "reads"), log, target_regions
    )

    log("aligning reads (translated search)")
    rng = np.random.default_rng(args.seed)
    valid_targets = set(inputs.positive_aa)
    datasets: Dict[int, Alignments] = {}
    for rl, reads_path in sorted(read_files.items()):
        tsv = os.path.join(work, f"alignments_len_{rl}.tsv")
        blastx(
            diamond, reads_path, db, tsv,
            threads=args.threads, sensitivity=args.sensitivity, evalue=args.evalue,
        )
        aln = load_alignments(tsv, valid_targets, rng)
        # Simulation already resolved every background read by coordinate overlap:
        # those hitting a target gene copy were labelled positive there, so whatever
        # still carries the Background tag is a confounder.
        aln.label = np.where(aln.label == "Background", "Negative", aln.label)
        n_pos = int(np.sum(aln.label == "Positive"))
        datasets[rl] = aln
        log(
            f"  read length {rl}: {len(aln):,} best-hit alignments "
            f"({n_pos:,} positive, {len(aln) - n_pos:,} confounder)"
        )
        if n_pos == 0 or n_pos == len(aln):
            raise ValueError(
                f"read length {rl} produced only one class of alignment. "
                "Thresholding needs both target and confounder reads to hit the "
                "positive proteins; your negatives may be too divergent to align."
            )

    log("fitting cross-validated cutoffs")
    axes = build_axes(datasets, ma_width, compat=args.compat)
    model = refine(
        datasets, offsets, axes,
        n_splits=args.splits,
        train_fraction=args.train_fraction,
        bias=args.cutoff_bias,
        seed=args.seed,
        compat=args.compat,
        log=log,
    )

    log("writing model")
    metadata = {
        "rockabye_version": __version__,
        "simulator": "bbmap",
        "coverage": args.coverage,
        "cutoff_bias": args.cutoff_bias,
        "compat": args.compat,
        "background_target_identity": (
            args.background_target_identity if inputs.n_background else None
        ),
        "background_labelling": (
            "coordinates" if target_regions is not None
            else ("identity-screen" if inputs.n_background else None)
        ),
        "splits": args.splits,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "protein_source": inputs.aa_source,
        "alignment_source": inputs.msa_source,
        "n_positive_sequences": inputs.n_positive,
        "n_negative_sequences": inputs.n_negative,
        "n_background_contigs": inputs.n_background,
        "background_coverage": args.background_coverage if inputs.n_background else None,
    }
    final = write_model(
        outdir, model, inputs.positive_aa, msa, diamond, args.threads, metadata, log
    )

    if not args.keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)
        log("  removed intermediates (pass --keep-intermediates to retain them)")

    mean_f1 = float(np.mean([r.f1 for r in model.reports])) if model.reports else 0.0
    print(f"\nModel written to {final}")
    print(f"Mean held-out F1 across all partitions: {mean_f1:.4f}")
    print("\nUse it with ROCkOut:")
    print(f"  python3 rockout_main.py align -d {outdir} -f <filter_dir> -1 <reads.fasta>")
    print(f"  python3 rockout_main.py filter -d {outdir} -f <filter_dir>")
    return 0


def cmd_validate(args) -> int:
    root = os.path.abspath(args.model_dir)
    final = os.path.join(root, "final_outputs")
    model_dir = os.path.join(final, "model")
    db_dir = os.path.join(final, "database")
    problems = []

    for name in MODEL_FILES:
        if not os.path.exists(os.path.join(model_dir, name)):
            problems.append(f"missing {os.path.join('final_outputs/model', name)}")
    for name in ("positive_proteins_aa.fasta", "positive_proteins_diamond_db.dmnd"):
        if not os.path.exists(os.path.join(db_dir, name)):
            problems.append(f"missing {os.path.join('final_outputs/database', name)}")

    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1

    from .fasta import read_fasta_dict

    msa = read_fasta_dict(os.path.join(model_dir, "complete_multiple_alignment_aa.fasta"))
    widths = {len(s) for s in msa.values()}
    if len(widths) != 1:
        print(f"FAIL  alignment rows have differing lengths: {sorted(widths)[:5]}")
        return 1
    width = widths.pop()
    print(f"OK    alignment: {len(msa)} sequences x {width} columns")

    proteins = read_fasta_dict(os.path.join(db_dir, "positive_proteins_aa.fasta"))
    if set(proteins) != set(msa):
        print("FAIL  protein FASTA and alignment contain different sequence IDs; "
              "the filter drops alignments whose target is absent from the MSA")
        return 1
    print(f"OK    protein database and alignment agree on {len(proteins)} IDs")

    ok = True
    read_length_sets = {}
    for fname, xname, expect_contiguous in (
        ("bitscore_vs_MA_pos.txt", "position_in_MA", True),
        ("pct_id_vs_MA_pos.txt", "position_in_MA", True),
        ("pct_id_vs_pct_aln.txt", "percent_aln", False),
    ):
        good, read_lengths = _validate_curve(
            os.path.join(model_dir, fname), xname, expect_contiguous, width
        )
        ok &= good
        read_length_sets[fname] = read_lengths

    # ROCkOut's import_filter assigns self.min_readlen/max_readlen on every call,
    # so after load_filters they hold whatever the *last* file specified -- and
    # that single row offset then indexes all three matrices. Files that disagree
    # on their read lengths therefore read thresholds off the wrong rows.
    distinct_ranges = {
        f: (min(v), max(v)) for f, v in read_length_sets.items() if v
    }
    if len(set(distinct_ranges.values())) > 1:
        print(
            "FAIL  cutoff tables disagree on their read-length range "
            f"{distinct_ranges}. The filter derives one row offset from the last "
            "table it loads and applies it to all three, so every threshold would "
            "be read off the wrong row."
        )
        ok = False
    elif len(set(map(tuple, (sorted(v) for v in read_length_sets.values())))) > 1:
        print(
            "WARN  cutoff tables share a read-length range but not the same set of "
            f"read lengths {({f: sorted(v) for f, v in read_length_sets.items()})}. "
            "Interpolation will fill the gaps, but the tables were probably not "
            "generated together."
        )

    return 0 if ok else 1


def _validate_curve(path: str, xname: str, contiguous: bool, ma_width: int):
    """-> (ok, set of read lengths present)."""
    per_rl: Dict[int, list] = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            if not line.strip():
                continue
            a, b, c = line.rstrip("\n").split("\t")
            per_rl.setdefault(int(a), []).append((float(b), float(c)))

    label = os.path.basename(path)
    if not per_rl:
        print(f"FAIL  {label}: no data rows")
        return False, set()

    read_lengths = set(per_rl)
    grids = {rl: tuple(sorted(x for x, _ in rows)) for rl, rows in per_rl.items()}
    distinct = set(grids.values())
    if len(distinct) != 1:
        sizes = {rl: len(g) for rl, g in grids.items()}
        print(
            f"FAIL  {label}: read lengths do not share one x grid ({sizes}). "
            "The filter interpolates across read lengths only, so missing x values "
            "stay at a cutoff of 0.0 and pass everything."
        )
        return False, read_lengths

    grid = distinct.pop()
    if contiguous:
        expected = tuple(float(i) for i in range(ma_width))
        if grid != expected:
            print(
                f"FAIL  {label}: {xname} must be the contiguous integers "
                f"0..{ma_width - 1} because the filter uses it as a column index; "
                f"got {len(grid)} values from {grid[0]:g} to {grid[-1]:g}"
            )
            return False, read_lengths

    print(
        f"OK    {label}: {len(per_rl)} read lengths x {len(grid)} {xname} values, "
        "grid consistent"
    )
    return True, read_lengths


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return cmd_build(args)
        if args.command == "validate":
            return cmd_validate(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
