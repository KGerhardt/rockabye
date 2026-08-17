"""Cross-validated model fitting.

Five partitions per read length, split *by source sequence* rather than by read,
so a protein never appears in both train and test. Each partition yields a full
three-classifier model; the shipped cutoffs are the F1-weighted average across
partitions, exactly as ROCkOut combines them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .align import Alignments
from .binning import (
    ALN_WINDOW_SIZE,
    MA_WINDOW_SIZE,
    PCT_ALN_RESOLUTION,
    Axes,
    MASpans,
    aln_histograms,
    ma_histograms,
)
from .youden import youden_cutoff_indices

REJECT_ALL = 101.0
MIN_MEANINGFUL_PCT_ID = 25.0


@dataclass
class CutoffSet:
    """One trained model: three cutoff curves for a single read length."""

    bitscore_vs_ma: np.ndarray
    pct_id_vs_ma: np.ndarray
    pct_id_vs_aln: np.ndarray


@dataclass
class SplitReport:
    split: int
    read_length: int
    n_train: int
    n_test: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / denom) if denom else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return (self.fp / denom) if denom else 0.0

    @property
    def fnr(self) -> float:
        denom = self.fn + self.tp
        return (self.fn / denom) if denom else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return ((self.tp + self.tn) / total) if total else 0.0


@dataclass
class RefinedModel:
    cutoffs: Dict[int, CutoffSet] = field(default_factory=dict)
    reports: List[SplitReport] = field(default_factory=list)
    axes: Optional[Axes] = None


def partition_sources(
    sources: Sequence[str], train_fraction: float, rng: np.random.Generator
) -> tuple[set, set]:
    uniq = np.array(sorted(set(sources)))
    if uniq.size < 2:
        # Cannot hold anything out; train and test on the same lone sequence.
        return set(uniq.tolist()), set(uniq.tolist())
    n_train = max(1, min(uniq.size - 1, int(round(uniq.size * train_fraction))))
    perm = rng.permutation(uniq.size)
    train = set(uniq[perm[:n_train]].tolist())
    test = set(uniq[perm[n_train:]].tolist())
    return train, test


def split_rows(
    aln: Alignments, train_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split reads by source sequence, positives and confounders independently."""
    is_pos = aln.label == "Positive"
    pos_train, pos_test = partition_sources(aln.source[is_pos], train_fraction, rng)
    neg_train, neg_test = partition_sources(aln.source[~is_pos], train_fraction, rng)

    train_ids = pos_train | neg_train
    test_ids = pos_test | neg_test
    in_train = np.array([s in train_ids for s in aln.source])
    in_test = np.array([s in test_ids for s in aln.source])
    return np.flatnonzero(in_train), np.flatnonzero(in_test)


def fit_cutoffs(
    aln: Alignments,
    spans: MASpans,
    rows: np.ndarray,
    axes: Axes,
    bias: str,
    compat: str = "hardened",
) -> CutoffSet:
    half_ma = MA_WINDOW_SIZE // 2
    half_aln = max(1, int(ALN_WINDOW_SIZE / PCT_ALN_RESOLUTION))

    # The two MA-position classifiers bin reads differently; see ma_histograms.
    bs_tgt, bs_con = ma_histograms(
        aln, spans, rows, axes, aln.bitscore, axes.bitscore, mode="median"
    )
    bs_idx = youden_cutoff_indices(bs_tgt, bs_con, half_ma, bias)
    bitscore_vs_ma = axes.bitscore_desc[bs_idx]

    id_tgt, id_con = ma_histograms(
        aln, spans, rows, axes, aln.pct_id, axes.pct_id, mode="span"
    )
    id_idx = youden_cutoff_indices(id_tgt, id_con, half_ma, bias)
    pct_id_vs_ma = axes.pct_id_desc[id_idx]

    aln_tgt, aln_con = aln_histograms(aln, rows, axes)
    aln_idx = youden_cutoff_indices(aln_tgt, aln_con, half_aln, bias)
    pct_id_vs_aln = axes.pct_id_desc[aln_idx].astype(np.float64)

    # A percent-identity threshold below the twilight zone is not a threshold at
    # all; ROCkOut promotes those bins to the reject-everything sentinel.
    pct_id_vs_aln[pct_id_vs_aln < MIN_MEANINGFUL_PCT_ID] = REJECT_ALL

    if compat == "hardened":
        # Bins with no positive training reads must reject rather than fall through
        # to "accept everything". Only meaningful alongside the zero-anchored
        # percent-alignment axis, which is what creates those empty low bins.
        no_target = aln_tgt[-1, :] == 0
        pct_id_vs_aln[no_target] = REJECT_ALL

    return CutoffSet(
        bitscore_vs_ma=bitscore_vs_ma.astype(np.float64),
        pct_id_vs_ma=pct_id_vs_ma.astype(np.float64),
        pct_id_vs_aln=pct_id_vs_aln,
    )


def filter_midpoints(aln: Alignments, offsets: Dict[str, np.ndarray]) -> np.ndarray:
    """MA column each read is scored at, replicating `rocker_filter.filter_reads`.

    Note the asymmetry with training, which is inherited from ROCkOut: fitting
    spreads a read across every column it covers, but filtering evaluates it at a
    single midpoint.
    """
    out = np.empty(len(aln), dtype=np.int32)
    for i, (tgt, s, e) in enumerate(zip(aln.target, aln.sstart, aln.send)):
        mid = int((s + e - 1) / 2)
        off = offsets[tgt]
        mid = min(max(mid, 0), off.size - 1)
        out[i] = off[mid] + mid
    return out


def apply_model(
    aln: Alignments,
    rows: np.ndarray,
    midpoints: np.ndarray,
    cutoffs: CutoffSet,
    axes: Axes,
) -> np.ndarray:
    """2-of-3 majority vote, mirroring the deployed filter."""
    ma_pos = np.clip(midpoints[rows], 0, axes.ma_width - 1)

    aln_idx = np.searchsorted(axes.pct_aln, aln.pct_aln[rows], side="left") - 1
    aln_idx = np.clip(aln_idx, 0, axes.pct_aln.size - 1)

    passes_bs = aln.bitscore[rows] >= cutoffs.bitscore_vs_ma[ma_pos]
    passes_id = aln.pct_id[rows] >= cutoffs.pct_id_vs_ma[ma_pos]
    passes_aln = aln.pct_id[rows] >= cutoffs.pct_id_vs_aln[aln_idx]

    votes = passes_bs.astype(np.int8) + passes_id + passes_aln
    return votes >= 2


def score(aln: Alignments, rows: np.ndarray, called_positive: np.ndarray) -> dict:
    truth = aln.label[rows] == "Positive"
    return {
        "tp": int(np.sum(called_positive & truth)),
        "fp": int(np.sum(called_positive & ~truth)),
        "fn": int(np.sum(~called_positive & truth)),
        "tn": int(np.sum(~called_positive & ~truth)),
    }


def combine(
    sets: List[CutoffSet], weights: np.ndarray
) -> CutoffSet:
    """F1-weighted average, with sentinel-aware handling for the identity curve."""
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    w = weights / weights.sum()

    bs = np.average([s.bitscore_vs_ma for s in sets], axis=0, weights=w)
    idma = np.average([s.pct_id_vs_ma for s in sets], axis=0, weights=w)

    # Averaging 101.0 with real thresholds would invent a meaningless in-between
    # value, so the sentinel is decided by weighted majority and only the
    # non-sentinel splits contribute to the averaged bins.
    stack = np.array([s.pct_id_vs_aln for s in sets])
    is_sentinel = stack >= 100.0
    sentinel_weight = (is_sentinel * w[:, None]).sum(axis=0)

    real_weight = ((~is_sentinel) * w[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = (np.where(is_sentinel, 0.0, stack) * w[:, None]).sum(axis=0) / real_weight
    idaln = np.where(sentinel_weight > 0.5, REJECT_ALL, averaged)
    idaln = np.where(np.isfinite(idaln), idaln, REJECT_ALL)

    return CutoffSet(bitscore_vs_ma=bs, pct_id_vs_ma=idma, pct_id_vs_aln=idaln)


def refine(
    datasets: Dict[int, Alignments],
    offsets: Dict[str, np.ndarray],
    axes: Axes,
    n_splits: int = 5,
    train_fraction: float = 0.75,
    bias: str = "balanced",
    seed: int = 1337,
    compat: str = "hardened",
    log=print,
) -> RefinedModel:
    model = RefinedModel(axes=axes)
    rng = np.random.default_rng(seed)

    for read_length in sorted(datasets):
        aln = datasets[read_length]
        spans = MASpans(aln, offsets)
        midpoints = filter_midpoints(aln, offsets)

        fitted: List[CutoffSet] = []
        weights: List[float] = []

        for split in range(1, n_splits + 1):
            train_rows, test_rows = split_rows(aln, train_fraction, rng)
            if train_rows.size == 0 or test_rows.size == 0:
                log(f"  [rl {read_length}] split {split}: empty partition, skipped")
                continue

            cutoffs = fit_cutoffs(aln, spans, train_rows, axes, bias, compat)
            called = apply_model(aln, test_rows, midpoints, cutoffs, axes)
            counts = score(aln, test_rows, called)

            report = SplitReport(
                split=split,
                read_length=read_length,
                n_train=int(train_rows.size),
                n_test=int(test_rows.size),
                **counts,
            )
            model.reports.append(report)
            fitted.append(cutoffs)
            weights.append(report.f1)
            log(
                f"  [rl {read_length}] split {split}: "
                f"F1={report.f1:.4f} acc={report.accuracy:.4f} "
                f"FPR={report.fpr * 100:.2f}% FNR={report.fnr * 100:.2f}%"
            )

        if not fitted:
            raise ValueError(
                f"every cross-validation split failed at read length {read_length}; "
                "you likely have too few distinct sequences to hold any out"
            )

        model.cutoffs[read_length] = combine(fitted, np.array(weights, dtype=np.float64))

    return model
