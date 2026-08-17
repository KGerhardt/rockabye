"""Axis construction and 2D cumulative histograms.

Every classifier is a 2D histogram of read counts: rows are descending y-bins
(bitscore or percent identity), columns are x positions (MA column or percent
alignment bin). Cumulative-summing down the rows turns cell (j, x) into "how many
reads at column x score at or above y_j" -- which is exactly the TP/FP count a
threshold at y_j would produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict

import numpy as np

from .align import Alignments

BITSCORE_RESOLUTION = 1.0
PCT_ID_RESOLUTION = 0.5
PCT_ALN_RESOLUTION = 2.5
MA_WINDOW_SIZE = 20
ALN_WINDOW_SIZE = 10.0


@dataclass
class Axes:
    """Shared across all read lengths and CV splits, as in ROCkOut."""

    bitscore: np.ndarray  # ascending
    pct_id: np.ndarray  # ascending
    pct_aln: np.ndarray  # ascending, x-axis of the third classifier
    ma_width: int

    @property
    def bitscore_desc(self) -> np.ndarray:
        return self.bitscore[::-1]

    @property
    def pct_id_desc(self) -> np.ndarray:
        return self.pct_id[::-1]


def build_axes(
    datasets: Dict[int, Alignments], ma_width: int, compat: str = "hardened"
) -> Axes:
    """Construct the shared binning axes.

    `compat="rockout"` reproduces ROCkOut's formulas exactly, including the
    half-open bitscore range and anchoring the percent-alignment axis at the
    minimum observed value.

    `compat="hardened"` anchors the percent-alignment axis at zero instead.
    ROCkOut's filter looks bins up with `searchsorted(..., 'left') - 1`, so a real
    read falling below the minimum seen during training yields index -1 and
    silently wraps to the most permissive bin. Anchoring at zero makes that
    unreachable; the resulting empty low bins are set to the reject sentinel by
    `refine.fit_cutoffs`.
    """
    if compat not in ("rockout", "hardened"):
        raise ValueError(f"unknown axis compat mode {compat!r}")

    min_bit = min(float(a.bitscore.min()) for a in datasets.values())
    max_bit = max(float(a.bitscore.max()) for a in datasets.values())
    min_id = min(float(a.pct_id.min()) for a in datasets.values())
    max_id = max(float(a.pct_id.max()) for a in datasets.values())
    min_aln = min(float(a.pct_aln.min()) for a in datasets.values())
    min_aln = min(min_aln, 100.0)

    if compat == "hardened":
        min_aln = 0.0

    bitscore = np.arange(min_bit, max_bit, BITSCORE_RESOLUTION)
    if bitscore.size == 0:
        bitscore = np.array([min_bit])

    id_steps = max(2, ceil((max_id - min_id) / PCT_ID_RESOLUTION))
    pct_id = np.round(np.linspace(min_id, max_id, num=id_steps), 2)

    aln_steps = max(2, ceil((100.0 - min_aln) / PCT_ALN_RESOLUTION))
    pct_aln = np.round(np.linspace(min_aln, 100.0, num=aln_steps), 2)

    return Axes(bitscore=bitscore, pct_id=pct_id, pct_aln=pct_aln, ma_width=ma_width)


def nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of ROCkOut's `find_nearest` (argmin of |axis - v|)."""
    if axis.size <= 1:
        # A degenerate axis (every read sharing one value) has only one bin; the
        # clip below would otherwise produce a wrapped negative index.
        return np.zeros(np.shape(values), dtype=np.int32)
    idx = np.searchsorted(axis, values, side="left")
    idx = np.clip(idx, 1, axis.size - 1)
    left = axis[idx - 1]
    right = axis[idx]
    take_left = (values - left) <= (right - values)
    return np.where(take_left, idx - 1, idx).astype(np.int32)


class MASpans:
    """CSR-style store of the MA columns each read's alignment covers.

    Built once per read length and reused across all five CV splits. Only columns
    holding an actual residue are filled -- gap columns inside the footprint are
    skipped, matching `refiner.bin_read`.
    """

    def __init__(self, aln: Alignments, offsets: Dict[str, np.ndarray]):
        cols: list[np.ndarray] = []
        indptr = np.zeros(len(aln) + 1, dtype=np.int64)
        medians = np.zeros(len(aln), dtype=np.int32)
        for i, (tgt, s, e) in enumerate(zip(aln.target, aln.sstart, aln.send)):
            lo, hi = (s, e) if s <= e else (e, s)
            span = np.arange(lo - 1, max(lo, hi - 1), dtype=np.int32)
            off = offsets[tgt]
            span = span[span < off.size]
            filled = off[span] + span
            cols.append(filled)
            indptr[i + 1] = indptr[i] + filled.size
            # int(np.median(...)) truncates, matching refiner.calculate_curves_for_one_set
            medians[i] = int(np.median(filled)) if filled.size else 0
        self.data = (
            np.concatenate(cols).astype(np.int32) if cols else np.zeros(0, np.int32)
        )
        self.indptr = indptr
        self.medians = medians

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (column indices, index of the owning read within `rows`)."""
        lengths = (self.indptr[rows + 1] - self.indptr[rows]).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int64)
        owner = np.repeat(np.arange(rows.size, dtype=np.int64), lengths)
        starts = self.indptr[rows]
        offsets_within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        flat = np.repeat(starts, lengths) + offsets_within
        return self.data[flat], owner


def cumulative_hist(
    y_index: np.ndarray,
    x_index: np.ndarray,
    n_y: int,
    n_x: int,
) -> np.ndarray:
    """Descending-row cumulative 2D histogram.

    Row 0 corresponds to the *highest* y bin, so row j is the count of reads with
    y >= axis_desc[j]: the TP (or FP) count for a threshold placed there.
    """
    flat = y_index.astype(np.int64) * n_x + x_index.astype(np.int64)
    counts = np.bincount(flat, minlength=n_y * n_x).reshape(n_y, n_x)
    counts = counts[::-1]
    return np.cumsum(counts, axis=0, dtype=np.int64)


def ma_histograms(
    aln: Alignments,
    spans: MASpans,
    rows: np.ndarray,
    axes: Axes,
    y_values: np.ndarray,
    y_axis: np.ndarray,
    mode: str = "span",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (target, confounder) cumulative histograms over MA columns.

    ROCkOut is asymmetric about how a read is spread over the alignment, and the
    two MA-position classifiers disagree deliberately (refiner.py:604-613):
    the bitscore histogram counts each read once at its median column, while the
    percent-identity histogram counts it at every column it covers. `mode`
    selects which.
    """
    if mode not in ("span", "median"):
        raise ValueError(f"unknown MA histogram mode {mode!r}")

    is_pos = aln.label[rows] == "Positive"
    y_idx_all = nearest_index(y_axis, y_values[rows])

    out = []
    for mask in (is_pos, ~is_pos):
        sel = rows[mask]
        if sel.size == 0:
            out.append(np.zeros((y_axis.size, axes.ma_width), dtype=np.int64))
            continue
        if mode == "median":
            cols = spans.medians[sel]
            y_rep = y_idx_all[mask]
        else:
            cols, owner = spans.gather(sel)
            y_rep = y_idx_all[mask][owner]
        out.append(cumulative_hist(y_rep, cols, y_axis.size, axes.ma_width))
    return out[0], out[1]


def aln_histograms(
    aln: Alignments, rows: np.ndarray, axes: Axes
) -> tuple[np.ndarray, np.ndarray]:
    """Build (target, confounder) cumulative histograms over percent-alignment bins."""
    is_pos = aln.label[rows] == "Positive"
    y_idx = nearest_index(axes.pct_id, aln.pct_id[rows])
    x_idx = nearest_index(axes.pct_aln, np.minimum(aln.pct_aln[rows], 100.0))

    out = []
    for mask in (is_pos, ~is_pos):
        if not mask.any():
            out.append(np.zeros((axes.pct_id.size, axes.pct_aln.size), dtype=np.int64))
            continue
        out.append(
            cumulative_hist(
                y_idx[mask], x_idx[mask], axes.pct_id.size, axes.pct_aln.size
            )
        )
    return out[0], out[1]
