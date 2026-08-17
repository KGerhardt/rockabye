"""Sliding-window Youden's J thresholding.

For each window along the x-axis we pick the y threshold maximising
J = sensitivity + specificity - 1, computed directly from the cumulative
histograms: at descending row j, TP = tgt[j], FP = con[j], FN = tgt_max - tgt[j],
TN = con_max - con[j], so both terms collapse to simple ratios.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Index into the tie-break vector produced below.
BIAS_CHOICES = {
    "balanced": 0,
    "favor_false_negatives": 1,
    "strongly_favor_false_negatives": 2,
    "favor_false_positives": 3,
    "strongly_favor_false_positives": 4,
}


def windowed_max(mat: np.ndarray, half: int, chunk: int = 256) -> np.ndarray:
    """Max over a sliding window of columns spanning [i-half, i+half).

    Counts are non-negative, so zero-padding the edges is equivalent to the
    truncated windows ROCkOut uses at the array boundaries.
    """
    n_y, n_x = mat.shape
    width = 2 * half
    if width <= 1:
        return mat.copy()
    padded = np.pad(mat, ((0, 0), (half, half - 1)))
    out = np.empty((n_y, n_x), dtype=mat.dtype)
    for start in range(0, n_x, chunk):
        end = min(start + chunk, n_x)
        block = padded[:, start : end + width - 1]
        win = sliding_window_view(block, width, axis=1)
        out[:, start:end] = win.max(axis=2)
    return out


def youden_cutoff_indices(
    tgt: np.ndarray, con: np.ndarray, half_window: int, bias: str = "balanced"
) -> np.ndarray:
    """-> index into the descending y-axis, one per x column.

    A lower index means a higher threshold (stricter, more false negatives).
    """
    if bias not in BIAS_CHOICES:
        raise ValueError(f"unknown cutoff bias {bias!r}; choose from {sorted(BIAS_CHOICES)}")
    pick = BIAS_CHOICES[bias]

    max_t = windowed_max(tgt, half_window).astype(np.float64)
    max_c = windowed_max(con, half_window).astype(np.float64)

    # Cumulative sums mean the final descending row holds the window totals.
    tgt_max = max_t[-1, :]
    con_max = max_c[-1, :]
    tn = con_max - max_c

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = max_t / tgt_max
        specificity = tn / con_max
    youden = sensitivity + specificity - 1.0

    n_y, n_x = youden.shape
    last = n_y - 1
    result = np.empty(n_x, dtype=np.int32)

    # argmax returns the first occurrence on ties, which is the highest threshold;
    # the tie-break vector below re-selects among all tied indices.
    best = np.argmax(np.nan_to_num(youden, nan=-np.inf), axis=0)
    best_vals = youden[best, np.arange(n_x)]

    for x in range(n_x):
        if not np.isfinite(best_vals[x]):
            # No confounders (or no targets) in this window: ROCkOut falls through
            # to the lowest threshold, i.e. accept everything here.
            result[x] = last
            continue
        ties = np.flatnonzero(youden[:, x] == best_vals[x])
        if ties.size == 1:
            result[x] = ties[0]
            continue
        vec = (
            int(np.median(ties)),
            int(np.quantile(ties, 0.25)),
            int(ties.min()),
            int(np.quantile(ties, 0.75)),
            int(ties.max()),
        )
        result[x] = vec[pick]

    return result


def indices_to_values(desc_axis: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return desc_axis[indices]
