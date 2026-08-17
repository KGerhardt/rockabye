"""Unit tests for the numeric core. Run: python3 tests/test_units.py"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.binning import (  # noqa: E402
    Axes,
    MASpans,
    cumulative_hist,
    ma_histograms,
    nearest_index,
)
from rockabye.align import Alignments, _merge_intervals  # noqa: E402
from rockabye.simulate import _BBMap, _overlaps  # noqa: E402
from rockabye.fasta import reverse_complement, translate  # noqa: E402
from rockabye.msa import compute_offsets  # noqa: E402
from rockabye.refine import REJECT_ALL, CutoffSet, combine  # noqa: E402
from rockabye.youden import windowed_max, youden_cutoff_indices  # noqa: E402

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"OK    {name}")
    else:
        print(f"FAIL  {name} {detail}")
        FAILURES.append(name)


def test_translate():
    check("translate handles a clean CDS", translate("ATGAAATTTTAA") == "MKF")
    # Internal stop becomes X; only a stop in the final codon is trimmed.
    check("internal stop becomes X", translate("ATGTAAAAA") == "MXK")
    check("trailing stop is trimmed", translate("ATGAAATAA") == "MK")
    check("reverse complement", reverse_complement("ATGC") == "GCAT")


def test_nearest_index():
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    got = nearest_index(axis, np.array([-5.0, 0.4, 0.6, 2.9, 99.0]))
    check("nearest_index matches argmin|axis-v|", list(got) == [0, 0, 1, 3, 3], f"got {list(got)}")
    # A degenerate single-bin axis arises when every read shares one value.
    got = nearest_index(np.array([7.0]), np.array([1.0, 7.0, 900.0]))
    check("single-bin axis maps everything to bin 0", list(got) == [0, 0, 0], f"got {list(got)}")


def test_cumulative_hist():
    # Two y bins, two x columns. y index 1 is the higher bin.
    y = np.array([1, 1, 0])
    x = np.array([0, 1, 0])
    mat = cumulative_hist(y, x, n_y=2, n_x=2)
    # Row 0 = highest bin: one read at each column. Row 1 adds the low-bin read.
    check("cumulative hist row 0 is the top bin", list(mat[0]) == [1, 1], f"got {list(mat[0])}")
    check("cumulative hist accumulates downward", list(mat[1]) == [2, 1], f"got {list(mat[1])}")


def test_windowed_max():
    rng = np.random.default_rng(0)
    mat = rng.integers(0, 50, size=(7, 33))
    half = 5
    fast = windowed_max(mat, half, chunk=4)
    naive = np.empty_like(mat)
    for i in range(mat.shape[1]):
        lo = max(0, i - half)
        hi = min(mat.shape[1], i + half)
        naive[:, i] = mat[:, lo:hi].max(axis=1)
    check("windowed_max equals truncated-window reference", np.array_equal(fast, naive))


def test_youden_separable():
    """Targets sit entirely above confounders -> cutoff lands between them."""
    n_y, n_x = 10, 8
    tgt = np.zeros((n_y, n_x), dtype=np.int64)
    con = np.zeros((n_y, n_x), dtype=np.int64)
    # Descending rows: targets appear from row 2, confounders only from row 7.
    tgt[2:, :] = 100
    con[7:, :] = 100
    idx = youden_cutoff_indices(tgt, con, half_window=1)
    check(
        "youden separates cleanly above the confounders",
        bool(np.all((idx >= 2) & (idx < 7))),
        f"got {idx.tolist()}",
    )


def test_youden_no_confounders():
    """No confounder data must fall through to the most permissive threshold."""
    n_y, n_x = 6, 4
    tgt = np.cumsum(np.ones((n_y, n_x), dtype=np.int64), axis=0)
    con = np.zeros((n_y, n_x), dtype=np.int64)
    idx = youden_cutoff_indices(tgt, con, half_window=1)
    check(
        "no confounders -> lowest threshold (accept everything)",
        bool(np.all(idx == n_y - 1)),
        f"got {idx.tolist()}",
    )


def test_combine_sentinel():
    def cs(vals):
        return CutoffSet(
            bitscore_vs_ma=np.array([10.0, 20.0]),
            pct_id_vs_ma=np.array([50.0, 60.0]),
            pct_id_vs_aln=np.array(vals, dtype=float),
        )

    # Column 0: majority say reject. Column 1: minority says reject.
    sets = [cs([REJECT_ALL, REJECT_ALL]), cs([REJECT_ALL, 40.0]), cs([60.0, 50.0])]
    weights = np.array([1.0, 1.0, 1.0])
    out = combine(sets, weights)
    check("sentinel wins on weighted majority", out.pct_id_vs_aln[0] == REJECT_ALL,
          f"got {out.pct_id_vs_aln[0]}")
    check(
        "minority sentinel excluded from the average",
        abs(out.pct_id_vs_aln[1] - 45.0) < 1e-9,
        f"got {out.pct_id_vs_aln[1]} (expected mean of 40 and 50)",
    )
    check(
        "plain averaging for the bitscore curve",
        np.allclose(out.bitscore_vs_ma, [10.0, 20.0]),
    )


def test_combine_all_sentinel():
    def cs(v):
        return CutoffSet(np.array([1.0]), np.array([1.0]), np.array([v], dtype=float))

    out = combine([cs(REJECT_ALL), cs(REJECT_ALL)], np.array([1.0, 1.0]))
    check("unanimous sentinel stays a sentinel", out.pct_id_vs_aln[0] == REJECT_ALL)


def _toy_alignments():
    """One positive read covering subject residues 1..5 of a gapless protein."""
    n = 1
    return Alignments(
        read_id=np.array(["r0"]),
        target=np.array(["p"]),
        pct_id=np.array([90.0]),
        aln_len=np.array([5], dtype=np.int32),
        sstart=np.array([1], dtype=np.int32),
        send=np.array([6], dtype=np.int32),
        bitscore=np.array([50.0]),
        qlen=np.array([15], dtype=np.int32),
        pct_aln=np.array([100.0]),
        label=np.array(["Positive"] * n),
        source=np.array(["s"]),
    )


def test_ma_histogram_modes():
    """ROCkOut spreads a read over its footprint for identity but not for bitscore.

    This asymmetry (refiner.py:604-613) is easy to miss and silently wrecks the
    bitscore curve, so it is pinned here.
    """
    aln = _toy_alignments()
    offsets = {"p": np.zeros(10, dtype=np.int32)}
    spans = MASpans(aln, offsets)
    axes = Axes(
        bitscore=np.array([50.0]),
        pct_id=np.array([90.0]),
        pct_aln=np.array([0.0, 100.0]),
        ma_width=10,
    )
    rows = np.arange(1)

    span_tgt, _ = ma_histograms(
        aln, spans, rows, axes, aln.bitscore, axes.bitscore, mode="span"
    )
    med_tgt, _ = ma_histograms(
        aln, spans, rows, axes, aln.bitscore, axes.bitscore, mode="median"
    )
    check(
        "span mode fills every covered column",
        int(span_tgt[-1].sum()) == 5 and list(span_tgt[-1][:6]) == [1, 1, 1, 1, 1, 0],
        f"got {list(span_tgt[-1])}",
    )
    check(
        "median mode fills exactly one column",
        int(med_tgt[-1].sum()) == 1 and med_tgt[-1][2] == 1,
        f"got {list(med_tgt[-1])}",
    )


def test_background_intervals():
    """Coordinate labelling of background reads (the ROCkOut annotation rule)."""
    check("merge overlapping regions",
          _merge_intervals([(0, 10), (5, 15), (20, 30)]) == [(0, 15), (20, 30)])
    check("merge unsorted regions", _merge_intervals([(20, 30), (0, 10)]) == [(0, 10), (20, 30)])
    check("merge abutting regions", _merge_intervals([(0, 10), (10, 20)]) == [(0, 20)])

    gene = [(100, 200)]
    check("read inside the gene is a target", _overlaps(gene, 120, 150))
    check("read straddling the gene start is a target", _overlaps(gene, 50, 110))
    check("read straddling the gene end is a target", _overlaps(gene, 190, 260))
    check("a single overlapping base is enough", _overlaps(gene, 199, 299))
    check("read entirely before the gene is not", not _overlaps(gene, 0, 100))
    check("read entirely after the gene is not", not _overlaps(gene, 200, 300))


def test_bbmap_read_names():
    name = "SYN_0_76_167_0_+_50382_1_._UNIPROT__Q70EF3_METSZ__Q70EF3__HE956757.3583"
    # Verified against the reference: contig[start:stop] recovers the read.
    check("bbmap coordinates parse 0-based half-open",
          _BBMap.read_coords(name) == (76, 167), f"got {_BBMap.read_coords(name)}")
    check("bbmap contig name parses after '_._'",
          _BBMap.source_of(name, set()) == "UNIPROT__Q70EF3_METSZ__Q70EF3__HE956757.3583")
    check("foreign read names yield no coordinates",
          _BBMap.read_coords("read1/1") is None)


def test_offsets():
    msa = {"a": "MK--LV", "b": "MKQQLV"}
    offsets, width = compute_offsets(msa)
    check("alignment width", width == 6, f"got {width}")
    # 'a' has residues M,K,L,V at MA columns 0,1,4,5 -> gaps before each: 0,0,2,2
    check("gap offsets for a gapped row", list(offsets["a"]) == [0, 0, 2, 2],
          f"got {list(offsets['a'])}")
    check("MA column reconstruction",
          list(offsets["a"] + np.arange(4)) == [0, 1, 4, 5])
    check("ungapped row has zero offsets", list(offsets["b"]) == [0, 0, 0, 0, 0, 0])


def test_zero_weights():
    def cs(v):
        return CutoffSet(np.array([v]), np.array([v]), np.array([v], dtype=float))

    out = combine([cs(10.0), cs(20.0)], np.array([0.0, 0.0]))
    check("all-zero F1 weights fall back to a plain mean",
          abs(out.bitscore_vs_ma[0] - 15.0) < 1e-9, f"got {out.bitscore_vs_ma[0]}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all unit tests passed")
