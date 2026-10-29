"""End-to-end check of `rockabye refpkg` and `rockabye place` with the real tools.

Builds a reference package from a positives directory (as made by
make_test_data.py), cuts reads of known origin from those genes -- half of them
reverse-complemented, as in a real metagenome -- places them, and scores how
many land on their source gene's leaf edge.

    python3 tests/test_placement.py <positives_dir> <workdir>

Needs muscle (v5), fasttree, taxit, mafft, pplacer and guppy on PATH.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.fasta import read_fasta, reverse_complement, write_fasta  # noqa: E402

READ_LENGTH = 150
READS_PER_GENE = 20
# Synthetic genes diverge strongly outside a shared domain, so a correctly
# aligned read almost always lands on its own leaf. Measured: 100% in fragments
# mode; add mode leaves reverse-strand reads unaligned.
MIN_ON_ORIGIN = 0.80


def rockabye(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "rockabye", *args], check=True)


def make_reads(positives: str, out_path: str) -> dict:
    rng = random.Random(11)
    genes = {}
    for f in sorted(os.listdir(positives)):
        genes.update(read_fasta(os.path.join(positives, f)))
    reads = {}
    for gene, seq in genes.items():
        for i in range(READS_PER_GENE):
            start = rng.randrange(0, len(seq) - READ_LENGTH)
            frag = seq[start : start + READ_LENGTH]
            if i % 2:
                frag = reverse_complement(frag)
            reads[f"{gene}|read{i}"] = frag
    write_fasta(out_path, reads)
    return reads


def on_origin(jplace: str) -> float:
    with open(jplace) as fh:
        d = json.load(fh)
    leaf_edge = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"([^(),:]+):[0-9.eE-]+\{(\d+)\}", d["tree"])
    }
    edge_col = d["fields"].index("edge_num")
    hits = total = 0
    for p in d["placements"]:
        for name in p.get("n") or [x[0] for x in p["nm"]]:
            gene = name.split("|")[0]
            total += 1
            hits += leaf_edge.get(gene) == p["p"][0][edge_col]
    return hits / total


def main() -> int:
    positives, work = sys.argv[1], os.path.abspath(sys.argv[2])
    reads_dir = os.path.join(work, "reads")
    reads = make_reads(positives, os.path.join(reads_dir, "reads.fasta"))

    rockabye("refpkg", positives, "-o", os.path.join(work, "ref"), "--force", "-q")
    pkg = os.path.join(work, "ref", "reference_package", "reference.refpkg")

    failures = []
    rates = {}
    for mode in ("fragments", "add"):
        out = os.path.join(work, f"place_{mode}")
        rockabye("place", "-c", pkg, reads_dir, "-o", out, "--mafft-mode", mode, "--force", "-q")
        tsv = os.path.join(out, "pplacer_tsv", "reads.jplace.tsv")
        with open(tsv) as fh:
            names = {line.split("\t")[1] for line in list(fh)[1:]}
        if names != set(reads):
            failures.append(f"{mode}: TSV read names differ from the input ({len(names ^ set(reads))} mismatched)")
        rates[mode] = on_origin(os.path.join(out, "pplacer_jplace", "reads.jplace"))
        print(f"{mode:>9}: {rates[mode]:.1%} of {len(reads)} reads on their origin leaf edge")

    if rates["fragments"] < MIN_ON_ORIGIN:
        failures.append(f"fragments mode placed only {rates['fragments']:.1%} on origin (< {MIN_ON_ORIGIN:.0%})")
    if rates["fragments"] <= rates["add"]:
        failures.append("fragments mode is no better than add mode")

    for f in failures:
        print(f"FAIL  {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
