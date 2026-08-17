"""Extract a ROCkOut project's cached gene sequences into rockabye input dirs.

A completed project caches the nucleotide CDS of every target protein it found
(`shared_files/combined_genomes/combined_proteins_nt.fasta`), named identically to
the multiple alignment. Sequences whose names appear in
`final_outputs/database/positive_proteins_aa.fasta` are the positives; the rest
came from the confounder genomes.

This lets rockabye be run on exactly the sequences ROCkOut trained on.

Usage:
    python3 tests/extract_rockout_genes.py <project_dir> <output_dir>
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rockabye.fasta import read_fasta_dict, write_fasta  # noqa: E402


def main() -> int:
    project, outdir = sys.argv[1], sys.argv[2]

    nt_path = os.path.join(
        project, "shared_files", "combined_genomes", "combined_proteins_nt.fasta"
    )
    pos_aa_path = os.path.join(
        project, "final_outputs", "database", "positive_proteins_aa.fasta"
    )
    msa_path = os.path.join(
        project, "final_outputs", "model", "complete_multiple_alignment_aa.fasta"
    )

    for p in (nt_path, pos_aa_path):
        if not os.path.exists(p):
            print(f"missing {p}")
            return 2

    nt = read_fasta_dict(nt_path)
    positive_names = set(read_fasta_dict(pos_aa_path))
    msa = read_fasta_dict(msa_path) if os.path.exists(msa_path) else {}

    missing = positive_names - set(nt)
    if missing:
        print(f"warning: {len(missing)} positive names absent from the nucleotide file")

    positives = {k: v for k, v in nt.items() if k in positive_names}
    negatives = {k: v for k, v in nt.items() if k not in positive_names}

    if not positives or not negatives:
        print(f"cannot split: {len(positives)} positives, {len(negatives)} negatives")
        return 2

    pos_dir = os.path.join(outdir, "positives")
    neg_dir = os.path.join(outdir, "negatives")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    write_fasta(os.path.join(pos_dir, "targets.fna"), positives)
    write_fasta(os.path.join(neg_dir, "confounders.fna"), negatives)

    # Also emit ROCkOut's own alignment, restricted to the positives, so a run can
    # optionally reuse it instead of realigning.
    if msa:
        pos_msa = {k: v for k, v in msa.items() if k in positive_names}
        if pos_msa:
            # Drop columns that are all-gap once the confounder rows are removed.
            width = len(next(iter(pos_msa.values())))
            keep = [
                i for i in range(width) if any(s[i] not in "-." for s in pos_msa.values())
            ]
            trimmed = {k: "".join(v[i] for i in keep) for k, v in pos_msa.items()}
            write_fasta(os.path.join(outdir, "rockout_positive_alignment.afa"), trimmed)
            print(
                f"ROCkOut alignment restricted to positives: {width} -> {len(keep)} columns "
                f"(written alongside, not inside positives/)"
            )

    print(f"positives: {len(positives)} sequences -> {pos_dir}")
    print(f"negatives: {len(negatives)} sequences -> {neg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
