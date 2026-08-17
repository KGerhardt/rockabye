# rockabye

Build [ROCkOut](https://github.com/KGerhardt/ROCkOut)-compatible read filtering models
from sequences you have already labelled.

ROCkOut works out its own labels: it downloads genomes from UniProt and decides a read
is positive only if it falls inside the target gene's coordinates. `rockabye` takes the
labels as given — you supply target genes, confounder genes, and the genomes they live
in. That drops the download, snapshot, and phylogenetic-placement machinery while
keeping the part that produces a filter, and the output loads directly into
`rockout_main.py filter`.

The thresholding is a faithful reimplementation, verified bin-exact against 15 real
ROCkOut projects covering 5.8 million alignments ([details](#fidelity-vs-rockout)).

## Installation

```bash
git clone https://github.com/KGerhardt/rockabye.git
cd rockabye
pip install -e .
```

Python ≥3.9 and numpy, plus three external tools:

```bash
conda install -c bioconda diamond bbmap muscle
```

| Tool | Used for |
|---|---|
| DIAMOND | translated alignment — the model's axes are `blastx` bitscore and identity |
| BBMap (`randomreads.sh`) | read simulation. ROCkOut's simulator, called with its parameters |
| MUSCLE or MAFFT | aligning the positive proteins (skippable, see below) |

BBMap is the only supported simulator. It is not merely a default: background reads
are labelled using the per-read contig coordinates BBMap reports, and a simulator
that does not report them cannot label them correctly — see
[How background reads get labelled](#how-background-reads-get-labelled).

## Usage

Three inputs: target genes, confounder genes, and genomes.

```
positives/
  targets.fna        # nucleotide CDS of the genes you want to detect
negatives/
  confounders.fna    # nucleotide CDS of genes you want excluded
genomes/
  *.fna              # whole genomes/contigs -- ideally the ones your targets live in
```

```bash
rockabye build \
    -p positives/ \
    -n negatives/ \
    --background genomes/ \
    -o my_model/ \
    -t 16
```

Then filter a metagenome with ROCkOut itself:

```bash
python3 rockout_main.py align  -d my_model/ -f filter_dir/ -1 reads.fasta
python3 rockout_main.py filter -d my_model/ -f filter_dir/
```

Check a model at any time — this works on any ROCkOut model, not only ones built here:

```bash
rockabye validate my_model/
```

### What each input does

**`--background` genomes are where the discriminative power comes from.** They are
simulated at depth so that every scrap of sequence which might spuriously align to your
target gets a chance to do so. Reads that land on a copy of a target gene are labelled
positive; everything else becomes a confounder. Supply the genomes your positives
actually live in — that is the point, since those carry the paralogues most likely to
misalign. Running without `--background` is supported but produces a much weaker model
([numbers below](#results)).

**Negatives** are curated near-homologues. They pin down the identity threshold, where
genomic background alone is sparse.

**Positives** define the target, the protein database, and the alignment axis.

### Expect this to take a while

Cost is dominated by background depth: 64 Mbp of genomes at the default 10x coverage
produces roughly 6 million reads per read length, so a run is tens of minutes, mostly
in `blastx`. The depth is not optional — only about 1 background read in 10⁵ aligns at
all, so shallow background yields too few confounders to constrain anything. A model is
built once and reused.

### Optional inputs

Drop these into `positives/` to skip a step:

- `proteins.faa` — use these instead of translating `targets.fna`
- `aligned.afa` — use this instead of running MUSCLE/MAFFT

Their sequence IDs must match the nucleotide IDs exactly; ROCkOut's filter silently
drops alignments whose target is missing from the alignment. IDs must be unique across
the project and contain no `;`.

### Options worth knowing

| Flag | Default | |
|---|---|---|
| `--background-coverage` | `10.0` | fold coverage for genomes; matches ROCkOut |
| `--background-target-identity` | `90.0` | identity at which a background region counts as a copy of a positive gene. Lower it if your genomes are divergent from your positive sequences |
| `--coverage` | `20` | coverage for the positive and confounder genes |
| `--read-lengths` | `100 150 250 300` | nominal lengths, ±10% jitter |
| `--snp-rate` | `0.01` | indel rates default to this ÷ 19, as ROCkOut does |
| `--splits` | `5` | cross-validation partitions |
| `--train-fraction` | `0.75` | fraction of *sequences* (not reads) used to train |
| `--cutoff-bias` | `balanced` | tie-breaking among equally-good thresholds |
| `--compat` | `hardened` | closes a lookup wrap in ROCkOut's filter; `rockout` reproduces its binning bit-for-bit |
| `--seed` | `1337` | the pipeline is deterministic given a seed |
| `--threads` | `1` | |

`--cutoff-bias` chooses among thresholds that score identically under Youden's J:
`strongly_favor_false_negatives` is strictest, `strongly_favor_false_positives` most
permissive, with `favor_*` variants at the quartiles.

## Results

Every model below was scored on the same ground truth: the labelled reads cached by the
real ROCkOut `AmoA_A_v2` project. `rockabye` was given that project's own gene sequences
and genomes, so the only thing varying is what the tool does with them.

| model | F1 | false positives | false negatives |
|---|---:|---:|---:|
| ROCkOut's own model (39×293 alignment) | 0.9982 | 0.00% | 0.35% |
| `rockabye`, **no** `--background` | 0.8820 | **20.86%** | 0.00% |
| `rockabye`, with `--background` (defaults) | 0.9899 | **0.09%** | 1.88% |
| `rockabye`, with `--background`, `--compat rockout` | 0.9977 | 0.27% | 0.12% |

Adding genomic background takes the false positive rate from 20.86% to 0.09% — a
230-fold reduction — and is the difference between a model that is unusable on real
data and one that matches ROCkOut. All `rockabye` rows use a 5×216 protein alignment
against ROCkOut's 39×293, so the coarser position axis costs very little once the
labels are right.

The last two rows show what `--compat hardened` (the default) actually trades. It
rejects reads in percent-alignment bins that training never populated, which removes
almost all remaining false positives (0.27% → 0.09%) at the cost of some sensitivity
(FNR 0.12% → 1.88%). Whether that is the right trade depends on your data: this
benchmark scores reads drawn from the same distribution the model was trained on, so
it can only show the cost of the stricter rule, never the benefit — which is
robustness to alignment fractions the training set never contained. Pass
`--compat rockout` if you would rather have ROCkOut's exact behaviour.

Two things drive the difference.

**Without genomic background the model is far too permissive.** Its confounders come
only from curated negative genes, which are separated from targets by *identity*. The
reads that actually cause false positives in a metagenome are separated by *alignment
fraction* — they match a short stretch of the protein very well. A model that never saw
one sets no useful threshold there.

This is not an artefact of the test. Measured across ROCkOut's own training data,
confounders are overwhelmingly genomic background rather than negative genes:

| Model | positive reads | from confounder genes | from genomic background |
|---|---:|---:|---:|
| nirk | 76,165 | 0 | 4,236,794 (100%) |
| nosz | 28,121 | 0 | 263,910 (100%) |
| blaA | 70,248 | 903 | 589,821 (99.8%) |
| mbl | 32,177 | 399 | 29,992 (98.7%) |
| AmoA_A | 852 | 229 | 864 (79%) |
| tem | 7,303 | 9,609 | 541 (5%) |

Refitting on ROCkOut's own reads while varying *only* which confounders are available
isolates the effect — genomic background alone is as good as everything, curated genes
alone are not:

| confounders used for fitting | F1 | FPR |
|---|---:|---:|
| all (ROCkOut's behaviour) | 0.9982 | 0.00% |
| genomic background only | 0.9982 | 0.00% |
| confounder genes only | 0.8623 | **24.89%** |

**Background reads must be labelled by coordinate, not by identity.** See below.

### How background reads get labelled

The genomes you supply contain the target gene, so their reads cannot all be treated as
confounders. ROCkOut resolves this with genome annotation: a read is positive if it
overlaps an annotated target gene *at all* — overlaps as small as 3% count — and a
confounder only if it overlaps nothing. `rockabye` reconstructs that rule without
annotation:

1. `diamond blastx` the background contigs against the positive proteins, keeping hits
   at or above `--background-target-identity` (default 90%). Those `qstart`/`qend`
   intervals are the copies of your positive genes. Distant paralogues fall far below
   the threshold and stay confounders — they are the whole point.
2. BBMap's `simplenames=t` encodes each read's contig coordinates in its name (0-based,
   half-open; verified empirically against the reference).
3. A read overlapping a target interval by one base is labelled positive; the rest are
   confounders.

This matters more than it sounds. Partially-overlapping boundary reads are **11.6% of
all true positives** in AmoA_A_v2, and they are the only positives with low alignment
fraction — a gene simulated in isolation can never produce a read running off its own
end. Train against them by mistake and the identity/alignment classifier turns too
strict: refitting ROCkOut's reads with exactly that error raises FNR from 0.35% to
6.34%.

This is why BBMap is the only supported simulator: step 2 needs per-read coordinates.
A simulator that reports only which contig a read came from forces a fallback — screen
reads by identity and discard the suspects — which throws away exactly the boundary
reads the model needs, and mislabels the ones it keeps.

## What the model is

`rocker_filter.py` reads exactly six files:

```
my_model/final_outputs/
├── database/
│   ├── positive_proteins_aa.fasta
│   └── positive_proteins_diamond_db.dmnd
└── model/
    ├── complete_multiple_alignment_aa.fasta
    ├── bitscore_vs_MA_pos.txt     read_length  position_in_MA  bitscore
    ├── pct_id_vs_MA_pos.txt       read_length  position_in_MA  percent_id
    └── pct_id_vs_pct_aln.txt      read_length  percent_aln     percent_id
```

Three cutoff curves, each a classifier. A read passes if it clears at least two of
the three. Thresholds are interpolated between trained read lengths and clamped
outside that range. A percent-identity cutoff of `101.0` is the reject-everything
sentinel, since identity cannot exceed 100.

`rockabye` also writes `cross_validation_performance.tsv` and
`rockabye_manifest.json`. ROCkOut ignores both.

### Three ways to build a broken-but-loadable model

Properties of ROCkOut's loader, so they apply to any model however made.
`rockabye validate` checks all three.

1. **`position_in_MA` must be contiguous integers `0..L-1`.** The filter uses the
   value directly as a column index. An offset or gap misaligns every threshold
   rather than raising an error.
2. **Every read length must repeat the full x grid.** The matrix is zero-filled and
   interpolated across read-length *rows* only, never columns. An x value present at
   one read length and absent at another keeps a cutoff of `0.0`, passing everything.
3. **All three tables must span the same read-length range.** `import_filter`
   reassigns `min_readlen`/`max_readlen` on each of its three calls, so the row
   offset comes from whichever file loaded last and is applied to all three matrices.

## Fidelity vs ROCkOut

Given the same reads, this produces the same model. Verified by feeding each real
ROCkOut project's own cached training reads and alignment through `rockabye`'s
binning and thresholding, then comparing against that project's shipped curves.
This runs under `--compat rockout`, which is what bit-exact agreement is defined
against. Duplicated project directories are collapsed, so 13 distinct models are
shown:

| Project | alignments | binning | MA mapping | same verdict | F1 rockabye / ROCkOut |
|---|---:|---:|---:|---:|---|
| AmoA_A_v2 | 1,945 | 100% | 100% | 100.00% | 0.9982 / 0.9982 |
| arch_amoa | 2,934 | 100% | 100% | 99.69% | 0.9981 / 0.9977 |
| bact_amoa | 11,848 | 100% | 100% | 99.86% | 0.9971 / 0.9963 |
| tem | 17,453 | 100% | 100% | 99.93% | 0.9918 / 0.9914 |
| mbl_s1_s2 | 42,148 | 100% | 100% | 100.00% | 0.9999 / 0.9999 |
| mbl | 62,568 | 100% | 100% | 99.52% | 0.9949 / 0.9927 |
| oxa | 68,707 | 100% | 100% | 99.91% | 0.9984 / 0.9987 |
| mbl_s3 | 110,815 | 100% | 100% | 99.90% | 0.9943 / 0.9928 |
| blac | 163,836 | 100% | 100% | 99.96% | 0.9978 / 0.9972 |
| nosz | 292,031 | 100% | 100% | 99.98% | 0.9946 / 0.9940 |
| blaA | 660,972 | 100% | 100% | 99.97% | 0.9953 / 0.9957 |
| nirk | 4,312,959 | 100% | 100% | 99.97% | 0.9805 / 0.9785 |

"binning" is agreement with ROCkOut's own cached bin assignments (`bitscore_bin`,
`id_bin`, `aln_indices`); "MA mapping" is agreement on each read's median alignment
column. Both are exact everywhere, so axis construction and position mapping are
identical, not merely close. Curves differ slightly because ROCkOut ships an
F1-weighted average over five partitions while the comparison fits once on all
data; the resulting *decisions* agree 99.5–100%.

Replicated deliberately, because each affects the output:

- window sizes (20 alignment columns, 10% alignment) and bin resolutions
  (1.0 bitscore, 0.5% identity, 2.5% alignment)
- a 0.75 train fraction: ROCkOut's default `sequence_outgroups` splitter hardcodes
  0.75 and discards the 0.4 its refiner passes in, so its own docs describing a
  60/40 split do not match that code path
- Youden's J with median tie-breaking, and the `101.0` sentinel for identity
  cutoffs below 25%
- the **asymmetry between the two position classifiers**: the bitscore histogram
  counts each read once at its median alignment column, while the percent-identity
  histogram counts it at every column it covers (`refiner.py:604-613`)
- the second asymmetry between fitting and filtering: fitting spreads a read across
  its footprint, filtering scores it at a single midpoint
- three-way labels collapsing to two — only `Positive` is a target; both `Negative`
  (the same gene in a confounder genome) and `Non_Target` (genomic background) are
  confounders (`refiner.py:603`)
- BBMap invoked with ROCkOut's parameters, including indel rates at 1/19 of the SNP rate

### Deliberate differences

- **Labels are trusted.** No genome download, no coordinate intersection.
- **The alignment covers positives only.** ROCkOut aligns positives together with
  homologues found in the confounder genomes (39 sequences × 293 columns for
  AmoA_A_v2, versus 5 × 216 here), which changes the position axis but not the
  method. Supply `aligned.afa` to control this yourself.
- **`--compat hardened` is the default**, and anchors the percent-alignment axis at
  0. ROCkOut starts it at the minimum observed value, but its filter looks bins up
  with `searchsorted(..., 'left') - 1`, so a read below that minimum yields index
  `-1` and wraps around to the most permissive bin — the flimsiest alignments judged
  by the standard meant for the strongest. Anchoring at zero makes that index
  unreachable, and the resulting empty low bins get the `101.0` reject sentinel
  rather than ROCkOut's NaN-path fallback of "accept everything". This changes the
  emitted numbers, so pass `--compat rockout` to reproduce ROCkOut bit-for-bit.
  How often the wrap fires on real metagenomic data is unmeasured; the fix is cheap
  and only ever makes the filter stricter at low alignment fractions.
- **Sentinel-aware averaging.** Averaging `101.0` against real thresholds would
  invent a meaningless in-between value, so the sentinel is decided by weighted
  majority and only non-sentinel partitions contribute to the average.

## Tests

```bash
python3 tests/test_units.py                     # numeric core
python3 tests/make_test_data.py /tmp/td          # synthetic family, shared domain
rockabye build -p /tmp/td/positives -n /tmp/td/negatives -o /tmp/proj
rockabye validate /tmp/proj

# load the model with ROCkOut's real filter code and classify fresh reads
python3 tests/test_rockout_interop.py \
    /path/to/ROCkOut/modules/rocker_filter.py \
    /tmp/proj $(which diamond) /tmp/td/positives /tmp/td/negatives /tmp/interop

# compare against real ROCkOut projects
python3 tests/compare_to_rockout.py /path/to/a/rockout/project
tests/compare_all.sh /path/to/dir/of/rockout/projects

# run rockabye on a project's own cached genes, then score both models on its reads
python3 tests/extract_rockout_genes.py <project> /tmp/genes
rockabye build -p /tmp/genes/positives -n /tmp/genes/negatives -o /tmp/rebuilt
python3 tests/evaluate_on_cached_reads.py <project> /tmp/rebuilt
```

`test_rockout_interop.py` is the one that proves compatibility: it imports
ROCkOut's real `rocker_filter.py` and runs its loading and classification path
unmodified.
