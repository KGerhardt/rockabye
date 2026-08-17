"""Read simulation, delegated entirely to established CLI simulators.

There is no in-house simulator: BBMap already models fragment sampling and
sequencing error properly, and reimplementing that badly is the fastest way to
build a model on reads that do not resemble data. BBMap is also what ROCkOut itself
calls, and it is invoked here with the same parameters, so a rockabye run and a
ROCkOut run see the same kind of reads.

The adapter indirection below is kept deliberately: a second backend needs only a
`command()` and the two read-name parsers. It must report per-read contig
coordinates, though -- without them background reads cannot be labelled correctly.

Reads are streamed straight to disk as they are produced. Deep genomic background
runs generate tens of millions of reads, which must never be held in memory.

Defline convention (the final field is what ROCkOut's optional simulated-read
scoring reads, splitting on ';'):

    >{source_id};{index};{nominal_read_length};{Positive|Negative|Background}
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, TextIO, Tuple

import numpy as np

from .fasta import read_fasta, write_fasta
from .project import BACKGROUND, NEGATIVE, POSITIVE, Inputs

DEFAULT_READ_LENGTHS = (100, 150, 250, 300)

# ROCkOut derives both indel rates from the SNP rate this way; keeping the same
# relationship means a matching --snp-rate reproduces its error model exactly.
INDEL_RATE_DIVISOR = 19

SIMULATORS = ("bbmap",)


@dataclass
class SimConfig:
    read_lengths: tuple = DEFAULT_READ_LENGTHS
    coverage: float = 20.0
    length_jitter: float = 0.10
    snp_rate: float = 0.01
    insertion_rate: Optional[float] = None
    deletion_rate: Optional[float] = None
    seed: int = 1337
    simulator: str = "bbmap"
    background_coverage: float = 10.0

    def indel_rates(self) -> Tuple[float, float]:
        ins = (
            self.insertion_rate
            if self.insertion_rate is not None
            else self.snp_rate / INDEL_RATE_DIVISOR
        )
        dele = (
            self.deletion_rate
            if self.deletion_rate is not None
            else self.snp_rate / INDEL_RATE_DIVISOR
        )
        return round(ins, 8), round(dele, 8)


def parse_defline(name: str) -> tuple[str, int, str]:
    """-> (source_id, nominal_read_length, label). Raises on malformed names."""
    segs = name.split(";")
    if len(segs) < 4:
        raise ValueError(f"read name {name!r} does not follow the tagged convention")
    return segs[0], int(segs[-2]), segs[-1]


def _read_fastq(path: str) -> Iterator[Tuple[str, str]]:
    """Stream (name, sequence). Quality lines may start with '@', so never grep."""
    with open(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                return
            seq = fh.readline().rstrip("\n")
            fh.readline()
            fh.readline()
            yield header[1:].split()[0], seq


def _read_any(path: str) -> Iterator[Tuple[str, str]]:
    if path.endswith((".fq", ".fastq")):
        return _read_fastq(path)
    return read_fasta(path)


class _BBMap:
    """BBMap's randomreads.sh -- ROCkOut's simulator, invoked the same way.

    Read names carry the source contig after a literal '_._' separator, e.g.
        SYN_0_76_167_0_+_50382_1_._UNIPROT__Q70EF3_METSZ__Q70EF3__HE956757.3583
    """

    name = "bbmap"
    executable = "randomreads.sh"

    def __init__(self, cfg: SimConfig, exe: str):
        self.cfg = cfg
        self.exe = exe

    def command(self, ref: str, out: str, read_length: int, coverage: float,
                seed: int, workdir: str) -> list:
        cfg = self.cfg
        ins, dele = cfg.indel_rates()
        lo = int(round(read_length * (1 - cfg.length_jitter)))
        hi = int(round(read_length * (1 + cfg.length_jitter)))
        return [
            self.exe,
            f"ref={ref}",
            f"out={out}",
            f"path={workdir}",
            f"coverage={coverage}",
            f"snprate={cfg.snp_rate}",
            f"insrate={ins}",
            f"delrate={dele}",
            f"minlength={lo}",
            f"maxlength={hi}",
            f"seed={seed}",
            "simplenames=t",
            "overwrite=t",
        ]

    @staticmethod
    def source_of(read_name: str, known: set) -> Optional[str]:
        if "_._" in read_name:
            return read_name.split("_._", 1)[1]
        return None

    @staticmethod
    def read_coords(read_name: str) -> Optional[Tuple[int, int]]:
        """-> (start, stop), 0-based half-open on the source contig.

        Verified empirically against the reference: `contig[start:stop]` recovers
        the read (median 97.5% identity at the default error rate).
        """
        parts = read_name.split("_", 4)
        if len(parts) < 4 or parts[0] != "SYN":
            return None
        try:
            return int(parts[2]), int(parts[3])
        except ValueError:
            return None


def get_simulator(cfg: SimConfig):
    if cfg.simulator not in SIMULATORS:
        raise ValueError(
            f"unknown simulator {cfg.simulator!r}; choose from {', '.join(SIMULATORS)}"
        )
    cls = _BBMap
    exe = shutil.which(cls.executable)
    if exe is None:
        raise FileNotFoundError(
            f"read simulation requires {cls.executable!r} (BBMap) on PATH.\n"
            "  conda install -c bioconda bbmap\n"
            "rockabye has no built-in simulator by design; simulation is delegated "
            "to a tool that models sequencing error properly."
        )
    return cls(cfg, exe)


def _overlaps(intervals, start: int, stop: int) -> bool:
    """Any overlap at all, matching ROCkOut: a read touching the gene is a target."""
    for lo, hi in intervals:
        if start < hi and lo < stop:
            return True
    return False


def _simulate_group(
    out_handle: TextIO,
    records: Dict[str, str],
    label: str,
    read_length: int,
    cfg: SimConfig,
    coverage: float,
    rng: np.random.Generator,
    sim,
    target_regions: Optional[dict] = None,
) -> tuple:
    """Simulate one labelled group and stream it into an open FASTA handle.

    When `target_regions` is supplied (background only, and only for a simulator
    that reports coordinates), each read is labelled by whether it overlaps a copy
    of a positive gene -- the same rule ROCkOut applies using genome annotation.
    Returns (n_written, n_relabelled_positive).
    """
    if not records:
        return 0, 0

    written = 0
    relabelled = 0
    known = set(records)
    with tempfile.TemporaryDirectory(prefix="rockabye_sim_") as tmp:
        ref = os.path.join(tmp, "ref.fna")
        write_fasta(ref, records)
        produced = os.path.join(tmp, "reads.fq")
        seed = int(rng.integers(1, 2**31 - 1))

        cmd = sim.command(ref, produced, read_length, coverage, seed, tmp)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{sim.name} failed (exit {proc.returncode}) for {label} reads at "
                f"length {read_length}:\n{proc.stderr[-3000:]}"
            )
        if not os.path.exists(produced):
            alternatives = [
                os.path.join(tmp, f) for f in os.listdir(tmp)
                if f.endswith((".fq", ".fastq", ".fa"))
            ]
            if not alternatives:
                raise RuntimeError(
                    f"{sim.name} reported success but produced no reads for {label} "
                    f"at length {read_length}"
                )
            produced = alternatives[0]

        counter: Dict[str, int] = {}
        unattributed = 0
        relabelled = 0
        for name, seq in _read_any(produced):
            src = sim.source_of(name, known)
            if src is None:
                unattributed += 1
                continue
            this_label = label
            if target_regions is not None:
                intervals = target_regions.get(src)
                if intervals:
                    coords = sim.read_coords(name)
                    if coords is not None and _overlaps(intervals, coords[0], coords[1]):
                        this_label = POSITIVE
                        relabelled += 1
            i = counter.get(src, 0)
            counter[src] = i + 1
            out_handle.write(f">{src};{i};{read_length};{this_label}\n{seq}\n")
            written += 1

        if written == 0:
            raise RuntimeError(
                f"{sim.name} produced {unattributed} reads for {label} at length "
                f"{read_length} but none could be attributed to a source sequence. "
                "This usually means the read-name format changed; please report it."
            )

    return written, relabelled


def simulate_all(
    inputs: Inputs,
    cfg: SimConfig,
    outdir: str,
    log=print,
    target_regions: Optional[dict] = None,
) -> Dict[int, str]:
    """Write one tagged FASTA per read length. Returns {read_length: path}."""
    sim = get_simulator(cfg)
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    paths: Dict[int, str] = {}

    ins, dele = cfg.indel_rates()
    log(
        f"  using {sim.name} (snprate={cfg.snp_rate}, insrate={ins}, delrate={dele})"
    )

    for rl in cfg.read_lengths:
        path = os.path.join(outdir, f"reads_len_{rl}.fasta")
        with open(path, "w") as out:
            n_pos, _ = _simulate_group(
                out, inputs.positive_nt, POSITIVE, rl, cfg, cfg.coverage, rng, sim
            )
            n_neg, _ = _simulate_group(
                out, inputs.negative_nt, NEGATIVE, rl, cfg, cfg.coverage, rng, sim
            )
            n_bg, n_from_target = _simulate_group(
                out, inputs.background_nt, BACKGROUND, rl, cfg,
                cfg.background_coverage, rng, sim, target_regions,
            )

        if n_pos == 0:
            raise ValueError(
                f"no positive reads simulated at length {rl}; are the positive "
                "sequences shorter than the read length?"
            )
        if n_neg == 0:
            raise ValueError(f"no confounder reads simulated at length {rl}")

        paths[rl] = path
        extra = f" + {n_bg - n_from_target:,} background" if n_bg else ""
        if n_from_target:
            extra += f" ({n_from_target:,} background reads hit a target gene copy)"
            n_pos += n_from_target
        log(
            f"  read length {rl}: {n_pos:,} positive + {n_neg:,} confounder{extra} reads"
        )

    return paths
