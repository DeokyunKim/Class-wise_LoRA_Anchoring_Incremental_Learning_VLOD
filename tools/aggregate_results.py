#!/usr/bin/env python
"""Aggregate CLA / DitHub run outputs into the headline metrics.

For each run directory this reads the per-dataset ``*.out`` files written at the
end of ``main.py`` (each contains ``<key>_cet <AP>``), then reports:

  * **Avg**   — mean AP over the benchmark's downstream tasks (COCO excluded)
  * **ZCOCO** — zero-shot COCO AP (forgetting / preservation proxy)
  * **hAP**   — harmonic mean of Avg and ZCOCO (the headline trade-off metric)

When several run directories are given (e.g. seeds 1/2/3) it also prints the
mean +/- sample-std across runs, matching the paper's reporting protocol.

Usage:
    python tools/aggregate_results.py output/odinw13_cla_seed1 \
                                       output/odinw13_cla_seed2 \
                                       output/odinw13_cla_seed3
    python tools/aggregate_results.py --glob 'output/odinwo_cla_seed*'
"""
from __future__ import annotations

import argparse
import glob
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunMetrics:
    """Headline metrics parsed from a single run directory."""

    name: str
    avg: float
    zcoco: float
    hap: float
    n_tasks: int


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def _read_last_ap(out_file: Path) -> Optional[float]:
    """Return the AP on the last non-empty line of a ``*.out`` file (``<key> <AP>``)."""
    last: Optional[float] = None
    for line in out_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[-1]
        try:
            last = float(token)
        except ValueError:
            continue
    return last


def parse_run(run_dir: Path) -> Optional[RunMetrics]:
    """Parse one run directory into Avg / ZCOCO / hAP, or ``None`` if incomplete."""
    out_files = sorted(run_dir.glob("*.out"))
    if not out_files:
        return None

    task_aps: list[float] = []
    zcoco: Optional[float] = None
    for out_file in out_files:
        ap = _read_last_ap(out_file)
        if ap is None:
            continue
        if "coco" in out_file.name.lower():
            zcoco = ap
        else:
            task_aps.append(ap)

    if not task_aps or zcoco is None:
        return None

    avg = statistics.fmean(task_aps)
    return RunMetrics(
        name=run_dir.name,
        avg=avg,
        zcoco=zcoco,
        hap=_harmonic_mean(avg, zcoco),
        n_tasks=len(task_aps),
    )


def _fmt_mean_std(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:6.2f} +/- {std:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run directories to aggregate")
    parser.add_argument("--glob", dest="glob_pat", default=None,
                        help="glob pattern for run directories (alternative to listing them)")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    if args.glob_pat:
        run_dirs += [Path(p) for p in sorted(glob.glob(args.glob_pat))]
    if not run_dirs:
        parser.error("provide run directories or --glob")

    results: list[RunMetrics] = []
    print(f"{'run':<42} {'#task':>5} {'Avg':>7} {'ZCOCO':>7} {'hAP':>7}")
    print("-" * 72)
    for run_dir in run_dirs:
        metrics = parse_run(run_dir)
        if metrics is None:
            print(f"{run_dir.name:<42} {'--':>5}  (incomplete: no parseable *.out)")
            continue
        results.append(metrics)
        print(f"{metrics.name:<42} {metrics.n_tasks:>5} "
              f"{metrics.avg:>7.2f} {metrics.zcoco:>7.2f} {metrics.hap:>7.2f}")

    if len(results) > 1:
        print("-" * 72)
        print(f"{'mean +/- std (n=' + str(len(results)) + ')':<42} {'':>5} "
              f"{_fmt_mean_std([r.avg for r in results])}  "
              f"{_fmt_mean_std([r.zcoco for r in results])}  "
              f"{_fmt_mean_std([r.hap for r in results])}")


if __name__ == "__main__":
    main()
