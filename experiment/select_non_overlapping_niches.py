#!/usr/bin/env python3
"""Select query niches from exported metric CSVs with bounded cell overlap.

Example::

    .venv/bin/python experiment/select_non_overlapping_niches.py -k 3 \
        --candidates experiment/query_niches/final_query_niche_candidates/large/simple/C57BL6J-638850.28.csv \
        --preprocessed experiment/query_niches/preprocessed/large/C57BL6J-638850.28.csv

Two niches are compatible only when their member-cell overlap is at most
``--max-cell-overlap-percent`` of the smaller niche. Candidate CSV order is
preserved when several maximum-size selections exist. The default threshold is
0, matching the previous strict non-overlap behavior.

len(members[A] & members[B]) / min(len(members[A]), len(members[B]))
    <= max_cell_overlap_percent / 100

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

csv.field_size_limit(100_000_000)


@dataclass(frozen=True)
class Niche:
    center: str
    members: frozenset[str]
    parcellations: frozenset[int]


def _json(value: str, field: str, center: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {field} for {center}") from exc


def load_parcellation_ids(preprocessed: Path) -> list[int]:
    ids_path = preprocessed.parent / "parcellation_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(f"missing shared parcellation IDs: {ids_path}")
    values = json.loads(ids_path.read_text())
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate parcellation IDs in {ids_path}")
    return ids


def load_niches(candidates: Path, preprocessed: Path) -> list[Niche]:
    ids = load_parcellation_ids(preprocessed)
    wanted: list[str] = []
    with candidates.open(newline="") as handle:
        for row in csv.DictReader(handle):
            center = str(row["center_cell_name"])
            if center in wanted:
                raise ValueError(f"duplicate candidate center: {center}")
            wanted.append(center)

    records: dict[str, dict[str, str]] = {}
    with preprocessed.open(newline="") as handle:
        for row in csv.DictReader(handle):
            center = str(row["center_cell_name"])
            if center in records:
                raise ValueError(f"duplicate preprocessed center: {center}")
            records[center] = row

    result: list[Niche] = []
    for center in wanted:
        if center not in records:
            raise ValueError(
                f"candidate center missing from preprocessed file: {center}"
            )
        row = records[center]
        members = _json(row["niche_cell_names"], "niche_cell_names", center)
        fractions = _json(
            row["parcellation_fractions"], "parcellation_fractions", center
        )
        if not isinstance(members, list) or not all(
            isinstance(x, str) for x in members
        ):
            raise ValueError(f"niche_cell_names must be a string list for {center}")
        if len(members) != len(set(members)):
            raise ValueError(f"duplicate niche cell IDs for {center}")
        if len(fractions) != len(ids):
            raise ValueError(f"fraction vector length mismatch for {center}")
        positive = frozenset(
            pid for pid, fraction in zip(ids, fractions) if float(fraction) > 0
        )
        result.append(Niche(center, frozenset(members), positive))
    return result


def cell_overlap_fraction(first: Niche, second: Niche) -> float:
    denominator = min(len(first.members), len(second.members))
    if denominator == 0:
        raise ValueError("niches must contain at least one member cell")
    return len(first.members & second.members) / denominator


def select(
    niches: list[Niche], k: int, *, max_cell_overlap_fraction: float = 0.0
) -> list[Niche]:
    print(
        f"precomputing pairwise compatibility for {len(niches)} candidates",
        file=sys.stderr,
        flush=True,
    )
    compatible_after: list[list[int]] = [[] for _ in niches]
    for i, first in enumerate(niches):
        if i and i % 250 == 0:
            print(
                f"  checked compatibility for {i}/{len(niches)} candidates",
                file=sys.stderr,
                flush=True,
            )
        for j in range(i + 1, len(niches)):
            second = niches[j]
            if first.parcellations & second.parcellations:
                continue
            if cell_overlap_fraction(first, second) <= max_cell_overlap_fraction:
                compatible_after[i].append(j)
    print("searching for compatible niche set", file=sys.stderr, flush=True)

    best: list[int] = []

    def search(options: list[int], chosen: list[int]) -> bool:
        nonlocal best
        if len(chosen) > len(best):
            best = chosen.copy()
        if len(chosen) == k:
            return True
        if len(chosen) + len(options) <= len(best):
            return False
        for option_pos, index in enumerate(options):
            compatible = set(compatible_after[index])
            next_options = [
                value
                for value in options[option_pos + 1 :]
                if value in compatible
            ]
            chosen.append(index)
            if search(next_options, chosen):
                return True
            chosen.pop()
        return False

    search(list(range(len(niches))), [])
    return [niches[index] for index in best]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--preprocessed", required=True, type=Path)
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument(
        "--max-cell-overlap-percent",
        type=float,
        default=0.0,
        help=(
            "Maximum pairwise member-cell overlap as a percent of the smaller "
            "niche. Default: 0."
        ),
    )
    parser.add_argument("--output", type=Path, help="JSON output; stdout by default")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k must be positive")
    if not 0 <= args.max_cell_overlap_percent <= 100:
        parser.error("--max-cell-overlap-percent must be between 0 and 100")
    niches = load_niches(args.candidates, args.preprocessed)
    selected = select(
        niches,
        args.k,
        max_cell_overlap_fraction=args.max_cell_overlap_percent / 100.0,
    )
    payload = {
        "requested_k": args.k,
        "achieved_k": len(selected),
        "shortfall": max(0, args.k - len(selected)),
        "max_cell_overlap_percent": args.max_cell_overlap_percent,
        "center_cell_names": [n.center for n in selected],
        "parcellation_ids": {n.center: sorted(n.parcellations) for n in selected},
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
