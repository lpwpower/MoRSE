#!/usr/bin/env python3
"""Build reproducible SciCode splits with domain+difficulty stratification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_metadata(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = str(row["problem_id"])
            out[pid] = {
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "subfield": row["subfield"],
                "steps": int(row["steps"]),
            }
    return out


def allocate(total: int, counts: Dict[str, int]) -> Dict[str, int]:
    """Largest-remainder allocation with per-bucket caps."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not counts:
        return {}
    denom = sum(counts.values())
    if total > denom:
        raise ValueError(f"cannot allocate {total} items from only {denom} available")

    raw = {k: (total * v / denom) for k, v in counts.items()}
    alloc = {k: min(int(math.floor(raw[k])), counts[k]) for k in counts}

    assigned = sum(alloc.values())
    remaining = total - assigned

    order = sorted(
        counts.keys(),
        key=lambda k: (raw[k] - math.floor(raw[k]), counts[k], k),
        reverse=True,
    )

    while remaining > 0:
        progressed = False
        for k in order:
            if alloc[k] < counts[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("allocation failed due to exhausted capacities")
    return alloc


def summarize(rows: List[dict], metadata: Dict[str, dict]) -> dict:
    domain_counts = Counter()
    difficulty_counts = Counter()
    domain_difficulty = defaultdict(Counter)
    for row in rows:
        pid = str(row["problem_id"])
        m = metadata[pid]
        d = m["domain"]
        diff = m["difficulty"]
        domain_counts[d] += 1
        difficulty_counts[diff] += 1
        domain_difficulty[d][diff] += 1
    return {
        "n_problems": len(rows),
        "domain_counts": dict(sorted(domain_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "domain_difficulty_counts": {
            d: dict(sorted(c.items())) for d, c in sorted(domain_difficulty.items())
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", type=Path, default=Path("problems_dev.jsonl"))
    p.add_argument("--test", type=Path, default=Path("problems_test.jsonl"))
    p.add_argument(
        "--metadata-tsv",
        type=Path,
        default=Path("domain_difficulty_stats.tsv"),
        help="TSV containing problem_id/domain/difficulty.",
    )
    p.add_argument("--out-dev", type=Path, default=Path("mydev.jsonl"))
    p.add_argument("--out-test", type=Path, default=Path("mytest.jsonl"))
    p.add_argument("--out-summary", type=Path, default=Path("my_split_summary.json"))
    p.add_argument("--dev-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = read_jsonl(args.dev) + read_jsonl(args.test)
    by_pid: Dict[str, dict] = {}
    for row in rows:
        pid = str(row["problem_id"])
        if pid in by_pid:
            raise ValueError(f"duplicate problem_id found: {pid}")
        by_pid[pid] = row
    rows = list(by_pid.values())
    n_total = len(rows)

    if args.dev_size <= 0 or args.dev_size >= n_total:
        raise ValueError(f"dev-size must be in [1, {n_total - 1}]")

    metadata = read_metadata(args.metadata_tsv)
    missing = sorted(str(r["problem_id"]) for r in rows if str(r["problem_id"]) not in metadata)
    if missing:
        raise ValueError(f"missing metadata for problem_id(s): {missing}")

    rng = random.Random(args.seed)

    domain_to_ids: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        pid = str(row["problem_id"])
        domain_to_ids[metadata[pid]["domain"]].append(pid)
    for d in domain_to_ids:
        domain_to_ids[d] = sorted(domain_to_ids[d], key=lambda x: int(x))

    domain_counts = {d: len(v) for d, v in domain_to_ids.items()}
    domain_quota = allocate(args.dev_size, domain_counts)

    selected: List[str] = []
    for domain in sorted(domain_to_ids):
        diff_to_ids: Dict[str, List[str]] = defaultdict(list)
        for pid in domain_to_ids[domain]:
            diff_to_ids[metadata[pid]["difficulty"]].append(pid)
        for diff in diff_to_ids:
            diff_to_ids[diff] = sorted(diff_to_ids[diff], key=lambda x: int(x))

        diff_counts = {k: len(v) for k, v in diff_to_ids.items()}
        diff_quota = allocate(domain_quota[domain], diff_counts)
        for diff in sorted(diff_to_ids):
            ids = diff_to_ids[diff][:]
            rng.shuffle(ids)
            selected.extend(ids[: diff_quota[diff]])

    if len(selected) != args.dev_size:
        raise RuntimeError(
            f"selected {len(selected)} problems, expected {args.dev_size}"
        )

    selected_set = set(selected)
    dev_rows = sorted(
        (by_pid[pid] for pid in selected_set), key=lambda r: int(str(r["problem_id"]))
    )
    test_rows = sorted(
        (row for row in rows if str(row["problem_id"]) not in selected_set),
        key=lambda r: int(str(r["problem_id"])),
    )

    write_jsonl(args.out_dev, dev_rows)
    write_jsonl(args.out_test, test_rows)

    summary = {
        "seed": args.seed,
        "dev_size": args.dev_size,
        "test_size": len(test_rows),
        "quotas": {
            "domain_quota_dev": dict(sorted(domain_quota.items())),
        },
        "stats": {
            "all": summarize(rows, metadata),
            "mydev": summarize(dev_rows, metadata),
            "mytest": summarize(test_rows, metadata),
        },
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dev} ({len(dev_rows)} rows)")
    print(f"Wrote {args.out_test} ({len(test_rows)} rows)")
    print(f"Wrote {args.out_summary}")
    print("mydev domain counts:", summary["stats"]["mydev"]["domain_counts"])
    print("mydev difficulty counts:", summary["stats"]["mydev"]["difficulty_counts"])
    print("mytest domain counts:", summary["stats"]["mytest"]["domain_counts"])
    print("mytest difficulty counts:", summary["stats"]["mytest"]["difficulty_counts"])


if __name__ == "__main__":
    main()

