#!/usr/bin/env python3
"""Reproduce the SciCode domain-level OOD split and verify byte-exact equality.

OOD split rule (see ``OOD_SPLIT_README.md``):

    train = Physics + Math + Material Science   (64 problems)
    test  = Chemistry + Biology  (held out)     (16 problems)

The script reads ``problems_dev.jsonl`` + ``problems_test.jsonl``, looks up each
problem's domain in ``domain_difficulty_stats.tsv`` (keyed by ``problem_id``),
partitions the problems by domain, writes ``ood_train.jsonl`` / ``ood_test.jsonl``
sorted by integer ``problem_id``, and then VERIFIES byte/line equality against the
shipped ``ood_train.jsonl`` / ``ood_test.jsonl``, printing PASS / FAIL.

Serialization note: rows are re-emitted with ``json.dumps(row, ensure_ascii=False)``
and a trailing ``\n`` per line. This matches the shipped OOD files exactly.

No GPU required. Run from this directory:

    python make_ood_split.py            # write + verify
    python make_ood_split.py --check    # verify only (does not overwrite shipped files)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Domains kept for training vs. held out for OOD test.
TRAIN_DOMAINS = {"Physics", "Math", "Material Science"}
TEST_DOMAINS = {"Chemistry", "Biology"}

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def read_domain_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[str(row["problem_id"])] = row["domain"]
    return out


def serialize(rows: List[dict]) -> str:
    """Emit one JSON object per line (UTF-8, no ASCII escaping)."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", type=Path, default=HERE / "problems_dev.jsonl")
    p.add_argument("--test", type=Path, default=HERE / "problems_test.jsonl")
    p.add_argument("--metadata-tsv", type=Path, default=HERE / "domain_difficulty_stats.tsv")
    p.add_argument("--out-train", type=Path, default=HERE / "ood_train.jsonl")
    p.add_argument("--out-test", type=Path, default=HERE / "ood_test.jsonl")
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify only: compare the regenerated split against the shipped files "
        "without overwriting them.",
    )
    args = p.parse_args()

    domain = read_domain_map(args.metadata_tsv)

    # Merge dev + test, de-duplicating by problem_id (keeping first occurrence).
    rows: List[dict] = []
    seen: set = set()
    for src in (args.dev, args.test):
        for d in read_jsonl(src):
            pid = str(d["problem_id"])
            if pid in seen:
                continue
            seen.add(pid)
            rows.append(d)

    missing = sorted(str(d["problem_id"]) for d in rows if str(d["problem_id"]) not in domain)
    if missing:
        raise ValueError(f"missing domain metadata for problem_id(s): {missing}")

    def by_pid(d: dict) -> int:
        return int(str(d["problem_id"]))

    train_rows = sorted((d for d in rows if domain[str(d["problem_id"])] in TRAIN_DOMAINS), key=by_pid)
    test_rows = sorted((d for d in rows if domain[str(d["problem_id"])] in TEST_DOMAINS), key=by_pid)

    # Sanity: every problem must land in exactly one bucket.
    assigned = len(train_rows) + len(test_rows)
    if assigned != len(rows):
        leftover = sorted(
            str(d["problem_id"])
            for d in rows
            if domain[str(d["problem_id"])] not in (TRAIN_DOMAINS | TEST_DOMAINS)
        )
        raise ValueError(
            f"{len(rows) - assigned} problem(s) have an unrecognized domain "
            f"(not in train/test domain sets): {leftover}"
        )

    train_blob = serialize(train_rows)
    test_blob = serialize(test_rows)

    train_domains = Counter(domain[str(d["problem_id"])] for d in train_rows)
    test_domains = Counter(domain[str(d["problem_id"])] for d in test_rows)
    print(f"OOD train: {len(train_rows)} problems  {dict(sorted(train_domains.items()))}")
    print(f"OOD test:  {len(test_rows)} problems  {dict(sorted(test_domains.items()))}")

    if not args.check:
        args.out_train.write_text(train_blob, encoding="utf-8")
        args.out_test.write_text(test_blob, encoding="utf-8")
        print(f"Wrote {args.out_train}")
        print(f"Wrote {args.out_test}")

    # ----- Verification against the shipped files (byte/line exact) -----
    ok = True
    for label, blob, ref in (
        ("ood_train.jsonl", train_blob, args.out_train),
        ("ood_test.jsonl", test_blob, args.out_test),
    ):
        if not ref.exists():
            print(f"[FAIL] {label}: shipped reference {ref} not found")
            ok = False
            continue
        shipped = ref.read_text(encoding="utf-8")
        byte_eq = blob == shipped
        line_eq = blob.splitlines() == shipped.splitlines()
        status = "PASS" if (byte_eq and line_eq) else "FAIL"
        print(
            f"[{status}] {label}: bytes={'equal' if byte_eq else 'DIFFER'} "
            f"({len(blob)} vs {len(shipped)}), lines={'equal' if line_eq else 'DIFFER'}"
        )
        ok = ok and byte_eq and line_eq

    print("OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
