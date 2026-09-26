#!/usr/bin/env python3
"""Reproduce and verify the SRDD dataset splits.

This script regenerates the four primary SRDD splits described in
``SPLIT_README.md`` directly from ``SRDD.csv`` and verifies that the
regenerated rows are byte-for-byte identical (same rows, same order) to the
shipped CSV files.

Splits
------
IID (stratified within-category):
    Within each of the 40 categories, the first 24 rows in CSV order go to
    ``iid_train`` and the last 6 go to ``iid_test``. The single exception is
    the ``Schedule`` category, which contains two near-duplicate pairs:
      * ``ScheduleAssistant`` (descriptions differing only by trailing
        ``...`` vs ``.``; 143 vs 145 chars)
      * ``ScheduleFocus`` (one description is a prefix truncation of the
        other; 128 vs 667 chars)
    For each pair the train-side instance falls in the first 24 rows while the
    test-side instance falls in the last 6. To prevent leakage, the test-side
    instance is moved into train, so ``Schedule`` ends up with 26 train / 4
    test instead of 24 / 6. Within ``Schedule`` only, the duplicates are
    detected by ``Name``: any of the last-6 rows whose ``Name`` already appears
    among the first 24 is moved to train. This reproduces exactly the two
    documented moves without hardcoding row indices.

    The fix is intentionally restricted to ``Schedule``: two other categories
    (``Role_Playing_Game`` with ``Quest_Difficulty_Analyzer`` and ``Security``
    with ``Security_Monitor``) also have a repeated ``Name`` straddling the
    24/6 boundary, but SPLIT_README.md documents the fix for ``Schedule`` only,
    and applying it elsewhere would not match the shipped CSVs.

OOD (category holdout):
    8 holdout categories are reserved entirely for ``ood_test``; the remaining
    32 categories form ``ood_train``. All 30 samples of a category go to a
    single split. Row order within each split follows the original SRDD.csv
    order.

Running
-------
    python make_splits.py            # verify against shipped CSVs (default)
    python make_splits.py --write    # also (re)write regenerated CSVs

No GPU is required; the script uses only the Python standard library.

Note on the ``_small`` variants
--------------------------------
``iid_train_small.csv`` (120 rows = 40 cats x 3) and ``iid_test_small.csv``
(80 rows = 40 cats x 2) are stratified subsamples of ``iid_train`` /
``iid_test`` (3 and 2 rows per category respectively). SPLIT_README.md does
not document how the within-category items are selected, and the observed
selection (train picks within-category indices [8, 16, 0] in that file order;
test picks the first two, or [1, 3] for the 4-row Schedule test set) does not
match a simple stride or any seeded ``random.sample`` we could recover. These
files are therefore copied as-is and NOT regenerated or verified here. See the
``issues`` section of the task report for details.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SRDD_CSV = os.path.join(DATA_DIR, "SRDD.csv")
IID_TRAIN = os.path.join(DATA_DIR, "iid_train.csv")
IID_TEST = os.path.join(DATA_DIR, "iid_test.csv")
OOD_TRAIN = os.path.join(DATA_DIR, "ood_train.csv")
OOD_TEST = os.path.join(DATA_DIR, "ood_test.csv")

# IID stratified split sizes (per category).
IID_TRAIN_PER_CAT = 24
IID_TEST_PER_CAT = 6  # last 6 rows; 30 - 24

# Category whose near-duplicate pairs are corrected (test-side instance moved
# to train). Restricted to "Schedule" per SPLIT_README.md.
IID_NEAR_DUP_CATEGORY = "Schedule"

# OOD holdout categories (entire categories reserved for the test split).
OOD_HOLDOUT = [
    "Strategy_Game",
    "Sport_Game",
    "Science",
    "Security",
    "Entertainment",
    "Health_Fitness",
    "Budgeting",
    "Graphics",
]


def load_csv(path):
    """Return (header, rows) where rows is a list of list[str]."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = list(csv.reader(fh))
    if not reader:
        raise ValueError("Empty CSV: {}".format(path))
    return reader[0], reader[1:]


def group_by_category(rows):
    """Group rows by their Category column, preserving first-seen order."""
    cats = OrderedDict()
    for row in rows:
        cats.setdefault(row[2], []).append(row)
    return cats


def make_iid_split(rows):
    """Reproduce the IID stratified within-category split.

    Returns (train_rows, test_rows) in the same row order the shipped CSVs use:
    categories in SRDD.csv first-seen order, and within a category all train
    rows (CSV order) precede all test rows.
    """
    cats = group_by_category(rows)
    train, test = [], []
    for cat, cat_rows in cats.items():
        head = cat_rows[:IID_TRAIN_PER_CAT]   # first 24 -> train
        tail = cat_rows[IID_TRAIN_PER_CAT:]   # last 6  -> test (candidate)
        cat_train = list(head)
        cat_test = []
        if cat == IID_NEAR_DUP_CATEGORY:
            # Near-duplicate fix (Schedule only): move any test-side row whose
            # Name already appears among the first 24 into train to prevent
            # leakage. This reproduces the two documented moves
            # (ScheduleAssistant, ScheduleFocus) -> 26 train / 4 test.
            head_names = {r[0] for r in head}
            for r in tail:
                if r[0] in head_names:
                    cat_train.append(r)
                else:
                    cat_test.append(r)
        else:
            cat_test = list(tail)
        train.extend(cat_train)
        test.extend(cat_test)
    return train, test


def make_ood_split(rows):
    """Reproduce the OOD category-holdout split.

    Holdout categories go to test; all others to train. Row order within each
    split follows the original SRDD.csv order.
    """
    holdout = set(OOD_HOLDOUT)
    train = [r for r in rows if r[2] not in holdout]
    test = [r for r in rows if r[2] in holdout]
    return train, test


def verify(name, regenerated_rows, shipped_path):
    """Compare regenerated rows against a shipped CSV. Returns True on PASS."""
    if not os.path.exists(shipped_path):
        print("[{:9s}] FAIL  shipped file missing: {}".format(name, shipped_path))
        return False
    _, shipped = load_csv(shipped_path)
    ok = regenerated_rows == shipped
    status = "PASS" if ok else "FAIL"
    print(
        "[{:9s}] {}  regenerated={} shipped={}".format(
            name, status, len(regenerated_rows), len(shipped)
        )
    )
    if not ok:
        # Report the first mismatch to aid debugging.
        if len(regenerated_rows) != len(shipped):
            print("           row-count mismatch")
        for i, (a, b) in enumerate(zip(regenerated_rows, shipped)):
            if a != b:
                print("           first diff at row {}:".format(i))
                print("             regenerated: {!r}".format(a))
                print("             shipped:     {!r}".format(b))
                break
    return ok


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote {} ({} rows)".format(path, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="(re)write the regenerated CSVs in addition to verifying them",
    )
    args = parser.parse_args()

    header, src = load_csv(SRDD_CSV)
    print("Loaded {}: {} rows, {} categories".format(
        os.path.basename(SRDD_CSV), len(src), len(group_by_category(src))))

    iid_train, iid_test = make_iid_split(src)
    ood_train, ood_test = make_ood_split(src)

    if args.write:
        write_csv(IID_TRAIN, header, iid_train)
        write_csv(IID_TEST, header, iid_test)
        write_csv(OOD_TRAIN, header, ood_train)
        write_csv(OOD_TEST, header, ood_test)

    print("\nVerification (regenerated vs shipped):")
    results = [
        verify("iid_train", iid_train, IID_TRAIN),
        verify("iid_test", iid_test, IID_TEST),
        verify("ood_train", ood_train, OOD_TRAIN),
        verify("ood_test", ood_test, OOD_TEST),
    ]

    all_pass = all(results)
    print("\nOverall: {}".format("ALL PASS" if all_pass else "FAILURE"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
