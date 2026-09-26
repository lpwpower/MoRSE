# SRDD Dataset Split

**Total**: 1200 samples, 40 categories × 30 samples each. All descriptions are unique.

---

## Setup 1 — IID (Stratified Within-Category)

**Files**: `iid_train.csv` (962 samples) / `iid_test.csv` (238 samples)

**Rule**: Within each category, the first 24 samples (by CSV order) go to train, the last 6 go to test.

**Near-duplicate fix**: The `Schedule` category contains two near-identical pairs:
- `ScheduleAssistant`: two entries whose descriptions differ only by trailing `...` vs `.` (143 vs 145 chars)
- `ScheduleFocus`: two entries where one is a prefix truncation of the other (128 vs 667 chars)

For both pairs, the test-side instance is moved to train to prevent leakage. As a result, `Schedule` has 26 train + 4 test instead of 24 + 6; all other 39 categories are exactly 24 + 6.

| Split | Samples | Categories | Per-category |
|-------|---------|------------|--------------|
| Train | 962 | 40 | 24 (39 cats) / 26 (Schedule) |
| Test  | 238 | 40 |  6 (39 cats) /  4 (Schedule) |

**Generalization tested**: In-distribution — all 40 categories appear in both train and test.

---

## Setup 2 — OOD (Category Holdout)

**Files**: `ood_train.csv` (960 samples) / `ood_test.csv` (240 samples)

**Rule**: 8 categories are held out entirely as the test set; the remaining 32 categories are used for training. All 30 samples per category go to their respective split.

| Split | Samples | Categories | Per-category |
|-------|---------|------------|--------------|
| Train | 960 | 32 | 30 each |
| Test  | 240 |  8 | 30 each |

**Holdout categories** (selected to balance domain type and difficulty):

| Category | Domain | Difficulty bias |
|----------|--------|----------------|
| Strategy_Game | Game | easy-heavy (E19/M6/H5) |
| Sport_Game | Game | hard-heavy (E4/M9/H17) |
| Science | Professional | easy-heavy (E20/M3/H7) |
| Security | Professional | hard-heavy (E5/M2/H23) |
| Entertainment | Social | balanced (E8/M11/H11) |
| Health_Fitness | Lifestyle | mid-hard (E10/M7/H13) |
| Budgeting | Tools | easy-heavy (E15/M9/H6) |
| Graphics | Tools | hard-heavy (E2/M8/H20) |

Test set difficulty distribution (proxy: description length + complexity keywords):
Easy 35% / Medium 23% / Hard 42%

**Generalization tested**: Out-of-distribution — test categories never appear in training.

---

## Difficulty Proxy

Used for category selection only, not for splitting. Each sample is scored as:

```
score = len(description) / 100 + keyword_count × 2
```

Keywords: `multiple`, `various`, `real-time`, `api`, `integration`, `authentication`,
`database`, `synchroni`, `concurrent`, `algorithm`, `machine learning`, `neural`,
`encrypt`, `multi-player`, `multiplayer`, `scalable`, `notification`, `analytics`,
`recommendation`, `dynamic`

Global thresholds (1200 samples): Easy ≤ 4.0, Medium ≤ 6.1, Hard > 6.1

Ground-truth difficulty (smoke_test pass rate) can be computed after running baseline inference.

---

## Correspondence with SciCode Splits

| Dataset | Train | Test | Ratio |
|---------|-------|------|-------|
| SciCode | 60 problems (~240 steps) | 20 problems (~98 steps) | 75/25 |
| SRDD IID | 962 samples | 238 samples | ~80/20 |
| SRDD OOD | 960 samples | 240 samples | 80/20 |

Both SRDD setups are consistent with SciCode's 75/25 training ratio.
