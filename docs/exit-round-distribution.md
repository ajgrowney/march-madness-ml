# Exit Round Distribution Export

This document defines the website-facing contract for the seeded-team exit-round distribution added to `data/web/ts/{team_id}_{season}.json`.

## Purpose

The website already has a historical-comps surface through `similar_teams`.

This export adds a second Team Insights surface: a model-based probability distribution over how far a current tournament team is expected to advance.

The intended product use cases are:

- estimate a team's floor and ceiling
- judge whether an Elite Eight or Final Four pick is aggressive for that seed
- compare teams within the same region on a common advancement scale

## Model Definition

Export id: `team-exit-round-xgb-v1`

Model family: XGBoost multiclass classifier

Objective: `multi:softprob`

Feature set: `stat_explorer_v1` plus `Seed`

Feature order:

- `Seed`
- `Q1_WinPct`
- `Q2_WinPct`
- `Q3_WinPct`
- `Q4_WinPct`
- `SOS`
- `SOV`
- `Poss_mean`
- `Fouls_mean`
- `AdjOE_mean`
- `AdjNE_mean`
- `EFG%_mean`
- `FG3%_mean`
- `FT%_mean`
- `FTA_mean`
- `TO_mean`
- `Ast_mean`
- `OR_mean`
- `FGA3_mean`
- `AdjDE_mean`
- `Stl_mean`
- `Blk_mean`
- `OppTO_mean`
- `DR_mean`
- `OppFGA3_mean`

Training population:

- seeded historical tournament teams only
- seasons 2003 through 2024 for the current 2026 export

Validation setup:

- 2025 is kept as the holdout season for the current export
- calibration was explored in the notebook and intentionally excluded from the website export

Target classes:

- `1 = Round of 64`
- `2 = Round of 32`
- `3 = Sweet Sixteen`
- `4 = Elite Eight`
- `5 = Final Four`
- `6 = Championship`
- `7 = Champion`

Historical exit labels are mapped as follows:

- `Play In` and `First Round` both map to `Round of 64`
- `Second Round` maps to `Round of 32`

Notebook reference:

- the export mirrors the uncalibrated multiclass setup built in `Notebooks/newmodel_2026_eda.ipynb`

## Export Shape

Each team page now includes `exit_round_distribution`.

For non-tournament teams:

```json
null
```

For seeded teams:

```json
{
  "model": {
    "id": "team-exit-round-xgb-v1",
    "family": "xgboost-multiclass",
    "feature_set": "stat_explorer_v1",
    "training_season_range": [2003, 2024],
    "validation_season": 2025,
    "calibrated": false
  },
  "expected_exit_round": 2.759,
  "most_likely_round_num": 2,
  "most_likely_round": "Round of 32",
  "floor_round_num": 2,
  "floor_round": "Round of 32",
  "ceiling_round_num": 3,
  "ceiling_round": "Sweet Sixteen",
  "seed_expected_round": 2.568,
  "seed_delta": 0.191,
  "region_rank": 3,
  "probabilities": {
    "round64": 0.219,
    "round32": 0.242,
    "sweet16": 0.315,
    "elite8": 0.162,
    "final4": 0.062,
    "championship": 0.028,
    "champion": 0.013
  },
  "threshold_probabilities": {
    "sweet16_plus": 0.539,
    "elite8_plus": 0.224,
    "final4_plus": 0.062,
    "title_game_plus": 0.041
  }
}
```

## Field Semantics

- `expected_exit_round`: continuous expectation across the seven classes
- `most_likely_round`: single highest-probability class
- `floor_round`: 25th percentile outcome from the exported probability mass
- `ceiling_round`: 75th percentile outcome from the exported probability mass
- `seed_expected_round`: historical expectation for the same seed across the training window
- `seed_delta`: model expectation minus same-seed historical expectation
- `region_rank`: current-region ordering by `expected_exit_round`
- `probabilities`: exact class probabilities
- `threshold_probabilities`: cumulative advancement probabilities for common bracket milestones

## Website Usage

Recommended default usage:

1. Lead the UI with `threshold_probabilities`.
2. Use `expected_exit_round`, `floor_round`, and `ceiling_round` for the narrative summary.
3. Use `seed_delta` to explain relative aggressiveness versus seed.
4. Use `region_rank` only inside a same-region comparison view.

Recommended copy patterns:

- `Sweet Sixteen+ 54% | Elite Eight+ 22% | Final Four+ 6%`
- `Model floor: Round of 32 | ceiling: Sweet Sixteen`
- `Profiles 0.19 rounds better than a typical 4-seed`

Guidance for interpretation:

- Treat these values as model-driven scenario support, not calibrated market probabilities.
- Prefer cumulative thresholds for user-facing summaries.
- Use the exact seven-class distribution for charts, hover states, or more advanced drilldowns.

## Empty State Rules

- If `tournament` is `null`, `exit_round_distribution` should also be `null` and the website should hide the section.
- If `tournament` is present but `exit_round_distribution` is `null`, treat it as missing enrichment rather than a hard error.

## Export Decision

Calibration is not part of `team-exit-round-xgb-v1`.

Reason:

- the notebook calibration pass used a single holdout season
- it was useful diagnostically but not stable enough to replace the raw multiclass probabilities in the export

If calibration is added later, bump the export model id rather than silently changing the semantics of this payload.