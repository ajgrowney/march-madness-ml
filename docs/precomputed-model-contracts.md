# Precomputed Model Contracts

This document defines the browser-facing contracts for offline-trained models such as RandomForest, sklearn MLP, Keras, XGBoost, and matchup-insight artifacts such as slot-aware clustering analogs.

Use these contracts when exporting 2026 Python models from the `march-madness-ml` repository.

## Recommended Artifact Strategy

Publish two files for each winner-probability model season:

1. `data/web/models/manifest.json`
Purpose: expose each available model in the website selector.

2. `data/web/models/predictions/{model_id}_{season}.json`
Purpose: provide matchup probability lookups for every supported ordered pair in the field.

The website should use the manifest to discover the model and the corresponding artifact to resolve matchup inference.

For insight-only models, publish:

1. `data/web/models/manifest.json`
Purpose: expose the insight model in the website selector.

2. `data/web/models/insights/{model_id}_{season}.json`
Purpose: provide per-matchup analogs, cluster summaries, and optional slot-aware filtering results.

## Why Pairwise Predictions

For Python-trained models, pairwise lookup is the cleanest browser contract.

- The browser does not need to execute sklearn, Keras, or XGBoost.
- The bracket builder can still autofill downstream rounds itself.
- Manual picks, slot probability drilldowns, and selector switching continue to work in one UX.

## Manifest Entry Example

The existing [model-manifest.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/model-manifest.schema.json) now supports richer metadata for offline models.

Example entry:

```json
{
  "id": "xgb-2026-v1",
  "name": "Andrew XGBoost 2026",
  "type": "precomputed-predictions",
  "model_family": "xgboost",
  "seasons": [2026],
  "training_seasons": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
  "feature_set": "stat_explorer_v1",
  "prediction_format": "precomputed",
  "mirrored_prediction_strategy": "average-forward-reverse",
  "description": "XGBoost model trained offline and exported as pairwise 2026 tournament probabilities.",
  "artifact_version": "1.0.0",
  "exported_at": "2026-03-15T18:42:00Z",
  "predictions_url": "https://raw.githubusercontent.com/ajgrowney/march-madness-ml/master/data/web/models/predictions/xgb-2026-v1_2026.json",
  "feature_importance_url": "https://raw.githubusercontent.com/ajgrowney/march-madness-ml/master/data/web/models/predictions/xgb-2026-v1_importance.json"
}
```

### Matchup Clustering Manifest Entry Example

Use a separate manifest entry when the artifact returns matchup context rather than winner probabilities.

```json
{
  "id": "matchup-cluster-2026-v1",
  "name": "2026 Matchup Clusters",
  "type": "precomputed-insights",
  "model_family": "matchup-clustering",
  "seasons": [2026],
  "training_seasons": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
  "feature_set": "stat_explorer_v1",
  "prediction_format": "precomputed-insights",
  "inference_task": "matchup-insights",
  "artifact_format": "pairwise-matchup-insights-v1",
  "supports_bracket_autofill": false,
  "description": "Slot-aware matchup clustering artifact that assigns each 2026 pairing to a historical game archetype and returns the nearest tournament analogs.",
  "artifact_version": "1.0.0",
  "exported_at": "2026-03-18T20:15:00Z",
  "matchup_insights_url": "data/web/models/insights/matchup-cluster-2026-v1_2026.json"
}
```

## Prediction Artifact Example

Validate these files against [precomputed-predictions.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/precomputed-predictions.schema.json).

Example:

```json
{
  "season": 2026,
  "model": {
    "id": "xgb-2026-v1",
    "name": "Andrew XGBoost 2026",
    "type": "precomputed-predictions",
    "model_family": "xgboost",
    "seasons": [2026],
    "training_seasons": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "feature_set": "stat_explorer_v1",
    "prediction_format": "precomputed",
    "mirrored_prediction_strategy": "average-forward-reverse",
    "description": "XGBoost model trained offline and exported as pairwise 2026 tournament probabilities.",
    "artifact_version": "1.0.0",
    "exported_at": "2026-03-15T18:42:00Z",
    "predictions_url": "https://raw.githubusercontent.com/ajgrowney/march-madness-ml/master/data/web/models/predictions/xgb-2026-v1_2026.json"
  },
  "prediction_format": "pairwise-probabilities-v1",
  "team_ids": [1104, 1112, 1181],
  "feature_importance": [
    { "featureKey": "AdjNE_mean", "importance": 0.231, "rank": 1 },
    { "featureKey": "AdjOE_mean", "importance": 0.184, "rank": 2 }
  ],
  "predictions": {
    "1104:1181": {
      "teamAId": 1104,
      "teamBId": 1181,
      "teamAProbability": 0.536421,
      "teamBProbability": 0.463579,
      "winnerId": 1104,
      "rawScore": 0.1459
    },
    "1181:1104": {
      "teamAId": 1181,
      "teamBId": 1104,
      "teamAProbability": 0.463579,
      "teamBProbability": 0.536421,
      "winnerId": 1104,
      "rawScore": -0.1459
    }
  }
}
```

## Matchup Insights Artifact Example

Validate these files against [matchup-insights.schema.json](/Users/andrewgrowney/Code/Python/march-madness-ml/docs/march-madness-ml/schemas/matchup-insights.schema.json).

```json
{
  "season": 2026,
  "model": {
    "id": "matchup-cluster-2026-v1",
    "name": "2026 Matchup Clusters",
    "type": "precomputed-insights",
    "model_family": "matchup-clustering",
    "seasons": [2026],
    "feature_set": "stat_explorer_v1",
    "prediction_format": "precomputed-insights",
    "inference_task": "matchup-insights",
    "artifact_format": "pairwise-matchup-insights-v1",
    "supports_bracket_autofill": false,
    "description": "Slot-aware matchup clustering artifact that returns analogs and cluster summaries for each 2026 pairing.",
    "matchup_insights_url": "data/web/models/insights/matchup-cluster-2026-v1_2026.json"
  },
  "inference_task": "matchup-insights",
  "artifact_format": "pairwise-matchup-insights-v1",
  "team_ids": [1104, 1112, 1181],
  "matchups": {
    "1104:1181": {
      "team_a_id": 1104,
      "team_b_id": 1181,
      "favorite_team_id": 1104,
      "underdog_team_id": 1181,
      "bracket_slot": "R2Z1",
      "bracket_slot_family": "R2:1",
      "same_bracket_slot": true,
      "cluster_summary": {
        "cluster_id": 2,
        "cluster_game_count": 355,
        "filtered_game_count": 15,
        "favorite_win_rate": 0.789,
        "upset_rate": 0.211,
        "avg_margin": 8.972,
        "avg_total_points": 139.242,
        "avg_seed_gap": 8.927,
        "common_round": "Round of 64"
      },
      "nearest_games": [
        {
          "season": 2003,
          "round": "Round of 32",
          "slot": "R2Z1",
          "slot_family": "R2:1",
          "favorite_team_id": 1112,
          "underdog_team_id": 1211,
          "favorite_won": true,
          "margin": 1,
          "total_points": 191,
          "similarity": 0.839
        }
      ]
    }
  }
}
```

## What It Takes To Use Matchup Clustering As A Website Model

If the goal is to let users pick this artifact in the same model selector used for 2026 matchup inference, the minimum pieces are:

1. Add a manifest entry like the example above with `type = precomputed-insights`, `inference_task = matchup-insights`, and `matchup_insights_url`.
2. Export a season-specific pairwise insights artifact for the 2026 field.
3. Teach the website model loader to branch on `inference_task` or `artifact_format` so it requests `matchup_insights_url` instead of `predictions_url`.
4. Treat the artifact as insight-only unless you explicitly derive a winner probability from cluster rates or nearest-neighbor vote. In practice that means `supports_bracket_autofill = false`.
5. If you later want bracket auto-pick support, publish a separate winner-probability model or add a calibrated probability field to the insight artifact and make that behavior explicit.

## Export Recommendations

Use these conventions when writing Python exports:

1. Export probabilities, not just winners.
2. Store both matchup directions explicitly.
3. Keep `teamAId` and `teamBId` aligned with the lookup key.
4. Keep the browser-facing feature set name identical to the website feature store name.
5. Preserve the mirrored prediction strategy in metadata so future debugging is straightforward.

## Validation Checklist

Before publishing:

1. Validate `manifest.json` against [model-manifest.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/model-manifest.schema.json).
2. Validate each pairwise predictions file against [precomputed-predictions.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/precomputed-predictions.schema.json).
3. Confirm every model manifest entry with `prediction_format = precomputed` includes `predictions_url`.
4. Confirm every team ID in `team_ids` exists in the exported team index for the same season.
5. Confirm both directions exist if your browser lookup path assumes ordered keys.
6. Confirm every model manifest entry with `prediction_format = precomputed-insights` includes `matchup_insights_url`.
7. Confirm insight artifacts declare whether same-slot filtering was applied so the UI can explain why a matchup returned fewer analogs.