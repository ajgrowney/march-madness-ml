# Precomputed Model Contracts

This document defines the browser-facing contracts for offline-trained models such as RandomForest, sklearn MLP, Keras, and XGBoost.

Use these contracts when exporting 2026 Python models from the `march-madness-ml` repository.

## Recommended Artifact Strategy

Publish two files for each season:

1. `data/web/models/manifest.json`
Purpose: expose each available model in the website selector.

2. `data/web/models/predictions/{model_id}_{season}.json`
Purpose: provide matchup probability lookups for every supported ordered pair in the field.

The website should use the manifest to discover the model and the predictions artifact to resolve matchup probabilities.

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
  "feature_set": "2026_initial",
  "prediction_format": "precomputed",
  "mirrored_prediction_strategy": "average-forward-reverse",
  "description": "XGBoost model trained offline and exported as pairwise 2026 tournament probabilities.",
  "artifact_version": "1.0.0",
  "exported_at": "2026-03-15T18:42:00Z",
  "predictions_url": "https://raw.githubusercontent.com/ajgrowney/march-madness-ml/master/data/web/models/predictions/xgb-2026-v1_2026.json",
  "feature_importance_url": "https://raw.githubusercontent.com/ajgrowney/march-madness-ml/master/data/web/models/predictions/xgb-2026-v1_importance.json"
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
    "feature_set": "2026_initial",
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