# March Madness ML Website Export Guide

This guide defines the browser-facing artifacts that the `march-madness-ml` repository should generate to power the March Madness 2026 website.

The core rule is simple:

- Keep the full team-season matrix for training and offline analysis.
- Export a separate set of JSON artifacts for the website.
- Validate those JSON artifacts against schemas before publishing them.

## Target Outputs

The website currently expects these JSON artifacts:

1. `data/web/index/{season}/teams.json`
Purpose: lightweight team search index and route bootstrap.

2. `data/web/features/{season}/base.json`
Purpose: compact feature store for client-side inference and model-aware UI.

3. `data/web/ts/{team_id}_{season}.json`
Purpose: rich Team Insights payload.

4. `data/web/brackets/{season}.json`
Purpose: bracket dependency graph and slot definitions.

5. `data/web/models/manifest.json`
Purpose: model discovery, compatibility, and loading instructions.

The training artifact should remain separate:

6. `TeamSeasons_{season}.csv`
Purpose: canonical model training and offline analysis matrix.

## Included Schemas

This folder contains JSON Schema files for the five website-facing JSON outputs:

- [team-index.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/team-index.schema.json)
- [team-page.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/team-page.schema.json)
- [feature-store.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/feature-store.schema.json)
- [bracket-definition.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/bracket-definition.schema.json)
- [model-manifest.schema.json](/Users/andrewgrowney/Code/React/andrewgrowney.com/docs/march-madness-ml/schemas/model-manifest.schema.json)

## Recommended Export Pipeline

Use a four-stage export process.

### 1. Build Internal Team-Season Objects

Create a single in-memory representation for each `(team_id, season)` containing:

- identity
- conference
- seed and region if applicable
- regular season record
- quadrant wins and losses
- stat means and rankings
- ordinal data such as NET or RPI
- tournament summary
- similar teams

This should be the source for both the training CSV and the website JSON artifacts.

### 2. Emit the Training Matrix

Write `TeamSeasons_{season}.csv` as the canonical training export.

This file is allowed to be wide and model-oriented. It should not be used directly by the website for normal page loads.

### 3. Emit Browser-Facing JSON Artifacts

From the same internal data, export the website contracts.

#### Team Index

Path:
`data/web/index/{season}/teams.json`

Rules:

- Keep it small.
- Include only the fields needed for search, filtering, route building, and bracket labels.
- Reference the rich team page JSON through `team_page_path`.

#### Team Page Payloads

Path:
`data/web/ts/{team_id}_{season}.json`

Rules:

- This is the only website artifact that should contain deeper resume and game-level detail.
- It is fetched on demand, so moderate size is acceptable.
- Use a consistent null strategy for tournament and similarity edge cases.

#### Feature Store

Path:
`data/web/features/{season}/base.json`

Rules:

- Use a compact named feature set.
- Include `feature_order` explicitly.
- Store team vectors in an object keyed by TeamID string for constant-time lookup in the browser.
- Keep values numeric so they can be consumed directly by runtime inference code.

#### Bracket Definition

Path:
`data/web/brackets/{season}.json`

Rules:

- Every slot must be explicit.
- Later rounds must refer to earlier slot winners through `source_slot`.
- Do not make the website infer bracket structure from seed labels alone.

#### Model Manifest

Path:
`data/web/models/manifest.json`

Rules:

- Be honest about model type.
- If a model is a heuristic or runtime config, say so.
- Do not label a scikit-learn artifact as TensorFlow.js unless it actually ships as a TFJS loadable model.

### 4. Validate Before Publish

Each generated JSON artifact should be validated against its schema before being committed or published.

Recommended checks:

1. Every `team_page_path` in the team index exists.
2. Every team referenced in the bracket exists in the season team index.
3. Every TeamID in the feature store exists in the team index.
4. Every `source_slot` in the bracket points to a valid slot.
5. `default_model_id` exists in the model manifest.
6. Every manifest `feature_set` corresponds to a real exported feature store.

## Source-to-Artifact Mapping

This is the intended mapping from your existing data work to the website exports.

### Kaggle Raw Data

Primary inputs:

- `MTeams.csv`
- `MTeamConferences.csv`
- `MTeamCoaches.csv`
- `MRegularSeasonDetailedResults.csv`
- `MNCAATourneyDetailedResults.csv`
- `MNCAATourneySeeds.csv`
- `MMasseyOrdinals.csv`
- `MSeasons.csv`

### Feature Engineering Layer

This is your existing `TeamSeason` and associated utilities layer.

Responsibilities:

- compute means and deviations
- compute adjusted metrics
- compute SOS and SOV
- compute quadrant splits
- compute ordinal snapshots
- compute historical similarity

### Website Export Layer

Add a dedicated export step after feature engineering.

Suggested functions:

1. `build_team_index(team_seasons, season)`
2. `build_feature_store(team_seasons, season, feature_set_name, feature_columns)`
3. `build_team_page_payload(team_season)`
4. `build_bracket_definition(seeds_df, bracket_source, season)`
5. `build_model_manifest(model_records)`
6. `validate_artifacts(output_root, season)`

## Recommended Feature Set Strategy

Do not expose the entire training matrix to the browser by default.

Start with one compact feature set such as `base_{season}`.

Example columns:

- `Seed`
- `WinPct`
- `SOS`
- `SOV`
- `NET_last`
- `AdjOE_mean`
- `AdjDE_mean`
- `AdjNE_mean`
- `FG%_mean`
- `FG3%_mean`
- `FT%_mean`

Later, if needed, add additional named feature stores such as:

- `resume_{season}`
- `shooting_{season}`
- `full_{season}`

The website should always refer to a named feature set, not an ad hoc list of columns.

## Suggested Folder Layout in `march-madness-ml`

```text
data/
  web/
    index/
      2026/
        teams.json
    features/
      2026/
        base.json
    ts/
      1242_2026.json
      1277_2026.json
    brackets/
      2026.json
    models/
      manifest.json
TeamSeasons_2026.csv
```

## Practical Notes for Your Repo

1. Keep `TeamSeasons_{season}.csv` because it is still the right training and audit artifact.

2. Add a browser export command instead of overloading the training export.

3. Avoid versioned folders such as `ts_v4` in the long-term published path.
Use stable paths and version your pipeline in git instead.

4. If you want a reference model before TensorFlow.js exists, prefer one of these:

- a runtime-config JSON built from manual weights
- a coefficient export for a linear model that can be reproduced in JavaScript
- a season-specific precomputed matchup table

5. Make schema validation part of CI or the export command itself so the website contract cannot drift silently.

## First Milestone

The most useful first milestone in `march-madness-ml` is:

1. export `teams.json`
2. export `base.json`
3. export `2026.json` bracket definition
4. validate them against the schemas

That is enough to switch the bracket page off mocked data.

The second milestone is exporting `ts/{team_id}_{season}.json`, which is enough to switch Team Insights off mocked data.

## Current V1 Scope

The current implementation scope for the first pass is:

- Season scope: `2026` only
- Publish artifacts from this repo under `data/web/`
- Use Kaggle `TeamName` for both `name` and `short_name`
- Build the team index from all teams appearing in 2026 regular-season data
- Use `2026_initial` as the first browser feature set
- Use historical similarity matches from `2003` through `2025`
- Emit top `5` similar teams
- Keep team pages thin for the first pass
- Use a mock model manifest for now
- Defer schema validation for the initial implementation

### Locked Initial Feature Set

`2026_initial` will contain:

- `Seed`
- `WinPct`
- `SOS`
- `SOV`
- `NET_last`
- `AdjOE_mean`
- `AdjDE_mean`
- `AdjNE_mean`
- `FG%_mean`
- `FG3%_mean`
- `FT%_mean`

### Placeholder 2026 Bracket Strategy

Until the official 2026 bracket is released, the exporter should synthesize a placeholder 2026 bracket by copying the 2025 Kaggle bracket structure and seed placement forward to season `2026`.

This includes:

- copying 2025 slot rows into a 2026 working bracket source
- using the same region and slot naming conventions
- marking the bracket as a placeholder in implementation notes and git history

Once Selection Sunday data is available, rerun the exporter against the real 2026 seeds and slots.

## Implementation Plan

### Phase 1. Export Scaffolding

Create a dedicated website export module and CLI entrypoint for 2026 website artifacts.

Tasks:

1. Add a dedicated export module under `src/mm_analytics/`.
2. Keep browser export separate from the training CSV flow.
3. Add a CLI command that targets season `2026` and an output root.
4. Establish stable output paths under `data/web/`.

### Phase 2. Shared Team-Season Build

Build the internal 2026 team-season objects once and reuse them across all website outputs.

Tasks:

1. Load 2026 regular-season team data from Kaggle inputs.
2. Build `TeamSeason` objects from the existing feature-engineering layer.
3. Compute similarity from the `2003-2025` historical pool.
4. Lock the `2026_initial` feature set for browser inference.
5. Apply feature-specific defaults only where missing values require them.

### Phase 3. First Artifact Exports

Export the first website contracts needed to switch the bracket page off mocked data.

Tasks:

1. Export `data/web/index/2026/teams.json`.
2. Export `data/web/features/2026/base.json`.
3. Export `data/web/brackets/2026.json` using the placeholder 2025-derived bracket.
4. Export `data/web/models/manifest.json` with a mock default model.
5. If low-cost, also export thin team pages under `data/web/ts/`.

### Phase 4. Contract Cleanup

Align the existing exporter code with the new published website contracts.

Tasks:

1. Replace long-term reliance on versioned output paths such as `ts_v4`.
2. Add adapter functions to map current internal objects into the new website schemas.
3. Preserve the existing stat, ordinal, and similarity logic unless a concrete export bug requires change.

### Phase 5. Smoke Checks

Before using the exported artifacts in the website, run lightweight consistency checks even though formal schema validation is deferred.

Tasks:

1. Confirm team index count matches the 2026 regular-season team universe.
2. Confirm every feature-store TeamID appears in the team index.
3. Confirm bracket references resolve to valid slots.
4. Confirm the manifest references the exported feature set.

### Recommended Execution Order

1. Export `teams.json`
2. Export `base.json`
3. Export `2026.json` bracket definition
4. Export the mock manifest
5. Add thin team pages