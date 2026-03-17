# Stat Explorer Enhancement

## Goal

Define the browser-facing data contract needed to power the stat explorer graph on `https://andrewgrowney.com/mm/model/`.

The stat explorer is a stat-centric view, not a team-detail view.

For a selected feature like `AdjOE_mean`, the UI should let a user:

1. See only 2026 tournament teams as the dot/rank list on the far left.
2. Filter those 2026 tournament teams by whole field, region, or one selected team.
3. Compare those 2026 teams against historical finish distributions from 2003 to 2025.
4. Support the full browser-training feature set so the same feature definitions can power both visualization and in-browser modeling.

## Product Constraints

These are hard requirements for the first stat explorer version.

1. Only 2026 tournament teams appear as the left-side dots/rank list.
2. The default view is the full 2026 tournament field.
3. The user can filter 2026 teams by region, one team, or whole field.
4. Non-tournament teams must not appear in the stat explorer dots, but their features should still be published for on-demand team pages and browser model workflows.
5. All features in the current notebook feature groups must be available to browser training and to the stat explorer.

## Feature Scope

The intended stat explorer feature set is:

```python
RESUME_FEATURES = ['Q1_WinPct', 'Q2_WinPct', 'Q3_WinPct', 'Q4_WinPct', 'SOS', 'SOV']

TEMPO_FEATURES = ['Poss_mean', 'Fouls_mean']

OFF_FEATURES = ['AdjOE_mean', 'AdjNE_mean', 'EFG%_mean', 'FG3%_mean', 'FT%_mean', 'FTA_mean', 'TO_mean', 'Ast_mean', 'OR_mean', 'FGA3_mean']

DEF_FEATURES = ['AdjDE_mean', 'Stl_mean', 'Blk_mean', 'OppTO_mean', 'DR_mean', 'OppFGA3_mean']

FEATURE_GROUPS = {
		'Resume': RESUME_FEATURES,
		'Tempo': TEMPO_FEATURES,
		'Offense': OFF_FEATURES,
		'Defense': DEF_FEATURES,
}
```

Flattened feature order for the explorer and browser training:

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

## What Exists Today

### Already Available in Current Contracts

#### Team index

Current artifact: `data/web/index/{season}/teams.json`

Current schema: `team-index.schema.json`

Useful fields already present:

- `team_id`
- `name`
- `seed`
- `region`
- `tournament_team`
- `team_page_path`

This is already enough to power:

- whole-field default filter
- region filter
- single-team filter list
- exclusion of non-tournament teams from the 2026 dots

#### Team page payload

Current artifact: `data/web/ts/{team_id}_{season}.json`

Current schema: `team-page.schema.json`

Useful fields already present:

- `stats`
- `stat_rankings`
- `tournament`
- `similar_teams`

This remains the right on-demand detail view for one team, but it is not the right primary contract for the stat explorer because the explorer is stat-first and needs cohort data for many teams at once.

#### Feature store

Current artifact: `data/web/features/{season}/stat_explorer_v1.json`

Current schema: `feature-store.schema.json`

Useful properties already present:

- `season`
- `feature_set`
- `feature_order`
- `teams`

This is already the right shape for browser inference and in-browser training.

#### Historical browser-training exports

Documented artifacts:

- `data/web/features/{season}/stat_explorer_v1.json`
- `data/web/training/stat_explorer_v1/{season}.json`
- `data/web/training/manifest.json`

These already establish the idea that browser training uses season feature stores plus a manifest-driven feature order.

### Gaps in Existing Contracts

The current exported contracts do not yet provide a clean browser-facing artifact for:

1. Historical team-level finish cohorts by stat.
2. A stat-first dataset that mixes 2026 tournament-team filters with historical finish distributions.
3. A browser feature set containing all 24 selected features.
4. A stable round-bucket definition for the explorer labels like `Round of 64`, `Round 32`, `Sweet 16`, `Elite 8`, `Final Four`, `Championship`, and `Champions`.

This is the main reason the generalized feature-store and browser-training exports are not sufficient by themselves for the visual shown in the mock.

## Design Recommendation

Use a two-layer approach.

### Layer 1: Expand the Browser Feature Stores

Keep using the existing `feature-store.schema.json` shape, but publish a richer feature set for both 2026 and historical seasons.

Recommended new feature set name:

- `stat_explorer_v1`

Recommended files:

- `data/web/features/2026/stat_explorer_v1.json`
- `data/web/features/{season}/stat_explorer_v1.json` for historical seasons used by browser training

Rules:

- Include all 24 selected features.
- Include all teams, not only tournament teams.
- Preserve numeric-only values.
- Use one canonical `feature_order` shared by serving and training.

This satisfies constraint 3 because non-tournament teams remain available to the website when a team page needs them.

### Layer 2: Add a Dedicated Stat Explorer Artifact

Create a new browser-facing artifact specifically for this visualization.

Recommended path:

- `data/web/stats/2026/stat_explorer_v1.json`

Reasoning:

- The explorer is not just a feature store lookup.
- It needs 2026 tournament-team membership and filters.
- It needs historical finish buckets for each feature.
- It should load quickly without forcing the browser to fetch 23 historical feature-store files and recompute every cohort live.

## Why a Dedicated Artifact Is Preferable

Alternative considered:

- Reconstruct the explorer in the browser from `team-index`, `feature-store`, and all historical feature files.

Why that is weaker:

- Requires many network requests.
- Historical feature stores do not themselves carry all explorer labels and finish-bucket metadata.
- Pushes cohort summarization work into the client for every page load.
- Makes UI logic too dependent on exporter internals instead of a stable contract.

The explorer should consume a purpose-built artifact while browser training continues to use the generalized feature stores and training manifest.

## Proposed Stat Explorer Payload

Proposed top-level shape:

```json
{
	"season": 2026,
	"feature_set": "stat_explorer_v1",
	"historical_range": [2003, 2025],
	"feature_groups": {
		"Resume": ["Q1_WinPct", "Q2_WinPct", "Q3_WinPct", "Q4_WinPct", "SOS", "SOV"],
		"Tempo": ["Poss_mean", "Fouls_mean"],
		"Offense": ["AdjOE_mean", "AdjNE_mean", "EFG%_mean", "FG3%_mean", "FT%_mean", "FTA_mean", "TO_mean", "Ast_mean", "OR_mean", "FGA3_mean"],
		"Defense": ["AdjDE_mean", "Stl_mean", "Blk_mean", "OppTO_mean", "DR_mean", "OppFGA3_mean"]
	},
	"feature_order": ["Q1_WinPct", "Q2_WinPct", "..."],
	"round_buckets": [
		{ "key": "round64", "label": "Round of 64", "exit_round_labels": ["Play In", "First Round"] },
		{ "key": "round32", "label": "Round 32", "exit_round_labels": ["Second Round"] },
		{ "key": "sweet16", "label": "Sweet 16", "exit_round_nums": [3] },
		{ "key": "elite8", "label": "Elite 8", "exit_round_nums": [4] },
		{ "key": "final4", "label": "Final Four", "exit_round_nums": [5] },
		{ "key": "championship", "label": "Championship", "exit_round_nums": [6] },
		{ "key": "champion", "label": "Champions", "exit_round_nums": [7] }
	],
	"filters": {
		"default_scope": "field",
		"regions": [
			{ "key": "W", "label": "East" },
			{ "key": "X", "label": "South" },
			{ "key": "Y", "label": "Midwest" },
			{ "key": "Z", "label": "West" }
		]
	},
	"current_field": {
		"teams": []
	},
	"historical_distributions": {},
	"historical_summary": {},
	"default_percentiles": {}
}
```

### `current_field.teams`

Purpose:

- power the far-left 2026 dots/ranks
- support whole field, region, and team filtering

Recommended entry shape:

```json
{
	"team_id": 1242,
	"name": "Kansas",
	"short_name": "Kansas",
	"seed": 4,
	"region": "Midwest",
	"region_key": "Y",
	"slot": "Y04",
	"tournament_team": true,
	"stats": {
		"AdjNE_mean": 37.34,
		"AdjOE_mean": 130.37,
		"AdjDE_mean": 93.03
	}
}
```

Notes:

- Include only 2026 tournament teams here.
- Reuse region/slot semantics from `team-index.schema.json`.
- Keep this section stat-dense and resume-light.

### `historical_distributions`

Purpose:

- power the right-hand box/whisker or percentile bands for each feature and finish bucket
- support browser-side year-range filtering where quartiles and percentiles must be recomputed from the filtered historical rows

Recommended shape:

```json
{
	"AdjNE_mean": [
		{ "season": 2003, "bucket": "round64", "value": 14.2 },
		{ "season": 2003, "bucket": "sweet16", "value": 27.8 },
		{ "season": 2004, "bucket": "final4", "value": 30.1 }
	]
}
```

This is the most flexible form for browser-side custom rendering, density estimation, and hover detail.

Design note:

- if the user can filter historical seasons in the UI, summary-only exports are not enough
- the browser needs raw historical values with season metadata so it can recompute quartiles for the selected year range
- this is not too heavy for in-browser computation at the current scale
- the heavier cost is payload design, not percentile math

Optional richer row shape:

```json
{
	"AdjNE_mean": [
		{ "season": 2003, "bucket": "round64", "value": 14.2, "team_id": 1437, "team_name": "Tulsa", "seed": 13, "exit_round": "First Round" },
		{ "season": 2003, "bucket": "sweet16", "value": 27.8, "team_id": 1242, "team_name": "Kansas", "seed": 2, "exit_round": "Sweet Sixteen" },
		{ "season": 2004, "bucket": "final4", "value": 30.1, "team_id": 1163, "team_name": "Connecticut", "seed": 2, "exit_round": "Final Four" }
	]
}
```

This is preferable to storing only naked numeric arrays because the frontend can:

- filter by historical season range
- recompute quartiles
- recompute percentiles for a selected 2026 team
- still derive boxplot summaries on demand

### `historical_summary`

Purpose:

- avoid recomputing quartiles and whiskers in the client when the chart only needs summary stats
- support tooltips and explanatory text

Recommended shape:

```json
{
	"AdjNE_mean": {
		"round32": {
			"count": 767,
			"min": -5.2,
			"p10": 8.1,
			"q1": 12.3,
			"median": 17.9,
			"q3": 22.8,
			"p90": 28.4,
			"max": 40.7
		}
	}
}
```

Recommendation:

- export both raw historical rows and `historical_summary` in v1
- use `historical_summary` for fast default render of the full 2003 to 2025 range
- use raw historical rows when the user changes the historical year filter
- if payload size becomes a problem, reduce repeated summary fields before dropping raw rows

## Mapping the Mock to the Data Contract

### Left panel: `2026 Teams`

Needed data:

- `team_id`
- `name`
- `seed`
- `region`
- selected feature value

Source recommendation:

- `current_field.teams`

### Stat selector

Needed data:

- feature keys
- display grouping

Source recommendation:

- `feature_groups`
- `feature_order`

### Historical finishes columns

Needed data:

- stable round bucket order
- distribution values or summary stats by feature and round bucket

Source recommendation:

- `round_buckets`
- `historical_distributions`
- `historical_summary`

Recommended rendering behavior:

- initial page load can use precomputed `historical_summary` for the default `2003 - 2025` range
- when the user changes the historical year range, recompute quartiles in the browser from raw historical rows

## Round Bucket Decisions

The mock uses these columns:

- `Round of 64`
- `Round 32`
- `Sweet 16`
- `Elite 8`
- `Final Four`
- `Championship`
- `Champions`

Recommended mapping from notebook data:

- `Round of 64` = `ExitRound in {"Play In", "First Round"}`
	- this explicitly combines play-in losers and first-round losers into one explorer bucket
- `Round 32` = `ExitRound == "Second Round"`
- `Sweet 16` = `ExitRoundNum == 3`
- `Elite 8` = `ExitRoundNum == 4`
- `Final Four` = `ExitRoundNum == 5`
- `Championship` = `ExitRoundNum == 6`
- `Champions` = `ExitRoundNum == 7`

This bucket mapping should be exported as data, not hardcoded in the frontend.

Implementation note:

- for the first two buckets, the exporter should group by `ExitRound` label instead of relying only on `ExitRoundNum`, because both `Play In` and `First Round` currently map to `ExitRoundNum = 1`

## What Can Reuse Existing Schemas

### Keep using existing schemas without change

- `team-index.schema.json`
	- still correct for season-level team filters and route bootstrap
- `feature-store.schema.json`
	- still correct for richer browser feature stores and in-browser model training
- `team-page.schema.json`
	- still correct for on-demand team detail payloads

### Missing schema that should likely be added

- `stat-explorer.schema.json`
	- added at `docs/march-madness-ml/schemas/stat-explorer.schema.json`

Reason:

- the explorer is a first-class published website artifact with its own contract
- it should be validated before publish like the other browser JSON outputs

## Questions Answered Now

### Can the current notebook EDA power the stat explorer conceptually?

Yes.

The notebook already proves the right analytical ingredients:

- 2026 team feature values
- historical tournament-team feature values
- exit-round mapping
- finish-bucket grouping

### Are the existing website artifacts enough by themselves?

No.

They cover parts of the problem, but not the stat explorer end-to-end.

### Should the explorer use all 2026 teams or only tournament teams?

Only tournament teams in the visible explorer dots.

### Should non-tournament teams still be exported?

Yes.

They should remain in the feature store and team pages for other website flows.

### Can region filtering already be supported?

Yes for 2026 tournament teams, because `team-index` already exports `region` for tournament entries.

### Is browser-side quartile recomputation too heavy?

No.

At this scale it is reasonable.

The explorer only needs to recompute distributions for one selected stat across a modest historical tournament dataset. The cost is far more about shipping raw historical rows in a clean contract than about CPU time in the browser.

### Should `Round of 64` include both `Play In` and `First Round` losers?

Yes.

That is now a fixed product rule for the explorer.

### Should `stat_explorer_v1` become the new browser default feature set?

Yes.

This should replace the current thin browser default so the website and browser training use the same canonical feature order.

### Do we need one explorer artifact per season?

No for v1.

The current use case is specifically to show how 2026 teams compare against a user-selected historical range, so one explorer artifact for 2026 backed by historical rows from 2003 to 2025 is sufficient.

### Should regions be pre-resolved to friendly names?

Yes.

The explorer artifact should publish friendly region labels for direct UI use.

## Outstanding Questions Before Implementation Planning

Only one design question remains before writing the build plan.

1. Should the explorer artifact precompute percentiles for every 2026 tournament team by feature and round bucket, or should the frontend compute percentiles from raw filtered historical rows?

Current recommendation:

- compute percentiles in the frontend from raw filtered historical rows
- keep the exporter responsible for raw rows, bucket definitions, and default summaries
- avoid precomputing percentiles in the artifact because those percentiles would become stale as soon as the user changes the historical year range

Exception:

- if the initial page needs very fast default rendering for all 2026 teams at once, the artifact could optionally include `default_percentiles` for the full `2003 - 2025` range only
- these should be treated as a render optimization, not as the canonical source of truth

## Recommended Next Design Step

Before implementation planning, define the new `stat-explorer.schema.json` contract around this payload strategy:

1. Raw historical rows plus default summaries
	 - supports browser-side year filtering
	 - supports browser-side percentile recomputation
	 - keeps the full-range default view fast

The current recommendation is to treat this as the default v1 design.

## Implementation Plan

This plan is scoped to the data-producing side of this repository.

The goal is to make the stat explorer payload available to the website with one canonical browser feature set, one 2026 explorer artifact, and one consistent historical export path for browser training.

### Phase 1: Introduce the New Browser Feature Set

Goal:

- make `stat_explorer_v1` the new default browser feature set for both 2026 serving and historical browser training

Primary code touchpoints:

- `src/mm_analytics/web_export.py`
- `cli.py`
- `model_2026.py`

Changes:

1. Add a new constant in `src/mm_analytics/web_export.py` for the stat explorer feature set name.
2. Add a new constant for the full 24-feature browser order.
3. Replace the current thin `INITIAL_FEATURE_COLUMNS` default in `bootstrap_web_export()` with the new 24-feature set.
4. Replace the current `HISTORICAL_FEATURE_COLUMNS` default in `bootstrap_historical_training_export()` with the same 24-feature set.
5. Keep the existing feature-store schema and file structure, but export:
	- `data/web/features/2026/stat_explorer_v1.json`
	- `data/web/features/{season}/stat_explorer_v1.json` for historical seasons
6. Update CLI defaults in `cli.py`:
	- `export-web --feature-set` default becomes `stat_explorer_v1`
	- `export-web-history --feature-set` default becomes `stat_explorer_v1`
7. Update browser-training defaults in `model_2026.py` so the local training path points to the new 2026 stat explorer feature store.

Acceptance criteria:

- 2026 browser feature store contains all 24 selected features.
- historical browser feature stores contain the same canonical feature order.
- browser training manifest uses the same feature order as the 2026 serving store.

### Phase 2: Add Stat Explorer Export Paths

Goal:

- teach the exporter how to write the dedicated explorer artifact

Primary code touchpoints:

- `src/mm_analytics/web_export.py`

Changes:

1. Extend the export path model with a new stat explorer output path:
	- `data/web/stats/2026/stat_explorer_v1.json`
2. Update directory creation helpers so `data/web/stats/2026/` is created automatically.
3. Add a small helper for stat explorer path resolution similar to `build_feature_store_path()`.

Acceptance criteria:

- running the exporter creates the stats output directory and writes a single 2026 explorer artifact path in a stable location.

### Phase 3: Build the 2026 Current Field Payload

Goal:

- export the left-side team list used by the stat explorer

Primary code touchpoints:

- `src/mm_analytics/web_export.py`

Changes:

1. Add a helper like `build_stat_explorer_current_field(...)`.
2. Source the 2026 field from the same season objects already used by `build_team_index()` and `build_feature_store()`.
3. Include only tournament teams.
4. Publish friendly regions in the explorer payload:
	- `region = East | South | Midwest | West`
	- optional `region_key = W | X | Y | Z`
5. Include the 24-feature stat map for each visible 2026 tournament team.

Notes:

- non-tournament teams remain in the feature store and team pages, but must not appear in `current_field.teams`.

Acceptance criteria:

- every `current_field.teams` entry is a 2026 tournament team.
- every 2026 seeded team from the team index appears exactly once in `current_field.teams`.
- friendly region labels match the bracket metadata.

### Phase 4: Build Historical Round Buckets and Raw Rows

Goal:

- export the historical rows needed for browser-side year filtering and percentile recomputation

Primary code touchpoints:

- `src/mm_analytics/web_export.py`

Changes:

1. Add a helper to normalize historical exit rounds into explorer buckets.
2. Use these bucket rules:
	- `round64` = `Play In` or `First Round`
	- `round32` = `Second Round`
	- `sweet16` = `ExitRoundNum == 3`
	- `elite8` = `ExitRoundNum == 4`
	- `final4` = `ExitRoundNum == 5`
	- `championship` = `ExitRoundNum == 6`
	- `champion` = `ExitRoundNum == 7`
3. Iterate through historical team-seasons from 2003 to 2025 and collect one raw row per team, per selected feature.
4. Store rows under `historical_distributions[feature_name]` with at least:
	- `season`
	- `bucket`
	- `value`
5. Include optional row fields if low-cost:
	- `team_id`
	- `team_name`
	- `seed`
	- `exit_round`

Notes:

- this phase should use label-aware logic for `Round of 64` rather than relying only on `ExitRoundNum`.

Acceptance criteria:

- every raw historical row maps to one valid explorer bucket.
- `Round of 64` includes both play-in losers and first-round losers.
- the artifact supports recomputing quartiles for arbitrary historical year subranges.

### Phase 5: Build Default Historical Summaries

Goal:

- precompute the default `2003 - 2025` summaries for fast first render

Primary code touchpoints:

- `src/mm_analytics/web_export.py`

Changes:

1. Add a helper like `build_stat_explorer_summary(...)`.
2. For each feature and each bucket, compute:
	- `count`
	- `min`
	- `q1`
	- `median`
	- `q3`
	- `max`
3. Optionally compute `p10`, `p90`, `mean`, and `std` while the rows are in memory.
4. Store these under `historical_summary` for the full default year range.

Recommendation:

- do not make summary generation conditional in v1
- the default page load benefits from always having this available

Acceptance criteria:

- every feature has summary entries for every bucket with at least one historical row.
- the website can render the default full-range explorer without recomputing summaries client-side.

### Phase 6: Decide and Implement Percentile Handling

Goal:

- keep the artifact correct under user-driven year filtering while allowing optional render optimization

Primary code touchpoints:

- `src/mm_analytics/web_export.py`
- `docs/march-madness-ml/schemas/stat-explorer.schema.json`

Recommended v1 behavior:

1. Treat raw historical rows as the source of truth for percentile computation.
2. Do not require exporter-side percentiles for correctness.
3. Optionally export `default_percentiles` for the default `2003 - 2025` range only if the website needs faster initial rendering for all 2026 teams.

Acceptance criteria:

- explorer behavior remains correct when the user changes the historical year range.
- optional default percentiles are clearly documented as an optimization only.

### Phase 7: Add Explorer Export Validation

Goal:

- validate the explorer artifact with the same rigor as the other browser outputs

Primary code touchpoints:

- `src/mm_analytics/web_export.py`
- `docs/march-madness-ml/schemas/stat-explorer.schema.json`

Changes:

1. Add smoke checks for the explorer payload:
	- all `current_field.teams` are tournament teams
	- all `region` values are friendly labels and all `region_key` values are valid
	- all bucket keys in `historical_distributions` exist in `round_buckets`
	- all bucket keys in `historical_summary` exist in `round_buckets`
	- all feature keys in `historical_distributions` and `historical_summary` exist in `feature_order`
2. Add a summary block to exporter output so the CLI reports explorer row counts and bucket coverage.

Acceptance criteria:

- malformed explorer payloads fail during export rather than at website runtime.

### Phase 8: Wire the Export into the Existing CLI Flow

Goal:

- make the new artifact part of the normal export workflow

Primary code touchpoints:

- `cli.py`
- `src/mm_analytics/web_export.py`

Changes:

1. Update `export-web` so it writes:
	- team index
	- 2026 feature store using `stat_explorer_v1`
	- bracket
	- team pages
	- stat explorer artifact
2. Keep `export-web-history` responsible for:
	- historical feature stores
	- historical matchup training files
	- training manifest
3. Decide whether `export-web` should depend on pre-existing historical files or compute the explorer’s historical rows directly from raw inputs.

Recommendation:

- compute the stat explorer’s historical rows directly from raw inputs inside `export-web`
- do not make `export-web` depend on prior `export-web-history` execution just to populate the explorer artifact

Acceptance criteria:

- a single `export-web` invocation can produce the 2026 website-serving artifacts including the explorer payload.

### Phase 9: Update Downstream Defaults and Documentation

Goal:

- keep docs and local tooling aligned with the new browser default

Primary code touchpoints:

- `model_2026.py`
- `docs/stat_explorer.md`
- `docs/march-madness-ml/march-madness-ml-guide.md`
- `docs/MM_2026.md`

Changes:

1. Replace remaining legacy references so the docs consistently point to `stat_explorer_v1` as the browser default.
2. Add the new explorer artifact to the website export guide.
3. Document that `stat_explorer_v1` is now the canonical browser feature set.
4. Document that the explorer artifact is 2026-serving data, while historical feature stores and training files remain separate browser-training artifacts.

Acceptance criteria:

- docs do not describe conflicting browser feature defaults.
- the exporter outputs and docs match the actual contract paths.

Status:

- completed for the core website docs and local browser defaults.
- `stat_explorer_v1` is now the canonical browser feature set for the serving feature store, stat explorer payload, and browser-training defaults.

## Proposed Build Order

Recommended implementation order:

1. Expand feature-set constants and defaults.
2. Add stat explorer export path support.
3. Build `current_field.teams`.
4. Build historical bucket normalization and raw rows.
5. Build default summaries.
6. Add smoke checks.
7. Update CLI summary output.
8. Update docs and local training defaults.

This order keeps the highest-risk data-shape work in the middle while preserving a simple path to incremental validation.

## Definition of Done

The stat explorer export is ready for website integration when all of the following are true:

1. `export-web` writes `data/web/stats/2026/stat_explorer_v1.json`.
2. `export-web` writes `data/web/features/2026/stat_explorer_v1.json` as the default 2026 browser feature store.
3. `export-web-history` writes `data/web/features/{season}/stat_explorer_v1.json` and the matching training manifest feature order.
4. The explorer payload validates against `stat-explorer.schema.json`.
5. The explorer payload contains only 2026 tournament teams on the left side.
6. The explorer payload contains raw historical rows that support year filtering from 2003 to 2025.
7. `Round of 64` correctly includes both `Play In` and `First Round` losers.
8. Friendly regions are published for direct UI use.
9. Local browser-training paths and docs point to the new default feature set.

## UI Adoption Handoff

Use these two artifacts as the primary UI inputs:

- `data/web/stats/2026/stat_explorer_v1.json`
- `data/web/features/2026/stat_explorer_v1.json`

The current exported stat explorer payload contains:

- `season: 2026`
- `feature_set: stat_explorer_v1`
- `historical_range: [2003, 2025]`
- `24` features in `feature_order`
- `68` tournament teams in `current_field.teams`
- `35244` historical rows across all features in `historical_distributions`
- `default_percentiles: {}` intentionally left empty in v1

### Recommended UI Fetch Model

Fetch once on page load:

```ts
const [explorer, featureStore, manifest] = await Promise.all([
	fetch('/mm/data/web/stats/2026/stat_explorer_v1.json').then((r) => r.json()),
	fetch('/mm/data/web/features/2026/stat_explorer_v1.json').then((r) => r.json()),
	fetch('/mm/data/web/models/manifest.json').then((r) => r.json()),
]);
```

Recommended usage split:

- `stat_explorer_v1.json` drives the stat explorer view, cohort summaries, year filtering, region filtering, and left-side team dots.
- `features/2026/stat_explorer_v1.json` remains the canonical numeric lookup table for model-aware UI and any browser-side feature access outside the explorer.
- `models/manifest.json` remains the source of truth for model selection and `feature_set` compatibility.

### Explorer Contract You Can Bind To

Top-level keys used directly by the UI:

- `feature_groups`: render grouped feature navigation.
- `feature_order`: preserve consistent stat ordering across tabs and selectors.
- `round_buckets`: render finish-series labels in the intended order.
- `filters.regions`: use directly for region pills or dropdown labels.
- `current_field.teams`: render the 2026 tournament field dots and team selector.
- `historical_distributions`: raw historical rows used to recompute filtered cohorts in the browser.
- `historical_summary`: default all-years summary stats for initial render.

`current_field.teams[]` shape:

```json
{
	"team_id": 1181,
	"name": "Duke",
	"short_name": "Duke",
	"seed": 1,
	"region": "East",
	"region_key": "W",
	"slot": "W01",
	"tournament_team": true,
	"stats": {
		"Q1_WinPct": 0.894737,
		"AdjNE_mean": 48.004961,
		"AdjDE_mean": 81.162401
	}
}
```

`historical_distributions[featureKey][]` row shape:

```json
{
	"season": 2003,
	"bucket": "elite8",
	"value": 0.75,
	"team_id": 1112,
	"team_name": "Arizona",
	"seed": 1,
	"exit_round": "Elite Eight"
}
```

`historical_summary[featureKey][bucket]` shape:

```json
{
	"count": 88,
	"min": 0.0,
	"p10": 0.3,
	"q1": 0.461538,
	"median": 0.583333,
	"q3": 0.666667,
	"p90": 0.805455,
	"max": 1.0,
	"mean": 0.562156,
	"std": 0.197289
}
```

### Browser-Side Recompute Rules

Use `historical_summary` only for the default `2003-2025` initial render. When the user changes historical year range:

- filter `historical_distributions[selectedFeature]` by `season`
- regroup by `bucket`
- recompute `count`, `min`, `p10`, `q1`, `median`, `q3`, `p90`, `max`, `mean`, and `std`
- treat `round64` as the union of `Play In` and `First Round`, which is already encoded in the exported rows

### Minimal TypeScript Types

```ts
type StatKey = string;
type RoundBucketKey =
	| 'round64'
	| 'round32'
	| 'sweet16'
	| 'elite8'
	| 'final4'
	| 'championship'
	| 'champion';

type CurrentFieldTeam = {
	team_id: number;
	name: string;
	short_name: string;
	seed: number | null;
	region: string | null;
	region_key: string | null;
	slot: string | null;
	tournament_team: boolean;
	stats: Record<StatKey, number | null>;
};

type HistoricalDistributionRow = {
	season: number;
	bucket: RoundBucketKey;
	value: number;
	team_id: number;
	team_name: string;
	seed: number | null;
	exit_round: string;
};

type HistoricalBucketSummary = {
	count: number;
	min: number;
	p10: number;
	q1: number;
	median: number;
	q3: number;
	p90: number;
	max: number;
	mean: number;
	std: number;
};

type StatExplorerPayload = {
	season: number;
	feature_set: string;
	historical_range: [number, number];
	feature_groups: Record<string, StatKey[]>;
	feature_order: StatKey[];
	round_buckets: Array<{
		key: RoundBucketKey;
		label: string;
		exit_round_labels?: string[];
		exit_round_nums?: number[];
	}>;
	filters: {
		default_scope: 'field' | string;
		regions: Array<{ key: string; label: string }>;
		historical_range_default: [number, number];
		historical_range_allowed: [number, number];
	};
	current_field: {
		teams: CurrentFieldTeam[];
	};
	historical_distributions: Record<StatKey, HistoricalDistributionRow[]>;
	historical_summary: Record<StatKey, Partial<Record<RoundBucketKey, HistoricalBucketSummary>>>;
	default_percentiles: Record<string, never>;
};
```

### UI Integration Checklist

1. Treat `feature_order` as canonical instead of hardcoding the 24 feature keys in the app.
2. Use `filters.regions` and `region_key` directly instead of maintaining a separate region-label map in the UI.
3. Bind the team selector to `current_field.teams` only so non-tournament teams never appear in the explorer.
4. Use `historical_summary` for first paint and recompute from `historical_distributions` after year-filter changes.
5. Treat `default_percentiles` as optional and empty in v1.