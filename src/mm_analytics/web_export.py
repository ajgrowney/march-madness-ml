from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from mm_analytics.objects import (
    ORDINALS_DF,
    REGULAR_SZN_DF,
    SEEDS_DF,
    SLOTS_DF,
    TEAM_COACH_DF,
    TEAM_CONF_DF,
    TOURNEY_RESULTS_DF,
    TeamSeason,
    get_season_ordinals,
    get_team_seasons_and_rankings,
    get_year_system,
)
from mm_analytics.utilities import NpEncoder

INITIAL_FEATURE_SET_NAME = "2026_initial"
HISTORICAL_FEATURE_SET_NAME = "historical_v1"
STAT_EXPLORER_FEATURE_SET_NAME = "stat_explorer_v1"
PLACEHOLDER_BRACKET_SOURCE_SEASON = 2025
REGION_DETAILS = {
    "W": {"friendly_name": "East", "abbrev": "East"},
    "X": {"friendly_name": "South", "abbrev": "South"},
    "Y": {"friendly_name": "Midwest", "abbrev": "MW"},
    "Z": {"friendly_name": "West", "abbrev": "West"},
    "Final Four": {"friendly_name": "Final Four", "abbrev": "FF"},
    "Championship": {"friendly_name": "Championship", "abbrev": "CH"},
}
INITIAL_FEATURE_COLUMNS = [
    "Seed",
    "WinPct",
    "SOS",
    "SOV",
    "NET_last",
    "AdjOE_mean",
    "AdjDE_mean",
    "AdjNE_mean",
    "FG%_mean",
    "FG3%_mean",
    "FT%_mean",
]
HISTORICAL_FEATURE_COLUMNS = [
    "Seed",
    "WinPct",
    "SOS",
    "SOV",
    "selection_ordinal_last",
    "AdjOE_mean",
    "AdjDE_mean",
    "AdjNE_mean",
    "FG%_mean",
    "FG3%_mean",
    "FT%_mean",
]
STAT_EXPLORER_FEATURE_COLUMNS = [
    "Q1_WinPct",
    "Q2_WinPct",
    "Q3_WinPct",
    "Q4_WinPct",
    "SOS",
    "SOV",
    "Poss_mean",
    "Fouls_mean",
    "AdjOE_mean",
    "AdjNE_mean",
    "EFG%_mean",
    "FG3%_mean",
    "FT%_mean",
    "FTA_mean",
    "TO_mean",
    "Ast_mean",
    "OR_mean",
    "FGA3_mean",
    "AdjDE_mean",
    "Stl_mean",
    "Blk_mean",
    "OppTO_mean",
    "DR_mean",
    "OppFGA3_mean",
]
STAT_EXPLORER_FEATURE_GROUPS = {
    "Resume": ["Q1_WinPct", "Q2_WinPct", "Q3_WinPct", "Q4_WinPct", "SOS", "SOV"],
    "Tempo": ["Poss_mean", "Fouls_mean"],
    "Offense": [
        "AdjOE_mean",
        "AdjNE_mean",
        "EFG%_mean",
        "FG3%_mean",
        "FT%_mean",
        "FTA_mean",
        "TO_mean",
        "Ast_mean",
        "OR_mean",
        "FGA3_mean",
    ],
    "Defense": ["AdjDE_mean", "Stl_mean", "Blk_mean", "OppTO_mean", "DR_mean", "OppFGA3_mean"],
}
STAT_EXPLORER_HISTORICAL_RANGE = (2003, 2025)
TEAM_PAGE_SIMILARITY_TOP_N = 10
TEAM_PAGE_SIMILARITY_SEED_WINDOW = 2
TEAM_PAGE_SIMILARITY_SEED_WEIGHT = 0.15
TEAM_PAGE_SIMILARITY_RESUME_COLUMNS = STAT_EXPLORER_FEATURE_GROUPS["Resume"]
TEAM_PAGE_SIMILARITY_STAT_COLUMNS = (
    STAT_EXPLORER_FEATURE_GROUPS["Tempo"]
    + STAT_EXPLORER_FEATURE_GROUPS["Offense"]
    + STAT_EXPLORER_FEATURE_GROUPS["Defense"]
)
STAT_EXPLORER_ROUND_BUCKETS = [
    {"key": "round64", "label": "Round of 64", "exit_round_labels": ["Play In", "First Round"]},
    {"key": "round32", "label": "Round 32", "exit_round_labels": ["Second Round"]},
    {"key": "sweet16", "label": "Sweet 16", "exit_round_nums": [3]},
    {"key": "elite8", "label": "Elite 8", "exit_round_nums": [4]},
    {"key": "final4", "label": "Final Four", "exit_round_nums": [5]},
    {"key": "championship", "label": "Championship", "exit_round_nums": [6]},
    {"key": "champion", "label": "Champions", "exit_round_nums": [7]},
]


@dataclass(frozen=True)
class ExportPaths:
    output_root: Path
    team_index_path: Path
    feature_store_path: Path
    stat_explorer_path: Path
    bracket_path: Path
    model_manifest_path: Path
    team_pages_dir: Path


@dataclass(frozen=True)
class FeatureDefaults:
    values: Dict[str, float]
    missing_counts: Dict[str, int]


@dataclass(frozen=True)
class SmokeCheckResults:
    team_pages_written: int
    team_index_count: int
    feature_store_count: int
    bracket_slot_count: int
    seeded_teams_in_index: int
    seeded_teams_in_feature_store: int
    stat_explorer_current_field_count: int
    stat_explorer_historical_feature_count: int
    stat_explorer_historical_row_count: int


@dataclass(frozen=True)
class HistoricalSeasonCheckResults:
    team_count: int
    example_count: int
    regular_season_examples: int
    tournament_examples: int


@dataclass(frozen=True)
class HistoricalExportPaths:
    output_root: Path
    training_manifest_path: Path


def build_export_paths(
    season: int,
    feature_set_name: str = STAT_EXPLORER_FEATURE_SET_NAME,
    output_root: str = "data/web",
) -> ExportPaths:
    root = Path(output_root)
    return ExportPaths(
        output_root=root,
        team_index_path=root / "index" / str(season) / "teams.json",
        feature_store_path=build_feature_store_path(root, season, feature_set_name),
        stat_explorer_path=build_stat_explorer_path(root, season, feature_set_name),
        bracket_path=root / "brackets" / f"{season}.json",
        model_manifest_path=root / "models" / "manifest.json",
        team_pages_dir=root / "ts",
    )


def build_historical_export_paths(output_root: str = "data/web") -> HistoricalExportPaths:
    root = Path(output_root)
    return HistoricalExportPaths(
        output_root=root,
        training_manifest_path=root / "training" / "manifest.json",
    )


def ensure_export_directories(paths: ExportPaths) -> None:
    paths.team_index_path.parent.mkdir(parents=True, exist_ok=True)
    paths.feature_store_path.parent.mkdir(parents=True, exist_ok=True)
    paths.stat_explorer_path.parent.mkdir(parents=True, exist_ok=True)
    paths.bracket_path.parent.mkdir(parents=True, exist_ok=True)
    paths.model_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.team_pages_dir.mkdir(parents=True, exist_ok=True)


def ensure_historical_export_directories(
    paths: HistoricalExportPaths,
    start_season: int,
    end_season: int,
    feature_set_name: str,
) -> None:
    paths.training_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for season in range(start_season, end_season + 1):
        (paths.output_root / "features" / str(season)).mkdir(parents=True, exist_ok=True)
        (paths.output_root / "training" / feature_set_name).mkdir(parents=True, exist_ok=True)


def load_team_seasons_for_export(
    season: int,
) -> Tuple[Dict[int, TeamSeason], Dict[str, List[Tuple[int, float]]]]:
    year_reg_season = REGULAR_SZN_DF[REGULAR_SZN_DF["Season"] == season]
    teams_conf_season = TEAM_CONF_DF[TEAM_CONF_DF["Season"] == season]
    teams_coach_season = TEAM_COACH_DF[TEAM_COACH_DF["Season"] == season]
    year_tourney = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"] == season]
    season_ordinals = get_season_ordinals(
        ORDINALS_DF[ORDINALS_DF["Season"] == season],
        [get_year_system(season)],
    )
    return get_team_seasons_and_rankings(
        season,
        year_reg_season,
        SEEDS_DF,
        teams_conf_season,
        teams_coach_season,
        season_ordinals,
        year_tourney,
    )


def parse_seed_label(seed_label: str) -> Tuple[str, int]:
    region = seed_label[0]
    seed_digits = "".join(ch for ch in seed_label if ch.isdigit())
    return region, int(seed_digits)


def resolve_bracket_source_season(season: int) -> int:
    has_seed_rows = not SEEDS_DF[SEEDS_DF["Season"] == season].empty
    has_slot_rows = not SLOTS_DF[SLOTS_DF["Season"] == season].empty
    if has_seed_rows and has_slot_rows:
        return season
    if season == 2026:
        return PLACEHOLDER_BRACKET_SOURCE_SEASON
    return season


def build_placeholder_seed_maps(season: int) -> Tuple[Dict[int, dict], Dict[str, dict], int]:
    source_season = resolve_bracket_source_season(season)
    season_seed_df = SEEDS_DF[SEEDS_DF["Season"] == source_season]

    by_team_id: Dict[int, dict] = {}
    by_seed_label: Dict[str, dict] = {}
    for row in season_seed_df.itertuples():
        region, seed_value = parse_seed_label(row.Seed)
        seed_record = {
            "season": season,
            "source_season": source_season,
            "seed_label": row.Seed,
            "seed": seed_value,
            "region": region,
            "slot": row.Seed,
            "team_id": int(row.TeamID),
        }
        by_team_id[int(row.TeamID)] = seed_record
        by_seed_label[row.Seed] = seed_record

    return by_team_id, by_seed_label, source_season


def feature_default_value(feature_name: str, observed_values: List[float]) -> float:
    if feature_name == "Seed":
        return 17.0
    if not observed_values:
        return 0.0
    return round(sum(observed_values) / len(observed_values), 6)


def is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value)
    return False


def get_feature_value(
    team_season: TeamSeason,
    feature_name: str,
    seed_by_team_id: Dict[int, dict],
) -> object:
    if feature_name == "Seed":
        seed_info = seed_by_team_id.get(team_season.id)
        return None if seed_info is None else seed_info["seed"]
    if feature_name == "selection_ordinal_last":
        return team_season.ordinal_data.get_system_data(get_year_system(team_season.year), "last")
    return team_season.get_data(columns=[feature_name])[0]


def build_feature_store_path(output_root: Path, season: int, feature_set_name: str) -> Path:
    file_name = "base.json" if feature_set_name == INITIAL_FEATURE_SET_NAME else f"{feature_set_name}.json"
    return output_root / "features" / str(season) / file_name


def build_stat_explorer_path(output_root: Path, season: int, feature_set_name: str) -> Path:
    return output_root / "stats" / str(season) / f"{feature_set_name}.json"


def get_friendly_region_label(region_key: str) -> str:
    region_details = REGION_DETAILS.get(region_key)
    return region_details["friendly_name"] if region_details is not None else region_key


def get_round_bucket_key(exit_round: Optional[str]) -> Optional[str]:
    if exit_round is None:
        return None

    for bucket in STAT_EXPLORER_ROUND_BUCKETS:
        exit_round_labels = bucket.get("exit_round_labels")
        if exit_round_labels and exit_round in exit_round_labels:
            return bucket["key"]

    exit_round_num_map = {
        "Play In": 1,
        "First Round": 1,
        "Second Round": 2,
        "Sweet Sixteen": 3,
        "Elite Eight": 4,
        "Final Four": 5,
        "Championship": 6,
        "Champion": 7,
    }
    exit_round_num = exit_round_num_map.get(exit_round)
    if exit_round_num is None:
        return None

    for bucket in STAT_EXPLORER_ROUND_BUCKETS:
        if exit_round_num in bucket.get("exit_round_nums", []):
            return bucket["key"]
    return None


def build_stat_explorer_current_field(
    season: int,
    team_seasons: Dict[int, TeamSeason],
    seed_by_team_id: Dict[int, dict],
) -> dict:
    current_field_teams = []

    for team_id, team_season in sorted(team_seasons.items(), key=lambda item: (item[1].tourney_seed or 99, item[1].name)):
        seed_info = seed_by_team_id.get(team_id)
        if seed_info is None:
            continue

        stats = {}
        for feature_name in STAT_EXPLORER_FEATURE_COLUMNS:
            value = get_feature_value(team_season, feature_name, seed_by_team_id)
            if is_missing_value(value):
                continue
            stats[feature_name] = round(float(value), 6)

        current_field_teams.append(
            {
                "team_id": team_id,
                "name": team_season.name,
                "short_name": team_season.name,
                "seed": int(seed_info["seed"]),
                "region": get_friendly_region_label(seed_info["region"]),
                "region_key": seed_info["region"],
                "slot": seed_info["slot"],
                "tournament_team": True,
                "stats": stats,
            }
        )

    return {"teams": current_field_teams}


def build_stat_explorer_historical_distributions(
    start_season: int,
    end_season: int,
) -> dict:
    historical_distributions = {
        feature_name: [] for feature_name in STAT_EXPLORER_FEATURE_COLUMNS
    }

    for historical_season in range(start_season, end_season + 1):
        team_seasons, _ = load_team_seasons_for_export(historical_season)
        seed_by_team_id, _, _ = build_placeholder_seed_maps(historical_season)

        for team_id, team_season in team_seasons.items():
            if team_season.tourney_seed is None or team_season.tourney_exit_round is None:
                continue

            bucket_key = get_round_bucket_key(team_season.tourney_exit_round)
            if bucket_key is None:
                continue

            for feature_name in STAT_EXPLORER_FEATURE_COLUMNS:
                value = get_feature_value(team_season, feature_name, seed_by_team_id)
                if is_missing_value(value):
                    continue
                historical_distributions[feature_name].append(
                    {
                        "season": historical_season,
                        "bucket": bucket_key,
                        "value": round(float(value), 6),
                        "team_id": team_id,
                        "team_name": team_season.name,
                        "seed": int(team_season.tourney_seed),
                        "exit_round": team_season.tourney_exit_round,
                    }
                )

    return historical_distributions


def build_stat_explorer_summary(historical_distributions: dict) -> dict:
    historical_summary = {}

    for feature_name, rows in historical_distributions.items():
        bucket_values: Dict[str, List[float]] = {}
        for row in rows:
            bucket_values.setdefault(row["bucket"], []).append(float(row["value"]))

        feature_summary = {}
        for bucket_key, values in bucket_values.items():
            if not values:
                continue
            value_array = np.array(sorted(values), dtype=float)
            feature_summary[bucket_key] = {
                "count": int(len(value_array)),
                "min": round(float(np.min(value_array)), 6),
                "p10": round(float(np.percentile(value_array, 10)), 6),
                "q1": round(float(np.percentile(value_array, 25)), 6),
                "median": round(float(np.percentile(value_array, 50)), 6),
                "q3": round(float(np.percentile(value_array, 75)), 6),
                "p90": round(float(np.percentile(value_array, 90)), 6),
                "max": round(float(np.max(value_array)), 6),
                "mean": round(float(np.mean(value_array)), 6),
                "std": round(float(np.std(value_array)), 6),
            }

        historical_summary[feature_name] = feature_summary

    return historical_summary


def run_stat_explorer_smoke_checks(stat_explorer_payload: dict, team_index_payload: dict) -> dict:
    round_bucket_keys = {bucket["key"] for bucket in stat_explorer_payload["round_buckets"]}
    feature_order = set(stat_explorer_payload["feature_order"])
    valid_region_keys = {region["key"] for region in stat_explorer_payload["filters"]["regions"]}
    valid_region_labels = {region["label"] for region in stat_explorer_payload["filters"]["regions"]}
    tournament_index_entries = {
        entry["team_id"]: entry for entry in team_index_payload["teams"] if entry["tournament_team"]
    }

    current_field_teams = stat_explorer_payload["current_field"]["teams"]
    current_field_team_ids = set()
    for team_entry in current_field_teams:
        team_id = team_entry["team_id"]
        current_field_team_ids.add(team_id)
        if not team_entry["tournament_team"]:
            raise ValueError(f"Stat explorer current field contains non-tournament team: {team_id}")
        if team_id not in tournament_index_entries:
            raise ValueError(f"Stat explorer current field references unknown tournament team: {team_id}")
        if team_entry["region"] not in valid_region_labels:
            raise ValueError(
                f"Stat explorer current field has invalid friendly region: team_id={team_id}, region={team_entry['region']}"
            )
        if team_entry.get("region_key") not in valid_region_keys:
            raise ValueError(
                f"Stat explorer current field has invalid region key: team_id={team_id}, region_key={team_entry.get('region_key')}"
            )

    expected_tournament_team_ids = set(tournament_index_entries.keys())
    if current_field_team_ids != expected_tournament_team_ids:
        raise ValueError(
            "Stat explorer current field does not match tournament team index: "
            f"missing={sorted(expected_tournament_team_ids - current_field_team_ids)[:5]}, "
            f"extra={sorted(current_field_team_ids - expected_tournament_team_ids)[:5]}"
        )

    historical_row_count = 0
    for feature_name, rows in stat_explorer_payload["historical_distributions"].items():
        if feature_name not in feature_order:
            raise ValueError(f"Stat explorer historical distributions contain unknown feature: {feature_name}")
        for row in rows:
            historical_row_count += 1
            if row["bucket"] not in round_bucket_keys:
                raise ValueError(
                    f"Stat explorer historical row has unknown bucket: feature={feature_name}, bucket={row['bucket']}"
                )

    for feature_name, bucket_map in stat_explorer_payload["historical_summary"].items():
        if feature_name not in feature_order:
            raise ValueError(f"Stat explorer historical summary contains unknown feature: {feature_name}")
        for bucket_key in bucket_map.keys():
            if bucket_key not in round_bucket_keys:
                raise ValueError(
                    f"Stat explorer historical summary has unknown bucket: feature={feature_name}, bucket={bucket_key}"
                )

    for team_id, feature_map in stat_explorer_payload.get("default_percentiles", {}).items():
        if int(team_id) not in expected_tournament_team_ids:
            raise ValueError(f"Stat explorer default percentiles reference unknown tournament team: {team_id}")
        for feature_name, bucket_map in feature_map.items():
            if feature_name not in feature_order:
                raise ValueError(f"Stat explorer default percentiles contain unknown feature: {feature_name}")
            for bucket_key in bucket_map.keys():
                if bucket_key not in round_bucket_keys:
                    raise ValueError(
                        f"Stat explorer default percentiles have unknown bucket: team_id={team_id}, feature={feature_name}, bucket={bucket_key}"
                    )

    return {
        "current_field_count": len(current_field_teams),
        "historical_feature_count": len(stat_explorer_payload["historical_distributions"]),
        "historical_row_count": historical_row_count,
    }


def build_stat_explorer_payload(
    season: int,
    feature_set_name: str,
    team_seasons: Dict[int, TeamSeason],
    seed_by_team_id: Dict[int, dict],
) -> dict:
    start_season, end_season = STAT_EXPLORER_HISTORICAL_RANGE
    historical_distributions = build_stat_explorer_historical_distributions(start_season, end_season)
    historical_summary = build_stat_explorer_summary(historical_distributions)
    return {
        "season": season,
        "feature_set": feature_set_name,
        "historical_range": [start_season, end_season],
        "feature_groups": STAT_EXPLORER_FEATURE_GROUPS,
        "feature_order": STAT_EXPLORER_FEATURE_COLUMNS,
        "round_buckets": STAT_EXPLORER_ROUND_BUCKETS,
        "filters": {
            "default_scope": "field",
            "regions": [
                {"key": region_key, "label": get_friendly_region_label(region_key)}
                for region_key in ["W", "X", "Y", "Z"]
            ],
            "historical_range_default": [start_season, end_season],
            "historical_range_allowed": [start_season, end_season],
        },
        "current_field": build_stat_explorer_current_field(season, team_seasons, seed_by_team_id),
        "historical_distributions": historical_distributions,
        "historical_summary": historical_summary,
        "default_percentiles": {},
    }


def build_training_file_path(output_root: Path, season: int, feature_set_name: str) -> Path:
    return output_root / "training" / feature_set_name / f"{season}.json"


def build_feature_defaults(
    team_seasons: Dict[int, TeamSeason],
    feature_columns: List[str],
    seed_by_team_id: Dict[int, dict],
) -> FeatureDefaults:
    observed_values = {feature_name: [] for feature_name in feature_columns}
    missing_counts = {feature_name: 0 for feature_name in feature_columns}

    for team_id, team_season in team_seasons.items():
        for feature_name in feature_columns:
            value = get_feature_value(team_season, feature_name, seed_by_team_id)
            if is_missing_value(value):
                missing_counts[feature_name] += 1
                continue
            observed_values[feature_name].append(float(value))

    defaults = {
        feature_name: feature_default_value(feature_name, observed_values[feature_name])
        for feature_name in feature_columns
    }
    return FeatureDefaults(values=defaults, missing_counts=missing_counts)


def build_team_index(
    season: int,
    team_seasons: Dict[int, TeamSeason],
    seed_by_team_id: Dict[int, dict],
) -> dict:
    team_entries = []
    for team_id, team_season in sorted(team_seasons.items(), key=lambda item: item[1].name):
        seed_info = seed_by_team_id.get(team_id)
        team_entries.append(
            {
                "team_id": team_id,
                "name": team_season.name,
                "short_name": team_season.name,
                "conference": team_season.conf,
                "seed": None if seed_info is None else seed_info["seed"],
                "region": None if seed_info is None else seed_info["region"],
                "slot": None if seed_info is None else seed_info["slot"],
                "tournament_team": seed_info is not None,
                "team_page_path": f"data/web/ts/{team_id}_{season}.json",
            }
        )

    return {
        "season": season,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "teams": team_entries,
    }


def build_tournament_loss_context(team_season: TeamSeason) -> dict:
    losing_games = [game for game in team_season.tourney_games if not game.is_win()]
    if not losing_games:
        return {
            "lost_to_team_id": None,
            "lost_to": None,
            "loss_score": None,
        }

    loss_game = losing_games[0]
    return {
        "lost_to_team_id": int(loss_game.opponent_id),
        "lost_to": loss_game.opponent_name,
        "loss_score": f"{int(loss_game.team_score)}-{int(loss_game.opp_score)}",
    }


def build_historical_similarity_frame(
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    rows: List[dict] = []

    for historical_season in range(start_season, end_season + 1):
        historical_team_seasons, _ = load_team_seasons_for_export(historical_season)
        historical_seed_by_team_id, _, _ = build_placeholder_seed_maps(historical_season)

        for team_id, team_season in historical_team_seasons.items():
            seed_info = historical_seed_by_team_id.get(team_id)
            if seed_info is None or team_season.tourney_exit_round is None:
                continue

            row = {
                "id": int(team_id),
                "name": team_season.name,
                "year": int(historical_season),
                "seed": int(seed_info["seed"]),
                "er": team_season.tourney_exit_round,
                **build_tournament_loss_context(team_season),
            }
            for feature_name in STAT_EXPLORER_FEATURE_COLUMNS:
                value = get_feature_value(team_season, feature_name, historical_seed_by_team_id)
                row[feature_name] = 0.0 if is_missing_value(value) else float(value)
            rows.append(row)

    return pd.DataFrame(rows)


def compute_similarity_scores(
    current_values: dict,
    candidates: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    scaler = StandardScaler()
    candidate_matrix = scaler.fit_transform(candidates[feature_columns])
    current_matrix = scaler.transform(pd.DataFrame([current_values], columns=feature_columns))
    return cosine_similarity(current_matrix, candidate_matrix).ravel()


def build_team_page_similarity_map(
    season: int,
    team_seasons: Dict[int, TeamSeason],
    seed_by_team_id: Dict[int, dict],
    top_n: int = TEAM_PAGE_SIMILARITY_TOP_N,
    seed_window: Optional[int] = TEAM_PAGE_SIMILARITY_SEED_WINDOW,
    seed_weight: float = TEAM_PAGE_SIMILARITY_SEED_WEIGHT,
) -> Dict[int, List[dict]]:
    start_season, end_season = STAT_EXPLORER_HISTORICAL_RANGE
    historical_end = min(end_season, season - 1)
    if historical_end < start_season:
        return {}

    candidates = build_historical_similarity_frame(start_season, historical_end)
    if candidates.empty:
        return {}

    similarity_map: Dict[int, List[dict]] = {}
    for team_id, team_season in team_seasons.items():
        seed_info = seed_by_team_id.get(team_id)
        if seed_info is None:
            continue

        team_seed = int(seed_info["seed"])
        team_values = {}
        for feature_name in STAT_EXPLORER_FEATURE_COLUMNS:
            value = get_feature_value(team_season, feature_name, seed_by_team_id)
            team_values[feature_name] = 0.0 if is_missing_value(value) else float(value)

        team_candidates = candidates
        if seed_window is not None:
            seed_filtered = candidates.loc[candidates["seed"].between(team_seed - seed_window, team_seed + seed_window)].copy()
            if len(seed_filtered) >= top_n:
                team_candidates = seed_filtered

        feature_similarity = compute_similarity_scores(team_values, team_candidates, STAT_EXPLORER_FEATURE_COLUMNS)
        resume_similarity = compute_similarity_scores(team_values, team_candidates, TEAM_PAGE_SIMILARITY_RESUME_COLUMNS)
        stat_similarity = compute_similarity_scores(team_values, team_candidates, TEAM_PAGE_SIMILARITY_STAT_COLUMNS)

        seed_distance = (team_candidates["seed"] - team_seed).abs().astype(float)
        max_seed_distance = max(float(seed_distance.max()), 1.0)
        seed_similarity = 1 - (seed_distance / max_seed_distance)
        combined_similarity = ((1 - seed_weight) * feature_similarity) + (seed_weight * seed_similarity.to_numpy())

        ranked = team_candidates.assign(
            avg=combined_similarity,
            feature_similarity=feature_similarity,
            res=resume_similarity,
            st=stat_similarity,
            seed_similarity=seed_similarity.to_numpy(),
        ).sort_values("avg", ascending=False)

        similarity_map[team_id] = [
            {
                "id": int(row.id),
                "year": int(row.year),
                "name": row.name,
                "seed": int(row.seed),
                "avg": round(float(row.avg), 3),
                "feature_similarity": round(float(row.feature_similarity), 3),
                "res": round(float(row.res), 3),
                "st": round(float(row.st), 3),
                "seed_similarity": round(float(row.seed_similarity), 3),
                "er": row.er,
                "lost_to_team_id": None if pd.isna(row.lost_to_team_id) else int(row.lost_to_team_id),
                "lost_to": None if pd.isna(row.lost_to) else row.lost_to,
                "loss_score": None if pd.isna(row.loss_score) else row.loss_score,
            }
            for row in ranked.head(top_n).itertuples(index=False)
        ]

    return similarity_map


def build_team_page_payload(team_season: TeamSeason, seed_by_team_id: Dict[int, dict]) -> dict:
    payload = team_season.to_web_json()
    seed_info = seed_by_team_id.get(team_season.id)
    if seed_info is None:
        return payload

    tournament_payload = payload.get("tournament") or {
        "seed": None,
        "games": [],
        "exit_round": None,
    }
    tournament_payload["seed"] = seed_info["seed"]
    payload["tournament"] = tournament_payload
    return payload


def write_team_pages(
    season: int,
    paths: ExportPaths,
    team_seasons: Dict[int, TeamSeason],
    seed_by_team_id: Dict[int, dict],
) -> int:
    written_count = 0
    for team_id, team_season in sorted(team_seasons.items()):
        payload = build_team_page_payload(team_season, seed_by_team_id)
        write_json(paths.team_pages_dir / f"{team_id}_{season}.json", payload)
        written_count += 1
    return written_count


def build_feature_store(
    season: int,
    team_seasons: Dict[int, TeamSeason],
    feature_set_name: str,
    feature_columns: List[str],
    seed_by_team_id: Dict[int, dict],
) -> Tuple[dict, FeatureDefaults]:
    defaults = build_feature_defaults(team_seasons, feature_columns, seed_by_team_id)
    teams = {}

    for team_id, team_season in sorted(team_seasons.items()):
        feature_values = {}
        for feature_name in feature_columns:
            value = get_feature_value(team_season, feature_name, seed_by_team_id)
            feature_values[feature_name] = (
                defaults.values[feature_name]
                if is_missing_value(value)
                else round(float(value), 6)
            )
        teams[str(team_id)] = feature_values

    return {
        "season": season,
        "feature_set": feature_set_name,
        "feature_order": feature_columns,
        "teams": teams,
    }, defaults


def slot_round_details(slot_name: str) -> Tuple[int, str, str]:
    if slot_name.startswith("R6"):
        return 7, "Championship", "Championship"
    if slot_name.startswith("R5"):
        return 6, "Final Four", "Final Four"
    if slot_name.startswith("R4"):
        return 5, slot_name[2], "Elite Eight"
    if slot_name.startswith("R3"):
        return 4, slot_name[2], "Sweet Sixteen"
    if slot_name.startswith("R2"):
        return 3, slot_name[2], "Second Round"
    if slot_name.startswith("R1"):
        return 2, slot_name[2], "First Round"
    return 1, slot_name[0], "Play In"


def slot_sort_key(slot_payload: dict) -> Tuple[int, int, str]:
    region_order = {
        "W": 0,
        "X": 1,
        "Y": 2,
        "Z": 3,
        "Final Four": 4,
        "Championship": 5,
    }
    return (
        int(slot_payload["round"]),
        region_order.get(slot_payload["region"], 99),
        str(slot_payload["slot"]),
    )


def build_region_order(slots: List[dict]) -> List[str]:
    ordered_regions = []
    for region_name in ["W", "X", "Y", "Z", "Final Four", "Championship"]:
        if any(slot["region"] == region_name for slot in slots):
            ordered_regions.append(region_name)
    return ordered_regions


def build_region_details(region_order: List[str]) -> dict:
    return {
        region_name: REGION_DETAILS[region_name]
        for region_name in region_order
        if region_name in REGION_DETAILS
    }


def resolve_slot_side(reference: str, slot_names: set[str], seed_by_label: Dict[str, dict]) -> dict:
    if reference in slot_names:
        return {"source": "winner", "source_slot": reference}

    seed_info = seed_by_label[reference]
    return {
        "source": "seed",
        "team_id": seed_info["team_id"],
        "seed": seed_info["seed"],
    }


def build_bracket_definition(season: int, seed_by_label: Dict[str, dict], source_season: int) -> dict:
    season_slots = SLOTS_DF[SLOTS_DF["Season"] == source_season].copy()
    season_slots["ExportSeason"] = season

    slot_names = set(season_slots["Slot"].tolist())
    next_slot_map = {}
    for row in season_slots.itertuples():
        for upstream in (row.StrongSeed, row.WeakSeed):
            if upstream in slot_names:
                next_slot_map[upstream] = row.Slot

    slots = []
    for row in season_slots.itertuples():
        round_number, region, label = slot_round_details(row.Slot)
        slots.append(
            {
                "slot": row.Slot,
                "label": label,
                "region": region,
                "round": round_number,
                "team_1": resolve_slot_side(row.StrongSeed, slot_names, seed_by_label),
                "team_2": resolve_slot_side(row.WeakSeed, slot_names, seed_by_label),
                "next_slot": next_slot_map.get(row.Slot),
            }
        )

    slots = sorted(slots, key=slot_sort_key)
    region_order = build_region_order(slots)

    return {
        "season": season,
        "regions": region_order,
        "region_details": build_region_details(region_order),
        "slots": slots,
        "final_slots": {
            "left_semifinal": "R5WX",
            "right_semifinal": "R5YZ",
            "championship": "R6CH",
        },
    }


def load_model_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise ValueError(
            "Model manifest is missing. Create a real website model manifest before running export-web."
        )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest_payload.get("models", [])
    if not models:
        raise ValueError(
            "Model manifest has no models. Add a real model entry before running export-web."
        )

    default_model_id = manifest_payload.get("default_model_id")
    if not any(entry.get("id") == default_model_id for entry in models):
        raise ValueError(
            "Model manifest default_model_id does not reference an existing model entry."
        )

    return manifest_payload


def write_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2, cls=NpEncoder), encoding="utf-8")


def build_matchup_vector(
    feature_store_payload: dict,
    team_1_id: int,
    team_2_id: int,
) -> List[float]:
    team_1_features = feature_store_payload["teams"][str(team_1_id)]
    team_2_features = feature_store_payload["teams"][str(team_2_id)]
    return [
        round(team_1_features[feature_name] - team_2_features[feature_name], 6)
        for feature_name in feature_store_payload["feature_order"]
    ]


def append_training_examples(
    examples: List[dict],
    feature_store_payload: dict,
    season: int,
    source: str,
    day_num: int,
    winner_id: int,
    loser_id: int,
) -> None:
    examples.append(
        {
            "season": season,
            "source": source,
            "day_num": day_num,
            "team_1_id": loser_id,
            "team_2_id": winner_id,
            "x": build_matchup_vector(feature_store_payload, loser_id, winner_id),
            "y": 1,
        }
    )
    examples.append(
        {
            "season": season,
            "source": source,
            "day_num": day_num,
            "team_1_id": winner_id,
            "team_2_id": loser_id,
            "x": build_matchup_vector(feature_store_payload, winner_id, loser_id),
            "y": 0,
        }
    )


def build_historical_training_payload(
    season: int,
    feature_store_payload: dict,
) -> dict:
    examples: List[dict] = []
    regular_season_games = REGULAR_SZN_DF[REGULAR_SZN_DF["Season"] == season]
    tournament_games = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"] == season]

    for row in regular_season_games[["DayNum", "WTeamID", "LTeamID"]].itertuples(index=False):
        append_training_examples(
            examples,
            feature_store_payload,
            season,
            "regular_season",
            int(row.DayNum),
            int(row.WTeamID),
            int(row.LTeamID),
        )

    for row in tournament_games[["DayNum", "WTeamID", "LTeamID"]].itertuples(index=False):
        append_training_examples(
            examples,
            feature_store_payload,
            season,
            "tournament",
            int(row.DayNum),
            int(row.WTeamID),
            int(row.LTeamID),
        )

    return {
        "season": season,
        "feature_set": feature_store_payload["feature_set"],
        "feature_order": feature_store_payload["feature_order"],
        "format": "diff",
        "include_regular_season": True,
        "include_tournament": True,
        "label_definition": {
            "0": "team_1_win",
            "1": "team_2_win",
        },
        "examples": examples,
    }


def build_training_manifest(
    feature_set_name: str,
    feature_columns: List[str],
    seasons: List[int],
    output_root: Path,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": feature_set_name,
        "feature_order": feature_columns,
        "format": "diff",
        "include_regular_season": True,
        "include_tournament": True,
        "label_definition": {
            "0": "team_1_win",
            "1": "team_2_win",
        },
        "seasons": seasons,
        "feature_files": {
            str(season): str(build_feature_store_path(output_root, season, feature_set_name))
            for season in seasons
        },
        "training_files": {
            str(season): str(build_training_file_path(output_root, season, feature_set_name))
            for season in seasons
        },
    }


def run_historical_smoke_checks(
    feature_store_payload: dict,
    training_payload: dict,
) -> HistoricalSeasonCheckResults:
    feature_team_ids = {int(team_id) for team_id in feature_store_payload["teams"].keys()}
    feature_length = len(feature_store_payload["feature_order"])
    regular_season_examples = 0
    tournament_examples = 0

    for example in training_payload["examples"]:
        if example["team_1_id"] not in feature_team_ids or example["team_2_id"] not in feature_team_ids:
            raise ValueError(
                "Training example references unknown team ids: "
                f"team_1_id={example['team_1_id']}, team_2_id={example['team_2_id']}"
            )
        if len(example["x"]) != feature_length:
            raise ValueError(
                f"Training example has incorrect feature length: expected={feature_length}, actual={len(example['x'])}"
            )
        if example["y"] not in (0, 1):
            raise ValueError(f"Training example has invalid label: {example['y']}")
        if example["source"] == "regular_season":
            regular_season_examples += 1
        elif example["source"] == "tournament":
            tournament_examples += 1
        else:
            raise ValueError(f"Training example has invalid source: {example['source']}")

    return HistoricalSeasonCheckResults(
        team_count=len(feature_team_ids),
        example_count=len(training_payload["examples"]),
        regular_season_examples=regular_season_examples,
        tournament_examples=tournament_examples,
    )


def run_smoke_checks(
    paths: ExportPaths,
    team_index_payload: dict,
    feature_store_payload: dict,
    stat_explorer_payload: dict,
    bracket_payload: dict,
    manifest_payload: dict,
) -> SmokeCheckResults:
    workspace_root = paths.output_root.parent.parent
    index_team_ids = {entry["team_id"] for entry in team_index_payload["teams"]}
    feature_team_ids = {int(team_id) for team_id in feature_store_payload["teams"].keys()}

    if feature_team_ids != index_team_ids:
        missing_in_index = sorted(feature_team_ids - index_team_ids)[:5]
        missing_in_features = sorted(index_team_ids - feature_team_ids)[:5]
        raise ValueError(
            "Feature store and team index differ: "
            f"missing_in_index={missing_in_index}, missing_in_feature_store={missing_in_features}"
        )

    for entry in team_index_payload["teams"]:
        team_page_path = workspace_root / entry["team_page_path"]
        if not team_page_path.exists():
            raise ValueError(f"Missing team page for team_id={entry['team_id']}: {entry['team_page_path']}")

    slot_names = {slot["slot"] for slot in bracket_payload["slots"]}
    bracket_team_ids = set()
    for slot in bracket_payload["slots"]:
        for side_name in ("team_1", "team_2"):
            side = slot[side_name]
            if side["source"] == "winner":
                if side["source_slot"] not in slot_names:
                    raise ValueError(
                        f"Bracket source slot missing: {slot['slot']} -> {side['source_slot']}"
                    )
            else:
                bracket_team_ids.add(int(side["team_id"]))

    unknown_bracket_teams = sorted(bracket_team_ids - index_team_ids)
    if unknown_bracket_teams:
        raise ValueError(f"Bracket references unknown teams: {unknown_bracket_teams[:5]}")

    for final_slot_name, slot_name in bracket_payload["final_slots"].items():
        if slot_name not in slot_names:
            raise ValueError(f"Final slot reference missing: {final_slot_name} -> {slot_name}")

    region_details = bracket_payload.get("region_details", {})
    missing_region_details = [region_name for region_name in bracket_payload["regions"] if region_name not in region_details]
    if missing_region_details:
        raise ValueError(f"Missing region metadata for: {missing_region_details}")

    feature_set_name = feature_store_payload["feature_set"]
    manifest_feature_sets = {model["feature_set"] for model in manifest_payload["models"]}
    if feature_set_name not in manifest_feature_sets:
        raise ValueError(f"Manifest does not reference exported feature set: {feature_set_name}")

    seeded_teams_in_index = {
        entry["team_id"] for entry in team_index_payload["teams"] if entry["seed"] is not None
    }
    if "Seed" in feature_store_payload["feature_order"]:
        seeded_teams_in_feature_store = {
            int(team_id)
            for team_id, team_values in feature_store_payload["teams"].items()
            if team_values["Seed"] != 17.0
        }
        if seeded_teams_in_index != seeded_teams_in_feature_store:
            raise ValueError(
                "Seeded team mismatch between index and feature store: "
                f"index_only={sorted(seeded_teams_in_index - seeded_teams_in_feature_store)[:5]}, "
                f"feature_only={sorted(seeded_teams_in_feature_store - seeded_teams_in_index)[:5]}"
            )
    else:
        seeded_teams_in_feature_store = seeded_teams_in_index

    stat_explorer_checks = run_stat_explorer_smoke_checks(stat_explorer_payload, team_index_payload)

    return SmokeCheckResults(
        team_pages_written=len(team_index_payload["teams"]),
        team_index_count=len(index_team_ids),
        feature_store_count=len(feature_team_ids),
        bracket_slot_count=len(bracket_payload["slots"]),
        seeded_teams_in_index=len(seeded_teams_in_index),
        seeded_teams_in_feature_store=len(seeded_teams_in_feature_store),
        stat_explorer_current_field_count=stat_explorer_checks["current_field_count"],
        stat_explorer_historical_feature_count=stat_explorer_checks["historical_feature_count"],
        stat_explorer_historical_row_count=stat_explorer_checks["historical_row_count"],
    )


def bootstrap_web_export(
    season: int = 2026,
    output_root: str = "data/web",
    feature_set_name: str = STAT_EXPLORER_FEATURE_SET_NAME,
) -> dict:
    paths = build_export_paths(season, feature_set_name=feature_set_name, output_root=output_root)
    ensure_export_directories(paths)
    team_seasons, _ = load_team_seasons_for_export(season)
    seed_by_team_id, seed_by_label, source_season = build_placeholder_seed_maps(season)
    similarity_map = build_team_page_similarity_map(season, team_seasons, seed_by_team_id)
    for team_id, similar_teams in similarity_map.items():
        team_seasons[team_id].similar_teams = similar_teams

    team_index_payload = build_team_index(season, team_seasons, seed_by_team_id)
    feature_store_payload, feature_defaults = build_feature_store(
        season,
        team_seasons,
        feature_set_name,
        STAT_EXPLORER_FEATURE_COLUMNS,
        seed_by_team_id,
    )
    stat_explorer_payload = build_stat_explorer_payload(
        season=season,
        feature_set_name=feature_set_name,
        team_seasons=team_seasons,
        seed_by_team_id=seed_by_team_id,
    )
    bracket_payload = build_bracket_definition(season, seed_by_label, source_season)
    manifest_payload = load_model_manifest(paths.model_manifest_path)

    write_json(paths.team_index_path, team_index_payload)
    write_json(paths.feature_store_path, feature_store_payload)
    write_json(paths.stat_explorer_path, stat_explorer_payload)
    write_json(paths.bracket_path, bracket_payload)
    team_pages_written = write_team_pages(season, paths, team_seasons, seed_by_team_id)
    smoke_checks = run_smoke_checks(
        paths,
        team_index_payload,
        feature_store_payload,
        stat_explorer_payload,
        bracket_payload,
        manifest_payload,
    )

    return {
        "season": season,
        "feature_set": feature_set_name,
        "feature_columns": STAT_EXPLORER_FEATURE_COLUMNS,
        "team_count": len(team_seasons),
        "stat_explorer": {
            "current_field_team_count": smoke_checks.stat_explorer_current_field_count,
            "historical_feature_count": smoke_checks.stat_explorer_historical_feature_count,
            "historical_row_count": smoke_checks.stat_explorer_historical_row_count,
        },
        "team_similarity": {
            "tournament_team_count": len(similarity_map),
            "matches_per_team": TEAM_PAGE_SIMILARITY_TOP_N,
            "historical_range": [STAT_EXPLORER_HISTORICAL_RANGE[0], min(STAT_EXPLORER_HISTORICAL_RANGE[1], season - 1)],
        },
        "placeholder_bracket_source_season": source_season,
        "feature_defaults": {
            "values": feature_defaults.values,
            "missing_counts": feature_defaults.missing_counts,
        },
        "smoke_checks": {
            "team_pages_written": team_pages_written,
            "team_index_count": smoke_checks.team_index_count,
            "feature_store_count": smoke_checks.feature_store_count,
            "bracket_slot_count": smoke_checks.bracket_slot_count,
            "seeded_teams_in_index": smoke_checks.seeded_teams_in_index,
            "seeded_teams_in_feature_store": smoke_checks.seeded_teams_in_feature_store,
            "stat_explorer_current_field_count": smoke_checks.stat_explorer_current_field_count,
            "stat_explorer_historical_feature_count": smoke_checks.stat_explorer_historical_feature_count,
            "stat_explorer_historical_row_count": smoke_checks.stat_explorer_historical_row_count,
        },
        "artifacts": {
            "team_index": str(paths.team_index_path),
            "feature_store": str(paths.feature_store_path),
            "stat_explorer": str(paths.stat_explorer_path),
            "bracket": str(paths.bracket_path),
            "model_manifest": str(paths.model_manifest_path),
            "team_pages_dir": str(paths.team_pages_dir),
        },
    }


def bootstrap_historical_training_export(
    start_season: int = 2003,
    end_season: int = 2025,
    output_root: str = "data/web",
    feature_set_name: str = STAT_EXPLORER_FEATURE_SET_NAME,
) -> dict:
    if start_season > end_season:
        raise ValueError(
            f"Invalid season range: start_season={start_season}, end_season={end_season}"
        )

    paths = build_historical_export_paths(output_root=output_root)
    ensure_historical_export_directories(
        paths,
        start_season=start_season,
        end_season=end_season,
        feature_set_name=feature_set_name,
    )

    seasons = list(range(start_season, end_season + 1))
    season_summaries = []

    for season in seasons:
        team_seasons, _ = load_team_seasons_for_export(season)
        seed_by_team_id, _, _ = build_placeholder_seed_maps(season)
        feature_store_payload, _ = build_feature_store(
            season=season,
            team_seasons=team_seasons,
            feature_set_name=feature_set_name,
            feature_columns=STAT_EXPLORER_FEATURE_COLUMNS,
            seed_by_team_id=seed_by_team_id,
        )
        training_payload = build_historical_training_payload(
            season=season,
            feature_store_payload=feature_store_payload,
        )
        checks = run_historical_smoke_checks(feature_store_payload, training_payload)

        write_json(
            build_feature_store_path(paths.output_root, season, feature_set_name),
            feature_store_payload,
        )
        write_json(
            build_training_file_path(paths.output_root, season, feature_set_name),
            training_payload,
        )

        season_summaries.append(
            {
                "season": season,
                "team_count": checks.team_count,
                "example_count": checks.example_count,
                "regular_season_examples": checks.regular_season_examples,
                "tournament_examples": checks.tournament_examples,
            }
        )

    manifest_payload = build_training_manifest(
        feature_set_name=feature_set_name,
        feature_columns=STAT_EXPLORER_FEATURE_COLUMNS,
        seasons=seasons,
        output_root=paths.output_root,
    )
    write_json(paths.training_manifest_path, manifest_payload)

    return {
        "status": "ok",
        "output_root": str(paths.output_root),
        "feature_set": feature_set_name,
        "season_range": {
            "start": start_season,
            "end": end_season,
            "count": len(seasons),
        },
        "manifest_path": str(paths.training_manifest_path),
        "season_summaries": season_summaries,
    }