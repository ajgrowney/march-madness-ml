from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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


@dataclass(frozen=True)
class ExportPaths:
    output_root: Path
    team_index_path: Path
    feature_store_path: Path
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


def build_export_paths(season: int, output_root: str = "data/web") -> ExportPaths:
    root = Path(output_root)
    return ExportPaths(
        output_root=root,
        team_index_path=root / "index" / str(season) / "teams.json",
        feature_store_path=root / "features" / str(season) / "base.json",
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


def build_model_manifest(season: int, feature_set_name: str) -> dict:
    return {
        "default_model_id": f"baseline_{season}",
        "models": [
            {
                "id": f"baseline_{season}",
                "name": f"Baseline {season}",
                "type": "runtime-config",
                "seasons": [season],
                "feature_set": feature_set_name,
                "prediction_format": "runtime-config",
                "description": "Placeholder manifest entry for the 2026 website export flow.",
                "model_url": None,
                "config_url": None,
                "predictions_url": None,
            }
        ],
    }


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

    return SmokeCheckResults(
        team_pages_written=len(team_index_payload["teams"]),
        team_index_count=len(index_team_ids),
        feature_store_count=len(feature_team_ids),
        bracket_slot_count=len(bracket_payload["slots"]),
        seeded_teams_in_index=len(seeded_teams_in_index),
        seeded_teams_in_feature_store=len(seeded_teams_in_feature_store),
    )


def bootstrap_web_export(
    season: int = 2026,
    output_root: str = "data/web",
    feature_set_name: str = INITIAL_FEATURE_SET_NAME,
) -> dict:
    paths = build_export_paths(season, output_root)
    ensure_export_directories(paths)
    team_seasons, _ = load_team_seasons_for_export(season)
    seed_by_team_id, seed_by_label, source_season = build_placeholder_seed_maps(season)

    team_index_payload = build_team_index(season, team_seasons, seed_by_team_id)
    feature_store_payload, feature_defaults = build_feature_store(
        season,
        team_seasons,
        feature_set_name,
        INITIAL_FEATURE_COLUMNS,
        seed_by_team_id,
    )
    bracket_payload = build_bracket_definition(season, seed_by_label, source_season)
    manifest_payload = build_model_manifest(season, feature_set_name)

    write_json(paths.team_index_path, team_index_payload)
    write_json(paths.feature_store_path, feature_store_payload)
    write_json(paths.bracket_path, bracket_payload)
    write_json(paths.model_manifest_path, manifest_payload)
    team_pages_written = write_team_pages(season, paths, team_seasons, seed_by_team_id)
    smoke_checks = run_smoke_checks(
        paths,
        team_index_payload,
        feature_store_payload,
        bracket_payload,
        manifest_payload,
    )

    return {
        "season": season,
        "feature_set": feature_set_name,
        "feature_columns": INITIAL_FEATURE_COLUMNS,
        "team_count": len(team_seasons),
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
        },
        "artifacts": {
            "team_index": str(paths.team_index_path),
            "feature_store": str(paths.feature_store_path),
            "bracket": str(paths.bracket_path),
            "model_manifest": str(paths.model_manifest_path),
            "team_pages_dir": str(paths.team_pages_dir),
        },
    }


def bootstrap_historical_training_export(
    start_season: int = 2003,
    end_season: int = 2025,
    output_root: str = "data/web",
    feature_set_name: str = HISTORICAL_FEATURE_SET_NAME,
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
            feature_columns=HISTORICAL_FEATURE_COLUMNS,
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
        feature_columns=HISTORICAL_FEATURE_COLUMNS,
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