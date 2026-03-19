from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from mm_analytics.utilities import NpEncoder
from mm_analytics.web_export import (
    EXIT_ROUND_MODEL_FEATURE_COLUMNS,
    EXIT_ROUND_NUM_MAP,
    STAT_EXPLORER_FEATURE_COLUMNS,
    STAT_EXPLORER_FEATURE_SET_NAME,
    STAT_EXPLORER_HISTORICAL_RANGE,
    TEAM_PAGE_EXIT_MODEL_VALIDATION_SEASON,
    SEEDS_DF,
    SLOTS_DF,
    TOURNEY_RESULTS_DF,
    build_historical_exit_round_frame,
    build_placeholder_seed_maps,
    exit_round_label,
    exit_round_quantile,
    get_feature_value,
    is_missing_value,
    load_team_seasons_for_export,
)


DEFAULT_SEASON = 2026
DEFAULT_MODEL_ID = "matchup-cluster-2026-v1"
DEFAULT_MODEL_NAME = "2026 Matchup Clusters"
DEFAULT_DESCRIPTION = (
    "Slot-aware matchup clustering artifact that assigns each 2026 pairing to a historical "
    "game archetype and returns the nearest tournament analogs without round-based filtering."
)
DEFAULT_TEAMS_INDEX = REPO_ROOT / "data/web/index/2026/teams.json"
DEFAULT_BRACKET = REPO_ROOT / "data/web/brackets/2026.json"
DEFAULT_INSIGHTS_DIR = REPO_ROOT / "data/web/models/insights"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data/web/models/manifest.json"
DEFAULT_MODEL_OUTPUT = REPO_ROOT / "Models/similar_matchup_model_2026.pkl"
DEFAULT_NEAREST_GAMES = 8
DEFAULT_CLUSTER_RANGE = range(4, 9)
EXIT_PROBABILITY_COLUMNS = {
    1: "p_round64",
    2: "p_round32",
    3: "p_sweet16",
    4: "p_elite8",
    5: "p_final4",
    6: "p_championship",
    7: "p_champion",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=NpEncoder), encoding="utf-8")


def build_model_urls(base_url: str | None, *relative_paths: Path) -> List[str | None]:
    urls: List[str | None] = []
    for relative_path in relative_paths:
        repo_relative = relative_path.relative_to(REPO_ROOT).as_posix()
        if base_url:
            urls.append(base_url.rstrip("/") + "/" + repo_relative)
        else:
            urls.append(repo_relative)
    return urls


def parse_training_seasons(raw_value: str | None, target_season: int) -> List[int]:
    if raw_value:
        seasons = [int(token.strip()) for token in raw_value.split(",") if token.strip()]
    else:
        start_season, end_season = STAT_EXPLORER_HISTORICAL_RANGE
        seasons = [season for season in range(start_season, min(end_season, target_season - 1) + 1)]

    if not seasons:
        raise ValueError("No training seasons selected.")
    return sorted(set(seasons))


def build_bracket_field_team_ids(bracket_payload: dict) -> List[int]:
    field_ids = set()
    for slot in bracket_payload["slots"]:
        for side_name in ("team_1", "team_2"):
            side = slot.get(side_name, {})
            team_id = side.get("team_id")
            if team_id is not None:
                field_ids.add(int(team_id))
    return sorted(field_ids)


def load_field_team_ids(teams_index_path: Path, bracket_path: Path) -> List[int]:
    teams_index = load_json(teams_index_path)
    bracket_payload = load_json(bracket_path)

    bracket_team_ids = build_bracket_field_team_ids(bracket_payload)
    index_team_ids = sorted(
        int(team["team_id"])
        for team in teams_index["teams"]
        if team.get("tournament_team")
    )

    if not bracket_team_ids:
        raise ValueError("No tournament teams were found in the published bracket artifact.")

    if index_team_ids != bracket_team_ids:
        missing_in_index = sorted(set(bracket_team_ids) - set(index_team_ids))
        extra_in_index = sorted(set(index_team_ids) - set(bracket_team_ids))
        raise ValueError(
            "Team index tournament field does not match published bracket field: "
            f"missing_in_index={missing_in_index}, extra_in_index={extra_in_index}"
        )

    return bracket_team_ids


def probability_summary_from_matrix(prob_matrix: np.ndarray) -> pd.DataFrame:
    summary = pd.DataFrame(prob_matrix, columns=[EXIT_PROBABILITY_COLUMNS[idx] for idx in range(1, 8)])
    summary["expected_exit_round"] = [
        float(np.dot(prob_vector, np.arange(1, 8)))
        for prob_vector in prob_matrix
    ]
    summary["most_likely_round_num"] = prob_matrix.argmax(axis=1) + 1
    summary["floor_round_num"] = [
        exit_round_quantile(prob_vector, 0.25)
        for prob_vector in prob_matrix
    ]
    summary["ceiling_round_num"] = [
        exit_round_quantile(prob_vector, 0.75)
        for prob_vector in prob_matrix
    ]
    summary["p_sweet16_plus"] = prob_matrix[:, 2:].sum(axis=1)
    summary["p_elite8_plus"] = prob_matrix[:, 3:].sum(axis=1)
    summary["p_final4_plus"] = prob_matrix[:, 4:].sum(axis=1)
    summary["p_title_game_plus"] = prob_matrix[:, 5:].sum(axis=1)
    summary["most_likely_round"] = summary["most_likely_round_num"].map(exit_round_label)
    summary["floor_round"] = summary["floor_round_num"].map(exit_round_label)
    summary["ceiling_round"] = summary["ceiling_round_num"].map(exit_round_label)
    return summary


def slot_round_number(slot_name: str | None) -> int | None:
    if not slot_name:
        return None
    match = re.match(r"R(\d+)", str(slot_name))
    return int(match.group(1)) if match else None


def normalize_slot_family(slot_name: str | None) -> str | None:
    if slot_name is None or pd.isna(slot_name):
        return None

    slot_value = str(slot_name)
    if ":" in slot_value and slot_value.startswith("R"):
        return slot_value

    round_number = slot_round_number(slot_value)
    if round_number is None:
        return slot_value

    suffix = re.sub(r"^R\d+", "", slot_value)
    if suffix and suffix[0] in "WXYZ" and any(character.isdigit() for character in suffix[1:]):
        suffix = suffix[1:]
    return f"R{round_number}:{suffix}" if suffix else f"R{round_number}"


def tourney_round_from_day(day_num: int) -> str:
    if day_num <= 137:
        return "Round of 64"
    if day_num <= 139:
        return "Round of 32"
    if day_num <= 144:
        return "Sweet Sixteen"
    if day_num <= 146:
        return "Elite Eight"
    if day_num <= 153:
        return "Final Four"
    return "Championship"


def build_exit_model_payload(
    season: int,
    training_seasons: Sequence[int],
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict], List[dict]]:
    historical_start = min(training_seasons)
    historical_end = max(training_seasons)
    exit_model_df = build_historical_exit_round_frame(historical_start, historical_end)
    exit_model_df = exit_model_df.loc[exit_model_df["season"].isin(training_seasons)].copy()
    if exit_model_df.empty:
        raise ValueError("Historical exit-round training frame is empty for the selected seasons.")

    validation_season = min(TEAM_PAGE_EXIT_MODEL_VALIDATION_SEASON, historical_end)
    train_exit_df = exit_model_df.loc[exit_model_df["season"] < validation_season].copy()
    if train_exit_df.empty:
        train_exit_df = exit_model_df.copy()

    exit_round_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=42,
    )
    exit_round_model.fit(
        train_exit_df[EXIT_ROUND_MODEL_FEATURE_COLUMNS],
        train_exit_df["exit_round_num"] - 1,
    )

    seed_distribution = (
        exit_model_df.groupby(["seed", "exit_round_num"]).size().unstack(fill_value=0)
        .reindex(columns=range(1, 8), fill_value=0)
    )
    seed_distribution = seed_distribution.div(seed_distribution.sum(axis=1), axis=0)
    seed_baseline_df = pd.DataFrame(
        {
            "seed": seed_distribution.index.astype(int),
            "seed_expected_round": [
                float(np.dot(row.to_numpy(), np.arange(1, 8)))
                for _, row in seed_distribution.iterrows()
            ],
        }
    )

    historical_probs = exit_round_model.predict_proba(exit_model_df[EXIT_ROUND_MODEL_FEATURE_COLUMNS])
    historical_team_prior_df = pd.concat(
        [
            exit_model_df[["id", "season", "seed"]].rename(columns={"id": "team_id"}).reset_index(drop=True),
            probability_summary_from_matrix(historical_probs),
        ],
        axis=1,
    )
    historical_team_prior_df = historical_team_prior_df.merge(seed_baseline_df, on="seed", how="left")
    historical_team_prior_df["seed_delta"] = (
        historical_team_prior_df["expected_exit_round"] - historical_team_prior_df["seed_expected_round"]
    )

    team_seasons, _ = load_team_seasons_for_export(season)
    seed_by_team_id, _, _ = build_placeholder_seed_maps(season)

    current_rows: List[dict] = []
    for team_id, team_season in team_seasons.items():
        seed_info = seed_by_team_id.get(team_id)
        if seed_info is None:
            continue

        row = {
            "team_id": int(team_id),
            "season": int(season),
            "team_name": team_season.name,
            "seed": int(seed_info["seed"]),
            "seed_label": seed_info["seed_label"],
            "region": seed_info["region"],
        }
        for feature_name in EXIT_ROUND_MODEL_FEATURE_COLUMNS:
            value = get_feature_value(team_season, feature_name, seed_by_team_id)
            row[feature_name] = 0.0 if is_missing_value(value) else float(value)
        current_rows.append(row)

    current_exit_df = pd.DataFrame(current_rows)
    if current_exit_df.empty:
        raise ValueError(f"No seeded current-season teams were found for {season}.")

    current_probs = exit_round_model.predict_proba(current_exit_df[EXIT_ROUND_MODEL_FEATURE_COLUMNS])
    current_exit_summary = pd.concat(
        [current_exit_df.reset_index(drop=True), probability_summary_from_matrix(current_probs)],
        axis=1,
    )
    current_exit_summary = current_exit_summary.merge(seed_baseline_df, on="seed", how="left")
    current_exit_summary["seed_delta"] = (
        current_exit_summary["expected_exit_round"] - current_exit_summary["seed_expected_round"]
    )
    current_exit_summary["region_rank"] = (
        current_exit_summary.groupby("region")["expected_exit_round"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    historical_tourney_rows: List[dict] = []
    for historical_season in training_seasons:
        historical_team_seasons, _ = load_team_seasons_for_export(historical_season)
        historical_seed_by_team_id, _, _ = build_placeholder_seed_maps(historical_season)
        for team_id, team_season in historical_team_seasons.items():
            seed_info = historical_seed_by_team_id.get(team_id)
            exit_round_num = EXIT_ROUND_NUM_MAP.get(team_season.tourney_exit_round)
            if seed_info is None or exit_round_num is None:
                continue

            row = {
                "team_id": int(team_id),
                "season": int(historical_season),
                "team_name": team_season.name,
                "seed": int(seed_info["seed"]),
                "seed_label": seed_info["seed_label"],
                "region": seed_info["region"],
                "exit_round": team_season.tourney_exit_round,
                "exit_round_num": int(exit_round_num),
            }
            for feature_name in STAT_EXPLORER_FEATURE_COLUMNS:
                value = get_feature_value(team_season, feature_name, historical_seed_by_team_id)
                row[feature_name] = 0.0 if is_missing_value(value) else float(value)
            historical_tourney_rows.append(row)

    historical_tourney_df = pd.DataFrame(historical_tourney_rows)
    if historical_tourney_df.empty:
        raise ValueError("Historical tournament team dataset is empty for the selected seasons.")

    return (
        exit_round_model,
        historical_team_prior_df,
        seed_baseline_df,
        current_exit_summary,
        current_rows,
        historical_tourney_rows,
    )


def matchup_row_team_id(team_row: pd.Series) -> int:
    if "team_id" in team_row.index and pd.notna(team_row["team_id"]):
        return int(team_row["team_id"])
    if isinstance(team_row.name, tuple):
        return int(team_row.name[1])
    return int(team_row.name)


def matchup_row_seed_label(team_row: pd.Series) -> str | None:
    if "seed_label" in team_row.index and pd.notna(team_row["seed_label"]):
        return str(team_row["seed_label"])
    if "region" in team_row.index and pd.notna(team_row["region"]) and pd.notna(team_row["seed"]):
        return f"{team_row['region']}{int(team_row['seed']):02d}"
    return None


def canonicalize_matchup(team_one: pd.Series, team_two: pd.Series) -> Tuple[pd.Series, pd.Series]:
    team_one_seed = float(team_one["seed"] if pd.notna(team_one["seed"]) else 99)
    team_two_seed = float(team_two["seed"] if pd.notna(team_two["seed"]) else 99)

    team_one_key = (team_one_seed, -float(team_one["expected_exit_round"]), matchup_row_team_id(team_one))
    team_two_key = (team_two_seed, -float(team_two["expected_exit_round"]), matchup_row_team_id(team_two))

    if team_one_key <= team_two_key:
        return team_one, team_two
    return team_two, team_one


def build_season_slot_structure(
    season: int,
    season_game_winner_lookup: Dict[Tuple[int, Tuple[int, int]], int],
) -> Tuple[List[dict], Dict[str, set[str]]]:
    season_slots = SLOTS_DF.loc[SLOTS_DF["Season"] == season, ["Slot", "StrongSeed", "WeakSeed"]].copy()
    if season_slots.empty:
        return [], {}

    season_slots["slot_round"] = season_slots["Slot"].map(slot_round_number)
    season_slots = season_slots.sort_values(["slot_round", "Slot"]).reset_index(drop=True)
    slot_source_map = season_slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict("index")

    season_seed_df = SEEDS_DF.loc[SEEDS_DF["Season"] == season, ["Seed", "TeamID"]].copy()
    seed_to_team = dict(zip(season_seed_df["Seed"], season_seed_df["TeamID"]))

    descendant_cache: Dict[str, set[str]] = {}

    def descendant_seed_labels(source: str) -> set[str]:
        if source in descendant_cache:
            return descendant_cache[source]
        if source in seed_to_team:
            descendant_cache[source] = {source}
            return descendant_cache[source]
        if source not in slot_source_map:
            descendant_cache[source] = set()
            return descendant_cache[source]

        slot_sources = slot_source_map[source]
        descendant_cache[source] = descendant_seed_labels(slot_sources["StrongSeed"]) | descendant_seed_labels(slot_sources["WeakSeed"])
        return descendant_cache[source]

    slot_winner_map: Dict[str, int] = {}
    season_slot_rows: List[dict] = []

    for slot_row in season_slots.itertuples():
        strong_source = str(slot_row.StrongSeed)
        weak_source = str(slot_row.WeakSeed)
        strong_team_id = seed_to_team.get(strong_source, slot_winner_map.get(strong_source))
        weak_team_id = seed_to_team.get(weak_source, slot_winner_map.get(weak_source))

        if strong_team_id is None or weak_team_id is None:
            continue

        game_key = (int(season), tuple(sorted((int(strong_team_id), int(weak_team_id)))))
        winner_team_id = season_game_winner_lookup.get(game_key)
        if winner_team_id is None:
            continue

        slot_winner_map[str(slot_row.Slot)] = int(winner_team_id)
        loser_team_id = int(weak_team_id if int(winner_team_id) == int(strong_team_id) else strong_team_id)
        season_slot_rows.append(
            {
                "season": int(season),
                "slot": str(slot_row.Slot),
                "slot_family": normalize_slot_family(str(slot_row.Slot)),
                "slot_round": int(slot_row.slot_round) if pd.notna(slot_row.slot_round) else None,
                "strong_team_id": int(strong_team_id),
                "weak_team_id": int(weak_team_id),
                "winner_team_id": int(winner_team_id),
                "loser_team_id": loser_team_id,
            }
        )

    season_slot_descendants = {
        str(slot_name): descendant_seed_labels(str(slot_name))
        for slot_name in season_slots["Slot"]
    }
    return season_slot_rows, season_slot_descendants


def infer_matchup_slot_from_seed_labels(
    season: int,
    seed_label_one: str | None,
    seed_label_two: str | None,
    slot_descendants_by_season: Dict[int, Dict[str, set[str]]],
    default_slot_template_season: int,
) -> Tuple[str | None, str | None]:
    if not seed_label_one or not seed_label_two:
        return None, None

    season_slot_descendants = slot_descendants_by_season.get(int(season))
    if not season_slot_descendants:
        season_slot_descendants = slot_descendants_by_season.get(default_slot_template_season, {})

    candidate_slots = [
        slot_name
        for slot_name, descendant_seed_set in season_slot_descendants.items()
        if seed_label_one in descendant_seed_set and seed_label_two in descendant_seed_set
    ]
    if not candidate_slots:
        return None, None

    best_slot = min(
        candidate_slots,
        key=lambda slot_name: (slot_round_number(slot_name) or 99, slot_name),
    )
    return best_slot, normalize_slot_family(best_slot)


def build_model_metadata(
    *,
    model_id: str,
    model_name: str,
    season: int,
    training_seasons: Sequence[int],
    feature_set: str,
    matchup_insights_url: str | None,
    description: str,
) -> dict:
    return {
        "id": model_id,
        "name": model_name,
        "type": "precomputed-insights",
        "model_family": "matchup-clustering",
        "seasons": [season],
        "training_seasons": list(training_seasons),
        "feature_set": feature_set,
        "prediction_format": "precomputed-insights",
        "inference_task": "matchup-insights",
        "artifact_format": "pairwise-matchup-insights-v1",
        "supports_bracket_autofill": False,
        "description": description,
        "artifact_version": "1.0.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "matchup_insights_url": matchup_insights_url,
    }


def update_manifest(manifest_path: Path, model_metadata: dict, make_default: bool) -> None:
    if manifest_path.exists():
        manifest_payload = load_json(manifest_path)
    else:
        manifest_payload = {"default_model_id": model_metadata["id"], "models": []}

    models = [entry for entry in manifest_payload.get("models", []) if entry.get("id") != model_metadata["id"]]
    models.append(model_metadata)
    models.sort(key=lambda item: item["name"])
    manifest_payload["models"] = models
    if make_default or not manifest_payload.get("default_model_id"):
        manifest_payload["default_model_id"] = model_metadata["id"]

    write_json(manifest_path, manifest_payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a website-ready 2026 matchup-clustering insights artifact."
    )
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--teams-index", type=Path, default=DEFAULT_TEAMS_INDEX)
    parser.add_argument("--bracket-path", type=Path, default=DEFAULT_BRACKET)
    parser.add_argument("--insights-dir", type=Path, default=DEFAULT_INSIGHTS_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--training-seasons", default=None)
    parser.add_argument("--nearest-games", type=int, default=DEFAULT_NEAREST_GAMES)
    parser.add_argument("--artifact-base-url", default=None)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--make-default", action="store_true")
    parser.add_argument("--skip-model-save", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    training_seasons = parse_training_seasons(args.training_seasons, args.season)

    field_team_ids = load_field_team_ids(args.teams_index, args.bracket_path)

    (
        exit_round_model,
        historical_team_prior_df,
        seed_baseline_df,
        current_exit_summary,
        current_rows,
        historical_tourney_rows,
    ) = build_exit_model_payload(args.season, training_seasons)

    historical_tourney_df = pd.DataFrame(historical_tourney_rows)
    if historical_tourney_df.empty:
        raise ValueError("Historical tournament rows are empty for the selected seasons.")

    top_signal_features = (
        historical_tourney_df[STAT_EXPLORER_FEATURE_COLUMNS + ["seed", "exit_round_num"]]
        .fillna(0)
        .corr()
        .loc[STAT_EXPLORER_FEATURE_COLUMNS, "exit_round_num"]
        .abs()
        .sort_values(ascending=False)
        .head(6)
        .index
        .tolist()
    )
    matchup_signal_features = list(top_signal_features)
    matchup_prior_columns = [
        "expected_exit_round",
        "seed_delta",
        "p_sweet16_plus",
        "p_elite8_plus",
        "p_final4_plus",
    ]

    historical_matchup_team_df = (
        historical_tourney_df[
            [
                "team_id",
                "season",
                "team_name",
                "seed",
                "seed_label",
                "region",
                "exit_round",
                "exit_round_num",
            ]
            + matchup_signal_features
        ]
        .merge(
            historical_team_prior_df[
                [
                    "team_id",
                    "season",
                    "expected_exit_round",
                    "seed_delta",
                    "p_sweet16_plus",
                    "p_elite8_plus",
                    "p_final4_plus",
                ]
            ],
            on=["team_id", "season"],
            how="left",
        )
        .drop_duplicates(subset=["season", "team_id"])
        .set_index(["season", "team_id"])
    )

    season_game_winner_lookup = {
        (int(game.Season), tuple(sorted((int(game.WTeamID), int(game.LTeamID))))): int(game.WTeamID)
        for game in TOURNEY_RESULTS_DF.itertuples()
        if int(game.Season) in training_seasons
    }

    historical_slot_rows: List[dict] = []
    slot_descendants_by_season: Dict[int, Dict[str, set[str]]] = {}
    slot_structure_seasons = sorted(set(training_seasons + [args.season]))
    for season in slot_structure_seasons:
        season_slot_rows, season_slot_descendants = build_season_slot_structure(
            season,
            season_game_winner_lookup,
        )
        if season in training_seasons:
            historical_slot_rows.extend(season_slot_rows)
        slot_descendants_by_season[int(season)] = season_slot_descendants

    default_slot_template_season = int(SLOTS_DF["Season"].max())
    historical_slot_lookup = {
        (row["season"], tuple(sorted((row["strong_team_id"], row["weak_team_id"])))): {
            "slot": row["slot"],
            "slot_family": row["slot_family"],
            "slot_round": row["slot_round"],
        }
        for row in historical_slot_rows
    }

    matchup_rows: List[dict] = []
    historical_games = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"].isin(training_seasons)].copy()
    for game in historical_games.itertuples():
        game_key_winner = (int(game.Season), int(game.WTeamID))
        game_key_loser = (int(game.Season), int(game.LTeamID))
        if game_key_winner not in historical_matchup_team_df.index or game_key_loser not in historical_matchup_team_df.index:
            continue

        winner_team = historical_matchup_team_df.loc[game_key_winner]
        loser_team = historical_matchup_team_df.loc[game_key_loser]
        favorite_team, underdog_team = canonicalize_matchup(winner_team, loser_team)
        favorite_won = int(matchup_row_team_id(favorite_team) == int(game.WTeamID))
        favorite_score = int(game.WScore if favorite_won else game.LScore)
        underdog_score = int(game.LScore if favorite_won else game.WScore)
        slot_info = historical_slot_lookup.get(
            (int(game.Season), tuple(sorted((int(game.WTeamID), int(game.LTeamID)))))
        )

        row = {
            "season": int(game.Season),
            "day_num": int(game.DayNum),
            "round": tourney_round_from_day(int(game.DayNum)),
            "slot": None if slot_info is None else slot_info["slot"],
            "slot_family": None if slot_info is None else slot_info["slot_family"],
            "slot_round": None if slot_info is None else slot_info["slot_round"],
            "favorite_team_id": matchup_row_team_id(favorite_team),
            "favorite_team": favorite_team["team_name"],
            "favorite_seed": int(favorite_team["seed"]),
            "favorite_seed_label": matchup_row_seed_label(favorite_team),
            "favorite_exit_round": favorite_team["exit_round"],
            "underdog_team_id": matchup_row_team_id(underdog_team),
            "underdog_team": underdog_team["team_name"],
            "underdog_seed": int(underdog_team["seed"]),
            "underdog_seed_label": matchup_row_seed_label(underdog_team),
            "underdog_exit_round": underdog_team["exit_round"],
            "favorite_won": favorite_won,
            "upset": int(not favorite_won),
            "favorite_score": favorite_score,
            "underdog_score": underdog_score,
            "margin": favorite_score - underdog_score,
            "total_points": favorite_score + underdog_score,
            "num_ot": int(game.NumOT),
        }
        row["seed_gap"] = row["underdog_seed"] - row["favorite_seed"]
        for column in matchup_prior_columns + matchup_signal_features:
            row[f"{column}_gap"] = float(favorite_team[column]) - float(underdog_team[column])
        matchup_rows.append(row)

    historical_matchup_df = pd.DataFrame(matchup_rows)
    if historical_matchup_df.empty:
        raise ValueError("Historical matchup dataset is empty for the selected seasons.")

    matchup_feature_columns = [
        "favorite_seed",
        "underdog_seed",
        "seed_gap",
        *[f"{column}_gap" for column in matchup_prior_columns + matchup_signal_features],
    ]
    matchup_feature_matrix = historical_matchup_df[matchup_feature_columns].fillna(0)
    matchup_scaler = StandardScaler()
    matchup_feature_scaled = matchup_scaler.fit_transform(matchup_feature_matrix)

    cluster_candidates: List[dict] = []
    for cluster_count in DEFAULT_CLUSTER_RANGE:
        candidate_model = KMeans(n_clusters=cluster_count, random_state=42, n_init=25)
        candidate_labels = candidate_model.fit_predict(matchup_feature_scaled)
        cluster_candidates.append(
            {
                "cluster_count": int(cluster_count),
                "silhouette": round(float(silhouette_score(matchup_feature_scaled, candidate_labels)), 4),
            }
        )

    matchup_cluster_search = pd.DataFrame(cluster_candidates).sort_values("silhouette", ascending=False).reset_index(drop=True)
    best_matchup_cluster_count = int(matchup_cluster_search.iloc[0]["cluster_count"])
    matchup_cluster_model = KMeans(n_clusters=best_matchup_cluster_count, random_state=42, n_init=25)
    historical_matchup_df["matchup_cluster"] = matchup_cluster_model.fit_predict(matchup_feature_scaled)

    cluster_round_mode = (
        historical_matchup_df.groupby("matchup_cluster")["round"]
        .agg(lambda rounds: rounds.mode().iat[0])
        .rename("common_round")
    )
    matchup_cluster_summary = (
        historical_matchup_df.groupby("matchup_cluster")
        .agg(
            cluster_game_count=("season", "size"),
            favorite_win_rate=("favorite_won", "mean"),
            upset_rate=("upset", "mean"),
            avg_margin=("margin", "mean"),
            avg_total_points=("total_points", "mean"),
            avg_seed_gap=("seed_gap", "mean"),
            avg_expected_round_gap=("expected_exit_round_gap", "mean"),
            avg_final4_gap=("p_final4_plus_gap", "mean"),
        )
        .join(cluster_round_mode)
        .sort_values(["upset_rate", "avg_seed_gap"], ascending=[False, False])
        .round(6)
    )

    current_exit_summary = current_exit_summary[current_exit_summary["team_id"].isin(field_team_ids)].copy()
    current_team_lookup = current_exit_summary.set_index("team_id")
    if set(field_team_ids) != set(current_team_lookup.index.tolist()):
        missing = sorted(set(field_team_ids) - set(current_team_lookup.index.tolist()))
        raise ValueError(f"Current exit summary is missing field teams: {missing}")

    insights_path = args.insights_dir / f"{args.model_id}_{args.season}.json"
    (insights_url,) = build_model_urls(args.artifact_base_url, insights_path)
    model_metadata = build_model_metadata(
        model_id=args.model_id,
        model_name=args.model_name,
        season=args.season,
        training_seasons=training_seasons,
        feature_set=STAT_EXPLORER_FEATURE_SET_NAME,
        matchup_insights_url=insights_url,
        description=args.description,
    )

    matchups_payload: Dict[str, dict] = {}
    sorted_field_team_ids = sorted(int(team_id) for team_id in field_team_ids)
    for team_a_id in sorted_field_team_ids:
        for team_b_id in sorted_field_team_ids:
            if team_a_id == team_b_id:
                continue

            team_one = current_team_lookup.loc[team_a_id]
            team_two = current_team_lookup.loc[team_b_id]
            favorite_team, underdog_team = canonicalize_matchup(team_one, team_two)

            bracket_slot, bracket_slot_family = infer_matchup_slot_from_seed_labels(
                args.season,
                matchup_row_seed_label(favorite_team),
                matchup_row_seed_label(underdog_team),
                slot_descendants_by_season,
                default_slot_template_season,
            )

            feature_row = {
                "favorite_seed": int(favorite_team["seed"]),
                "underdog_seed": int(underdog_team["seed"]),
                "seed_gap": int(underdog_team["seed"] - favorite_team["seed"]),
            }
            for column in matchup_prior_columns + matchup_signal_features:
                feature_row[f"{column}_gap"] = float(favorite_team[column]) - float(underdog_team[column])
            matchup_features = pd.DataFrame([feature_row])[matchup_feature_columns]

            scaled_features = matchup_scaler.transform(matchup_features)
            assigned_cluster = int(matchup_cluster_model.predict(scaled_features)[0])
            cluster_games = historical_matchup_df.loc[
                historical_matchup_df["matchup_cluster"] == assigned_cluster
            ].copy()
            cluster_game_scaled = matchup_scaler.transform(cluster_games[matchup_feature_columns].fillna(0))
            cluster_games["similarity"] = cosine_similarity(scaled_features, cluster_game_scaled).ravel()
            nearest_games = cluster_games.sort_values("similarity", ascending=False).head(args.nearest_games)

            cluster_summary_row = matchup_cluster_summary.loc[assigned_cluster]
            cluster_summary_payload = {
                "cluster_id": assigned_cluster,
                "cluster_game_count": int(cluster_summary_row["cluster_game_count"]),
                "filtered_game_count": int(cluster_summary_row["cluster_game_count"]),
                "favorite_win_rate": round(float(cluster_summary_row["favorite_win_rate"]), 6),
                "upset_rate": round(float(cluster_summary_row["upset_rate"]), 6),
                "avg_margin": round(float(cluster_summary_row["avg_margin"]), 6),
                "avg_total_points": round(float(cluster_summary_row["avg_total_points"]), 6),
                "avg_seed_gap": round(float(cluster_summary_row["avg_seed_gap"]), 6),
                "common_round": str(cluster_summary_row["common_round"]),
            }

            nearest_games_payload = [
                {
                    "season": int(row.season),
                    "round": str(row.round),
                    "slot": None if pd.isna(row.slot) else str(row.slot),
                    "slot_family": None if pd.isna(row.slot_family) else str(row.slot_family),
                    "favorite_team_id": int(row.favorite_team_id),
                    "underdog_team_id": int(row.underdog_team_id),
                    "favorite_won": bool(row.favorite_won),
                    "margin": int(row.margin),
                    "total_points": int(row.total_points),
                    "similarity": round(float(row.similarity), 6),
                }
                for row in nearest_games.itertuples()
            ]

            matchups_payload[f"{team_a_id}:{team_b_id}"] = {
                "team_a_id": int(team_a_id),
                "team_b_id": int(team_b_id),
                "favorite_team_id": int(matchup_row_team_id(favorite_team)),
                "underdog_team_id": int(matchup_row_team_id(underdog_team)),
                "bracket_slot": bracket_slot,
                "bracket_slot_family": bracket_slot_family,
                "same_bracket_slot": False,
                "cluster_summary": cluster_summary_payload,
                "nearest_games": nearest_games_payload,
            }

    insights_payload = {
        "season": int(args.season),
        "model": model_metadata,
        "inference_task": "matchup-insights",
        "artifact_format": "pairwise-matchup-insights-v1",
        "team_ids": sorted_field_team_ids,
        "matchups": matchups_payload,
    }
    write_json(insights_path, insights_payload)

    if not args.skip_model_save:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        with args.model_output.open("wb") as handle:
            pickle.dump(
                {
                    "season": args.season,
                    "model_id": args.model_id,
                    "training_seasons": training_seasons,
                    "feature_set": STAT_EXPLORER_FEATURE_SET_NAME,
                    "matchup_feature_columns": matchup_feature_columns,
                    "matchup_signal_features": matchup_signal_features,
                    "matchup_prior_columns": matchup_prior_columns,
                    "seed_baseline": seed_baseline_df.to_dict(orient="records"),
                    "cluster_search": cluster_candidates,
                    "cluster_summary": matchup_cluster_summary.reset_index().to_dict(orient="records"),
                    "model_state": {
                        "exit_round_model": exit_round_model,
                        "matchup_cluster_model": matchup_cluster_model,
                        "matchup_scaler": matchup_scaler,
                    },
                },
                handle,
            )

    if args.update_manifest:
        update_manifest(args.manifest_path, model_metadata, make_default=args.make_default)

    print(
        f"Built matchup insights for {len(sorted_field_team_ids)} tournament teams "
        f"across {len(matchups_payload)} ordered matchups."
    )
    print(f"Training seasons: {training_seasons}")
    print(f"Selected signal features: {', '.join(matchup_signal_features)}")
    print(f"Chosen cluster count: {best_matchup_cluster_count}")
    print(f"Exported insights: {insights_path.relative_to(REPO_ROOT)}")
    if not args.skip_model_save:
        print(f"Saved model state: {args.model_output.relative_to(REPO_ROOT)}")
    if args.update_manifest:
        print(f"Updated manifest: {args.manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()