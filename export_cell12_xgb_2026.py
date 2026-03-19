from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from model_2026 import (
    DEFAULT_BRACKET,
    DEFAULT_BROWSER_FEATURE_STORE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_TEAMS_INDEX,
    REPO_ROOT,
    build_feature_importance_payload,
    build_model_urls,
    load_json,
    update_manifest,
    write_json,
)


DEFAULT_MODEL_OUTPUT = REPO_ROOT / "Models/cell12_xgb_2026_browser.pkl"
DEFAULT_MODEL_ID = "cell12-xgb-2026-v1"
DEFAULT_MODEL_NAME = "Cell 12 XGBoost 2026"
DEFAULT_DESCRIPTION = "XGBoost model matching notebook cell 12 and exported as pairwise 2026 tournament probabilities."
DEFAULT_GRID_PARAMS = {
    "learning_rate": [0.15, 0.1],
    "gamma": [0, 0.25],
    "reg_lambda": [0, 1],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}


def load_feature_lookup(feature_order: Sequence[str]) -> Dict[Tuple[int, int], np.ndarray]:
    ts_df = pd.read_csv(REPO_ROOT / "TeamSeasons_2026.csv")
    indexed = ts_df.set_index(["Season", "TeamID"])[list(feature_order)].fillna(0)
    return {
        (int(season), int(team_id)): row.to_numpy(dtype=float)
        for (season, team_id), row in indexed.iterrows()
    }


def build_training_examples(
    games_df: pd.DataFrame,
    feature_lookup: Dict[Tuple[int, int], np.ndarray],
) -> Tuple[List[List[float]], List[int], List[int]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    seasons_seen = set()

    for game in games_df.itertuples():
        winner_key = (int(game.Season), int(game.WTeamID))
        loser_key = (int(game.Season), int(game.LTeamID))
        if winner_key not in feature_lookup or loser_key not in feature_lookup:
            continue

        winner_features = feature_lookup[winner_key]
        loser_features = feature_lookup[loser_key]
        x_rows.append((winner_features - loser_features).tolist())
        y_rows.append(1)
        x_rows.append((loser_features - winner_features).tolist())
        y_rows.append(0)
        seasons_seen.add(int(game.Season))

    return x_rows, y_rows, sorted(seasons_seen)


def train_cell12_model(
    feature_order: Sequence[str],
    grid_params: dict,
) -> Tuple[xgb.XGBClassifier, List[int], float]:
    feature_lookup = load_feature_lookup(feature_order)
    reg_szn_df = pd.read_csv(
        REPO_ROOT / "data/kaggle-v2/MRegularSeasonDetailedResults.csv",
        usecols=["Season", "WTeamID", "LTeamID"],
    )
    tourney_df = pd.read_csv(
        REPO_ROOT / "data/kaggle-v2/MNCAATourneyDetailedResults.csv",
        usecols=["Season", "WTeamID", "LTeamID"],
    )

    train_inputs, train_results, training_seasons = build_training_examples(reg_szn_df, feature_lookup)
    validation_inputs, validation_results, _ = build_training_examples(tourney_df, feature_lookup)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )
    grid_cv = GridSearchCV(model, grid_params, cv=5, n_jobs=-1)
    grid_cv.fit(train_inputs, train_results)

    best_model = xgb.XGBClassifier(
        **grid_cv.best_params_,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
    )
    best_model.fit(train_inputs, train_results)
    validation_accuracy = float(accuracy_score(validation_results, best_model.predict(validation_inputs)))
    return best_model, training_seasons, validation_accuracy


def build_model_metadata(
    *,
    model_id: str,
    model_name: str,
    season: int,
    training_seasons: Sequence[int],
    feature_set: str,
    predictions_url: str | None,
    feature_importance_url: str | None,
    description: str,
    validation_accuracy: float,
) -> dict:
    return {
        "id": model_id,
        "name": model_name,
        "type": "precomputed-predictions",
        "model_family": "xgboost",
        "seasons": [season],
        "training_seasons": list(training_seasons),
        "feature_set": feature_set,
        "prediction_format": "precomputed",
        "mirrored_prediction_strategy": "average-forward-reverse",
        "description": description,
        "artifact_version": "0.1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "predictions_url": predictions_url,
        "feature_importance_url": feature_importance_url,
        "validation_accuracy": round(validation_accuracy, 6),
    }


def build_predictions_payload(
    model: xgb.XGBClassifier,
    feature_store_payload: dict,
    team_ids: Sequence[int],
    model_metadata: dict,
    feature_importance: List[dict],
) -> dict:
    feature_order = feature_store_payload["feature_order"]
    teams = feature_store_payload["teams"]
    predictions: Dict[str, List[float | int]] = {}

    sorted_team_ids = sorted(int(team_id) for team_id in team_ids)
    for index, team_a_id in enumerate(sorted_team_ids):
        for team_b_id in sorted_team_ids[index + 1 :]:
            team_a_features = teams[str(team_a_id)]
            team_b_features = teams[str(team_b_id)]
            vector_ab = [
                round(float(team_a_features[name]) - float(team_b_features[name]), 6)
                for name in feature_order
            ]
            vector_ba = [
                round(float(team_b_features[name]) - float(team_a_features[name]), 6)
                for name in feature_order
            ]

            probs_ab = model.predict_proba([vector_ab])[0]
            probs_ba = model.predict_proba([vector_ba])[0]
            team_a_prob = (float(probs_ab[1]) + float(probs_ba[0])) / 2.0
            team_b_prob = 1.0 - team_a_prob

            winner_id = team_a_id if team_a_prob >= team_b_prob else team_b_id
            winner_prob = round(max(team_a_prob, team_b_prob), 6)
            predictions[f"{team_a_id}:{team_b_id}"] = [winner_id, winner_prob]
            predictions[f"{team_b_id}:{team_a_id}"] = [winner_id, winner_prob]

    return {
        "season": int(feature_store_payload["season"]),
        "model": model_metadata,
        "prediction_format": "pairwise-winner-probability",
        "team_ids": sorted_team_ids,
        "feature_importance": feature_importance,
        "predictions": predictions,
    }


def load_export_field_team_ids(
    teams_index_path: Path,
    bracket_path: Path,
    feature_store_payload: dict,
) -> List[int]:
    teams_index = load_json(teams_index_path)
    bracket_payload = load_json(bracket_path)

    bracket_team_ids = sorted(
        {
            int(entry["team_id"])
            for slot in bracket_payload["slots"]
            for side in ("team_1", "team_2")
            for entry in [slot.get(side, {})]
            if entry.get("team_id") is not None
        }
    )
    index_team_ids = sorted(
        int(team["team_id"])
        for team in teams_index["teams"]
        if team.get("tournament_team")
    )
    feature_store_team_ids = sorted(
        int(team_id)
        for team_id in feature_store_payload["teams"].keys()
        if int(team_id) in set(index_team_ids)
    )

    if index_team_ids != bracket_team_ids:
        raise ValueError(
            "Team index tournament field does not match published bracket field."
        )
    if feature_store_team_ids != index_team_ids:
        missing_ids = sorted(set(index_team_ids) - set(feature_store_team_ids))
        raise ValueError(
            "Feature store is missing tournament teams: " + ", ".join(str(team_id) for team_id in missing_ids)
        )

    return index_team_ids


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the notebook cell 12 XGBoost model as a browser-ready 2026 precomputed predictions artifact."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--browser-feature-store", type=Path, default=DEFAULT_BROWSER_FEATURE_STORE)
    parser.add_argument("--teams-index", type=Path, default=DEFAULT_TEAMS_INDEX)
    parser.add_argument("--bracket-path", type=Path, default=DEFAULT_BRACKET)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--artifact-base-url", default=None)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--make-default", action="store_true")
    parser.add_argument("--skip-model-save", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    browser_feature_store = load_json(args.browser_feature_store)
    feature_order = browser_feature_store["feature_order"]
    feature_set_name = browser_feature_store["feature_set"]

    model, training_seasons, validation_accuracy = train_cell12_model(
        feature_order=feature_order,
        grid_params=DEFAULT_GRID_PARAMS,
    )

    field_team_ids = load_export_field_team_ids(
        args.teams_index,
        args.bracket_path,
        browser_feature_store,
    )

    predictions_path = args.predictions_dir / f"{args.model_id}_{args.season}.json"
    importance_path = args.predictions_dir / f"{args.model_id}_importance.json"
    predictions_url, importance_url = build_model_urls(
        args.artifact_base_url,
        predictions_path,
        importance_path,
    )

    feature_importance = build_feature_importance_payload(
        feature_order,
        model.feature_importances_,
    )
    model_metadata = build_model_metadata(
        model_id=args.model_id,
        model_name=args.model_name,
        season=args.season,
        training_seasons=training_seasons,
        feature_set=feature_set_name,
        predictions_url=predictions_url,
        feature_importance_url=importance_url,
        description=args.description,
        validation_accuracy=validation_accuracy,
    )

    predictions_payload = build_predictions_payload(
        model,
        feature_store_payload=browser_feature_store,
        team_ids=field_team_ids,
        model_metadata=model_metadata,
        feature_importance=feature_importance,
    )
    write_json(predictions_path, predictions_payload)
    write_json(
        importance_path,
        {
            "model_id": args.model_id,
            "season": args.season,
            "feature_set": feature_set_name,
            "feature_importance": feature_importance,
        },
    )

    if not args.skip_model_save:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        with args.model_output.open("wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "feature_order": feature_order,
                    "feature_set": feature_set_name,
                    "training_seasons": training_seasons,
                    "validation_accuracy": validation_accuracy,
                    "grid_params": DEFAULT_GRID_PARAMS,
                },
                handle,
            )

    if args.update_manifest:
        update_manifest(args.manifest_path, model_metadata, make_default=args.make_default)

    print(f"Validation accuracy: {validation_accuracy:.6f}")
    print(f"Exported predictions: {predictions_path.relative_to(REPO_ROOT)}")
    print(f"Exported importance: {importance_path.relative_to(REPO_ROOT)}")
    if not args.skip_model_save:
        print(f"Saved model: {args.model_output.relative_to(REPO_ROOT)}")
    if args.update_manifest:
        print(f"Updated manifest: {args.manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()