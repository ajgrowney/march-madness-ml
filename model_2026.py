"""
Build out 
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BROWSER_FEATURE_STORE = REPO_ROOT / "data/web/features/2026/stat_explorer_v1.json"
DEFAULT_TRAINING_MANIFEST = REPO_ROOT / "data/web/training/manifest.json"
DEFAULT_TEAMS_INDEX = REPO_ROOT / "data/web/index/2026/teams.json"
DEFAULT_BRACKET = REPO_ROOT / "data/web/brackets/2026.json"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "data/web/models/predictions"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data/web/models/manifest.json"
DEFAULT_MODEL_OUTPUT = REPO_ROOT / "Models/rf_2026_browser.pkl"

# Prototype alias: the historical export uses a normalized ordinal slot that is
# semantically closest to the browser's current NET_last feature.
TRAINING_FEATURE_ALIASES = {
    "NET_last": "selection_ordinal_last",
}


@dataclass(frozen=True)
class TrainingDataset:
    feature_order: List[str]
    training_feature_order: List[str]
    seasons: List[int]
    x_rows: List[List[float]]
    y_rows: List[int]
    sample_weights: List[float]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_training_seasons(raw_value: str | None, available_seasons: Iterable[int], target_season: int) -> List[int]:
    if raw_value:
        seasons = [int(token.strip()) for token in raw_value.split(",") if token.strip()]
    else:
        seasons = sorted(season for season in available_seasons if season < target_season)
    if not seasons:
        raise ValueError("No training seasons selected.")
    return seasons


def resolve_training_feature_order(browser_feature_order: Sequence[str], training_feature_order: Sequence[str]) -> List[str]:
    resolved: List[str] = []
    missing: List[str] = []
    training_feature_set = set(training_feature_order)

    for browser_feature in browser_feature_order:
        if browser_feature in training_feature_set:
            resolved.append(browser_feature)
            continue
        alias = TRAINING_FEATURE_ALIASES.get(browser_feature)
        if alias and alias in training_feature_set:
            resolved.append(alias)
            continue
        missing.append(browser_feature)

    if missing:
        raise ValueError(
            "Training feature contract is missing browser features: " + ", ".join(missing)
        )

    return resolved


def load_training_dataset(
    manifest_path: Path,
    browser_feature_order: Sequence[str],
    seasons: Sequence[int],
    regular_season_weight: float,
    tournament_weight: float,
) -> TrainingDataset:
    manifest = load_json(manifest_path)
    training_feature_order = manifest["feature_order"]
    resolved_training_order = resolve_training_feature_order(browser_feature_order, training_feature_order)
    feature_index = {feature_name: index for index, feature_name in enumerate(training_feature_order)}

    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    sample_weights: List[float] = []

    for season in seasons:
        training_file = resolve_repo_path(manifest["training_files"][str(season)])
        season_payload = load_json(training_file)
        for example in season_payload["examples"]:
            vector = example["x"]
            x_rows.append([float(vector[feature_index[name]]) for name in resolved_training_order])
            y_rows.append(int(example["y"]))
            sample_weights.append(
                tournament_weight if example["source"] == "tournament" else regular_season_weight
            )

    if not x_rows:
        raise ValueError("No training rows were loaded from the historical export manifest.")

    return TrainingDataset(
        feature_order=list(browser_feature_order),
        training_feature_order=resolved_training_order,
        seasons=list(seasons),
        x_rows=x_rows,
        y_rows=y_rows,
        sample_weights=sample_weights,
    )


def get_probability_by_class(model: RandomForestClassifier, class_probabilities: Sequence[float], label: int) -> float:
    class_index = list(model.classes_).index(label)
    return float(class_probabilities[class_index])


def build_matchup_vector(team_a_features: Dict[str, float], team_b_features: Dict[str, float], feature_order: Sequence[str]) -> List[float]:
    return [round(float(team_a_features[name]) - float(team_b_features[name]), 6) for name in feature_order]


def build_bracket_field_team_ids(bracket_payload: dict) -> List[int]:
    field_ids = set()
    for slot in bracket_payload["slots"]:
        for side in ("team_1", "team_2"):
            entry = slot.get(side, {})
            team_id = entry.get("team_id")
            if team_id is not None:
                field_ids.add(int(team_id))
    return sorted(field_ids)


def load_field_team_ids(teams_index_path: Path, bracket_path: Path, feature_store_payload: dict) -> List[int]:
    teams_index = load_json(teams_index_path)
    bracket_payload = load_json(bracket_path)
    bracket_team_ids = build_bracket_field_team_ids(bracket_payload)
    index_team_ids = sorted(int(team["team_id"]) for team in teams_index["teams"] if team["tournament_team"])
    feature_store_team_ids = sorted(
        int(team_id)
        for team_id, team_values in feature_store_payload["teams"].items()
        if float(team_values["Seed"]) != 17.0
    )

    if not bracket_team_ids:
        raise ValueError("No tournament teams were found in the published bracket artifact.")

    if index_team_ids != bracket_team_ids:
        missing_in_predictions = sorted(set(bracket_team_ids) - set(index_team_ids))
        extra_in_predictions = sorted(set(index_team_ids) - set(bracket_team_ids))
        raise ValueError(
            "Team index tournament field does not match published bracket field: "
            f"missing_in_index={missing_in_predictions}, extra_in_index={extra_in_predictions}"
        )

    if feature_store_team_ids != bracket_team_ids:
        missing_in_features = sorted(set(bracket_team_ids) - set(feature_store_team_ids))
        extra_in_features = sorted(set(feature_store_team_ids) - set(bracket_team_ids))
        raise ValueError(
            "Feature store seeded teams do not match published bracket field: "
            f"missing_in_feature_store={missing_in_features}, extra_in_feature_store={extra_in_features}"
        )

    return bracket_team_ids


def build_feature_importance_payload(feature_order: Sequence[str], importances: Sequence[float]) -> List[dict]:
    ranked = sorted(
        zip(feature_order, importances),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "featureKey": feature_name,
            "importance": round(float(importance), 6),
            "rank": rank,
        }
        for rank, (feature_name, importance) in enumerate(ranked, start=1)
    ]


def build_model_urls(base_url: str | None, *relative_paths: Path) -> List[str | None]:
    urls: List[str | None] = []
    for relative_path in relative_paths:
        repo_relative = relative_path.relative_to(REPO_ROOT).as_posix()
        if base_url:
            urls.append(base_url.rstrip("/") + "/" + repo_relative)
        else:
            urls.append(repo_relative)
    return urls


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
) -> dict:
    return {
        "id": model_id,
        "name": model_name,
        "type": "precomputed-predictions",
        "model_family": "random_forest",
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
    }


def build_predictions_payload(
    model: RandomForestClassifier,
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

            vector_ab = build_matchup_vector(team_a_features, team_b_features, feature_order)
            vector_ba = build_matchup_vector(team_b_features, team_a_features, feature_order)

            probs_ab = model.predict_proba([vector_ab])[0]
            probs_ba = model.predict_proba([vector_ba])[0]

            team_a_prob = (
                get_probability_by_class(model, probs_ab, 0)
                + get_probability_by_class(model, probs_ba, 1)
            ) / 2.0
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
        description="Prototype RandomForest trainer and exporter for 2026 browser-compatible precomputed predictions."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model-id", default="rf-2026-v1")
    parser.add_argument("--model-name", default="Andrew RandomForest 2026")
    parser.add_argument(
        "--description",
        default="RandomForest model trained offline and exported as pairwise 2026 tournament probabilities.",
    )
    parser.add_argument("--browser-feature-store", type=Path, default=DEFAULT_BROWSER_FEATURE_STORE)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--teams-index", type=Path, default=DEFAULT_TEAMS_INDEX)
    parser.add_argument("--bracket-path", type=Path, default=DEFAULT_BRACKET)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--training-seasons", default=None)
    parser.add_argument("--regular-season-weight", type=float, default=1.0)
    parser.add_argument("--tournament-weight", type=float, default=3.0)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--artifact-base-url", default=None)
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--make-default", action="store_true")
    parser.add_argument("--skip-model-save", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    browser_feature_store = load_json(args.browser_feature_store)
    browser_feature_order = browser_feature_store["feature_order"]
    feature_set_name = browser_feature_store["feature_set"]

    training_manifest = load_json(args.training_manifest)
    available_seasons = [int(season) for season in training_manifest["seasons"]]
    training_seasons = parse_training_seasons(args.training_seasons, available_seasons, args.season)

    dataset = load_training_dataset(
        args.training_manifest,
        browser_feature_order=browser_feature_order,
        seasons=training_seasons,
        regular_season_weight=args.regular_season_weight,
        tournament_weight=args.tournament_weight,
    )
    print("[debug]loaded dataset")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
        oob_score=True,
    )
    model.fit(dataset.x_rows, dataset.y_rows, sample_weight=dataset.sample_weights)
    print("[debug]fit model")

    field_team_ids = load_field_team_ids(
        args.teams_index,
        args.bracket_path,
        browser_feature_store,
    )
    if not field_team_ids:
        raise ValueError("No 2026 tournament teams were found in the team index or bracket artifact.")

    predictions_path = args.predictions_dir / f"{args.model_id}_{args.season}.json"
    importance_path = args.predictions_dir / f"{args.model_id}_importance.json"
    predictions_url, importance_url = build_model_urls(
        args.artifact_base_url,
        predictions_path,
        importance_path,
    )
    print("[debug]built urls")

    feature_importance = build_feature_importance_payload(
        browser_feature_order,
        model.feature_importances_,
    )
    print("[debug]built feature importance payload")
    model_metadata = build_model_metadata(
        model_id=args.model_id,
        model_name=args.model_name,
        season=args.season,
        training_seasons=training_seasons,
        feature_set=feature_set_name,
        predictions_url=predictions_url,
        feature_importance_url=importance_url,
        description=args.description,
    )
    model_metadata["training_feature_aliases"] = TRAINING_FEATURE_ALIASES
    model_metadata["oob_score"] = round(float(model.oob_score_), 6)

    predictions_payload = build_predictions_payload(
        model,
        feature_store_payload=browser_feature_store,
        team_ids=field_team_ids,
        model_metadata=model_metadata,
        feature_importance=feature_importance,
    )
    print("[debug]built feature predictions payload")
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
                    "feature_order": browser_feature_order,
                    "feature_set": feature_set_name,
                    "training_seasons": training_seasons,
                    "training_feature_order": dataset.training_feature_order,
                    "training_feature_aliases": TRAINING_FEATURE_ALIASES,
                },
                handle,
            )

    if args.update_manifest:
        update_manifest(args.manifest_path, model_metadata, make_default=args.make_default)

    print(f"Trained RandomForest on {len(dataset.x_rows)} mirrored matchup rows across seasons {training_seasons}.")
    print(f"Browser feature set: {feature_set_name} ({', '.join(browser_feature_order)})")
    print(f"Exported predictions: {predictions_path.relative_to(REPO_ROOT)}")
    print(f"Exported importance: {importance_path.relative_to(REPO_ROOT)}")
    if not args.skip_model_save:
        print(f"Saved model: {args.model_output.relative_to(REPO_ROOT)}")
    if args.update_manifest:
        print(f"Updated manifest: {args.manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()