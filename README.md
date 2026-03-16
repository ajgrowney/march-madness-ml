# March Madness ML Prediction
Year by year progression of my attempt to let data try to beat my fanaticism for NCAA basketball <br />
<i>Hint: haven't beat it yet</i>

## 2026

- Goal: interactive website for model building and hosting

## 2025

2025 shifted away from the older CLI submission workflow and into a notebook-driven export pipeline centered on precomputed tournament predictions.

### Model Types
1. `clf_2025.pkl`
    - Primary classifier used for the default 2025 tournament predictions.
    - RandomForestClassifier with 100 trees and random_state = 42
    - Loaded with `scaler_2025.pkl` and used to populate the main `slots` and `predictions` fields in the exported tournament JSON.

2. `clf_2025_v2.pkl`
    - Second classifier variant used as an alternative prediction branch: Feed forward neural net with 2 hidden layers of 100 units each. 1000 max iters, alpha = 0.001, solver = adam
    - Loaded with `scaler_2025_v2.pkl` and exported alongside the default classifier as `slots_clf_v2` and `predictions_clf_v2`.

3. `nn_model_v2.h5`
    - Neural-network alternative to the classifier-based pipelines.
    - Exported as `slots_neural` and `predictions_neural` for side-by-side comparison against the classifier outputs.

### Data Pipeline
1. Team season features
    - The 2025 workflow reads `TeamSeasons_2025.csv` as the base feature table for tournament teams.
    - The feature set is a compact team-quality vector built from offensive, defensive, and schedule metrics:
        - offense: `AdjOE_mean`, `EFG%_mean`, `FGA3_mean`, `TO_mean`, `OR_mean`, `FT%_mean`
        - defense: `AdjDE_mean`, `OppEFG%_mean`, `OppFGA3_mean`, `OppTO_mean`, `OppOR_mean`
        - context: `AdjNE_mean`, `Poss_mean`, `SOS`, `Q1_WinPct`, `Q2_WinPct`

2. Matchup construction
    - Predictions are made from pairwise matchup vectors rather than raw team rows.
    - For each matchup, the pipeline subtracts Team B's feature vector from Team A's feature vector, then also evaluates the reverse ordering to reduce directional bias.

3. Scaling and inference
    - Both classifier pipelines use `MinMaxScaler` artifacts saved beside the models.
    - The scikit-learn classifiers call `predict_proba` on both matchup directions, then average the mirrored probabilities into a final win/loss estimate.
    - The neural model runs both matchup directions through the Keras model and averages the forward probability with the reverse complement.

4. Tournament export
    - The notebook `Notebooks/export_tourney.ipynb` builds the full 2025 bracket tree, including play-in handling and round-by-round slot probabilities.
    - It exports a combined website payload to `data/web/tourney_v4/2025.json`.
    - It also writes standalone prediction maps:
        - `Notebooks/clf_2025.json`
        - `Notebooks/nn_2025.json`
        - `Notebooks/base_2025.json`

### Notes
- The default 2025 bracket payload is classifier-first: the unlabeled `slots` and `predictions` entries come from `clf_2025.pkl`.
- The neural and `clf_v2` branches were preserved as alternate views, not replacements for the primary export.

## 2024

- Data Cleaning
    - 2021 tournamnet: manually adjust dates to match other tournaments

## 2022
Currently building models based on last tournaments. This year I built a CLI around the model training, exploring, and submission process.
Submitted grid_xgb_2 to kaggle

### ML Pipeline
1) Data Collection
    - Using exclusively Kaggle provided data so simple download of the CSVs into a Data folder

2) Exploratory Data Analysis
    - Used my `eda.py` file rather dynamically to explore different pieces of the data

3) Feature Extraction
    - After doing some visual analysis of the data set we were working with, used the `objects.py` file to structure the feature based representation of a team's performance for a single season. Using that class, I dumped the data into a simple vector and used that as the feature for that team. The `build_features.py` file was used to save those to a training data folder to cache the feature sets.

4) Model Training - train
    - `python cli.py train --models basic_svc,grid_xgb --save`
        - Outputs a run id that the models are saved to

5) Model Evaluation - evalute
    - `python cli.py evaluate --run 12341234 --models basic_svc`

6) Kaggle Submission - submit

## 2021
[Archives for 2021](Archives/2021/README.md)

## 2020 
Covid...

## 2019
[Archives for 2019](Archives/2019/README.md)
