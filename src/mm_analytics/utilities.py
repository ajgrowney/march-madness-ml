from typing import List, Tuple, Dict
import json
import uuid
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

DATA_ROOT = os.getenv("MM_DATA_ROOT")
MODELS_ROOT = os.getenv("MM_MODELS_ROOT")
SUBMISSIONS_ROOT = os.getenv("MM_SUBMISSIONS_ROOT", "./Results/2023")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

ROUND_DAYS = {
    134: "Play In",
    135: "Play In",
    136: "First Round",
    137: "First Round",
    138: "Second Round",
    139: "Second Round",
    143: "Sweet Sixteen",
    144: "Sweet Sixteen",
    145: "Elite Eight",
    146: "Elite Eight",
    152: "Final Four",
    154: "Championship"
}

# ---- Similarity Metrics ----
def get_historical_resume_stats_similarity(df, num_teams:int = 3) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """Using resume and stats data
    calculate the similarity between teams
    and return the top n most similar teams
    :param df { DataFrame }: The dataframe containing the team data
    :param num_teams { int }: The number of similar teams to return
    :return { Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] }: A dictionary of (team id, season)
        mapped to and their most similar (team id, season) with similarity score
    """
    # Build Resume Similarity
    resume_cols = ["Q1_WinPct", "Q2_WinPct", "Q3_WinPct", "Q4_WinPct", "SOS", "SOV", "NET_last"]
    resume_df = df[resume_cols]
    resume_df.fillna(0, inplace=True)
    resume_scaler = MinMaxScaler()
    resume_df = resume_scaler.fit_transform(resume_df)
    resume_similarity = cosine_similarity(resume_df)

    # Build Stat Similarity
    stat_cols = ["AdjOE_mean", "AdjDE_mean", "Poss_mean", "FGA3_mean", "FTA_mean", "FG%_mean","FG3%_mean","FT%_mean","Ast_mean","TO_mean","OR_mean", "OppFGA3_mean", "OppFTA_mean", "OppFG%_mean","OppFG3%_mean","OppFT%_mean","OppAst_mean","OppTO_mean","OppOR_mean"]
    stat_df = df[stat_cols]
    stat_scaler = MinMaxScaler()
    stat_df = stat_scaler.fit_transform(stat_df)
    stat_similarity = cosine_similarity(stat_df)

    # Average the two similarities
    avg_similarity = (resume_similarity + stat_similarity) / 2

    similar_teams = {}
    for i in range(len(avg_similarity)):
        avg_similarity[i][i] = 0

        arr = avg_similarity[i]
        # Get the indices of the most similar n values
        top_n_indices = arr.argsort()[-num_teams:][::-1]

        # Set the most similar teams
        similar_teams[df.iloc[i]['TeamID']] = [
            (df.iloc[j]['TeamID'], avg_similarity[i][j]) for j in top_n_indices]
    return similar_teams

def evaluate_model_on_tournament(model, scaler, year, data_version, teams_df, tourney_df):
    correct, incorrect = [], []

    # Load auxillary data
    features = fetch_features(year, data_version)
    tourney_df = tourney_df[tourney_df["Season"] == year]
    tourney_df = tourney_df[["WTeamID", "LTeamID"]]
    for i, wt_id, lt_id in tourney_df.itertuples():
        wt_name = teams_df.loc[teams_df['TeamID'] == wt_id]['TeamName'].values[0]
        lt_name = teams_df.loc[teams_df['TeamID'] == lt_id]['TeamName'].values[0]
        pred_id, pred_prob = make_prediction(model, scaler, wt_id, lt_id, features)
        if(pred_id != wt_id):
            incorrect.append(f"{wt_name} vs {lt_name}")
        else:
            correct.append(f"{wt_name} vs {lt_name}")
    return {
        "correct": correct, "incorrect": incorrect, 
        "score": (len(correct) / (len(correct) + len(incorrect)))
    }


def fetch_training_data(years = list(range(2003,2020))+[2021], version:str = "Stage2", format:str = "diff", inclued_reg_season = False):
    """
    :param format { str }: diff, stacked
    """
    tourney_game_start_index = 0
    dfs = {}
    for year in years:
        dfs[year] = pd.read_csv(f"{DATA_ROOT}/Training/{version}/features_{year}.csv")
    feature_list = list(list(dfs.values())[0].columns.values)[1:]

    if inclued_reg_season:
        reg_season_games = pd.read_csv(f"{DATA_ROOT}/Stage2/MNCAATourneyDetailedResults.csv")
        reg_season_games_years = reg_season_games[reg_season_games["Season"].isin(years)]
    
    tourney_games = pd.read_csv(f"{DATA_ROOT}/Stage2/MNCAATourneyDetailedResults.csv")
    tourney_games_years = tourney_games[tourney_games["Season"].isin(years)]


    X, Y = [], []

    print("Fill Training and Testing Data")
    if inclued_reg_season:
        for game in reg_season_games_years.itertuples():
            season = game[1]
            winner, loser = game[3], game[5]
            t1_win_data = get_matchup_data(loser, winner, dfs[season], format)
            t2_win_data = get_matchup_data(winner, loser, dfs[season], format)
            X.append(t1_win_data), Y.append(1)
            X.append(t2_win_data), Y.append(0)
            tourney_game_start_index += 1
    
    for tournament_game in tourney_games_years.itertuples():
        season = tournament_game[1]
        winner, loser = tournament_game[3], tournament_game[5]
        t1_win_data = get_matchup_data(loser, winner, dfs[season], format)
        t2_win_data = get_matchup_data(winner, loser, dfs[season], format)
        X.append(t1_win_data), Y.append(1)
        X.append(t2_win_data), Y.append(0)
    return (X,Y, feature_list, tourney_game_start_index)

def fetch_features(year, version = None):
    train_data_dir = "Training"
    if version is not None:
        train_data_dir += f"/{version}"
    df = pd.read_csv(f"{DATA_ROOT}/{train_data_dir}/features_{year}.csv")
    return df

def find_team_id(name:str):
    teams_df = pd.read_csv(f"{DATA_ROOT}/Stage2/MTeams.csv")
    team_id = teams_df[teams_df["TeamName"] == name]["TeamID"].values[0]
    return team_id

def fill_submission(model, scaler, model_id, submission_id = str(uuid.uuid4())[0:8], matchup_format:str = "diff", stage="2", data_dir = "Training", model_type = "sav"):
    stage_folder = "Stage2" if stage == "1" else "Stage2"
    submission_template = pd.read_csv(f"{DATA_ROOT}/{stage_folder}/SampleSubmission2023.csv")
    print(submission_template)
    feature_dfs = {}
    submission_df = pd.DataFrame(columns=["Id","Pred"])
    for i, id, _ in submission_template.itertuples():
        year, t1, t2 = id.split("_")
        year, t1, t2 = int(year), int(t1), int(t2)

        if year not in feature_dfs.keys():
            print(year)
            feature_dfs[year] = pd.read_csv(f"{DATA_ROOT}/{data_dir}/features_{year}.csv")

        matchup = get_matchup_data(t1, t2, feature_dfs[year], matchup_format)
        matchup = np.array(matchup).reshape(1,-1)
        print(matchup)
        if scaler is not None:
            matchup = scaler.transform(matchup)
        try:
            if model_type == "sav":
                prediction = model.predict_proba(matchup).flatten()
            else:
                prediction = model.predict(matchup).flatten()
        except ValueError:
            prediction = (0.5, 0.5)
        submission_df = submission_df.append({'Id': id, 'Pred': prediction[0]}, ignore_index=True)
    
    submission_root = f"{SUBMISSIONS_ROOT}/{submission_id}"
    if not os.path.exists(submission_root):
        os.makedirs(submission_root)
    submission_df.to_csv(f"{submission_root}/{model_id}_{stage}.csv", index=False)

def get_matchup_data(team1, team2, feature_df, format:str = "diff"):
    team1Data = np.delete(np.array(feature_df[feature_df['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(feature_df[feature_df['TeamID'] == float(team2)]), 0)
    results = []

    if format == "diff":
        for a,b in zip(team1Data,team2Data):
            diff = a-b
            # if diff > 0:
            #     result = diff**2
            # else:
            #     result = (-1)*(diff**2)
            results.append(diff)
    elif format == "stack":
        results = list(team1Data) + list(team2Data)
    else:
        raise Exception(f"Invalid get_matchup_data format: {format}")
    return results

def get_scaler(run_id:str, models_folder:str = MODELS_ROOT, scaler_name:str = "scaler.pkl"):
    """Attempt to load the scaler that this model was trained with
    """
    scaler_path = os.path.join(models_folder, run_id, "scaler.pkl")

    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
    else:
        scaler = None
    return scaler

def make_prediction(model, scaler, t1: int, t2: int, feature_set_df):
    matchup = get_matchup_data(t1, t2, feature_set_df)
    matchup = np.array(matchup).reshape(1,-1)
    if scaler is not None:
        matchup = scaler.transform(matchup)
    prediction = model.predict_proba(matchup).flatten()
    winner = t1 if (prediction[0] > prediction[1]) else t2
    return winner, max(prediction[0], prediction[1])
    
def f_importances(coef, names):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "search":
        print(find_team_id(sys.argv[2]))