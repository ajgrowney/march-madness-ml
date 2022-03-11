import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

DATA_ROOT = os.getenv("MM_DATA_ROOT")
MODELS_ROOT = os.getenv("MM_MODELS_ROOT")

def find_team_id(name:str):
    teams_df = pd.read_csv("./Data/Raw/MTeams.csv")
    team_id = teams_df[teams_df["TeamName"] == name]["TeamID"].values[0]
    return team_id


def fetch_training_data(years = range(2003,2020)):
    dfs = {}
    for year in years:
        dfs[year] = pd.read_csv(f"{DATA_ROOT}/Training/features_{year}.csv")
    feature_list = list(list(dfs.values())[0].columns.values)[1:]

    tourney_games = pd.read_csv(f"{DATA_ROOT}/Raw/MNCAATourneyDetailedResults.csv")
    tourney_games_years = tourney_games[tourney_games["Season"].isin(years)]

    X, Y = [], []

    print("Fill Training and Testing Data")

    for tournament_game in tourney_games_years.itertuples():
        season = tournament_game[1]
        winner, loser = tournament_game[3], tournament_game[5]
        X.append(get_matchup_data(winner, loser, dfs[season])), Y.append(0)
        X.append(get_matchup_data(loser, winner, dfs[season])), Y.append(1)
    return (X,Y, feature_list)

def fetch_features(year):
    df = pd.read_csv(f"{DATA_ROOT}/Training/features_{year}.csv")
    return df

def get_matchup_data(team1, team2, dataframe):
    team1Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team2)]), 0)
    dataDifference = []

    for a,b in zip(team1Data,team2Data):
        diff = a-b
        # if diff > 0:
        #     result = diff**2
        # else:
        #     result = (-1)*(diff**2)
        dataDifference.append(diff)
    return dataDifference

def make_prediction(model, scaler, t1: int, t2: int, feature_set_df):
    matchup = get_matchup_data(t1, t2, feature_set_df)
    matchup = np.array(matchup).reshape(1,-1)
    if scaler is not None:
        matchup = scaler.transform(matchup)
    prediction = model.predict_proba(matchup).flatten()
    winner = t1 if (prediction[0] > prediction[1]) else t2
    return winner, max(prediction[0], prediction[1])
    
def evaluate_model_on_tournament(model, scaler, year):
    correct, incorrect = [], []

    # Load auxillary data
    features = fetch_features(year)
    teams_df = pd.read_csv(f"{DATA_ROOT}/Raw/MTeams.csv")

    tourney_df = pd.read_csv(f"{DATA_ROOT}/Raw/MNCAATourneyCompactResults.csv")
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

def f_importances(coef, names):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "search":
        find_team_id(sys.argv[2])