import os
import pickle
import sys
import pandas as pd
import numpy as np
import json
from utilities import get_matchup_data
DATA_ROOT = "/Users/andrewgrowney/Data/kaggle/marchmadness-2021"
MODELS_ROOT = os.getenv("MM_MODELS_ROOT")
MODEL_SELECTED = "poly_model" if len(sys.argv) < 2 else sys.argv[1]
year = 2022 if len(sys.argv) < 3 else int(sys.argv[2])
features_df = pd.read_csv(f"{DATA_ROOT}/Training/V1/features_{year}.csv")
teams_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
model = pickle.load(open(f'Models/{MODEL_SELECTED}.sav', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', "rb"))

# Param: t1 { int } - Team 1's TeamID
# Param: t2 { int } - Team 2's TeamID
# Return: { Numpy Array }
def make_prediction(t1: int, t2: int, feature_set_df):
    matchup = get_matchup_data(t1, t2, feature_set_df)
    matchup = np.array(matchup).reshape(1,-1)
    matchup = scaler.transform(matchup)
    prediction = model.predict_proba(matchup).flatten()
    winner = t1 if (prediction[0] > prediction[1]) else t2
    return winner, max(prediction[0], prediction[1])
    



def user_in():
    done=False
    while(not done):
        team1 = input("Team 1: ")
        team2 = input("Team 2: ")
        try:
            t1_id = teams_df.loc[teams_df['TeamName'] == team1]['TeamID'].values[0]
            t2_id = teams_df.loc[teams_df['TeamName'] == team2]['TeamID'].values[0]
            w_id, prob = make_prediction(t1_id, t2_id,features_df)
            print(w_id,prob)
        except IndexError as er:
            print("Invalid Team Selection")
        
        done=(input("Continue (y/n): ") == "n")

def calculate_tourney_percentage(mm_df, features):
    misses = []
    for i, wt_id, lt_id in mm_df.itertuples():
        pred_id, pred_prob = make_prediction(wt_id, lt_id, features)
        if(pred_id != wt_id):
            wt_name = teams_df.loc[teams_df['TeamID'] == wt_id]['TeamName'].values[0]
            lt_name = teams_df.loc[teams_df['TeamID'] == lt_id]['TeamName'].values[0]
            misses.append([str(wt_name),str(lt_name)])

    return misses

if "userin" in sys.argv:
    user_in()
else:
    ncaa_tourney_games = pd.read_csv(f"{DATA_ROOT}/Stage2/MNCAATourneyCompactResults.csv")
    ncaa_tourney_games = ncaa_tourney_games[ncaa_tourney_games["Season"] == year]
    ncaa_tourney_games = ncaa_tourney_games[["WTeamID", "LTeamID"]]
    misses = calculate_tourney_percentage(ncaa_tourney_games, features_df)
    print(len(misses))
    output = { year: misses }
    print(json.dumps(output))