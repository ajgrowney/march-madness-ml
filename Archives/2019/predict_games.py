import pickle
import sys
import pandas as pd
import numpy as np
import json
year = 2018 if len(sys.argv) < 2 else int(sys.argv[1])
df = pd.read_csv("./Data/Training/features_{:d}.csv".format(year))
teams_df = pd.read_csv('Data/Raw/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
model = pickle.load(open('Results/model.sav', 'rb'))

def get_matchup_data(team1, team2, dataframe):
    team1Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team2)]), 0)
    dataDifference = [(a-b) for a, b in zip(team1Data, team2Data)]
    return [dataDifference]

# Param: t1 { int } - Team 1's TeamID
# Param: t2 { int } - Team 2's TeamID
# Return: { Numpy Array }
def make_prediction(t1: int, t2: int):
    matchup = get_matchup_data(t1, t2, df)

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
            tms = {t1_id: team1, t2_id: team2}
            w_id, prob = make_prediction(t1_id, t2_id)
            print(tms[w_id], prob)
        except IndexError as er:
            print("Invalid Team Selection")
        
        done=(input("Continue (y/n): ") == "n")

def calculate_tourney_percentage(mm_df):
    misses = []
    for i, wt_id, lt_id in mm_df.itertuples():
        pred_id, pred_prob = make_prediction(wt_id, lt_id)
        if(pred_id != wt_id):
            wt_name = teams_df.loc[teams_df['TeamID'] == wt_id]['TeamName'].values[0]
            lt_name = teams_df.loc[teams_df['TeamID'] == lt_id]['TeamName'].values[0]
            misses.append([str(wt_name),str(lt_name)])

    return misses

if "userin" in sys.argv:
    user_in()
else:
    ncaa_tourney_games = pd.read_csv("./Data/Raw/MNCAATourneyCompactResults.csv")
    ncaa_tourney_games = ncaa_tourney_games[ncaa_tourney_games["Season"] == year]
    ncaa_tourney_games = ncaa_tourney_games[["WTeamID", "LTeamID"]]
    misses = calculate_tourney_percentage(ncaa_tourney_games)
    print(len(misses))
    output = { year: misses }
    print(json.dumps(output))