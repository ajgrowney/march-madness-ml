import pickle
import pandas as pd
import numpy as np

year = 2018
df = pd.read_csv("phase1_{:d}.csv".format(year))
teams_df = pd.read_csv('Data/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])

def get_matchup_data(team1, team2):

    team1Data = np.delete(np.array(df[df['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(df[df['TeamID'] == float(team2)]), 0)
    return team1Data, team2Data

# Param: t1 { int } - Team 1's TeamID
# Param: t2 { int } - Team 2's TeamID
# Return: { Numpy Array }
def make_prediction(t1: int, t2: int):
        t1_data, t2_data = get_matchup_data(t1, t2)
        make_pred = np.append(t1_data, t2_data).reshape(1,-1)
        return model.predict_proba(make_pred)

model = pickle.load(open('model.sav', 'rb'))
done=False
while(not done):
    team1 = input("Team 1: ")
    team2 = input("Team 2: ")
    try:
        t1_id = teams_df.loc[teams_df['TeamName'] == team1]['TeamID'].values[0]
        t2_id = teams_df.loc[teams_df['TeamName'] == team2]['TeamID'].values[0]
        print(make_prediction(t1_id, t2_id))
    
    except IndexError as er:
        print("Invalid Team Selection")
    
    done=(input("Continue (y/n): ") == "n")