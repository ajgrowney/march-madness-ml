import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def find_team_id(name:str):
    teams_df = pd.read_csv("./Data/Raw/MTeams.csv")
    print(teams_df[teams_df["TeamName"] == name]["TeamID"].values[0])

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


def f_importances(coef, names):
    imp = coef[0]
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "search":
        find_team_id(sys.argv[2])