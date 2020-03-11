import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

years = [2015,2016,2017,2018,2019]
dfs = {}

for year in years:
    dfs[year] = pd.read_csv("TrainingData/phase1_{:d}.csv".format(year))

df = pd.concat(dfs, axis=0)
tourney_games = pd.read_csv("Data/MNCAATourneyDetailedResults.csv")
tourney_games_years = tourney_games[tourney_games["Season"].isin(years)]


def get_matchup_data(team1, team2, dataframe):
    team1Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(dataframe[dataframe['TeamID'] == float(team2)]), 0)
    dataDifference = [(a-b) for a, b in zip(team1Data, team2Data)]
    return dataDifference

X, Y = [], []
print("Fill Training and Testing Data")

for tournament_game in tourney_games_years.itertuples():
    season = tournament_game[1]
    winner, loser = tournament_game[3], tournament_game[5]
    X.append(get_matchup_data(winner,loser, dfs[season])), Y.append(0)
    X.append(get_matchup_data(loser, winner, dfs[season])), Y.append(1)

print("Training and Testing Data Finished")
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.8)

model = SVC(kernel="linear", probability=True)
print("Fitting model")
model.fit(x_train, y_train)
print("Model fit")
print("Score: ", model.score(x_test, y_test))

if len(sys.argv) > 1 and sys.argv[1] == "save": pickle.dump(model, open('model.sav', 'wb'))
