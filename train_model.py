import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

year = 2018
df = pd.read_csv("phase1_{:d}.csv".format(year))
tourney_games = pd.read_csv("Data/MNCAATourneyDetailedResults.csv")
tourney_games_year = tourney_games[tourney_games["Season"] == year]

model_data = []
X, Y = [], []
for tournament_game in tourney_games_year.itertuples():
    winner, loser = tournament_game[3], tournament_game[5]
    winnerData = np.delete(np.array(df[df['TeamID'] == float(winner)]), 0)
    loserData = np.delete(np.array(df[df['TeamID'] == float(loser)]), 0)
    datarow_1 = list(np.append(winnerData, loserData))
    datarow_2 = list(np.append(loserData, winnerData))
    X.append(datarow_1), Y.append(0)
    X.append(datarow_2), Y.append(1)

(x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.9)

model = SVC(kernel="linear")
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
pickle.dump(model, open('model.sav', 'wb'))