import sys
import pandas as pd
import numpy as np
from Teams import Team_Historical, TeamSeason

years = [2015,2016,2017,2018,2019]
tourney_games = pd.read_csv("Data/MNCAATourneyDetailedResults.csv")
tourney_games = tourney_games[tourney_games["Season"].isin(years)]
tourney_teams = set(tourney_games[["WTeamID", "LTeamID"]].values.flatten())


df_columns, df_rows = None, []
teams_data = {}
for yr in years:
    teams_data[yr] = []

for tm in tourney_teams:
    team = Team_Historical(tm, years)  
    if df_columns == None: df_columns = ['TeamID'] + team.get_data_columns(years[0])
    # Fill rows of DataFrame
    for year in years:
        teams_data[year].append(np.append(team.id, team.get_season_data(year)))


for year in years:
    df = pd.DataFrame(teams_data[year], columns=df_columns)
    df.to_csv('./TrainingData/phase1_{:d}.csv'.format(year), index=False)
