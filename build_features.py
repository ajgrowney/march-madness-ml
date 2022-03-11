import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from objects import TeamSeason, Team_Historical
from utilities import DATA_ROOT
raw_data_dir = f"{DATA_ROOT}/Stage2/"
season_res = pd.read_csv(raw_data_dir+"MRegularSeasonCompactResults.csv")
tourney_res = pd.read_csv(raw_data_dir+"MNCAATourneyCompactResults.csv")


years = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
tourney_games = pd.read_csv(raw_data_dir+"MNCAATourneyDetailedResults.csv")
season_games = season_res[season_res["Season"].isin(years)]
season_teams = set(season_games[["WTeamID", "LTeamID"]].values.flatten())


# Initialization
df_columns, df_rows = None, []
teams_data = {}
team_season_by_year = {}
for yr in years:
    teams_data[yr] = []
    team_season_by_year[yr] = {}

historical_teams = {}

for tm in season_teams:
    team = Team_Historical(tm, years)  
    if df_columns == None: 
        df_columns = ['TeamID'] + team.get_data_columns(list(team.valid_years)[0])
    for y in team.valid_years:
        team_season_by_year[y][tm] = team.team_seasons[y]

    historical_teams[tm] = team


for team in historical_teams.values():
    for y in team.valid_years:
        team.team_seasons[y].calculate_post_season_stats(team_season_by_year[y])
    # Fill rows of DataFrame
    for year in team.valid_years:
        teams_data[year].append(np.append(team.id, team.get_season_data(year)))




for year in years:
    df = pd.DataFrame(teams_data[year], columns=df_columns)
    df.to_csv('./Data/Training/features_{:d}.csv'.format(year), index=False)
