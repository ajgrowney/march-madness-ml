import numpy as np
import pandas as pd
from mm_analytics.objects import TeamSeason, TeamHistorical
DATA_ROOT = "/Users/andrewgrowney/Data/kaggle/marchmadness-2021"
raw_data_dir = f"{DATA_ROOT}/Stage2/"
season_res = pd.read_csv(raw_data_dir+"MRegularSeasonCompactResults.csv")
tourney_res = pd.read_csv(raw_data_dir+"MNCAATourneyCompactResults.csv")

years = [2022]
# years = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
tourney_games = pd.read_csv(raw_data_dir+"MNCAATourneyDetailedResults.csv")
print(tourney_games.describe())
season_games = season_res[season_res["Season"].isin(years)]
season_teams = set(season_games[["WTeamID", "LTeamID"]].values.flatten())
print(season_teams)

# Initialization
df_columns, df_rows = None, []
teams_data = {}
team_season_by_year = {}
for yr in years:
    teams_data[yr] = []
    team_season_by_year[yr] = {}

historical_teams = {}

for tm in season_teams:
    team = TeamHistorical(tm, years)  
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
    print(teams_data[year])
    df = pd.DataFrame(teams_data[year], columns=df_columns)
    df.to_csv(f'{DATA_ROOT}/Training/features_{year}.csv', index=False)
