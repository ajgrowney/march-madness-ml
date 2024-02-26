from typing import Dict
import numpy as np
import pandas as pd
from mm_analytics.objects import get_team_seasons, calculate_season_rankings, TeamSeason
from mm_analytics.utilities import DATA_ROOT
season_res = pd.read_csv(f"{DATA_ROOT}/Stage2/MRegularSeasonCompactResults.csv")
tourney_res = pd.read_csv(f"{DATA_ROOT}/Stage2/MNCAATourneyCompactResults.csv")
SEEDS_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv')

# years = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
years = [2021]

teams_conf_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamConferences.csv')
teams_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
teams_coach_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamCoaches.csv') 
regularseasonresults_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MRegularSeasonDetailedResults.csv')

for year in years:
    year_reg_season = regularseasonresults_df[regularseasonresults_df["Season"] == year]
    teams_conf_season = teams_conf_df[teams_conf_df["Season"] == year]
    teams_coach_season = teams_coach_df[teams_coach_df["Season"] == year]
    
    ts: Dict[int, TeamSeason] = get_team_seasons(year, year_reg_season, SEEDS_DF, teams_conf_season, teams_coach_season)
    sr = calculate_season_rankings(ts)
    df = pd.DataFrame(columns = ["TeamID"] + list(ts.values())[0].get_data_columns())
    for tid, t in ts.items():
        team_row = np.array([tid] + t.get_data().tolist())
        df = pd.concat([df, pd.DataFrame([team_row], columns = df.columns)], ignore_index=True)
    df.to_csv(f'./features_{year}.csv', index=False)
