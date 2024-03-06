from copy import deepcopy
from typing import Dict
import numpy as np
import pandas as pd
from mm_analytics.objects import get_team_seasons_and_rankings, team_seasons_to_df, get_season_ordinals
DATA_ROOT = "/Users/andrewgrowney/data/kaggle/marchmadness-2023"
TOURNEY_RESULTS_DF = pd.read_csv(f"{DATA_ROOT}/Stage2/MNCAATourneyDetailedResults.csv")
SEEDS_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv')
ORDINALS_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MMasseyOrdinals_thru_Season2023_Day128.csv')

TEAM_COACH_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamCoaches.csv')
teams_df = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
TEAM_CONF_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MTeamConferences.csv') 
RS_DF = pd.read_csv(f'{DATA_ROOT}/Stage2/MRegularSeasonDetailedResults.csv')
FEATURE_COLUMNS = ["Points_mean","Poss_mean","OE_mean","DE_mean","NE_mean","FGM_mean","FGA_mean","FGM3_mean","FGA3_mean","FTM_mean","FTA_mean","OR_mean","DR_mean","Ast_mean","TO_mean","Stl_mean","Blk_mean","Fouls_mean","FG%_mean","FG3%_mean","FT%_mean","OppPoints_mean","OppFGM_mean","OppFGA_mean","OppFGM3_mean","OppFGA3_mean","OppFTM_mean","OppFTA_mean","OppOR_mean","OppDR_mean","OppAst_mean","OppTO_mean","OppStl_mean","OppBlk_mean","OppFouls_mean","OppFG%_mean","OppFG3%_mean","OppFT%_mean","AdjOE_mean","AdjDE_mean","AdjNE_mean","Points_stdev","Poss_stdev","OE_stdev","DE_stdev","NE_stdev","FGM_stdev","FGA_stdev","FGM3_stdev","FGA3_stdev","FTM_stdev","FTA_stdev","OR_stdev","DR_stdev","Ast_stdev","TO_stdev","Stl_stdev","Blk_stdev","Fouls_stdev","FG%_stdev","FG3%_stdev","FT%_stdev","OppPoints_stdev","OppFGM_stdev","OppFGA_stdev","OppFGM3_stdev","OppFGA3_stdev","OppFTM_stdev","OppFTA_stdev","OppOR_stdev","OppDR_stdev","OppAst_stdev","OppTO_stdev","OppStl_stdev","OppBlk_stdev","OppFouls_stdev","OppFG%_stdev","OppFG3%_stdev","OppFT%_stdev","AdjOE_stdev","AdjDE_stdev","AdjNE_stdev","Q1_WinPct","Q2_WinPct","Q3_WinPct","Q4_WinPct","WinPct","SOS","SOV","Seed", "ExitRound"]
full_df = pd.DataFrame(columns=FEATURE_COLUMNS)


for year in [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]:
    season_ordinal_sys  = "NET" if year >= 2019 else "RPI"
    season_features     = deepcopy(FEATURE_COLUMNS) + [f"{season_ordinal_sys}_last"]
    year_reg_season     = RS_DF[RS_DF["Season"] == year]
    teams_conf_season   = TEAM_CONF_DF[TEAM_CONF_DF["Season"] == year]
    teams_coach_season  = TEAM_COACH_DF[TEAM_COACH_DF["Season"] == year]
    year_tourney        = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"] == year]

    so = get_season_ordinals(ORDINALS_DF[ORDINALS_DF["Season"] == year], [season_ordinal_sys])
    (ts, sr) = get_team_seasons_and_rankings(year, year_reg_season, SEEDS_DF, teams_conf_season, teams_coach_season, so, year_tourney)
    
    ts_df = team_seasons_to_df(ts, season_features)
    if season_ordinal_sys == "RPI":
        ts_df.rename(columns={"RPI_last": "NET_last"}, inplace=True)
    full_df = pd.concat([full_df, ts_df], ignore_index=True)
full_df.to_csv("TeamSeasons.csv", index=False)