from copy import deepcopy
import json
from typing import Dict, Tuple
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from mm_analytics.objects import TeamSeason, get_team_seasons_and_rankings, team_seasons_to_df, get_season_ordinals
from mm_analytics.utilities import NpEncoder, get_historical_similarity

DATA_ROOT = "/Users/andrewgrowney/data/kaggle/marchmadness-2024"
TOURNEY_RESULTS_DF = pd.read_csv(f"{DATA_ROOT}/MNCAATourneyDetailedResults.csv")
SEEDS_DF = pd.read_csv(f'{DATA_ROOT}/MNCAATourneySeeds.csv')
ORDINALS_DF = pd.read_csv(f'{DATA_ROOT}/MMasseyOrdinals.csv')

TEAM_COACH_DF = pd.read_csv(f'{DATA_ROOT}/MTeamCoaches.csv')
teams_df = pd.read_csv(f'{DATA_ROOT}/MTeams.csv').drop(columns=['FirstD1Season', 'LastD1Season'])
TEAM_CONF_DF = pd.read_csv(f'{DATA_ROOT}/MTeamConferences.csv') 
RS_DF = pd.read_csv(f'{DATA_ROOT}/MRegularSeasonDetailedResults.csv')

FEATURE_COLUMNS = ["Points_mean","Poss_mean","OE_mean","DE_mean","NE_mean","FGM_mean","FGA_mean","FGM3_mean","FGA3_mean","FTM_mean","FTA_mean","OR_mean","DR_mean","Ast_mean","TO_mean","Stl_mean","Blk_mean","Fouls_mean","FG%_mean","FG3%_mean", "EFG%_mean", "FT%_mean","OppPoints_mean","OppFGM_mean","OppFGA_mean","OppFGM3_mean","OppFGA3_mean","OppFTM_mean","OppFTA_mean","OppOR_mean","OppDR_mean","OppAst_mean","OppTO_mean","OppStl_mean","OppBlk_mean","OppFouls_mean","OppFG%_mean","OppFG3%_mean", "OppEFG%_mean", "OppFT%_mean","AdjOE_mean","AdjDE_mean","AdjNE_mean","Points_stdev","Poss_stdev","OE_stdev","DE_stdev","NE_stdev","FGM_stdev","FGA_stdev","FGM3_stdev","FGA3_stdev","FTM_stdev","FTA_stdev","OR_stdev","DR_stdev","Ast_stdev","TO_stdev","Stl_stdev","Blk_stdev","Fouls_stdev","FG%_stdev","FG3%_stdev","EFG%_stdev","FT%_stdev","OppPoints_stdev","OppFGM_stdev","OppFGA_stdev","OppFGM3_stdev","OppFGA3_stdev","OppFTM_stdev","OppFTA_stdev","OppOR_stdev","OppDR_stdev","OppAst_stdev","OppTO_stdev","OppStl_stdev","OppBlk_stdev","OppFouls_stdev","OppFG%_stdev","OppFG3%_stdev","OppEFG%_stdev","OppFT%_stdev","AdjOE_stdev","AdjDE_stdev","AdjNE_stdev","Q1_WinPct","Q2_WinPct","Q3_WinPct","Q4_WinPct","WinPct","SOS","SOV","Seed", "ExitRound"]
    

def year_team_seasons(year:int):
    year_reg_season     = RS_DF[RS_DF["Season"] == year]
    teams_conf_season   = TEAM_CONF_DF[TEAM_CONF_DF["Season"] == year]
    teams_coach_season  = TEAM_COACH_DF[TEAM_COACH_DF["Season"] == year]
    year_tourney        = TOURNEY_RESULTS_DF[TOURNEY_RESULTS_DF["Season"] == year]

    so = get_season_ordinals(ORDINALS_DF[ORDINALS_DF["Season"] == year], [season_ordinal_sys])
    (ts, sr) = get_team_seasons_and_rankings(year, year_reg_season, SEEDS_DF, teams_conf_season, teams_coach_season, so, year_tourney)
    return ts

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--export", type=str, default="csv")
    args = parser.parse_args()

    # ---- Build Features ----
    team_seasons: Dict[Tuple[int,int], TeamSeason] = {}
    for year in range(args.start_year, args.end_year+1):
        season_ordinal_sys  = "NET" if year >= 2019 else "RPI"
        ts = year_team_seasons(year)
        for tid, team_season in ts.items():
            team_seasons[(int(tid), int(year))] = team_season
    ts_df = team_seasons_to_df(team_seasons, FEATURE_COLUMNS, add_season_ordinal=True)

    # ---- Export ----
    if args.export == "csv":
        ts_df.to_csv("TeamSeasons_cust.csv", index=False)
    else:
        ts_df.to_csv("tsdf.csv")
        similar = get_historical_similarity(ts_df, num_teams=3, precision=3)
        print(len(similar))
        print(len(ts_df))
        for (tid, tyear), sim in similar.items():
            team_seasons[(int(tid), int(tyear))].similar_teams = [s + (team_seasons[(s[0], s[1])].tourney_exit_round, ) for s in sim]
        # Dump as json files to data/web/ts
        for (tid, tyear), team_season in team_seasons.items():
            with open(f"data/web/ts_v2/{tid}_{tyear}.json", "w") as f:
                f.write(json.dumps(team_season.to_web_json(), cls=NpEncoder))