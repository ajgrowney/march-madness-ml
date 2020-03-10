import sys
import pandas as pd
import numpy as np
from Teams import Team_Historical, TeamSeason

year = 2018

tournament_teams = {
    2015: [

    ],
    2016: [

    ],
    2017: [

    ],
    2018: [
        "Virginia", "Cincinnati", "Tennessee", "Arizona", "Kentucky", "Miami FL", "Nevada", "Creighton", "Kansas St", "Texas", "Loyola-Chicago", "Davidson", "Buffalo", "Wright St", "Georgia St", "UMBC",
        "Xavier", "North Carolina", "Michigan", "Gonzaga", "Ohio St", "Houston", "Texas A&M", "Missouri", "Florida St", "Providence", "San Diego St", "S Dakota St", "UNC Greensboro", "Montana", "NC Central", "TX Southern",
        "Villanova", "Purdue", "Texas Tech", "Wichita St", "West Virginia", "Florida", "Arkansas", "Virginia Tech", "Alabama", "Butler", "St Bonaventure", "UCLA", "Murray St", "Marshall", "SF Austin", "CS Fullerton", "LIU Brooklyn", "Radford",
        "Kansas", "Duke", "Michigan St", "Auburn", "Clemson", "TCU", "Rhode Island", "Seton Hall", "NC State", "Oklahoma", "Arizona St", "Syracuse", "New Mexico St", "Col Charleston", "Bucknell", "Iona", "Penn", "Lipscomb"
    ],
    2019: [

    ]
}
df_columns, df_rows = None, []

for tm in tournament_teams[year]:
    team = Team_Historical(tm, [year])
    # Initialize Columns of DataFrame based on what is coming from data
    if(not df_columns):
        df_columns = ['TeamID'] + team.get_data_columns(year)
    
    # Fill rows of DataFrame
    df_rows.append(np.append(team.id, team.get_season_data(year)))

df = pd.DataFrame(df_rows, columns=df_columns)
df.to_csv('./TrainingData/phase1_{:d}.csv'.format(year), index=False)
