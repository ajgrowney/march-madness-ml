import sys
import pandas as pd
import numpy as np
from Teams import Team_Historical, TeamSeason


teams2018 = [
    "Virginia", "Cincinnati", "Tennessee", "Arizona", "Kentucky", "Miami FL", "Nevada", "Creighton", "Kansas St", "Texas", "Loyola-Chicago", "Davidson", "Buffalo", "Wright St", "Georgia St", "UMBC",
    "Xavier", "North Carolina", "Michigan", "Gonzaga", "Ohio St", "Houston", "Texas A&M", "Missouri", "Florida St", "Providence", "San Diego St", "S Dakota St", "UNC Greensboro", "Montana", "NC Central", "TX Southern",
    "Villanova", "Purdue", "Texas Tech", "Wichita St", "West Virginia", "Florida", "Arkansas", "Virginia Tech", "Alabama", "Butler", "St Bonaventure", "UCLA", "Murray St", "Marshall", "SF Austin", "CS Fullerton", "LIU Brooklyn", "Radford",
    "Kansas", "Duke", "Michigan St", "Auburn", "Clemson", "TCU", "Rhode Island", "Seton Hall", "NC State", "Oklahoma", "Arizona St", "Syracuse", "New Mexico St", "Col Charleston", "Bucknell", "Iona", "Penn", "Lipscomb"
]
df_columns, df_rows = None, []

for tm in teams2018:
    team = Team_Historical(tm, [2018, 2019])
    # Initialize Columns of DataFrame based on what is coming from data
    if(not df_columns):
        df_columns = ['TeamID'] + team.get_data_columns(2018)
    
    # Fill rows of DataFrame
    df_rows.append(np.append(team.id, team.get_season_data(2018)))

df = pd.DataFrame(df_rows, columns=df_columns)
df.to_csv('./Data/phase1_2018.csv', index=False)
