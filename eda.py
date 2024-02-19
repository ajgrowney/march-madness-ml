import pandas as pd
import matplotlib.pyplot as plt
import sys
from mm_analytics.utilities import get_matchup_data, DATA_ROOT

def compare_teams(year, team_1, team_2):
    df = pd.read_csv(f'{DATA_ROOT}/Training/features_{year}.csv')
    team_1_df = df[df['TeamID'] == float(team_1)]
    team_2_df = df[df['TeamID'] == float(team_2)]
    print(team_1_df)
    print(team_2_df)
    matchup = get_matchup_data(team_1, team_2, df)
    print(matchup)


y, t1, t2 = sys.argv[1], sys.argv[2], sys.argv[3]
compare_teams(y,t1,t2)