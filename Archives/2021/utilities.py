
import numpy as np

def get_matchup_data(team1, team2, feature_df):
    team1Data = np.delete(np.array(feature_df[feature_df['TeamID'] == float(team1)]), 0)
    team2Data = np.delete(np.array(feature_df[feature_df['TeamID'] == float(team2)]), 0)
    feature_diff = []

    for a,b in zip(team1Data,team2Data):
        diff = a-b
        # if diff > 0:
        #     result = diff**2
        # else:
        #     result = (-1)*(diff**2)
        feature_diff.append(diff)
    return feature_diff
