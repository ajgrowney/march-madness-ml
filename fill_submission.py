import pickle
import sys
import pandas as pd
import numpy as np
import json
from utilities import get_matchup_data
from bracketeer import build_bracket

write = True
MODEL_SELECTED = "poly_model" if len(sys.argv) < 2 else sys.argv[1]
model = pickle.load(open(f'Results/{MODEL_SELECTED}.sav', 'rb'))
scaler = pickle.load(open('Results/scaler.pkl', "rb"))
stage_1_comp = pd.read_csv("./Data/Stage2/MSampleSubmissionStage2.csv")
feature_dfs = {}
# features_df = pd.read_csv("./Data/Training/features_{:d}.csv".format(year))
if write:
    with open(f"./Results/2021_Stage2_{MODEL_SELECTED}.csv","w") as f:
        f.write("ID,Pred\n")

with open(f"./Results/2021_Stage2_{MODEL_SELECTED}.csv","a") as f:
    for i, id, _ in stage_1_comp.itertuples():
        year, t1, t2 = id.split("_")
        year, t1, t2 = int(year), int(t1), int(t2)

        if year not in feature_dfs.keys():
            print(year)
            feature_dfs[year] = pd.read_csv("./Data/Training/features_{:d}.csv".format(year))

        matchup = get_matchup_data(t1, t2, feature_dfs[year])
        matchup = np.array(matchup).reshape(1,-1)
        matchup = scaler.transform(matchup)
        prediction = model.predict_proba(matchup).flatten()
        if write == True:
            f.write(f"{id},{prediction[0]}\n")
        else:
            print(id, prediction[0])

if write:
    b = build_bracket(
        outputPath=f'Results/2021_bracket_{MODEL_SELECTED}.png',
        teamsPath='Data/Stage2/MTeams.csv',
        seedsPath='Data/Stage2/MNCAATourneySeeds.csv',
        submissionPath=f'Results/2021_Stage2_{MODEL_SELECTED}.csv',
        slotsPath='Data/Stage2/MNCAATourneySlots.csv',
        year=2021
    )