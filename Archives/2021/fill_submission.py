import pickle
import sys
import pandas as pd
import numpy as np
import json
from utilities import get_matchup_data
from bracketeer import build_bracket

write = True
DATA_ROOT = "/Users/andrewgrowney/Data/kaggle/marchmadness-2021"
MODEL_SELECTED = "poly_model" if len(sys.argv) < 2 else sys.argv[1]
model = pickle.load(open(f'Models/{MODEL_SELECTED}.sav', 'rb'))
scaler = pickle.load(open('Models/scaler.pkl', "rb"))
stage_comp = pd.read_csv(f"{DATA_ROOT}/Stage2/MSampleSubmissionStage2.csv")
feature_dfs = {}
# features_df = pd.read_csv("./Data/Training/features_{:d}.csv".format(year))
if write:
    with open(f"./Results/2021_Stage2_{MODEL_SELECTED}_first_round.csv","w") as f:
        f.write("ID,Pred\n")

with open(f"./Results/2021_Stage2_{MODEL_SELECTED}_first_round.csv","a") as f:
    for i, id, _ in stage_comp.itertuples():
        year, t1, t2 = id.split("_")
        year, t1, t2 = int(year), int(t1), int(t2)

        if year not in feature_dfs.keys():
            print(year)
            feature_dfs[year] = pd.read_csv(f"{DATA_ROOT}/Training/V1/features_{year}.csv")

        matchup = get_matchup_data(t1, t2, feature_dfs[year])
        matchup = np.array(matchup).reshape(1,-1)
        matchup = scaler.transform(matchup)
        prediction = model.predict_proba(matchup).flatten()
        if write == True:
            f.write(f"{id},{prediction[0]}\n")
        else:
            print(id, prediction[0])

if write:
    _ = build_bracket(
        outputPath=f'Results/{MODEL_SELECTED}.png',
        teamsPath=f'{DATA_ROOT}/Stage2/MTeams.csv',
        seedsPath=f'{DATA_ROOT}/Stage2/MNCAATourneySeeds.csv',
        submissionPath=f'Results/{MODEL_SELECTED}.csv',
        slotsPath=f'{DATA_ROOT}/Stage2/MNCAATourneySlots.csv',
        year=2022
    )