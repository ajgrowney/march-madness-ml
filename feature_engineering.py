import os
import pandas as pd
from utilities import DATA_ROOT

def feature_transform(subfolder:str):
    """Take raw features and create computed values
    """
    raw_feature_dir = os.path.join(DATA_ROOT, "Training")
    new_feature_dir = os.path.join(raw_feature_dir, subfolder)
    for feature_file in [f for f in os.listdir(raw_feature_dir) if f[-3:] == "csv"]:
        df = pd.read_csv(os.path.join(raw_feature_dir, feature_file))
        df["MOV_mean"] = df["Points_mean"] - df["OppPoints_mean"]
        df.to_csv(os.path.join(new_feature_dir, feature_file))




if __name__ == "__main__":
    feature_transform("ComputedV1")