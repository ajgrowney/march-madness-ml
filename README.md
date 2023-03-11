# March Madness ML Prediction
Year by year progression of my attempt to let data try to beat my fanaticism for NCAA basketball <br />
<i>Hint: haven't beat it yet</i>

## 2022
Currently building models based on last tournaments. This year I built a CLI around the model training, exploring, and submission process.
Submitted grid_xgb_2 to kaggle

### ML Pipeline
1) Data Collection
    - Using exclusively Kaggle provided data so simple download of the CSVs into a Data folder

2) Exploratory Data Analysis
    - Used my `eda.py` file rather dynamically to explore different pieces of the data

3) Feature Extraction
    - After doing some visual analysis of the data set we were working with, used the `objects.py` file to structure the feature based representation of a team's performance for a single season. Using that class, I dumped the data into a simple vector and used that as the feature for that team. The `build_features.py` file was used to save those to a training data folder to cache the feature sets.

4) Model Training - train
    - `python cli.py train --models basic_svc,grid_xgb --save`
        - Outputs a run id that the models are saved to

5) Model Evaluation - evalute
    - `python cli.py evaluate --run 12341234 --models basic_svc`

6) Kaggle Submission - submit

## 2021
[Archives for 2021](Archives/2021/README.md)

## 2020 
Covid...

## 2019
[Archives for 2019](Archives/2019/README.md)
