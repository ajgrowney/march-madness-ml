# March Madness ML Prediction
Year by year progression of my attempt to let data try to beat my fanaticism for NCAA basketball <br />
<i>Hint: haven't beat it yet</i>

## 2021
Results coming in

### ML Pipeline
1) Data Collection
    - Using exclusively Kaggle provided data so simple download of the CSVs into a Data folder

2) Exploratory Data Analysis
    - Used my `eda.py` file rather dynamically to explore different pieces of the data

3) Feature Extraction
    - After doing some visual analysis of the data set we were working with, used the `objects.py` file to structure the feature based representation of a team's performance for a single season. Using that class, I dumped the data into a simple vector and used that as the feature for that team. The `build_features.py` file was used to save those to a training data folder to cache the feature sets.

4) Model Training
    - Model training was the big place of exploration I did this year, spent time implementing GridSearch to do hyperparameter exploration and experimented with different model architectures. You can find this in the `train_model.py` file.

5) Filling out the submission
    - Unfortunately I forgot to upload the submission to Kaggle, but I did fill out the submission template and used that in tandem with the `bracketeer` pip moudle to fill out the model's bracket. This is done in the `fill_submission.py` file

## 2020 
Covid...

## 2019
[Archives for 2019](Archives/2019/README.md)
