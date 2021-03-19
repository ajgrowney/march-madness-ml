# March Madness Bracket Predictor

## 2021


## 2020 
Covid...

## 2019 Model Information
## Find source code in Archives/2019
| Tournament Year | Accuracy |
| --- | --- |
| 2015 | 69% |
| 2016 | 73% |
| 2017 | 70% |
| 2018 | 73% |
| 2019 | 76% |

This model was based just a LinearSVC based on a team's averages throughout the season

## Tutorial
### Write Data (write_data.py)
Handles the creation of the data that the mdoels will train off of.
Uses the data from the Kaggle competition to generate a new set of CSVs with meaningful statistics about each team.
These stats will be used as the way to compare teams.

Ex: `py write_data.py`

### Train Model
Using the data stored in the TrainingData folder, train the model(s).
You can save the model as well.

Ex: `py train_model.py save`

### Predict Games
Using a saved model, run predictions for games between teams of your choosing, or against a full March Madness set.

Ex 1: `py predict_games.py 2018` will run predictions for the 2018 NCAA Tournament 

Ex 2: `py predict_games.py 2019 userin` will pop open a menu that allows you to continuously predict outcomes of games between two teams 

## Objectives by Phase
1. Data driven prediction with confidence of who will win any NCAA tournament games based on Regular Season Stats


## Data Classes
Each team has
1. Name
2. ID
3. Conference / Year
4. Coach(es) / Year

Each team plays
1. Regular Season Games
2. Conference Tourney Games
3. NCAA Tourney Games

Each game has
1. Winning team
2. Losing team
3. Location
    - Distance from each team's home
4. Statistics
