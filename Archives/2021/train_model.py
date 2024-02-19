import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from mm_analytics.utilities import get_matchup_data, f_importances
scaler = StandardScaler()

years = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
dfs = {}

for year in years:
    dfs[year] = pd.read_csv("Data/Training/features_{:d}.csv".format(year))
feature_list = list(list(dfs.values())[0].columns.values)[1:]

df = pd.concat(dfs, axis=0)
tourney_games = pd.read_csv("Data/Raw/MNCAATourneyDetailedResults.csv")
tourney_games_years = tourney_games[tourney_games["Season"].isin(years)]


X, Y = [], []
print("Fill Training and Testing Data")

for tournament_game in tourney_games_years.itertuples():
    season = tournament_game[1]
    winner, loser = tournament_game[3], tournament_game[5]
    X.append(get_matchup_data(winner, loser, dfs[season])), Y.append(0)
    X.append(get_matchup_data(loser, winner, dfs[season])), Y.append(1)

# Scale the Data
X, Y = np.array(X), np.array(Y).reshape(-1,1)
X = scaler.fit_transform(X)
print("Training and Testing Data Finished")
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.85)

# DecisionTree
tree_clf = DecisionTreeClassifier()
print("Fitting Decision Tree")
tree_clf.fit(x_train, y_train.ravel())
print("Decision Tree Score: ", tree_clf.score(x_test, y_test.ravel()))


# Basic LinearSVC
# model = SVC(kernel="linear", probability=True)
# print("Fitting LinearSVC")
# model.fit(x_train, y_train.ravel())
# print("LinearSVC Score: ", model.score(x_test, y_test.ravel()))
# f_importances(model.coef_, feature_list)

# rfe = RFE(model, 15)
# fit = rfe.fit(x_train, y_train.ravel())
# print("Num Features: %d" % fit.n_features_)
# print("Selected Features:")
# [print(n,r) for n,r in zip(feature_list, fit.support_)]
# print("Feature Ranking: %s" % fit.ranking_)
# print("RFE Score: ", fit.score(x_test, y_test.ravel()))


# Basic Non-linearSVC
# poly_model = SVC(kernel="poly", degree=3, probability=True)
# print("Fitting Non-linearSVC")
# poly_model.fit(x_train, y_train.ravel())
# print("Non-LinearSVC Score: ", poly_model.score(x_test, y_test.ravel()))

# gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.15, max_depth=1, random_state=0).fit(x_train, y_train.ravel())
# print("Gradiant Boosting Score: ", gb_clf.score(x_test, y_test.ravel()))

# rf_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.15, max_depth=1, random_state=0).fit(x_train, y_train.ravel())
# print("Random Forest Score: ", rf_clf.score(x_test, y_test.ravel()))

# GridSearchCV Classifier
# linear_steps = [('svc', SVC(kernel='linear', probability=True))]
# linear_pipeline = Pipeline(linear_steps) # define the pipeline object.
# linear_parameters = {'svc__C':[0.001,0.01,0.1,1,10], 'svc__gamma':[10, 1,0.1,0.01,0.001]}
# grid_linear = GridSearchCV(linear_pipeline, param_grid=linear_parameters, cv=5, verbose=2)

# print("Fitting Linear GridSearch")
# grid_linear.fit(x_train, y_train.ravel())
# print("Linear Grid Score: ", grid_linear.score(x_test, y_test.ravel()))
# print(grid_linear.best_estimator_)

poly_steps = [('svc', SVC(kernel='poly', probability=True))]
poly_pipeline = Pipeline(poly_steps) # define the pipeline object.
poly_parameters = {'svc__C':[0.1], 'svc__gamma':[10,1,0.1,0.001], 'svc__degree':[5]}
grid_poly = GridSearchCV(poly_pipeline, param_grid=poly_parameters, cv=5, verbose=2)

print("Fitting Poly GridSearch")
grid_poly.fit(x_train, y_train.ravel())
print("Linear Grid Score: ", grid_poly.score(x_test, y_test.ravel()))
print(grid_poly.best_estimator_)


# # Neural Net
# neural_net = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,2))

# print("Fitting MLP Classifier")
# neural_net.fit(x_train, y_train.ravel())
# print("Neural Net Score: ", neural_net.score(x_test, y_test.ravel()))

if len(sys.argv) > 1 and sys.argv[1] == "save": 
    # pickle.dump(model, open('model.sav', 'wb'))
    # pickle.dump(poly_model, open("poly_model.sav", "wb"))
    # pickle.dump(gb_clf, open("gb_clf.sav", "wb"))
    pickle.dump(grid_poly, open('grid_poly.sav', 'wb'))
    # pickle.dump(grid_linear, open('grid_linear.sav', 'wb'))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
else:
    save_inp = input("Want to save this model? [y|n]\n")
    if save_inp.lower() == "y":
        # pickle.dump(model, open('model.sav', 'wb'))
        # pickle.dump(poly_model, open("poly_model.sav", "wb"))
        # pickle.dump(gb_clf, open("gb_clf.sav", "wb"))
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        pickle.dump(grid_poly, open('grid_poly.sav', 'wb'))
        # pickle.dump(grid_linear, open('grid_linear.sav', 'wb'))
    