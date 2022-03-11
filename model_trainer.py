from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb

default_grid_params = {
    "learning_rate": [0.1, 0.01, 0.05],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0,1,10],
    "scale_pos_weight": [1,3,5],
    "subsample": [0.8],
    "colsample_bytree": [0.5]
}

def train_basic_svc(x_train, x_test, y_train, y_test):
    model = SVC(kernel="linear", probability=True)
    print("Fitting LinearSVC")
    model.fit(x_train, y_train.ravel())
    score = model.score(x_test, y_test.ravel())
    # f_importances(model.coef_, feature_list)
    return model, score

def train_xgb_basic(x_train, x_test, y_train, y_test):
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.fit(x_train, y_train.ravel())
    preds = model.predict(x_test)
    score = accuracy_score(y_test, preds)
    return model, score


def train_xgb_grid(x_train, x_test, y_train, y_test, grid_params = default_grid_params):
    model = xgb.XGBClassifier(objective="binary:logistic")
    grid_cv = GridSearchCV(model, grid_params, cv=5)
    _ = grid_cv.fit(x_train, y_train.ravel())
    best_model = xgb.XGBClassifier(
        **grid_cv.best_params_,
        objective="binary:logistic"
    )
    _ = best_model.fit(x_train, y_train.ravel())
    preds = best_model.predict(x_test)
    score = accuracy_score(y_test, preds)
    return best_model, score, grid_cv.best_params_