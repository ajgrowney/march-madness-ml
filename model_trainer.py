from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import xgboost as xgb
from keras import Sequential
from keras.layers import InputLayer, Dense, Activation, Dropout

default_grid_params = {
    "learning_rate": [0.15, 0.1, 0.01],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0,1,10],
    "scale_pos_weight": [1,3,5],
    "subsample": [0.8],
    "colsample_bytree": [0.5]
}

def train_basic_svc(x_train, x_test, y_train, y_test, sample_weight):
    model = SVC(kernel="linear", probability=True, C=0.01, gamma=10)
    print("Fitting LinearSVC")
    model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    print("Fitted")
    score = model.score(x_test, y_test.ravel())
    return model, score


def train_grid_svc(x_train, x_test, y_train, y_test, sample_weight):
    linear_steps = [('svc', SVC(kernel='linear', probability=True))]
    linear_pipeline = Pipeline(linear_steps) # define the pipeline object.
    linear_parameters = {'svc__C':[0.1,1,3], 'svc__gamma':[100, 10, 1, 0.1]}
    grid_linear = GridSearchCV(linear_pipeline, param_grid=linear_parameters, cv=5, verbose=2,n_jobs=-1)

    print("Fitting Linear GridSearch")
    grid_linear.fit(x_train, y_train.ravel())
    best_c, best_g = grid_linear.best_params_['svc__C'], grid_linear.best_params_['svc__gamma']
    best_model = SVC(kernel='linear', probability=True, C=best_c, gamma=best_g)
    best_model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    score = best_model.score(x_test, y_test.ravel())
    print("Linear Grid Score: ", score)
    return best_model, score, grid_linear.best_params_

def train_poly_svc(x_train, x_test, y_train, y_test, degree = 5, c = 0.01, g = 0.1, sample_weight=None):
    model = SVC(kernel="poly", degree=degree, probability=True, C=c, gamma=g)
    model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    score = model.score(x_test, y_test.ravel())
    return model, score

def train_grid_poly_svc(x_train, x_test, y_train, y_test):
    poly_steps = [('svc', SVC(kernel='poly', probability=True))]
    poly_pipeline = Pipeline(poly_steps) # define the pipeline object.
    poly_parameters = {'svc__C':[0.005, 0.01,0.1,1], 'svc__gamma':[10,5, 1,0.1], 'svc__degree':[2,3,5]}
    cv_model = GridSearchCV(poly_pipeline, param_grid=poly_parameters, cv=5, verbose=2, n_jobs=-1)
    cv_model.fit(x_train, y_train.ravel())
    score = cv_model.score(x_test, y_test.ravel())
    return cv_model, score, cv_model.best_params_
    

def train_xgb_basic(x_train, x_test, y_train, y_test, sample_weight):
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False)
    model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    preds = model.predict(x_test)
    score = accuracy_score(y_test, preds)
    return model, score


def train_xgb_grid(x_train, x_test, y_train, y_test, grid_params = default_grid_params, sample_weight = None):
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False)
    grid_cv = GridSearchCV(model, grid_params, cv=5, n_jobs=-1)
    _ = grid_cv.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    best_model = xgb.XGBClassifier(
        **grid_cv.best_params_,
        objective="binary:logistic",
        eval_metric='logloss',
        use_label_encoder=False
    )
    _ = best_model.fit(x_train, y_train.ravel(), sample_weight=sample_weight)
    preds = best_model.predict(x_test)
    score = accuracy_score(y_test, preds)
    return best_model, score, grid_cv.best_params_


def train_dnn(x_train, x_test, y_train, y_test, sample_weight=None):
    clf = Sequential()
    clf.add(InputLayer(input_shape=x_train[0].shape))
    clf.add(Dense(64))
    clf.add(Dropout(0.2))
    clf.add(Dense(64))
    clf.add(Dense(32))
    clf.add(Dropout(0.2))
    clf.add(Dense(16))
    clf.add(Activation('relu'))
    clf.add(Dropout(0.1))
    clf.add(Dense(1))
    clf.add(Activation('sigmoid'))
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    clf.fit(x_train, y_train, epochs=300, validation_data=(x_test,y_test), sample_weight=sample_weight)
    predicted = clf.predict(x_test)
    y_pred = [int(y_p > 0.5) for y_p in predicted]
    score = accuracy_score(y_test, y_pred)
    return clf, score
