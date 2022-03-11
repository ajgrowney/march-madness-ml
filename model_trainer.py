from sklearn.svm import SVC

def train_basic_svc(x_train, x_test, y_train, y_test):
    model = SVC(kernel="linear", probability=True)
    print("Fitting LinearSVC")
    model.fit(x_train, y_train.ravel())
    score = model.score(x_test, y_test.ravel())
    # f_importances(model.coef_, feature_list)
    return model, score
