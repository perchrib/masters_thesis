from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def logisitc_regression(data):
    print("Using Logistic Regression")
    x_train = data['x_train']
    y_train = data['y_train']

    x_val = data['x_val']
    y_val = data['y_val']

    x_test = data['x_test']
    y_test = data['y_test']

    lr = LogisticRegression(verbose=1)
    lr.fit(x_train, y_train)

    print("Predict Validation Set")
    y_val_predicted = lr.predict(x_val)
    print("Predict Test Set")
    y_test_predicted = lr.predict(x_test)

    print(metrics.classification_report(y_val, y_val_predicted, digits=3))

    print(metrics.classification_report(y_test, y_test_predicted, digits=3))


def random_forests(data):
    print("Using Random Forests")
    x_train = data['x_train']
    y_train = data['y_train']

    x_val = data['x_val']
    y_val = data['y_val']

    x_test = data['x_test']
    y_test = data['y_test']

    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(x_train, y_train)

    print("Predict Validation Set")
    y_val_predicted = clf.predict(x_val)

    print("Predict Test Set")
    y_test_predicted = clf.predict(x_test)

    print(metrics.classification_report(y_val, y_val_predicted, digits=3))

    print(metrics.classification_report(y_test, y_test_predicted, digits=3))


def svm(data):
    print("Using SVM")
    x_train = data['x_train']
    y_train = data['y_train']

    x_val = data['x_val']
    y_val = data['y_val']

    x_test = data['x_test']
    y_test = data['y_test']

    # normalize
    norm_value = np.max(x_train)
    v = np.max(x_val)
    t = np.max(x_test)
    print("train, ", norm_value, " val, ", v, " test, ", t)
    x_train = x_train / float(norm_value)
    x_val = x_val / float(norm_value)
    x_test = x_test / float(norm_value)

    svc = SVC(C=0.01, kernel='linear', verbose=True)

    svc.fit(x_train, y_train)

    print("Predict Validation Set")
    y_val_predicted = svc.predict(x_val)

    print("Predict Test Set")
    y_test_predicted = svc.predict(x_test)

    print(metrics.classification_report(y_val, y_val_predicted, digits=3))

    print(metrics.classification_report(y_test, y_test_predicted, digits=3))



