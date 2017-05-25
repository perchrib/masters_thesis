from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

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



def svm(data):
    print("Using SVM")
    x_train = data['x_train']
    y_train = data['y_train']

    x_val = data['x_val']
    y_val = data['y_val']

    x_test = data['x_test']
    y_test = data['y_test']

    svc = SVC(kernel='linear')

    svc.fit(x_train, y_train)

    print("Predict Validation Set")
    y_val_predicted = svc.predict(x_val)

    print("Predict Test Set")
    y_test_predicted = svc.predict(x_test)

    print(metrics.classification_report(y_val, y_val_predicted, digits=3))

    print(metrics.classification_report(y_test, y_test_predicted, digits=3))



