from __future__ import print_function
import os
import time
import itertools
import numpy as np

from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model, Model
from global_constants import GENDER

import matplotlib.pyplot as plt


def get_model_checkpoint(model_name, model_dir, model_optimizer):
    if not os.path.exists(model_dir):
        os.makedirs(os.path.join(model_dir, model_name))

    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model_name + "_" + model_optimizer + "_{epoch:02d}_{val_acc:.4f}.h5"

    checkpoint = ModelCheckpoint(os.path.join(model_dir, model_name, model_file_name), save_best_only=True)

    return checkpoint


def save_trained_model(model, model_dir, model_optimizer):
    """

    :type: model: Model
    :param model:
    :param model_dir:
    :param model_optimizer:
    :return:
    """

    print("Saving trained model")

    if not os.path.exists(model_dir):
        os.makedirs(os.path.join(model_dir, model.name))

    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + model_optimizer + "_{epoch:02d}_{val_acc:.4f}.h5"

    model.save(os.path.join(model_dir, model.name, model_file_name))
    print("Model saved")


def load_and_evaluate(model_path, data):
    """
    Load model and evaluate a dataset --> Gives accuracy
    :param model_path: file path to model
    :param data: dictionary of data; need data['x_test']
    :return:
    """

    model = load_model(model_path)
    test_results = model.evaluate(data['x_test'], data['y_test'])

    print("\n--------------Test results---------------")
    print("%s: %f" % (model.metrics_names[0], round(test_results[0], 5)))
    print("%s: %f" % (model.metrics_names[1], round(test_results[1], 5)))


def load_and_predict(model_path, data, prediction_type):
    """
    Load model and predict/classify dataset; then graph confusion matrix
    :param model_path: file path to model
    :param data: dictionary of data; need data['x_test']
    :param prediction_type:
    :return:
    """

    model = load_model(model_path)
    y_pred = model.predict(data['x_test'])

    class_names = get_class_names(prediction_type)

    create_and_plot_confusion_matrix(y_test=data['y_test'], y_pred=y_pred, class_names=class_names)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_and_plot_confusion_matrix(y_test, y_pred, class_names):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()


def get_class_names(prediction_type):
    if prediction_type == GENDER:
        return ["Male", "Female"]

# if __name__ == '__main__':
#     create_and_plot_confusion_matrix([1, 1, 1], [1, 0, 0], ["Male", "Female"])