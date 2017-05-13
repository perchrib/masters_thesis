from __future__ import print_function
import os
import time
import itertools
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model, Model
from global_constants import GENDER, OVERALL_MACRO, OVERALL_MICRO

import matplotlib.pyplot as plt


def get_model_checkpoint(model_name, model_dir, model_optimizer):
    if not os.path.exists(model_dir, model_name):
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

    if not os.path.exists(os.path.join(model_dir, model.name)):
        os.makedirs(os.path.join(model_dir, model.name))

    # _{epoch:02d}_{val_acc:.4f}
    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + model_optimizer + ".h5"

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


def load_and_predict(model_path, data, prediction_type, normalize):
    """
    Load model and predict/classify dataset; then graph confusion matrix
    :param model_path: file path to model
    :param data: dictionary of data; need data['x_test']
    :param prediction_type: generalization in case of multiple prediction types; 'GENDER'
    :param normalize: True if values should be normalized by number of elements in the class
    :return:
    """

    model = load_model(model_path)
    predictions = model.predict(data['x_test'])  # List of lists with confidence in each class, for each test sample

    # List of indices of highest value in each list in predictions. Corresponds to the prediction class
    y_pred = get_argmax_classes(predictions)
    y_true = get_argmax_classes(data['y_test'])

    class_names = get_class_names(prediction_type)

    create_and_plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=class_names, normalize=normalize)


def create_and_plot_confusion_matrix(y_true, y_pred, class_names, normalize):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize)

    plt.show()


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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_class_names(prediction_type):
    if prediction_type == GENDER:
        return ["Male", "Female"]


def get_precision_recall_f_score(model, x_data, y_data, prediction_type):
    classes = get_class_names(prediction_type)
    predictions = model.predict(x_data)

    # List of indices of highest value in each list in predictions. Corresponds to the prediction class
    y_pred = get_argmax_classes(predictions)
    y_true = get_argmax_classes(y_data)

    # Calculate precision, recall, f-scores for all classes and
    prf_scores = {OVERALL_MICRO: precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro'),
                  OVERALL_MACRO: precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')}

    prf_each = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)

    for i in range(len(classes)):
        prf_scores[classes[i]] = tuple(metric[i] for metric in prf_each)

    return prf_scores


def get_argmax_classes(y_values):
    """
    Given list of multiclass predictions or categorical labels, return list of indicative class;
    i.e., single values labels
    :param y_values: list of list with multiclass confidence values/categorical labels
    :return: list of single class values
    """

    return np.asarray([np.argmax(confidence) for confidence in y_values])

# if __name__ == '__main__':
#     preds = [[0.4, 0.6], [0.7, 0.3], [0.8, 0.2]]
#     y_pred = [np.argmax(x) for x in preds]
#     create_and_plot_confusion_matrix([1, 1, 0], y_pred, ["Male", "Female"], normalize=False)