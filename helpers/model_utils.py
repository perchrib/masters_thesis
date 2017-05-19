from __future__ import print_function
import os
import time
import itertools
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model, Model
from global_constants import GENDER, OVERALL_MACRO, OVERALL_MICRO, TRAIN_ACC, TRAIN_LOSS, VAL_ACC, VAL_LOSS
from helper_functions import save_pickle

import matplotlib.pyplot as plt


def get_model_checkpoint(model_name, model_dir, model_optimizer):
    dir_path = os.path.join(model_dir, model_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model_name + "_" + model_optimizer + "_{epoch:02d}_{val_acc:.4f}.h5"

    checkpoint = ModelCheckpoint(os.path.join(dir_path, model_file_name), save_best_only=True)

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
    dir_path = os.path.join(model_dir, model.name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # _{epoch:02d}_{val_acc:.4f}
    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + model_optimizer + ".h5"

    model.save(os.path.join(model_dir, model.name, model_file_name))
    print("Model saved")


def save_term_index(term_index, model_name, index_dir):
    print("Saving term index")

    if not os.path.exists(os.path.join(index_dir, model_name)):
        os.makedirs(os.path.join(index_dir, model_name))

    index_file_name = time.strftime("%d.%m.%Y_%H:%M:%S") + "_" + model_name
    save_pickle(os.path.join(index_dir, index_file_name), term_index)

    print("Term index saved")


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


def plot_models(log_path_list, graph_metric):
    """
    Given a list of log file paths, plot the training histories for the specified graph_metric
    :param log_path_list: List of log file paths
    :param graph_metric: Which metric to plot from provided logs. Allowed values are TRAIN_ACC; TRAIN_LOSS; VAL_ACC and VAL_LOSS
    :return: 
    """

    # Get statistics for all models in log_path_list
    # List of lists (for each models) with corresponding history of values
    model_names, statistics = _get_log_statistics(log_path_list)

    print("Specified models for plotting: %s" % model_names)
    # Plot correct metric
    plt.style.use("seaborn-darkgrid")

    for i in range(len(statistics[graph_metric])):
        plt.plot(statistics[graph_metric][i], label=model_names[i])

    plt.ylabel(graph_metric)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    print("")


def _get_log_statistics(log_path_list):
    """
    Read log files and mine training statistics
    :param log_path_list: list of log file paths
    :return: list of model names and dictionary with training statistics
    """

    model_names = []
    # List of lists (for each models) with corresponding history of values
    model_accs = []
    model_losses = []
    model_val_accs = []
    model_val_losses = []

    for path in log_path_list:
        with open(path) as log_file:
            eof = False  # End of file
            while not eof:
                line = log_file.readline()
                if line == "":
                    eof = True

                # Model Name
                elif line.startswith("Model name"):
                    model_names.append(line.split(":")[1].strip())

                elif "Training statistics" in line:
                    log_file.readline()  # Jump to next line containing first line of stats
                    end_of_training_stats = False

                    acc = []
                    loss = []
                    val_acc = []
                    val_loss = []

                    while not end_of_training_stats:
                        line = log_file.readline()
                        if line == "\n":
                            end_of_training_stats = True
                        else:
                            all_metrics = line.split()  # [acc, loss, val_acc, val_loss]
                            all_metrics.pop(0)  # Remove history index from the list

                            acc.append(all_metrics[0])
                            loss.append(all_metrics[1])
                            val_acc.append(all_metrics[2])
                            val_loss.append(all_metrics[3])

                    model_accs.append(acc)
                    model_losses.append(loss)
                    model_val_accs.append(val_acc)
                    model_val_losses.append(val_loss)

    statistics = {
        TRAIN_ACC: model_accs,
        TRAIN_LOSS: model_losses,
        VAL_ACC: model_val_accs,
        VAL_LOSS: model_val_losses
    }

    return model_names, statistics

if __name__ == '__main__':
    # preds = [[0.4, 0.6], [0.7, 0.3], [0.8, 0.2]]
    # y_pred = [np.argmax(x) for x in preds]
    # create_and_plot_confusion_matrix([1, 1, 0], y_pred, ["Male", "Female"], normalize=False)

    #
    paths = ['../logs/character_level_classification/Conv_BiLSTM/18.05.2017_17:47:45_Conv_BiLSTM.txt',
             '../logs/character_level_classification/BiLSTM/16.05.2017_14:27:27_BiLSTM_adam.txt']
    plot_models(paths, TRAIN_LOSS)