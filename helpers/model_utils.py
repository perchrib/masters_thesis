from __future__ import print_function
import os
import time
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, save_model, Model
from global_constants import GENDER, OVERALL_MACRO, OVERALL_MICRO, TRAIN_ACC, TRAIN_LOSS, VAL_ACC, VAL_LOSS
from helper_functions import save_pickle, get_time_format

import matplotlib.pyplot as plt
from collections import Counter

def get_model_checkpoint(model_name, model_dir, model_optimizer=None):
    dir_path = os.path.join(model_dir, model_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model_name + "_{epoch:02d}_{val_loss:.4f}.h5"

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
        "%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_model.h5"

    model.save(os.path.join(model_dir, model.name, model_file_name))
    print("Model saved")


def save_term_index(term_index, model_name, index_dir):
    print("Saving term index")

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    index_file_name = time.strftime("%d.%m.%Y_%H:%M:%S") + "_" + model_name + "_index"
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


def load_and_predict(model_path, x_data, y_data):
    """
    Load model and predict/classify dataset; then graph confusion matrix
    :param model_path: file path to model
    :param x_data: data samples
    :param y_data: data labels
    :return:
    """

    model = load_model(model_path)
    start_time = time.time()
    print("Generating predictions with model from path: %s..." % model_path, end="")
    categorical_preds = model.predict(x_data)  # List of lists with confidence in each class, for each test sample

    # List of indices of highest value in each list in predictions. Corresponds to the prediction class
    y_pred = get_argmax_classes(categorical_preds)
    y_true = get_argmax_classes(y_data)

    end_time = get_time_format(time.time() - start_time)
    print("Done. Elapsed time: %s" % end_time)
    return y_pred, categorical_preds, model

    # create_and_plot_confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)


def create_and_plot_confusion_matrix(y_true, y_pred, normalize, class_names=None):
    """
    
    :param y_true: 
    :param y_pred: 
    :param normalize: True if values should be normalized by number of elements in the class
    :param class_names: 
    :return: 
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)

    if class_names is None:
        class_names = get_class_names(GENDER)

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


def predict_and_get_precision_recall_f_score(model, x_data, y_data, prediction_type=GENDER):
    """
    Predict using model and return PRF-scores
    :param model: ANN model. h5 file
    :param x_data: formatted data samples
    :param y_data: categorical labels
    :param prediction_type: GENDER by default
    :return: PRF dictionary
    """
    predictions = model.predict(x_data)

    # List of indices of highest value in each list in predictions. Corresponds to the prediction class
    y_pred = get_argmax_classes(predictions)
    y_true = get_argmax_classes(y_data)

    return get_precision_recall_f_score(y_pred=y_pred, y_true=y_true, prediction_type=prediction_type)


def get_precision_recall_f_score(y_pred, y_true, prediction_type=GENDER):
    """
    Given predictions and true labels, return PRF
    :param y_pred: Single class predictions
    :param y_true: Single class true labels
    :param prediction_type: GENDER by default
    :return: PRF dictionary
    """

    classes = get_class_names(prediction_type)

    # Calculate precision, recall, f-scores for all classes and
    prf_scores = {OVERALL_MICRO: precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro'),
                  OVERALL_MACRO: precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')}

    prf_each = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)

    for i in range(len(classes)):
        prf_scores[classes[i]] = tuple(metric[i] for metric in prf_each)

    return prf_scores


def get_prf_repr(prf_scores):
    return pd.DataFrame(data=prf_scores, index=pd.Index(["Precision", "Recall", "F-score", "Support"])).__repr__()


def get_argmax_classes(y_values):
    """
    Given list of multiclass predictions or categorical labels, return list of indicative class;
    i.e., single values labels
    :param y_values: list of list with multiclass confidence values/categorical labels
    :return: list of single class values
    """

    return np.asarray([np.argmax(confidence) for confidence in y_values])


def plot_models(log_path_list, graph_metric, save_path=None, title=None):
    """
    Given a list of log file paths, plot the training histories for the specified graph_metric
    :param log_path_list: List of log file paths
    :param graph_metric: Which metric to plot from provided logs. Allowed values are TRAIN_ACC; TRAIN_LOSS; VAL_ACC and VAL_LOSS. Can be a list to plot multiple metrics
    :return: 
    """

    # Get statistics for all models in log_path_list
    # List of lists (for each models) with corresponding history of values
    model_names, statistics = _get_log_statistics(log_path_list)

    print("Specified models for plotting: %s" % model_names)
    # Plot correct metric
    plt.style.use("seaborn-darkgrid")

    if type(graph_metric) is str:
        for i in range(len(statistics[graph_metric])):  # Iterate over models
            plt.plot(statistics[graph_metric][i], label=model_names[i])
            plt.ylabel(graph_metric)

    elif type(graph_metric) is list:
        for metric in(graph_metric):
            for i in range(len(statistics[metric])): # Iterate over models
                plt.plot(statistics[metric][i], label=model_names[i] + " " + metric)
                plt.ylabel(metric)


    plt.xlabel("Epochs")
    plt.legend()

    if title:
        plt.title(title)

    if save_path:
        print("Plot saved")
        plt.savefig(save_path, format='png', dpi=600)

    plt.show()

def plot_prediction_confidence(predictions, name, plot_type='graph', common_plot=True, truth=None):
    if not common_plot:
        figure = plt.figure()

    confidence_values = []
    num_of_preds = len(predictions)
    predicted_correct_values = []
    for i in range(len(predictions)):
        preds = predictions[i]
        confidence = preds[0] - preds[1]
        confidence_values.append(round(confidence, 2))

        if truth:
            correct = truth[i]

            if confidence > 0 and correct == 0:
                predicted_correct_values.append(round(confidence, 2))

            elif confidence < 0 and correct == 1:
                predicted_correct_values.append(round(confidence, 2))
            else:
                print("conf: ", round(confidence, 2), " correct: ", correct)



    # confidence_values = sorted(confidence_values)
    #
    # for c in confidence_values:
    #     print("C: ", c)

    counts = Counter(confidence_values)
    x_values = sorted(counts.keys())
    y_values = [counts[i] for i in x_values]

    y_values_norm = np.asarray(y_values) / float(num_of_preds)
    y_values = []
    for y in y_values_norm:
        y_values.append(round(y, 4))
    y_max = max(y_values)
    print("Y_mAX: ", y_max)

    if truth:
        truth_counts = Counter(predicted_correct_values)
        truth_x_values = sorted(truth_counts.keys())
        truth_y_values = [truth_counts[i] for i in truth_x_values]

        truth_y_values_norm = np.asarray(truth_y_values) / float(num_of_preds)
        truth_y_values = []
        for y in truth_y_values_norm:
            truth_y_values.append(round(y, 4))






    mu = np.mean(confidence_values) # mean of distribution
    sigma = np.std(confidence_values) # standard deviation of distribution


    if "Doc" in name:
        name = "Document-level"
        color = "C0"
    elif "Char" in name:
        name = "Character-level"
        color = "C1"
    elif "Word" in name:
        name = "Word-level"
        color = "C2"

    if common_plot:
        title_name = "All Models"

    if plot_type == 'graph':
        x_values = np.linspace(-1, 1, 201)
        # for smoothing
        x = np.linspace(-1, 1, 1000)
        poly_deg = 3
        coefs = np.polyfit(x_values, y_values, poly_deg)
        y_poly = np.polyval(coefs, x)
        plt.plot(x, y_poly, label=name, color=color)

        if truth:
            coefs_ = np.polyfit(x_values, truth_y_values, poly_deg)
            y_poly_ = np.polyval(coefs_, x)
            plt.plot(x, y_poly_, label="Truth", color="C3", alpha=0.7)

    elif plot_type == 'bar':
        x_ = np.linspace(-1, 1, 201)
        plt.bar(x_, y_values, width=0.01, color=color, alpha=0.5, label=name)

    else:
        fig, ax = plt.subplots()
        import matplotlib.mlab as mlab
        weights = np.ones_like(confidence_values) / float(len(confidence_values))
        n, bins, patches = ax.hist(confidence_values, 400, weights=weights)
        y = mlab.normpdf(bins, mu, sigma)
        #ax.plot(bins, y, '--')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Probability density')
        print("MU_exp ", mu, " sigma_std: ", sigma)
        ax.set_title('Histogram of Confidence: ' + name)
        print(n)
        print(bins)


    plt.xlabel('Degree of Confidence')
    plt.ylabel('Probability Density, Number of Tweets')


    if common_plot:
        plt.title('Confidence Distributions of ' + title_name)
    else:
        plt.title('Confidence Distributions of ' + name)

    file_format = 'png'
    quality = 750
    plt.legend()
    if common_plot:
        plt.savefig(os.path.join("../wfigs", "conf_distribution_" + title_name.lower() + "." + file_format), format=file_format, dpi=quality)
    else:
        plt.savefig(os.path.join("../wfigs", "conf_distribution_" + name.lower() + "." + file_format),
                    format=file_format, dpi=quality)


def find_differences_in_prediction(y_preds, y_true, m_conf=1, f_conf=-1, true_positive=True):
    from helpers.global_constants import CHAR_MODEL, DOC_MODEL, WORD_MODEL
    word_preds = y_preds[WORD_MODEL]
    char_preds = y_preds[CHAR_MODEL]
    doc_preds = y_preds[DOC_MODEL]
    index = dict()
    print("length y_true. ", len(y_true), " length w_pred: ", len(word_preds))
    for index_pred in range(len(y_true)):
        correct_pred_models = []
        word_pred = word_preds[index_pred]
        doc_pred = doc_preds[index_pred]
        char_pred = char_preds[index_pred]

        models_name = [WORD_MODEL, DOC_MODEL, CHAR_MODEL]
        models_pred = [word_pred, doc_pred, char_pred]

        for i in range(len(models_pred)):
            y = y_true[index_pred]
            m_pred = models_pred[i]
            confidence = round(m_pred[0] - m_pred[1], 2)

            if true_positive:
                if confidence == m_conf and y == 0:
                    correct_pred_models.append(models_name[i] + " -> " + "M " + str(m_conf))

                elif confidence == f_conf and y == 1:
                    correct_pred_models.append(models_name[i] + " -> " + "F " + str(f_conf))

            else:
                if confidence == m_conf and y == 1:
                    correct_pred_models.append(models_name[i] + " -> " + "F " + str(m_conf))

                elif confidence == f_conf and y == 0:
                    correct_pred_models.append(models_name[i] + " -> " + "M " + str(f_conf))

        if correct_pred_models:
            index[index_pred] = correct_pred_models

    return index


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


def print_prf_scores(y_pred, y_true):
    """
    Print PRF_scores in a readable fashion given model path, samples and labels
    :param model_path: path to h5 model file
    :param x_data: data samples
    :param y_data: categorical labels
    :return: 
    """

    prf = get_precision_recall_f_score(y_pred=y_pred, y_true=y_true)
    print(get_prf_repr(prf))

if __name__ == '__main__':
    # preds = [[0.4, 0.6], [0.7, 0.3], [0.8, 0.2]]
    # y_pred = [np.argmax(x) for x in preds]
    # create_and_plot_confusion_matrix([1, 1, 0], y_pred, ["Male", "Female"], normalize=False)

    # Char model plotting
    # char_paths = \
    #     [
    #         '../logs/character_level_classification/model_comp/16.05.2017_11:33:15_2x512LSTM_adam.txt',     # 2x512LSTM
    #         '../logs/character_level_classification/model_comp/16.05.2017_14:27:27_BiLSTM_adam.txt',        # BiLSTM
    #         '../logs/character_level_classification/model_comp/18.05.2017_17:47:45_Conv_BiLSTM_base.txt',        # Conv_BiLSTM
    #         '../logs/character_level_classification/model_comp/16.05.2017_07:05:29_2xConv_BiLSTM_adam.txt', # 2xConv_BiLSTM
    #         '../logs/character_level_classification/model_comp/20.05.2017_09:38:23_Conv_2xBiLSTM.txt'       # Conv_2xBiLSTM
    #     ]
    #
    # plot_models(char_paths, VAL_LOSS, save_path='../../images/experiments/char_model_base.png', title="Character model comparison")
    # plot_models(char_paths, TRAIN_LOSS, title="Character model comparison")



    ##REGULARIZATION AND DROPOUT##

    char_paths = \
        [
            '../logs/character_level_classification/Ablation/18.05.2017_17:47:45_Conv_BiLSTM_base.txt',  # Base
            '../logs/character_level_classification/Regularizer/23.05.2017_10:18:48_Conv_BiLSTM_l1.txt',      # Regularizer
            '../logs/character_level_classification/No dropout/22.05.2017_16:39:37_Conv_BiLSTM.txt'   # No dropout
         ]

    plot_models(char_paths, VAL_LOSS, save_path='../../images/experiments/char_model_comp.png', title="Character model comparison")

    # path = ['../logs/word_embedding_classification/BiLSTM/22.05.2017_16:37:14_BiLSTM.txt', '../logs/character_level_classification/Ablation/20.05.2017_21:30:39_Conv_BiLSTM_em_lower.txt']
    # plot_models(path, [VAL_LOSS, TRAIN_LOSS], title="Conv_BiLSTM training loss and validation loss")  #save_path="../../images/experiments/char_train_val_loss_.png"


    # Word model plotting
    # model_comp_path = '../logs/word_embedding_classification/model_comp'
    # word_paths = list(map(lambda file_name: os.path.join(model_comp_path, file_name), os.listdir(model_comp_path)))
    #
    # plot_models(word_paths, VAL_LOSS, save_path='../../images/experiments/word_model_base.png', title="Word model comparison")
    #



    print("")