from keras_diagram import ascii
import pickle
import sys
import os
import time
import pandas as pd
import yaml
import numpy as np

def get_time_format(seconds):
    """
    
    :param seconds: int seconds 
    :return: a string with time format "hours:minutes:seconds"
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    training_time = "%dh:%02dm:%02ds" % (h, m, s)
    return training_time


def shuffle(x_input, y_label):
    """shuffles a texts of list with the given labels"""
    if len(x_input) != len(y_label):
        raise TypeError("Not Same Length")
    else:
        x_input, y_label = np.asarray(x_input), np.asarray(y_label)
        indices = np.arange(len(x_input))

        texts_and_indices = list(zip(x_input, indices))
        np.random.seed(1337)
        np.random.shuffle(texts_and_indices)
        x_input, indices = zip(*texts_and_indices)
        x_input, indices = np.asarray(x_input), np.asarray(indices)
        y_label = y_label[indices]
        return x_input, y_label, indices


def save_pickle(file_path, data):
    """
    Save data to a pickle file
    :param file_path: path of save location with .pkl omitted
    :param data: data to be saved
    """
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path, feedback=False):
    """
    Load data from pickle file
    :param file_path: path to .pkl file
    :param feedback: if True print messages indicating when loading starts and finishes
    :return: data from pickle file
    """
    if feedback:
        print('Loading %s', file_path)

    with open(file_path + '.pkl', 'rb') as f:
        data = pickle.load(f)

    if feedback:
        print('Done loading')

    return data


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f)


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    source: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = 'X' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def log_session(log_dir, model, history, training_time, num_train, num_val, optimizer, batch_size, max_epochs, prf_val,
                test_results=None, model_info=None, extra_info=None, num_test=0, max_sequence_length=None, prf_test=None):

    if not os.path.exists((os.path.join(log_dir, model.name))):
        os.makedirs((os.path.join(log_dir, model.name)))

    file_name = time.strftime("%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + optimizer + ".txt"

    print("Writing log file - %s..." % file_name)

    with open(os.path.join(log_dir, model.name, file_name), 'wb') as log_file:
        log_file.write("Training_log - %s" % time.strftime("%d/%m/%Y %H:%M:%S"))

        log_file.write("\n\nModel name: %s" % model.name)
        log_file.write("\nElapsed training time: %s" % training_time)

        log_file.write("\n\nTraining set size: %i" % num_train)
        log_file.write("\nValidation set size: %i" % num_val)

        total_dataset_size = num_train + num_val + num_test
        val_frac = float(num_val) / total_dataset_size  # Fraction of dataset used for validation
        log_file.write("\nValidation set fraction: %f" % val_frac)
        test_frac = float(num_test) / total_dataset_size
        log_file.write("\nTest set fraction: %f" % test_frac)

        log_file.write("\n\nHyperparameters\n=========================================")
        log_file.write("\nOptimizer: %s" % optimizer)
        log_file.write("\nBatch size: %i" % batch_size)
        log_file.write("\nMax number of epochs: %i" % max_epochs)

        if max_sequence_length:
            log_file.write("\nMax sequence length: %s" % max_sequence_length)

        # Write accuracies for training and validation set from callback history
        log_file.write("\n\n-----------Training statistics-----------\n")
        log_file.write(pd.DataFrame(history).__repr__())

        if prf_val:
            log_file.write("\n\nValidation PRF\n")
            prf_val_df = pd.DataFrame(data=prf_val, index=pd.Index(["Precision", "Recall", "F-score", "Support"]))
            log_file.write(pd.DataFrame(prf_val_df).__repr__())

        # Write Test results
        if test_results:
            log_file.write("\n\n--------------Test results---------------\n")
            log_file.write("%s: %f" % (model.metrics_names[0], round(test_results[0], 5)))  # loss
            log_file.write(" %s: %f" % (model.metrics_names[1], round(test_results[1], 5)))  # accuracy

        if prf_test:
            log_file.write("\n\nTest PRF\n")
            prf_test_df = pd.DataFrame(data=prf_test, index=pd.Index(["Precision", "Recall", "F-score", "Support"]))
            log_file.write(pd.DataFrame(prf_test_df).__repr__())

        # Write model diagram
        log_file.write("\n\n--------------Model Diagram---------------\n")
        log_file.write(ascii(model))
        log_file.write("\n")

        if model_info:
            log_file.write("\nModel information:\n=========================================")
            for info in model_info:
                log_file.write("\n %s" % info)

        if extra_info:
            log_file.write("\nExtra information:\n=========================================")
            for info in extra_info:
                log_file.write("\n %s" % info)

    print("Done")


def remove_texts_shorter_than_threshold(texts, labels, metadata, threshold=2):
    """
    Remove texts shorter than threshold from list of texts, labels and metadata
    :param modified_texts:
    :param modified_labels:
    :param modified_metadata:
    :param threshold:
    :return:
    """
    removal_count = 0

    modified_texts = []
    modified_labels = []
    modified_metadata = []

    for i in range(len(texts)):
        if len(texts[i]) >= threshold:
            modified_texts.append(texts[i])
            modified_labels.append(labels[i])
            modified_metadata.append(metadata[i])

            removal_count += 1

    print("Removed %i tweets" % removal_count)

    return modified_texts, modified_labels, modified_metadata


def sum_col(array):
    sum_ = []
    row_dim = array.shape[0]
    col_dim = array.shape[1]
    for i in range(col_dim):
        sum_i = 0
        for j in range(row_dim):
            sum_i += array[j, i]
        sum_.append(sum_i)
    return np.asarray(sum_)

