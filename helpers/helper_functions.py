from keras_diagram import ascii
import pickle
import sys
import os


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


# Print iterations progress
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
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


def log_session(model_ref, history, training_time, num_train, num_val, optimizer, batch_size, max_epochs):
    print("Saving log file")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    if not os.path.exists(os.path.join(LOGS_DIR, model_ref.name)):
        os.makedirs(os.path.join(LOGS_DIR, model_ref.name))

    # TODO: Construct filename from details
    file_name = time.strftime("%d.%m.%Y_%H:%M:%S") + "_" + model_ref.name +  ".txt"
    with open(os.path.join(LOGS_DIR, model_ref.name, file_name), 'wb') as log_file:
        log_file.write("Training_log - %s" % time.strftime("%d/%m/%Y %H:%M:%S"))

        log_file.write("\n\nModel name: %s" % model_ref.name)
        log_file.write("\nElapsed training time: %i minutes" % training_time)

        log_file.write("\n\nTraining set size: %i" % num_train)
        log_file.write("\nValidation set size: %i" % num_val)

        val_frac = float(num_val) / (num_train + num_val)  # Fraction of dataset used for validation
        log_file.write("\nValidation set fraction: %f" % val_frac)

        log_file.write("\n\nHyperparameters\n=========================================")
        log_file.write("\nOptimizer: %s" % optimizer)
        log_file.write("\nBatch size: %i" % batch_size)
        log_file.write("\nMax number of epochs: %i" % max_epochs)



        # Write accuracies for training and validation set from callback history
        log_file.write("\n\n-----------Training statistics-----------\n")
        log_file.write(pandas.DataFrame(history).__repr__())
        log_file.write("\n\n--------------Model Diagram---------------\n")
        log_file.write(ascii(model))