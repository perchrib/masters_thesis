from __future__ import print_function
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.models import *
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, PREDICTION_TYPE, LOGS_DIR, MODEL_DIR

import time
from helpers.helper_functions import log_session, get_time_format, get_model_checkpoint
import numpy as np

np.random.seed(1337)


def train(model, model_info, data, save_model=False, extra_info=None):
    """

    :type model: Model

    :param model:
    :param model_info:
    :param data:
    :param extra_info:
    :return:
    """

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    callbacks = [early_stopping]

    if save_model:
        if not os.path.exists(MODEL_DIR):
            os.makedirs(os.path.join(MODEL_DIR, model.name))
        model_file_name = time.strftime("%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + MODEL_OPTIMIZER + "_{epoch:02d}_{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, model_file_name), save_best_only=True)
        callbacks.append(get_model_checkpoint(model.name, MODEL_DIR, MODEL_OPTIMIZER))

    # Time
    start_time = time.time()

    print('\nCommence training %s model' % model.name)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks).history

    seconds = time.time() - start_time
    training_time = get_time_format(seconds)

    print("Training time: %s" % training_time)

    # Evaluate on test set
    test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)

    log_session(LOGS_DIR, model, history, training_time, len(data['x_train']), len(data['x_val']), len(data['x_test']), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH,test_results, model_info, extra_info)
