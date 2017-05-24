
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from character_level_classification.models import *

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset

import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from helpers.model_utils import get_model_checkpoint, save_trained_model, predict_and_get_precision_recall_f_score, save_term_index
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, PREDICTION_TYPE, LOGS_DIR, MODEL_DIR, CHAR_INDEX_DIR
from helpers.helper_functions import log_session, get_time_format
import numpy as np

np.random.seed(1337)


def train(model, model_info, data, save_model=False, extra_info=None, log_sess=True):
    """
    Train a given character model with given data
    :type model: Model

    :param model:
    :param model_info:
    :param data:
    :param save_model:
    :param extra_info:
    :param log_sess:
    :return:
    """

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    callbacks = [early_stopping]

    # if save_model:
    #     callbacks.append(get_model_checkpoint(model.name, MODEL_DIR, MODEL_OPTIMIZER))

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

    # Compute prf for val set
    prf_val = predict_and_get_precision_recall_f_score(model, data['x_val'], data['y_val'], PREDICTION_TYPE)

    # Evaluate on test set, if supplied
    if 'x_test' in data:
        test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)
        prf_test = predict_and_get_precision_recall_f_score(model, data['x_test'], data['y_test'], PREDICTION_TYPE)
        num_test = len(data['x_test'])
    else:
        test_results = None
        num_test = 0
        prf_test = None

    if log_sess:
        log_session(log_dir=LOGS_DIR,
                    model=model,
                    history=history,
                    training_time=training_time,
                    num_train=len(data['x_train']),
                    num_val=len(data['x_val']),
                    num_test=num_test,
                    optimizer=MODEL_OPTIMIZER,
                    batch_size=BATCH_SIZE,
                    max_epochs=NB_EPOCHS,
                    prf_val=prf_val,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    test_results=test_results,
                    model_info=model_info,
                    extra_info=extra_info,
                    prf_test=prf_test)

    if save_model:
        save_trained_model(model, MODEL_DIR, MODEL_OPTIMIZER)
        save_term_index(data['char_index'], model.name, CHAR_INDEX_DIR)