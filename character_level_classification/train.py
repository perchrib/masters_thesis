
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.models import *
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    PREDICTION_TYPE, LOGS_DIR, MODEL_DIR

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset

import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from helpers.model_utils import get_model_checkpoint, save_trained_model
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, PREDICTION_TYPE, LOGS_DIR, MODEL_DIR
from helpers.helper_functions import log_session, get_time_format
import numpy as np

np.random.seed(1337)


def train(model, model_info, data, save_model=False, extra_info=None, log_sess=True):
    """

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

    # Evaluate on test set
    test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)

    if log_sess:
        log_session(log_dir=LOGS_DIR,
                    model=model,
                    history=history,
                    training_time=training_time,
                    num_train=len(data['x_train']),
                    num_val=len(data['x_val']),
                    num_test=len(data['x_test']),
                    optimizer=MODEL_OPTIMIZER,
                    batch_size=BATCH_SIZE,
                    max_epochs=NB_EPOCHS,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                    test_results=test_results,
                    model_info=model_info,
                    extra_info=extra_info)

    if save_model:
        save_trained_model(model, MODEL_DIR, MODEL_OPTIMIZER)