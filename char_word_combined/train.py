from __future__ import print_function
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from char_word_combined.constants import MAX_CHAR_SEQUENCE_LENGTH, MAX_WORD_SEQUENCE_LENGTH

import time
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from helpers.model_utils import get_model_checkpoint, save_trained_model
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, PREDICTION_TYPE, LOGS_DIR, MODEL_DIR
from helpers.helper_functions import log_session, get_time_format
import numpy as np

np.random.seed(1337)


def train(model, model_info, c_data, w_data, save_model=False, extra_info=None, log_sess=True):
    """

    :param model:
    :param model_info:
    :param c_data:
    :param save_model:
    :param extra_info:
    :param log_sess:
    
    :type model: Model
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
    history = model.fit({'c_input': c_data['x_train'], 'w_input': w_data['x_train']}, c_data['y_train'],
                        validation_data=[{'c_input': c_data['x_val'], 'w_input': w_data['x_val']}, c_data['y_val']],
                        # validation_data=[[c_data['x_val'], w_data['x_val']], [c_data['y_val'], w_data['y_val']]],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks).history

    seconds = time.time() - start_time
    training_time = get_time_format(seconds)

    print("Training time: %s" % training_time)

    # Evaluate on test set
    test_results = model.evaluate([c_data['x_train'], w_data['x_train']], [c_data['y_train'], w_data['y_train']], batch_size=BATCH_SIZE)

    if log_sess:
        log_session(log_dir=LOGS_DIR,
                    model=model,
                    history=history,
                    training_time=training_time,
                    num_train=len(c_data['x_train']),
                    num_val=len(c_data['x_val']),
                    num_test=len(c_data['x_test']),
                    optimizer=MODEL_OPTIMIZER,
                    batch_size=BATCH_SIZE,
                    max_epochs=NB_EPOCHS,
                    max_sequence_length="(Char:%i, Word:%i)" % (MAX_CHAR_SEQUENCE_LENGTH, MAX_WORD_SEQUENCE_LENGTH),
                    test_results=test_results,
                    model_info=model_info,
                    extra_info=extra_info)

    if save_model:
        save_trained_model(model, MODEL_DIR, MODEL_OPTIMIZER)