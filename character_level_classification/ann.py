from __future__ import print_function
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.models import *
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from keras.callbacks import EarlyStopping
from character_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    PREDICTION_TYPE, LOGS_DIR
from time import time
from helpers.helper_functions import log_session
import numpy as np

np.random.seed(1337)


def train(model, model_info, data, extra_info=None):

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Time
    start_time = time()

    print('\nCommence training %s model' % model.name)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping]).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)
    log_session(LOGS_DIR, model, history, training_time, len(data['x_train']), len(data['x_val']), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH, model_info, extra_info)
