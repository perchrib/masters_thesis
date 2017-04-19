import os
import sys

import numpy as np

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from word_level_classification.dataset_formatting import format_dataset_word_level, construct_embedding_matrix, \
    get_embedding_dim
from word_level_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    EMBEDDINGS_INDEX, MAX_SEQUENCE_LENGTH, LOGS_DIR
from word_level_classification.models import *
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from time import time
from helpers.helper_functions import log_session

np.random.seed(1337)


def train(model, extra_info, data):

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # TODO: Add intermediate model saving

    # Time
    start_time = time()

    print('\nCommence training %s model' % model.name)
    print('Embeddings from: %s' % EMBEDDINGS_INDEX)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping],
                        verbose=1).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)

    log_session(LOGS_DIR, model, history, training_time, len(data['x_train']), len(data['x_val']), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH, extra_info)


def get_embedding_layer(word_index):
    embedding_matrix = construct_embedding_matrix(word_index)
    return Embedding(len(word_index) + 1, get_embedding_dim(),
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)  # Not trainable to prevent weights from being updated during training


