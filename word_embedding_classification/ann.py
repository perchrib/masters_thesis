from __future__ import print_function
import os
import sys

import numpy as np

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from word_embedding_classification.dataset_formatting import format_dataset_word_level, construct_embedding_matrix, \
    get_embedding_dim
from word_embedding_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    EMBEDDINGS_INDEX, MAX_SEQUENCE_LENGTH, LOGS_DIR
from word_embedding_classification.models import *
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import time, strftime
from helpers.helper_functions import log_session

np.random.seed(1337)


def train(model, model_info, data, save_model=False, extra_info=None):

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    callbacks = [early_stopping]

    if save_model:
        weights_file_path = strftime("%d.%m.%Y_%H:%M:%S") + "_" + model.name + "_" + MODEL_OPTIMIZER + "_{epoch:02d}_{val_acc:.2f}" + ".txt"
        checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, weights_file_path), save_best_only=True)
        callbacks.append(checkpoint)

    # Time
    start_time = time()

    print('\nCommence training %s model' % model.name)
    print('Embeddings from: %s' % EMBEDDINGS_INDEX)
    history = model.fit(data['x_train'], data['y_train'],
                        validation_data=[data['x_val'], data['y_val']],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=1).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)

    # Evaluate on test set
    test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)

    log_session(LOGS_DIR, model, history, training_time, len(data['x_train']), len(data['x_val']), len(data['x_test']), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH, test_results, model_info, extra_info)


def get_embedding_layer(word_index):
    embedding_matrix = construct_embedding_matrix(word_index)
    return Embedding(len(word_index) + 1, get_embedding_dim(),
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)  # Not trainable to prevent weights from being updated during training


