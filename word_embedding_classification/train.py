from __future__ import print_function
import os
import sys

import numpy as np

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from word_embedding_classification.dataset_formatting import format_dataset_word_level, construct_embedding_matrix, \
    get_embedding_dim
from word_embedding_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, \
    EMBEDDINGS_INDEX, MAX_SEQUENCE_LENGTH, LOGS_DIR, MODEL_DIR, WORD_INDEX_DIR
from helpers.model_utils import get_model_checkpoint, save_trained_model, save_term_index
from word_embedding_classification.models import *
from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from time import time, strftime
from helpers.helper_functions import log_session
from helpers.model_utils import get_precision_recall_f_score

np.random.seed(1337)


def train(model, model_info, data, save_model=False, extra_info=None):

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)  # TODO: Patience is 1
    callbacks = [early_stopping]

    # if save_model:
    #     callbacks.append(get_model_checkpoint(model.name, MODEL_DIR, MODEL_OPTIMIZER))

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

    # Compute prf for val set
    prf_val = get_precision_recall_f_score(model, data['x_val'], data['y_val'], PREDICTION_TYPE)

    if 'x_test' in data:
        # Evaluate on test set
        test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=BATCH_SIZE)
        prf_test = get_precision_recall_f_score(model, data['x_test'], data['y_test'], PREDICTION_TYPE)
        num_test = len(data['x_test'])
    else:
        test_results = None
        num_test = 0
        prf_test = None

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
        save_term_index(data['word_index'], model.name, WORD_INDEX_DIR)


def get_embedding_layer(word_index):
    embedding_matrix = construct_embedding_matrix(word_index)
    return Embedding(len(word_index) + 1, get_embedding_dim(),
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
                     # mask_zero=True)  # Ignore unknown words
    # Not trainable to prevent weights from being updated during training

