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


def train(model_name, extra_info=None):
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)

    x_train, y_train, meta_train, x_val, y_val, meta_val, word_index = format_dataset_word_level(texts, labels,
                                                                                                 metadata)

    embedding_layer = get_embedding_layer(word_index)
    model = get_model(model_name, embedding_layer, len(labels_index))

    if model is None:
        print("Error: Model is none. Breaking execution")
        return

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
    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping],
                        verbose=1).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)

    log_session(LOGS_DIR, model, history, training_time, len(x_train), len(x_val), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH, extra_info)


def get_model(model_name, embedding_layer, num_output_nodes):
    if model_name == '3xSimpleLSTM':
        return get_word_model_3xsimple_lstm(embedding_layer, num_output_nodes)
    elif model_name == '2x512_256LSTM':
        return get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes)
    elif model_name == '3x512_LSTM':
        return get_word_model_3x512_lstm(embedding_layer, num_output_nodes)
    elif model_name == '3x512_recDropoutLSTM':
        return get_word_model_3x512_rec_dropout_lstm(embedding_layer, num_output_nodes)
    elif model_name == '2x1024_512_LSTM':
        return get_word_model_2x1024_512_lstm(embedding_layer, num_output_nodes)
    elif model_name == '2x512_256LSTM_128full':
        return get_word_model_2x512_256_lstm_128_full(embedding_layer, num_output_nodes)
    elif model_name == '2x512_256GRU':
        return get_word_model_2x512_256_gru(embedding_layer, num_output_nodes)
    else:
        print("Error: Invalid model name")
        return None


def get_embedding_layer(word_index):
    embedding_matrix = construct_embedding_matrix(word_index)
    return Embedding(len(word_index) + 1, get_embedding_dim(),
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)  # Not trainable to prevent weights from being updated during training


