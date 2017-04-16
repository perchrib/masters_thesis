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


def train(model_name, extra_info=None):
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)

    x_train, y_train, meta_train, x_val, y_val, meta_val, char_index = format_dataset_char_level(texts, labels,
                                                                                                 metadata)

    model = get_model(model_name, len(labels_index))

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Time
    start_time = time()

    print('\nCommence training %s model' % model.name)
    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping]).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)
    log_session(LOGS_DIR, model, history, training_time, len(x_train), len(x_val), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS, MAX_SEQUENCE_LENGTH, extra_info)


def get_model(model_name, num_output_nodes):
    if model_name == '3xConv_2xBiLSTM':
        return get_char_model_3xConv_2xBiLSTM(num_output_nodes)
    elif model_name == '2x512_256LSTM':
        return get_model_2x512_256_lstm(num_output_nodes)
    elif model_name == 'BiLSTM_full':
        return get_char_model_BiLSTM_full(num_output_nodes)


if __name__ == '__main__':
    train("BiLSTM_full")