import sys
import os
import numpy as np
# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from word_level_classification.dataset_formatting import format_dataset_word_level, construct_embedding_matrix, get_embedding_dim
from word_level_classification.constants import *
from preprocessors.dataset_preparation import prepare_dataset
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from word_level_classification.models import SeqLSTM
from time import time

np.random.seed(1337)


def train():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset()
    x_train, y_train, meta_train, x_val, y_val, meta_val, word_index = format_dataset_word_level(texts, labels, metadata)

    embedding_layer = get_embedding_layer(word_index)
    model = get_model(embedding_layer, len(labels_index))
    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # TODO: Add intermediate model saving

    # Time
    start_time = time()

    print('\nCommence training %s model' % MODEL_NAME)
    print('Embeddings from: %s' % EMBEDDINGS_INDEX)
    model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              nb_epoch=NB_EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True,
              callbacks=[early_stopping],
              verbose=1)

    print('Training time: %i' % (time() - start_time))
    # TODO: Write and save log

def get_model(embedding_layer, nb_output_nodes):
    if MODEL_NAME == 'SeqLSTM':
        return SeqLSTM(embedding_layer, nb_output_nodes)


def get_embedding_layer(word_index):
    embedding_matrix = construct_embedding_matrix(word_index)
    return Embedding(len(word_index) + 1, get_embedding_dim(),
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)  # Not trainable to prevent weights from being updated during training


def write_and_save_log(save_dir=LOG_DIR, ):
    pass

train()