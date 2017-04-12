import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.models import get_char_model
from preprocessors.dataset_preparation import prepare_dataset
from keras.callbacks import EarlyStopping
from character_level_classification.constants \
    import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE, LOGS_DIR
from time import time
from helpers.helper_functions import log_session
import numpy as np

np.random.seed(1337)


def train():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset()
    x_train, y_train, meta_train, x_val, y_val, meta_val, char_index = format_dataset_char_level(texts, labels,
                                                                                                 metadata)

    model = get_char_model(len(labels_index))

    print("Model name: %s" % model.name)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    model.compile(optimizer=MODEL_OPTIMIZER,
                  loss=MODEL_LOSS,
                  metrics=MODEL_METRICS)

    # Time
    start_time = time()

    print('\nCommence training model')
    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        epochs=NB_EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[early_stopping]).history

    training_time = (time() - start_time) / 60
    print('Training time: %i' % training_time)
    log_session(LOGS_DIR, model, history, training_time, len(x_train), len(x_val), MODEL_OPTIMIZER, BATCH_SIZE,
                NB_EPOCHS)


if __name__ == '__main__':
    train()
