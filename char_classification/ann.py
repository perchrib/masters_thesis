import sys
import os
# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from char_classification.text_preprocessing import prepare_dataset
from char_classification.models import get_char_model, get_char_model_2
from keras.callbacks import EarlyStopping
from char_classification.constants import MODEL_OPTIMIZER, MODEL_LOSS, MODEL_METRICS, NB_EPOCHS, BATCH_SIZE
from time import time
import numpy as np
np.random.seed(1337)

def train():
    # Load dataset
    x_train, y_train, meta_train, x_val, y_val, meta_val, char_index, labels_index = prepare_dataset()

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
    model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              nb_epoch=NB_EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True,
              callbacks=[early_stopping])

    print('Training time: %i' % (time() - start_time))


train()