from __future__ import print_function
import os
import time

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def get_model_checkpoint(model_name, model_dir, model_optimizer):
    if not os.path.exists(model_dir):
        os.makedirs(os.path.join(model_dir, model_name))

    model_file_name = time.strftime(
        "%d.%m.%Y_%H:%M:%S") + "_" + model_name + "_" + model_optimizer + "_{epoch:02d}_{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(os.path.join(model_dir, model_name, model_file_name), save_best_only=True)

    return checkpoint


def load_and_evaluate(model_path, data, batch_size):

    model = load_model(model_path)
    test_results = model.evaluate(data['x_test'], data['y_test'], batch_size=batch_size)

    print("\n--------------Test results---------------")
    print("%s: %f" % (model.metrics_names[0], round(test_results[0], 5)))
    print("%s: %f" % (model.metrics_names[1], round(test_results[1], 5)))

