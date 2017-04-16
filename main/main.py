import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from word_level_classification.ann import train as w_train
from character_level_classification.ann import train as c_train

from word_level_classification.constants import MODELS as word_models
from character_level_classification.constants import MODELS as char_models

if __name__ == '__main__':
    if word_models:
        for model in word_models:
            w_train(model, [])

    if char_models:
        for model in char_models:
            c_train(model, ["No consume_less"])

