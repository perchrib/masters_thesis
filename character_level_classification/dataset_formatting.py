import os
from functools import reduce

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from character_level_classification.constants import *

np.random.seed(1337)


def format_dataset_char_level(texts, labels, metadata):
    """
    This is a sub-procedure.
    Split into training set and validation set
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :return:
    """

    print('\n-------Creating character set...')
    all_text = ''.join(texts)

    chars = set(all_text)
    print('Total Chars: %i' % len(chars))
    char_index = dict((char, i) for i, char in enumerate(chars))

    # TODO: Why -1?
    data = np.ones((len(texts), MAX_SEQUENCE_LENGTH), dtype=np.int64) * -1
    # data = np.zeros((len(texts), MAX_SEQUENCE_LENGTH), dtype=np.int64)
    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    #TODO: Try chars reversed
    for i, tweet in enumerate(texts):
        for j, char in enumerate(tweet):
            if j < MAX_SEQUENCE_LENGTH:
                data[i, j] = char_index[char]

    # shuffle and split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    metadata = [metadata[i] for i in indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    meta_train = metadata[:-nb_validation_samples]

    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    meta_val = metadata[-nb_validation_samples:]

    return x_train, y_train, meta_train, x_val, y_val, meta_val, char_index

