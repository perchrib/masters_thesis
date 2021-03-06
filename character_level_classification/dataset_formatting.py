from __future__ import print_function
import os
from functools import reduce

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from nltk import sent_tokenize

from preprocessors.dataset_preparation import split_dataset
from character_level_classification.constants import *
from helpers.global_constants import VALIDATION_SPLIT


def format_dataset_char_level(texts, labels, metadata, trained_char_index=None):
    """
    Format dataset into char indices for one hot encodings.
    Split into training set, validation set and test set.
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :param trained_char_index: whether or not to split training data. Should be false for test data
    :return:
    """

    if not trained_char_index:
        print('\n-------Creating character set...')
        all_text = ''.join(texts)

        chars = set(all_text)
        print("CHARACTER SET:", chars)
        print('Total Chars: %i' % len(chars))
        char_index = dict((char, i) for i, char in enumerate(chars))
    else:
        char_index = trained_char_index

    # Matrix of -1 because char_index can be both 0 and -1
    formatted_data = np.ones((len(texts), MAX_SEQUENCE_LENGTH), dtype=np.int64) * -1

    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', formatted_data.shape)
    print('Shape of label tensor:', labels.shape)

    for i, tweet in enumerate(texts):
        for j, char in enumerate(tweet):
            if j < MAX_SEQUENCE_LENGTH:
                if char in char_index:
                    formatted_data[i, j] = char_index[char]

    # Training data
    if not trained_char_index:
        # split the data into a training set and a validation set
        x_train, y_train, meta_train, x_val, y_val, meta_val = split_dataset(formatted_data, labels, metadata)

        all_data = {
            'x_train': x_train,
            'y_train': y_train,
            'meta_train': meta_train,
            'x_val': x_val,
            'y_val': y_val,
            'meta_val': meta_val,
            'char_index': char_index
        }

        return all_data

    # Test data
    else:
        return formatted_data, labels





