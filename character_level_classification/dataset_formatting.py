import os
from functools import reduce

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from nltk import sent_tokenize

from preprocessors.dataset_preparation import split_dataset
from character_level_classification.constants import *
from helpers.global_constants import VALIDATION_SPLIT

np.random.seed(1337)


def format_dataset_char_level(texts, labels, metadata):
    """
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

    # Matrix of -1 because char_index can be both 0 and -1
    data = np.ones((len(texts), MAX_SEQUENCE_LENGTH), dtype=np.int64) * -1

    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    #TODO: Try chars reversed
    for i, tweet in enumerate(texts):
        for j, char in enumerate(tweet):
            if j < MAX_SEQUENCE_LENGTH:
                data[i, j] = char_index[char]
                # data[i, MAX_SEQUENCE_LENGTH-1-j] = char_index[char]  # Input reversed

    # split the data into a training set and a validation set
    x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test = split_dataset(data, labels, metadata)

    return x_train, y_train, meta_train, x_val, y_val, meta_val,  x_test, y_test, meta_test, char_index


# PUT ON HOLD -- METHOD FOR SPLITTING TWEETS IN SENTENCES FOR ENCODING, BUT TWEETS CONTAIN ON AVERAGE 1.3 SENTENCES

def format_dataset_char_level_sentences(texts, labels, metadata):
    """
    Split into training set and validation set.
    Also splits each tweet into sentences to encode the sentences by themselves.
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

    data = np.ones((len(texts), MAX_SENTENCE_LENGTH, MAX_CHAR_SENT_LENGTH), dtype=np.int64) * -1

    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    #TODO: Try chars reversed
    for i, tweet in enumerate(texts):
        sentences = sent_tokenize(tweet)
        for j, sent in enumerate(sentences):
            if j < MAX_SENTENCE_LENGTH:
                for k, char in enumerate(sent[:MAX_CHAR_SENT_LENGTH]):
                    # data[i, j] = char_index[char]
                    data[i, j, MAX_CHAR_SENT_LENGTH-1-k] = char_index[char]  # Input reversed

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
