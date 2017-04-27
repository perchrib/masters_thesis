import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from preprocessors.dataset_preparation import split_dataset
from helpers.global_constants import EMBEDDINGS_INDEX_DIR
from helpers.helper_functions import load_pickle
from word_embedding_classification.constants import *

np.random.seed(1337)


def format_dataset_word_level(texts, labels, metadata):
    """
    This is a sub-procedure.
    Format text samples and labels into tensors. Convert text samples into sequences of word indices.
    Split into training set and validation set
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :return:
    """
    print("------Formatting text samples into tensors...")

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # construct word index sequences of the texts

    word_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
    print('Found %s unique tokens / length of word index.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # zero-pad sequences that are too short
    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test = split_dataset(data, labels,
                                                                                                    metadata)

    return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test, word_index


def construct_embedding_matrix(word_index):
    """
    Prepare an "embedding matrix" which will contain at index i the embedding vector for the word of index i in our word index.
    :param word_index: dictionary mapping words (str) to their rank/index (int)

    :type word_index: dict
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(word_index) + 1, get_embedding_dim()))
    embeddings_index = load_pickle(os.path.join(EMBEDDINGS_INDEX_DIR, EMBEDDINGS_INDEX))

    number_of_missing_occurrences = 0
    total_number_of_words = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
        else:
            number_of_missing_occurrences += 1
        total_number_of_words += 1

    print("Total number of word occurrences: %i" % total_number_of_words)
    print('Number of missing word occurrences / words with no embedding: %i' % number_of_missing_occurrences)

    return embedding_matrix


def get_embedding_dim():
    if EMBEDDINGS_INDEX == 'glove.twitter.27B.200d':
        return 200
    elif EMBEDDINGS_INDEX == 'word_vectors_300' or EMBEDDINGS_INDEX == 'word_vectors_300_stop_words':
        return 300
    else:
        raise Exception("Embedding-index not specified in get_embedding_dim method")
