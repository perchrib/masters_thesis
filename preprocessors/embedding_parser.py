import os
from helpers.helper_functions import save_pickle
from word_level_classification.constants import *
import numpy as np


def parse_glove(filename='glove.twitter.27B.200d.txt', save_file=True):
    """
    Parse glove text file and create dictionary of word embeddings (embeddings_index)
    :param filename: name of glove file. Should be in embeddings_native directory
    :param save_file: boolean. True if dictionary should be saved to pickle file
    :return: embeddings_index
    """
    embeddings_index = {}

    if not os.path.isdir(EMBEDDINGS_NATIVE_DIR):
        os.makedirs(EMBEDDINGS_NATIVE_DIR)
        print(EMBEDDINGS_NATIVE_DIR, " directory created")

    with open(os.path.join(EMBEDDINGS_NATIVE_DIR, filename), 'r') as glove_file:
        print("Reading Glove File...")
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    if save_file:
        if not os.path.isdir(EMBEDDINGS_INDEX_DIR):
            os.makedirs(EMBEDDINGS_INDEX_DIR)
            print(EMBEDDINGS_INDEX_DIR, " directory created")

        save_pickle(os.path.join(EMBEDDINGS_INDEX_DIR, filename.strip('.txt')), embeddings_index)

parse_glove()