import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from helpers.helper_functions import save_pickle, load_pickle
from helpers.global_constants import EMBEDDINGS_INDEX_DIR, EMBEDDINGS_NATIVE_DIR
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

    if not os.path.isfile(os.path.join(EMBEDDINGS_NATIVE_DIR, filename)):
        print("Error: Embeddings file missing in embeddings_native directory")
        return

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


def save_embedding_dict_from_index(word_index, embedding_dict):
    """
    Given a word index dictionary of (word, index) (key,value) pairs,
    create a dictionary of (word, embedding) (key, value) pairs with words available in embedding_dict
    :param word_index:
    :param embedding_dict:
    :return:
    """

    pass

if __name__ == '__main__':
    # parse_glove()

    word_index_path = os.path.join(, '')
    embedding_index_path = os.path.join(EMBEDDINGS_NATIVE_DIR, 'glove.twitter.27B.200d')
    save_embedding_dict_from_index(word_index=load_pickle(word_index_path),
                                   embedding_dict=load_pickle(embedding_index_path)
                                   )