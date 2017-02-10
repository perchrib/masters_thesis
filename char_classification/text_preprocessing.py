import os
from char_classification.helper_functions import load_pickle
from char_classification.constants import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from functools import reduce
import tensorflow as tf
import numpy as np


np.random.seed(1337)


def prepare_dataset(folder_path=TEXT_DATA_DIR):
    """
    Iterate over dataset folder and create sequences of word indices
    Expecting a directory of text files, one for each author. Each line in files corresponds to a tweet
    :return: results of format_dataset: training set, validation set, word_index
    """

    texts = []  # list of text samples
    labels_index = construct_labels_index(PREDICTION_TYPE)  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    metadata = []  # list of dictionaries with author information (age, gender)

    print("------Parsing txt files...")
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r') as txt_file:
                data_samples = [line.strip() for line in txt_file]

            author_data = data_samples.pop(0).split(':::')  # ID, gender and age of author

            # Remaining lines correspond to the tweets by the author
            for tweet in data_samples:
                texts.append(tweet)
                metadata.append({GENDER: author_data[1], AGE: author_data[2]})
                labels.append(labels_index[metadata[-1][PREDICTION_TYPE]])

    print('Found %s texts.' % len(texts))

    x_train, y_train, meta_train, x_val, y_val, meta_val, char_index = format_dataset_chars(texts, labels, metadata)

    return x_train, y_train, meta_train, x_val, y_val, meta_val, char_index, labels_index


def construct_labels_index(prediction_type):
    """
    Constuct appropriate dictionary mappings class labels to IDs
    :param prediction_type: constants.PREDICT_GENDER or constants.PREDICT_AGE
    :return: dictionary mapping label name to numeric ID
    """
    if prediction_type == GENDER:
        return {'MALE': 0, 'FEMALE': 1}
    elif prediction_type == AGE:
        return {'18-24': 0, '25-34': 1, '35-49': 2, '50-64': 3, '65-xx': 4}


def format_dataset_chars(texts, labels, metadata):
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


def display_dataset_statistics(texts):
    """
    Given a dataset as a list of texts, display statistics: Number of tweets, avg length of characters and tokens.
    :param texts: List of string texts
    """

    # Number of tokens per tweet
    tokens_all_texts = list(map(lambda tweet: tweet.split(" "), texts))
    avg_token_len = reduce(lambda total_len, tweet_tokens: total_len + len(tweet_tokens), tokens_all_texts, 0) / len(tokens_all_texts)

    # Number of characters per tweet
    char_length_all_texts = list(map(lambda tweet: len(tweet), texts))
    avg_char_len = reduce(lambda total_len, tweet_len: total_len + tweet_len, char_length_all_texts) / len(texts)

    print("Number of tweets: %i" % len(texts))
    print("Average number of tokens per tweet: %f" % avg_token_len)
    print("Average number of characters per tweet: %f" % avg_char_len)



def binarize(x, sz=71):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

if __name__ == "__main__":
    prepare_dataset()

