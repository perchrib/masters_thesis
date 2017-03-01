import os
from helpers.helper_functions import load_pickle
from premaster_work.constants import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np

np.random.seed(1337)


def prepare_dataset_word_level(folder_path=TEXT_DATA_DIR):
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
    for sub_folder_name in sorted(os.listdir(folder_path)):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        for file_name in sorted(os.listdir(sub_folder_path)):
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(sub_folder_path, file_name)

                with open(file_path, 'r') as txt_file:
                    data_samples = [line.strip() for line in txt_file]

                author_data = data_samples.pop(0).split(':::')  # ID, gender and age of author

                # Remaining lines correspond to the tweets by the author
                for tweet in data_samples:
                    texts.append(tweet)
                    metadata.append({GENDER: author_data[1], AGE: author_data[2]})
                    labels.append(labels_index[metadata[-1][PREDICTION_TYPE]])

    print('Found %s texts.' % len(texts))

    x_train, y_train, meta_train, x_val, y_val, meta_val, word_index = format_dataset(texts, labels, metadata)

    return x_train, y_train, meta_train, x_val, y_val, meta_val, word_index, labels_index


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


def format_dataset(texts, labels, metadata):
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

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # construct word index sequences of the texts

    word_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
    print('Found %s unique tokens / length of word index.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # zero-pad sequences that are too short
    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

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

    return x_train, y_train, meta_train, x_val, y_val, meta_val, word_index


def construct_embedding_matrix(word_index):
    """
    Prepare an "embedding matrix" which will contain at index i the embedding vector for the word of index i in our word index.
    :param word_index: dictionary mapping words (str) to their rank/index (int)

    :type word_index: dict
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(word_index) + 1, get_embedding_dim()))
    embeddings_index = load_pickle(os.path.join(EMBEDDINGS_INDEX_DIR, EMBEDDINGS_INDEX))
    print(len(embeddings_index))

    number_of_missing_occurrences = 0
    total_number_of_words = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            number_of_missing_occurrences += 1
        total_number_of_words += 1

    print("Total number of word occurences: %i" % total_number_of_words)
    print('Number of missing word occurences: %i' % number_of_missing_occurrences)

    return embedding_matrix


def get_embedding_dim():
    if EMBEDDINGS_INDEX == 'glove.twitter.27B.200d':
        return 200
    elif EMBEDDINGS_INDEX == 'word_vectors_300' or EMBEDDINGS_INDEX == 'word_vectors_300_stop_words':
        return 300
    else:
        raise Exception("Embedding-index not specified in get_embedding_dim method")
