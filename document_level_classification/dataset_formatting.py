import numpy as np
from keras.utils import to_categorical

from document_level_classification.features import TF_IDF
from constants import MAX_FEATURE_LENGTH
from preprocessors.dataset_preparation import split_dataset


def format_dataset_doc_level(texts, labels, metadata):
    """
    Split into training set, validation and test set. It also transform the text into doc_level features ie TFIDF
     POS-Tags etc
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :return:
    """

    x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test = split_dataset(texts,
                                                                                                    labels,
                                                                                                    metadata,
                                                                                                    data_type_is_string=True)



    tfidf = TF_IDF(x_train, MAX_FEATURE_LENGTH)
    x_train = tfidf.fit_to_training_data()
    x_test = tfidf.fit_to_new_data(x_test)
    x_val = tfidf.fit_to_new_data(x_val)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test


