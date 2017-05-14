from __future__ import print_function
from keras.utils import to_categorical

from document_level_classification.features import TF_IDF, BOW
from constants import MAX_FEATURE_LENGTH, N_GRAM, DIM_REDUCTION, \
    DIM_REDUCTION_SIZE, CATEGORICAL, FEATURE_MODEL, C_BAG_OF_WORDS, C_TF_IDF
from preprocessors.dataset_preparation import split_dataset
import time
from helpers.helper_functions import get_time_format
from sklearn.decomposition import SparsePCA
from helpers.dimension_reduction import DimReduction
import numpy as np
#
from preprocessors.dataset_preparation import prepare_dataset
from preprocessors.parser import Parser

def format_dataset_doc_level(texts, labels, metadata, is_test=False, feature_model=None, reduction_model=None):
    """
    Split into training set, validation and test set. It also transform the text into doc_level features ie TFIDF
     POS-Tags etc
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :return:
    """

    if not is_test:
        x_train, y_train, meta_train, x_val, y_val, meta_val = split_dataset(texts, labels, metadata, data_type_is_string=True)

        if FEATURE_MODEL == C_TF_IDF:
            feature_model = TF_IDF(x_train, y_train, MAX_FEATURE_LENGTH, N_GRAM)
        elif FEATURE_MODEL == C_BAG_OF_WORDS:
            feature_model = BOW(x_train, N_GRAM, max_features=MAX_FEATURE_LENGTH)

        start = time.time()

        x_train = feature_model.fit_to_training_data()
        x_val = feature_model.fit_to_new_data(x_val)

        if DIM_REDUCTION:
            start = time.time()
            print("Starting With Dimensionality Reduction From Size %i to %i..." % (x_train.shape[1], DIM_REDUCTION_SIZE))
            reduction_model = DimReduction(DIM_REDUCTION_SIZE, train=True)
            x_train = reduction_model.fit_transform(x_train, x_val)
            x_val = reduction_model.fit_transform(x_val)

            print("Reduction Time: ", get_time_format(time.time() - start))
            # for x in x_train[:3]:
            #     print("Length: ", len(x))
            #     print(x)
            #     print("")

        if CATEGORICAL:
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)

        else:
            y_train = np.asarray([[i] for i in y_train])
            y_val = np.asarray([[i] for i in y_val])

        return x_train, y_train, meta_train, x_val, y_val, meta_val, feature_model, reduction_model

    elif is_test:
        x_test = feature_model.fit_to_new_data(texts)

        if DIM_REDUCTION:
            x_test = reduction_model.fit_transform(x_test)

        if CATEGORICAL:
            y_test = to_categorical(labels)

        else:
            y_test = np.asarray([[i] for i in labels])

        return x_test, y_test, metadata,


