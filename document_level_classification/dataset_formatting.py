from __future__ import print_function
from keras.utils import to_categorical

from document_level_classification.features import TF_IDF
from constants import MAX_FEATURE_LENGTH, N_GRAM, DIM_REDUCTION
from preprocessors.dataset_preparation import split_dataset
import time
from helpers.helper_functions import get_time_format
from sklearn.decomposition import SparsePCA
from helpers.dimension_reduction import DimReduction

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

    # create vocabulary for n words!!!

    start = time.time()

    tfidf = TF_IDF(x_train, y_train, MAX_FEATURE_LENGTH, N_GRAM)
    x_train = tfidf.fit_to_training_data()

    print(get_time_format(time.time()-start))
    x_val = tfidf.fit_to_new_data(x_val)
    x_test = tfidf.fit_to_new_data(x_test)

    start = time.time()
    # PCA reduction
    # print("Starting With PCA Reduction...")
    # pca = SparsePCA(n_components=50, verbose=True)
    # print("Transforming x_train")
    # x_train = pca.fit_transform(x_train)
    # print("Print first index.........")
    # for val in x_train[0]:
    #     print(val)
    # print("Transforming x_test")
    # x_test = pca.fit_transform(x_test)
    #
    # print("Transforming x_val")
    # x_val = pca.fit_transform(x_val)
    if DIM_REDUCTION:
        print("Starting With Dimensonality Reduction...")
        dr = DimReduction(1000)
        x_train = dr.fit_transform(x_train, x_val)
        x_val = dr.fit_transform(x_val)
        x_test = dr.fit_transform(x_test)
        #print("Reduction Time: ", get_time_format(time.time()-start))
        #for x in x_train[:3]:
        #    print("Length: ", len(x))
        #    print(x)
        #    print("")

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test


