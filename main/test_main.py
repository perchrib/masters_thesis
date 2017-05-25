from __future__ import print_function
from collections import Counter
import sys
import os
import random
import numpy as np

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset, filter_dataset
from helpers.global_constants import TEST_DATA_DIR, TRAIN_DATA_DIR, TEST, TRAIN, REM_PUNCTUATION, REM_STOPWORDS, \
    REM_EMOTICONS, LEMMATIZE, REM_INTERNET_TERMS, CHAR_MODEL, DOC_MODEL, WORD_MODEL, X_TEST, Y_TEST, LOWERCASE, \
    MAJORITY, AVERAGE_CONF, MAX_CONF

from keras.models import load_model

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.constants import PREDICTION_TYPE as c_PREDICTION_TYPE, MODEL_DIR as c_MODEL_DIR, \
    FILTERS as c_FILTERS

from word_embedding_classification.dataset_formatting import format_dataset_word_level
from word_embedding_classification.constants import PREDICTION_TYPE as w_PREDICTION_TYPE, MODEL_DIR as w_MODEL_DIR, \
    FILTERS as w_FILTERS

import keras.backend.tensorflow_backend as k_tf
from helpers.model_utils import load_and_evaluate, load_and_predict, get_precision_recall_f_score, get_prf_repr, \
    print_prf_scores, get_argmax_classes
from helpers.helper_functions import load_pickle

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE, FILTERS as d_FILTERS
from document_level_classification.models import get_ann_model, get_logistic_regression

from document_level_classification.dataset_formatting import format_dataset_doc_level


def pre_process_test_word(trained_word_index_path, specified_filters=None):
    print("\nPre-processing data for word model")
    filters = w_FILTERS if specified_filters is None else specified_filters

    # Load word index
    word_index = load_pickle(trained_word_index_path)

    # Load datasets
    test_texts, test_labels, test_metadata, _ = prepare_dataset(w_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    test_texts, test_labels, test_metadata, _ = filter_dataset(texts=test_texts,
                                                               labels=test_labels,
                                                               metadata=test_metadata,
                                                               filters=filters,
                                                               train_or_test=TEST)
    data = {}
    data[X_TEST], data[Y_TEST] = format_dataset_word_level(test_texts,
                                                           test_labels,
                                                           test_metadata,
                                                           trained_word_index=word_index)

    return data

    # return load_and_predict(model_path, data)


def pre_process_test_char(trained_char_index_path, specified_filters=None):
    print("\nPre-processing data for char model")
    filters = c_FILTERS if specified_filters is None else specified_filters

    # Load char index
    char_index = load_pickle(trained_char_index_path)

    test_texts, test_labels, test_metadata, _ = prepare_dataset(c_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    test_texts, test_labels, test_metadata, _ = \
        filter_dataset(texts=test_texts,
                       labels=test_labels,
                       metadata=test_metadata,
                       filters=filters,
                       train_or_test=TEST)

    data = {}
    data[X_TEST], data[Y_TEST] = format_dataset_char_level(test_texts, test_labels, test_metadata,
                                                           trained_char_index=char_index)

    return data


def pre_process_test_doc(vocabulary_path, specified_filters=None):
    # TODO: Implement appropriate document level pre-processing
    print("\nPre-processing data for doc model")
    filters = d_FILTERS if specified_filters is None else specified_filters

    feature_model = load_pickle(vocabulary_path)

    test_texts, test_labels, test_metadata, _ = prepare_dataset(DOC_PREDICTION_TYPE,
                                                                folder_path=TEST_DATA_DIR)

    test_texts, test_labels, test_metadata, _ = \
        filter_dataset(texts=test_texts,
                       labels=test_labels,
                       metadata=test_metadata,
                       filters=filters,
                       train_or_test=TEST)

    data = {}
    print("Format Dataset to Document Level")
    data['x_test'], data['y_test'], data['meta_test'] = format_dataset_doc_level(test_texts,
                                                                                 test_labels,
                                                                                 test_metadata,
                                                                                 is_test=True,
                                                                                 feature_model=feature_model)

    return data


def predict_stacked_model(model_paths, vocabularies, averaging_style, print_individual_prfs=False):
    """
    Use several models (char-, word-, doc-level) to predict as a stacked model
    :param model_paths: dictionary with key, value pairs of model name/type, model path
    :param vocabularies: corresponding word_index, char-index and bow vocabulary
    :param averaging_style: prediction averaging style. Can be:
        AVERAGE_CONF: Take the average of each system's confidence values and choose the class with highest average confidence
        MAJORITY: Each system votes on the max confidence class. The class with most votes is predicted
        MAX_CONF: Rely on the prediction of the system with highest confidence for each sample
    :type model_paths: dict
    :return: 
    """
    number_of_models = len(model_paths)
    print("Number of input models to stack: %i" % number_of_models)

    # Dictionary containing a dictionary for each sub-system, with appropriately formatted data vectors - dict[X_TEST], dict[Y_TEST]
    formatted_data = {
        WORD_MODEL: pre_process_test_word(trained_word_index_path=vocabularies[WORD_MODEL], specified_filters={
            REM_STOPWORDS: True,
            LEMMATIZE: False,
            REM_PUNCTUATION: True,
            REM_EMOTICONS: True
        }),
        CHAR_MODEL: pre_process_test_char(trained_char_index_path=vocabularies[CHAR_MODEL], specified_filters={
            REM_STOPWORDS: True,
            LEMMATIZE: False,
            REM_PUNCTUATION: False,
            REM_EMOTICONS: False,
            LOWERCASE: False
        }),
        DOC_MODEL: pre_process_test_doc(vocabulary_path=vocabularies[DOC_MODEL], specified_filters={
            REM_STOPWORDS: True,
            LEMMATIZE: False,
            REM_PUNCTUATION: False,
            REM_EMOTICONS: False,
        })
    }

    # TODO: Add d_data

    # Dictionary with predictions for each system. Predictions in categorical confidence form
    pred_dict_categorical = {}
    pred_dict = {}  # Dictionary with predictions for each system. Predictions with single class values
    loaded_models = {}  # TODO: May be removed if model is only used once in code sequence

    y_categorical = formatted_data.values()[0][Y_TEST]  # Categorical true labels - only used to get shape.
    y_true = get_argmax_classes(y_categorical)  # The correct test set labels - single class values

    # Load and predict with each model
    for name, path in model_paths.iteritems():
        pred_dict[name], pred_dict_categorical[name], loaded_models[name] = \
            load_and_predict(model_path=path, x_data=formatted_data[name][X_TEST], y_data=formatted_data[name][Y_TEST])

    if print_individual_prfs:
        for name, predictions in pred_dict.iteritems():
            print("\n---PRF for %s" % name)
            print_prf_scores(y_pred=predictions, y_true=y_true)

    print("Averaging predictions using: %s" % averaging_style)
    if averaging_style == AVERAGE_CONF:
        aggregated_preds = [[0 for _ in y_categorical[0]] for _ in y_categorical]

        # Summation
        for name, predictions in pred_dict_categorical.iteritems():
            for i in range(len(predictions)):  # For each sample
                for j in range(len(predictions[i])):  # For each confidence value
                    aggregated_preds[i][j] += predictions[i][j]

        # Average
        for sample in aggregated_preds:
            for i in range(len(sample)):  # For each confidence
                sample[i] /= float(number_of_models)

        aggregated_preds = get_argmax_classes(aggregated_preds)  # Single class values

    elif averaging_style == MAJORITY:
        votes = [[] for _ in range(len(y_true))]

        # Gather votes
        for name, predictions in pred_dict.iteritems():
            for i in range(len(predictions)):
                votes[i].append(predictions[i])

        # Count and predict
        aggregated_preds = []
        for sample in votes:
            aggregated_preds.append(Counter(sample).most_common(1)[0][0])

    elif averaging_style == MAX_CONF:
        all_confs = [[] for _ in y_true]

        # Gather all confidences
        for name, predictions in pred_dict_categorical.iteritems():
            for i in range(len(predictions)):  # For each sample
                all_confs[i].append(predictions[i])

        all_confs = np.asarray(all_confs)  # Numpy array for convenience in next step

        # Find highest confidence value and the corresponding gender
        aggregated_preds = []
        for confs in all_confs:
            max_confs = np.amax(confs, axis=1)  # Each system's max conf
            system_index = np.argmax(max_confs)  # Index of system with highest conf value
            aggregated_preds.append(np.argmax(confs[system_index]))  # Append gender index


    else:
        raise Exception("Invalid averaging style. Should be MAX_VOTE or AVERAGE_CONF")

    # PRF - Stacked model
    print_prf_scores(y_pred=aggregated_preds, y_true=y_true)


if __name__ == '__main__':
    # model_path = '../models/word_embedding_classification/BiLSTM/23.05.2017_18:12:26_BiLSTM_punct_em.h5'
    # w_data = pre_process_test_word(trained_word_index_path='../models/word_embedding_classification/word_index/23.05.2017_18:12:28_BiLSTM_punct_em.pkl')
    #

    # model_path = "../models/character_level_classification/Conv_BiLSTM/23.05.2017_05:36:06_Conv_BiLSTM_no_lower.h5"
    # c_data = pre_process_test_char(
    #     "../models/character_level_classification/char_index/23.05.2017_05:36:06_Conv_BiLSTM_no_lower.pkl",
    #     {
    #         REM_STOPWORDS: True,
    #         LEMMATIZE: False,
    #         REM_PUNCTUATION: False,
    #         REM_EMOTICONS: False,
    #         LOWERCASE: False
    #     })

    # load_and_evaluate(model_path, c_data)

    predict_stacked_model(
        model_paths={
            WORD_MODEL: '../models/word_embedding_classification/BiLSTM/23.05.2017_18:12:26_BiLSTM_punct_em.h5',
            CHAR_MODEL: '../models/character_level_classification/Conv_BiLSTM/23.05.2017_05:36:06_Conv_BiLSTM_no_lower.h5',
            DOC_MODEL: '../models/document_level_classification/final_2048_1024_512/25.05.2017_10:04:09_final_2048_1024_512_01_0.5349.h5'
        },
        vocabularies={
            WORD_MODEL: '../models/word_embedding_classification/word_index/23.05.2017_18:12:28_BiLSTM_punct_em.pkl',
            CHAR_MODEL: '../models/character_level_classification/char_index/23.05.2017_05:36:06_Conv_BiLSTM_no_lower.pkl',
            DOC_MODEL: '../models/document_level_classification/feature_models/bow_10k_most_freq.pkl'
        },
        averaging_style=MAX_CONF,
        print_individual_prfs=False
    )
