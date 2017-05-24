import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset, filter_dataset
from helpers.global_constants import TEST_DATA_DIR, TRAIN_DATA_DIR, TEST, TRAIN, REM_PUNCTUATION, REM_STOPWORDS, \
    REM_EMOTICONS, LEMMATIZE, REM_INTERNET_TERMS, CHAR, DOC, WORD

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.constants import PREDICTION_TYPE as c_PREDICTION_TYPE, MODEL_DIR as c_MODEL_DIR, \
    FILTERS as c_FILTERS


from word_embedding_classification.dataset_formatting import format_dataset_word_level
from word_embedding_classification.constants import PREDICTION_TYPE as w_PREDICTION_TYPE, MODEL_DIR as w_MODEL_DIR, \
    FILTERS as w_FILTERS

import keras.backend.tensorflow_backend as k_tf
from helpers.model_utils import load_and_evaluate, load_and_predict
from helpers.helper_functions import load_pickle

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE, FILTERS as d_FILTERS
from document_level_classification.models import get_ann_model, get_logistic_regression

from document_level_classification.dataset_formatting import format_dataset_doc_level


def get_word_predictions(model_path, trained_word_index_path, specified_filters=None):
    print("""TEST WORD MODEL""")

    # Load word index
    word_index = load_pickle(trained_word_index_path)

    # Load datasets
    test_texts, test_labels, test_metadata, _ = prepare_dataset(w_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    test_texts, test_labels, test_metadata, _ = filter_dataset(texts=test_texts,
                                                               labels=test_labels,
                                                               metadata=test_metadata,
                                                               filters=w_FILTERS,
                                                               train_or_test=TEST)
    data = {}
    data['x_test'], data['y_test'] = format_dataset_word_level(test_texts,
                                                               test_labels,
                                                               test_metadata,
                                                               trained_word_index=word_index)




if __name__ == '__main__':
    preds = get_word_predictions(model_path='models/word_embedding_classification/word_index/23.05.2017_18:12:28_BiLSTM_punct_em.pkl',
                                 trained_word_index_path='../models/word_embedding_classification/word_index/23.05.2017_18:12:28_BiLSTM_punct_em.pkl')