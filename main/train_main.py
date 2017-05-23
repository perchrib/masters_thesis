import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset, filter_dataset
from helpers.global_constants import TEST_DATA_DIR, TRAIN_DATA_DIR, TEST, TRAIN, REM_PUNCTUATION, REM_STOPWORDS, REM_EMOTICONS, LEMMATIZE, REM_INTERNET_TERMS, CHAR, DOC, WORD

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.constants import PREDICTION_TYPE as c_PREDICTION_TYPE, MODEL_DIR as c_MODEL_DIR, \
    FILTERS as c_FILTERS
from character_level_classification.train import train as c_train
from character_level_classification.models import *

from word_embedding_classification.dataset_formatting import format_dataset_word_level
from word_embedding_classification.constants import PREDICTION_TYPE as w_PREDICTION_TYPE, MODEL_DIR as w_MODEL_DIR, \
    FILTERS as w_FILTERS
from word_embedding_classification.train import train as w_train, get_embedding_layer
from word_embedding_classification.models import *

import keras.backend.tensorflow_backend as k_tf
from helpers.model_utils import load_and_evaluate, load_and_predict

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE, FILTERS as d_FILTERS
from document_level_classification.models import get_ann_model, get_logistic_regression

from document_level_classification.train import train as document_trainer
from document_level_classification.dataset_formatting import format_dataset_doc_level

from char_word_combined.models import get_cw_model
from char_word_combined.train import train as cw_train


def word_main(operation, trained_model_path=None, manual_filters=None):
    print("""WORD MODEL""")
    # Load datasets
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(w_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(w_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    filters = w_FILTERS if manual_filters is None else manual_filters

    # Filter datasets
    train_texts, train_labels, train_metadata, extra_info = \
        filter_dataset(texts=train_texts,
                       labels=train_labels,
                       metadata=train_metadata,
                       filters=filters,
                       train_or_test=TRAIN)
    test_texts, test_labels, test_metadata, _ = \
        filter_dataset(texts=test_texts,
                       labels=test_labels,
                       metadata=test_metadata,
                       filters=w_FILTERS,
                       train_or_test=TEST)

    print("Formatting dataset")
    data = format_dataset_word_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_word_level(test_texts, test_labels, test_metadata,
                                                               trained_word_index=data['word_index'])

    if operation == TRAIN:
        embedding_layer = get_embedding_layer(data['word_index'])
        num_output_nodes = len(labels_index)

        # ------- Insert models to txt here -----------
        # Remember star before model getter
        # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_Conv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_3xConv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_2x512_256_lstm_128_full(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)



        w_train(*get_word_model_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=True)  # TODO: Save model is True


        # w_train(*get_word_model_2xBiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info,
        #         save_model=False)


        # w_train(*get_word_model_3x512_128lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_4x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

    elif operation == TEST:
        # Evaluate model on te  st set
        load_and_evaluate(os.path.join(w_MODEL_DIR, trained_model_path), data=data)
        # load_and_predict(os.path.join(w_MODEL_DIR, trained_model_path), data=data, prediction_type=w_PREDICTION_TYPE,
        #                  normalize=True)


def char_main(operation, trained_model_path=None, manual_filters=None):
    print("""CHAR MODEL""")
    # Load dataset
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(c_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    filters = c_FILTERS if manual_filters is None else manual_filters

    # Filter datasets
    train_texts, train_labels, train_metadata, extra_info = \
        filter_dataset(texts=train_texts,
                       labels=train_labels,
                       metadata=train_metadata,
                       filters=filters,
                       train_or_test=TRAIN)
    test_texts, test_labels, test_metadata, _ = \
        filter_dataset(texts=test_texts,
                       labels=test_labels,
                       metadata=test_metadata,
                       filters=c_FILTERS,
                       train_or_test=TEST)

    print("Formatting dataset")
    data = format_dataset_char_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_char_level(test_texts, test_labels, test_metadata,
                                                               trained_char_index=data['char_index'])

    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    if operation == TRAIN:
        # ------- Insert models to txt here -----------
        # Remember star before model getter

        # c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)
        # c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)

        # c_train(*get_char_model_2xConv_BiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)

        c_train(*get_char_model_Conv_BiLSTM(num_output_nodes, num_chars), data=data, save_model=True, extra_info=extra_info)
        # c_train(*get_char_model_Conv_2_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False, extra_info=extra_info)


        # c_train(*get_char_model_Conv_2xBiLSTM(num_output_nodes, num_chars), data=data, save_model=False, extra_info=extra_info)

        # c_train(*get_char_model_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)

        # c_train(*get_char_model_512lstm(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)

        # c_train(*get_char_model_2x512lstm(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)


    elif operation == TEST:
        # Evaluate model on test set
        # load_and_evaluate(os.path.join(c_MODEL_DIR, trained_model_path), data=data)
        # load_and_predict(os.path.join(c_MODEL_DIR, trained_model_path), data=data, prediction_type=c_PREDICTION_TYPE, normalize=True)
        print("")

def document_main():
    # Load dataset
    from document_level_classification.constants import Log_Reg, TEST_DATA_DIR, LAYERS, EXPERIMENTS, N_GRAM, \
        MAX_FEATURE_LENGTH, FEATURE_MODEL, get_constants_info
    print("-"*20, " RUNNING DOCUMENT MODEL ", "-"*20)
    # Train and Validation
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(DOC_PREDICTION_TYPE)

    # This is the Test datset from Kaggle
    test_texts, test_labels, test_metadata, _ = prepare_dataset(DOC_PREDICTION_TYPE,
                                                                folder_path=TEST_DATA_DIR)

    # Filter datasets
    train_texts, train_labels, train_metadata, extra_info = \
        filter_dataset(texts=train_texts,
                       labels=train_labels,
                       metadata=train_metadata,
                       filters=d_FILTERS,
                       train_or_test=TRAIN)

    test_texts, test_labels, test_metadata, _ = \
        filter_dataset(texts=test_texts,
                       labels=test_labels,
                       metadata=test_metadata,
                       filters=d_FILTERS,
                       train_or_test=TEST)

    data = {}

    if EXPERIMENTS:

        for max_length in MAX_FEATURE_LENGTH:
            for n_gram in N_GRAM:
                info = extra_info
                print "-"*20, " Running: ", n_gram, " and max feature length: ", max_length, " ", "-"*20
                info.extend(get_constants_info(n_gram=n_gram, vocabulary_size=max_length))
                print("Format Dataset to Document Level")
                data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], \
                feature_model, reduction_model = format_dataset_doc_level(train_texts,
                                                                          train_labels,
                                                                          train_metadata,
                                                                          is_test=False,
                                                                          feature_model_type=FEATURE_MODEL,
                                                                          n_gram=n_gram,
                                                                          max_feature_length=max_length)

                # vocabulary = feature_model.train_vocabulary_counts
                # keys = sorted(vocabulary, key=vocabulary.get, reverse=True)
                # i = 0
                # for key in keys:
                #     if i < 100:
                #         print key, " ", vocabulary[key]
                #         i += 1
                #     else:
                #         break

                data['x_test'], data['y_test'], data['meta_test'] = format_dataset_doc_level(test_texts,
                                                                                             test_labels,
                                                                                             test_metadata,
                                                                                             is_test=True,
                                                                                             feature_model_type=FEATURE_MODEL,
                                                                                             n_gram=n_gram,
                                                                                             max_feature_length=max_length,
                                                                                             feature_model=feature_model,
                                                                                             reduction_model=reduction_model)

                input_size = data['x_train'].shape[1]
                output_size = data['y_train'].shape[1]

                document_trainer(*get_ann_model(input_size, output_size, LAYERS), data=data, extra_info=info, save_model=False)


    # This code are for test a saved model !!!!

    #c_MODEL_DIR, trained_model_path), data = data, prediction_type = c_PREDICTION_TYPE, normalize = True
    # model_path = "../models/document_level_classification/base_1024_512_256/16.05.2017_23:14:32_base_1024_512_256_01_0.5424.h5"
    # load_and_predict(model_path, )
    # load_and_evaluate(model_path, data=data)

    """STANDARD RUNNING"""
    # if not Log_Reg:
    #     if type(LAYERS[0]) == list:
    #         for layers_type in LAYERS:
    #             document_trainer(*get_ann_model(input_size, output_size, layers_type), data=data, extra_info=extra_info)
    #     else:
    #         # when running single models, checkpoint during training are set to True! (save_model=True)
    #         print("Running Single Model")
    #         document_trainer(*get_ann_model(input_size, output_size, LAYERS), data=data, extra_info=extra_info, save_model=False)

    # Logistic Regression
    if Log_Reg:
        document_trainer(*get_logistic_regression(input_size, output_size), data=data, extra_info=extra_info)


def char_word_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.lowercase(texts)
    texts = text_parser.replace_all_twitter_syntax_tokens(texts)
    texts = text_parser.remove_stopwords(texts)
    # texts = text_parser.replace_urls(texts)

    # Format for character model
    c_data = {}
    c_data['x_train'], c_data['y_train'], c_data['meta_train'], c_data['x_val'], c_data['y_val'], c_data['meta_val'], \
    c_data[
        'x_test'], c_data['y_test'], c_data['meta_test'], c_data['char_index'] = format_dataset_char_level(texts,
                                                                                                           labels,
                                                                                                           metadata)
    # Format for word model
    w_data = {}
    w_data['x_train'], w_data['y_train'], w_data['meta_train'], w_data['x_val'], w_data['y_val'], w_data['meta_val'], \
    w_data[
        'x_test'], w_data['y_test'], w_data['meta_test'], w_data[
        'word_index'] = format_dataset_word_level(texts, labels,
                                                  metadata)

    embedding_layer = get_embedding_layer(w_data['word_index'])
    num_chars = len(c_data['char_index'])
    num_output_nodes = len(labels_index)

    extra_info = []

    cw_train(*get_cw_model(embedding_layer, num_output_nodes, num_chars), c_data=c_data, w_data=w_data, save_model=True,
             extra_info=extra_info)


if __name__ == '__main__':
    # For more conservative memory usage
    tf_config = k_tf.tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    k_tf.set_session(k_tf.tf.Session(config=tf_config))

    # Ablation setup
    filter_list = [
        # Base
        # {REM_STOPWORDS: True,
        #  LEMMATIZE: False,
        #  REM_EMOTICONS: False,
        #  REM_PUNCTUATION: False},

        # Lemmatize
        {REM_STOPWORDS: True,
         LEMMATIZE: True,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: False},

        # Emoticons
        # {REM_STOPWORDS: True,
        #  LEMMATIZE: False,
        #  REM_EMOTICONS: True,
        #  REM_PUNCTUATION: False},

        # Punctuation
        {REM_STOPWORDS: True,
         LEMMATIZE: False,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: True},

        # Do not remove stopwords
        {REM_STOPWORDS: False,
         LEMMATIZE: False,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: False},

        # All true
        {REM_STOPWORDS: True,
         LEMMATIZE: True,
         REM_EMOTICONS: True,
         REM_PUNCTUATION: True},

        # Remove Internet terms instead of replacing
        {REM_STOPWORDS: True,
         LEMMATIZE: False,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: False,
         REM_INTERNET_TERMS: True}
    ]

    if sys.argv[1] == DOC:
        # Train all models in doc main
        """ DOCUMENT MODEL """
        document_main()

    elif sys.argv[1] == CHAR:
        # Train all models in character main
        """CHAR MODEL"""
        # Char ablation
        # for f in filter_list:
        #     char_main(operation=TRAIN, manual_filters=f)

        # Single char
        char_main(operation=TRAIN)

        # Load model and run test data on model
        # char_main(operation=TEST, trained_model_path="Conv_BiLSTM/27.04.2017_21:07:34_Conv_BiLSTM_adam_31_0.70.h5")

    elif sys.argv[1] == WORD:
        # Train all models in word main
        """ WORD MODEL """
        # Word ablation
        # for f in filter_list:
        #     word_main(operation=TRAIN, manual_filters=f)

        # Single word
        word_main(operation=TRAIN)


        # Load model and run test data on model
        # word_main(operation=TEST, trained_model_path="Conv_BiLSTM/28.04.2017_18:59:55_Conv_BiLSTM_adam_{epoch:02d}_{val_acc:.4f}.h5")



