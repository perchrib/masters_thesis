import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser

from preprocessors.dataset_preparation import prepare_dataset, filter_dataset, filter_gender
from helpers.global_constants import TEST_DATA_DIR, TRAIN_DATA_DIR, TEST, TRAIN, REM_PUNCTUATION, REM_STOPWORDS, REM_EMOTICONS, LEMMATIZE, REM_INTERNET_TERMS, CHAR, DOC, WORD, MALE, FEMALE


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

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE, FILTERS as d_FILTERS
from document_level_classification.models import get_ann_model, get_logistic_regression

from document_level_classification.train import train as document_trainer
from document_level_classification.dataset_formatting import format_dataset_doc_level

from char_word_combined.models import get_cw_model
from char_word_combined.train import train as cw_train


def word_main(specified_filters=None, train_only_on=None, save_model=False):
    print("""WORD MODEL""")
    # Load datasets
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(w_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(w_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    filters = w_FILTERS if specified_filters is None else specified_filters

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
                       filters=filters,
                       train_or_test=TEST)

    print("Formatting dataset")
    data = format_dataset_word_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_word_level(test_texts, test_labels, test_metadata,
                                                               trained_word_index=data['word_index'])

    # Train on one gender only
    if train_only_on:
        data['x_train'], data['y_train'], data['meta_train'] = filter_gender(data['x_train'],
                                                                             data['y_train'],
                                                                             data['meta_train'],
                                                                             labels_index,
                                                                             train_only_on)
        extra_info.append("Data is only trained on %s" % train_only_on)
        extra_info.append("Training on %i training samples" % len(data['x_train']))

    embedding_layer = get_embedding_layer(data['word_index'])
    num_output_nodes = len(labels_index)

    # ------- Insert models to txt here -----------
    # Remember star before model getter
    # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)


    # w_train(*get_word_model_Conv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)


    # w_train(*get_word_model_3xConv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
    # w_train(*get_word_model_2x512_256_lstm_128_full(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)



    w_train(*get_word_model_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=save_model)


    # w_train(*get_word_model_2xBiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info,
    #         save_model=False)


    # w_train(*get_word_model_3x512_128lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
    # w_train(*get_word_model_4x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)



def char_main(specified_filters=None, train_only_on=None, save_model=False):
    print("""CHAR MODEL""")
    # Load dataset
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(c_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    filters = c_FILTERS if specified_filters is None else specified_filters

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
                       filters=filters,
                       train_or_test=TEST)

    print("Formatting dataset")
    data = format_dataset_char_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_char_level(test_texts, test_labels, test_metadata,
                                                               trained_char_index=data['char_index'])


    # Train on one gender only
    if train_only_on:
        data['x_train'], data['y_train'], data['meta_train'] = filter_gender(data['x_train'],
                                                                             data['y_train'],
                                                                             data['meta_train'],
                                                                             labels_index,
                                                                             train_only_on)
        extra_info.append("Data is only trained on %s" % train_only_on)
        extra_info.append("Training on %i training samples" % len(data['x_train']))

    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)


    # ------- Insert models to txt here -----------
    # Remember star before model getter


    # c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)
    # c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)


    # c_train(*get_char_model_2xConv_BiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)

    c_train(*get_char_model_Conv_BiLSTM(num_output_nodes, num_chars), data=data, save_model=save_model, extra_info=extra_info)
    # c_train(*get_char_model_Conv_2_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False, extra_info=extra_info)


    # c_train(*get_char_model_Conv_2xBiLSTM(num_output_nodes, num_chars), data=data, save_model=False, extra_info=extra_info)

    # c_train(*get_char_model_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False,
    #         extra_info=extra_info)

    # c_train(*get_char_model_512lstm(num_output_nodes, num_chars), data=data, save_model=False,
    #         extra_info=extra_info)

    # c_train(*get_char_model_2x512lstm(num_output_nodes, num_chars), data=data, save_model=False,
    #         extra_info=extra_info)


def document_main(train_only_on=None, pretrained_model = False):
    # Load dataset
    from document_level_classification.constants import Log_Reg, TEST_DATA_DIR, LAYERS, EXPERIMENTS, N_GRAM, \
        MAX_FEATURE_LENGTH, FEATURE_MODEL, get_constants_info, AUTOENCODER_DIR, SAVE_FEATUREMODEL,

    from keras.models import load_model




    print("-" * 20, " RUNNING DOCUMENT MODEL ", "-" * 20)
    # Train and Validation

    n_gram = N_GRAM
    max_length = MAX_FEATURE_LENGTH
    reduction_model = None
    # reduction_model_name = "10k_500_autoencoder_deep_tanh_softmax_categorical_crossentropy.h5"
    # reduction_model = load_model(os.path.join(AUTOENCODER_DIR, reduction_model_name))

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

    info = extra_info
    print "-" * 20, " Running: ", n_gram, " and max feature length: ", max_length, " ", "-" * 20

    info.extend(get_constants_info(n_gram=n_gram, vocabulary_size=max_length))

    print("Format Dataset to Document Level")

    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], \
    feature_model, reduction_model = format_dataset_doc_level(train_texts,
                                                              train_labels,
                                                              train_metadata,
                                                              is_test=False,
                                                              feature_model_type=FEATURE_MODEL,
                                                              n_gram=n_gram,
                                                              max_feature_length=max_length,
                                                              reduction_model=reduction_model)


    if pretrained_model:
        from helpers.model_utils import predict_and_get_precision_recall_f_score
        import pandas as pd
        model = load_model('../models/document_level_classification/final_2048_1024_512/25.05.2017_10:04:09_final_2048_1024_512_01_0.5349.h5')
        print("model loaded")
        preds = model.predict(data['x_val'])
        for p in preds:
            print(p)
        prf_val = predict_and_get_precision_recall_f_score(model, data['x_val'], data['y_val'], PREDICTION_TYPE)
        prf_val_df = pd.DataFrame(data=prf_val, index=pd.Index(["Precision", "Recall", "F-score", "Support"]))
        print(pd.DataFrame(prf_val_df).__repr__())
        return


    if SAVE_FEATUREMODEL:
        from helpers.helper_functions import save_pickle
        save_pickle("../models/document_level_classification/feature_models/bow_10k_most_freq", feature_model)
        print("Succesfully saved to file")

    data['x_test'], data['y_test'], data['meta_test'] = format_dataset_doc_level(test_texts,
                                                                                 test_labels,
                                                                                 test_metadata,
                                                                                 is_test=True,
                                                                                 feature_model_type=FEATURE_MODEL,
                                                                                 n_gram=n_gram,
                                                                                 max_feature_length=max_length,
                                                                                 feature_model=feature_model,
                                                                                 reduction_model=reduction_model)

    # Train on one gender only
    if train_only_on:
        print('Training only on: ', train_only_on)
        data['x_train'], data['y_train'], data['meta_train'] = filter_gender(data['x_train'],
                                                                             data['y_train'],
                                                                             data['meta_train'],
                                                                             labels_index,
                                                                             train_only_on)
        extra_info.append("Data is only trained on %s" % train_only_on)
        extra_info.append("Training on %i training samples" % len(data['x_train']))



    """STANDARD RUNNING"""
    if not Log_Reg:

        input_size = data['x_train'].shape[1]
        output_size = data['y_train'].shape[1]
        if type(LAYERS[0]) == list:
            for layers_type in LAYERS:
                document_trainer(*get_ann_model(input_size, output_size, layers_type), data=data, extra_info=extra_info, save_model=False)
        else:
            # when running single models, checkpoint during training are set to True! (save_model=True)
            print("Running Single Model")
            document_trainer(*get_ann_model(input_size, output_size, LAYERS), data=data, extra_info=extra_info, save_model=False)

    # Machine Learning Methods
    if Log_Reg:
        from ml_models.models import logisitc_regression, svm, random_forests, naive_bayes
        #logisitc_regression(data)

        #random_forests(data)
        #svm(data)
        naive_bayes(data)


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
        {REM_STOPWORDS: True,
         LEMMATIZE: False,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: False},

        # Lemmatize
        {REM_STOPWORDS: True,
         LEMMATIZE: True,
         REM_EMOTICONS: False,
         REM_PUNCTUATION: False},

        # Emoticons
        {REM_STOPWORDS: True,
         LEMMATIZE: False,
         REM_EMOTICONS: True,
         REM_PUNCTUATION: False},

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

        # All true
        {REM_STOPWORDS: True,
         LEMMATIZE: True,
         REM_EMOTICONS: True,
         REM_PUNCTUATION: True,
         LOWERCASE: True}

        # Remove Internet terms instead of replacing
        # {REM_STOPWORDS: True,
        #  LEMMATIZE: False,
        #  REM_EMOTICONS: False,
        #  REM_PUNCTUATION: False,
        #  REM_INTERNET_TERMS: True}
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
        char_main(save_model=True)

        # Load model and run test data on model
        # char_main(operation=TEST, trained_model_path="Conv_BiLSTM/27.04.2017_21:07:34_Conv_BiLSTM_adam_31_0.70.h5")

    elif sys.argv[1] == WORD:
        # Train all models in word main
        """ WORD MODEL """
        # Word ablation
        # for f in filter_list:
        #     word_main(operation=TRAIN, manual_filters=f)

        # Single word
        word_main(save_model=True)


        # Load model and run test data on model
        # word_main(operation=TEST, trained_model_path="Conv_BiLSTM/28.04.2017_18:59:55_Conv_BiLSTM_adam_{epoch:02d}_{val_acc:.4f}.h5")

