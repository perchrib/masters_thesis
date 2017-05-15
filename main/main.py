import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset
from helpers.global_constants import TEST_DATA_DIR, TRAIN_DATA_DIR

from character_level_classification.dataset_formatting import format_dataset_char_level
from character_level_classification.constants import PREDICTION_TYPE as c_PREDICTION_TYPE, MODEL_DIR as c_MODEL_DIR
from character_level_classification.train import train as c_train
from character_level_classification.models import *
from character_level_classification.model_sent import get_char_model_3xConv_Bi_lstm_sent


from word_embedding_classification.dataset_formatting import format_dataset_word_level
from word_embedding_classification.constants import PREDICTION_TYPE as w_PREDICTION_TYPE, MODEL_DIR as w_MODEL_DIR
from word_embedding_classification.train import train as w_train, get_embedding_layer
from word_embedding_classification.models import *

import keras.backend.tensorflow_backend as k_tf
from helpers.model_utils import load_and_evaluate, load_and_predict

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE
from document_level_classification.models import get_2048_1024_512, get_4096_2048_1024_512, get_1024_512,\
    get_logistic_regression, get_512_256_128
from document_level_classification.train import train as document_trainer
from document_level_classification.dataset_formatting import format_dataset_doc_level

from char_word_combined.models import get_cw_model
from char_word_combined.train import train as cw_train


TRAIN = "train"
TEST = "test"


def word_main(operation, trained_model_path=None):
    rem_stopwords = True
    lemmatize = False
    rem_punctuation = False
    rem_emoticons = False

    # Load dataset
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(w_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(w_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    # Clean texts
    text_parser = Parser()
    train_texts = text_parser.lowercase(train_texts)
    train_texts = text_parser.replace_all(train_texts)  # Base filtering, i.e lowercase and tags

    test_texts = text_parser.lowercase(test_texts)
    test_texts = text_parser.replace_all(test_texts)


    if rem_stopwords:
        train_texts = text_parser.remove_stopwords(train_texts)
        test_texts = text_parser.remove_stopwords(test_texts)

    if rem_punctuation:
        train_texts = text_parser.remove_punctuation(train_texts)
        test_texts = text_parser.remove_punctuation(test_texts)

    if lemmatize:
        train_texts = text_parser.lemmatize(train_texts)
        test_texts = text_parser.lemmatize(test_texts)

    if rem_emoticons:
        train_texts = text_parser.remove_emoticons(train_texts)
        test_texts = text_parser.remove_emoticons(test_texts)

    # Remove short texts from training
    train_texts, train_labels, train_metadata, count_removed = text_parser.remove_texts_shorter_than_threshold(train_texts, train_labels, train_metadata)

    # Add extra info, e.g., about parsing here
    extra_info = ["Remove stopwords %s" % rem_stopwords,
                  "Lemmatize %s" % lemmatize,
                  "Remove punctuation %s" % rem_punctuation,
                  "Remove emoticons %s" % rem_emoticons,
                  "All Internet terms are replaced with tags",
                  "Removed %i tweet because they were shorter than threshold" % count_removed]

    print("Formatting dataset")
    data = format_dataset_word_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_word_level(test_texts, test_labels, test_metadata, trained_word_index=data['word_index'])

    if operation == TRAIN:
        embedding_layer = get_embedding_layer(data['word_index'])
        num_output_nodes = len(labels_index)

        # ------- Insert models to train here -----------
        # Remember star before model getter
        # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_Conv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_3xConv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_2x512_256_lstm_128_full(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_3x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        w_train(*get_word_model_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

        # w_train(*get_word_model_3x512_128lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_4x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

    elif operation == TEST:
        # Evaluate model on te  st set
        load_and_evaluate(os.path.join(w_MODEL_DIR, trained_model_path), data=data)
        # load_and_predict(os.path.join(w_MODEL_DIR, trained_model_path), data=data, prediction_type=w_PREDICTION_TYPE,
        #                  normalize=True)


def char_main(operation, trained_model_path=None):

    rem_stopwords = True  # Part of base
    lemmatize = False
    rem_punctuation = False
    rem_emoticons = False

    # Load dataset
    train_texts, train_labels, train_metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)
    test_texts, test_labels, test_metadata, _ = prepare_dataset(c_PREDICTION_TYPE, folder_path=TEST_DATA_DIR)

    # Clean texts
    text_parser = Parser()
    train_texts = text_parser.lowercase(train_texts)
    train_texts = text_parser.replace_all(train_texts)  # Base filtering, i.e lowercase and tags

    test_texts = text_parser.lowercase(test_texts)
    test_texts = text_parser.replace_all(test_texts)

    if rem_stopwords:
        train_texts = text_parser.remove_stopwords(train_texts)
        test_texts = text_parser.remove_stopwords(test_texts)

    if rem_punctuation:
        train_texts = text_parser.remove_punctuation(train_texts)
        test_texts = text_parser.remove_punctuation(test_texts)

    if lemmatize:
        train_texts = text_parser.lemmatize(train_texts)
        test_texts = text_parser.lemmatize(test_texts)

    if rem_emoticons:
        train_texts = text_parser.remove_emoticons(train_texts)
        test_texts = text_parser.remove_emoticons(test_texts)

    # Remove short texts from training set
    train_texts, train_labels, train_metadata, count_removed = text_parser.remove_texts_shorter_than_threshold(train_texts, train_labels, train_metadata)


    # Add extra info, e.g., about parsing here
    extra_info = ["Remove stopwords %s" % rem_stopwords,
                  "Lemmatize %s" % lemmatize,
                  "Remove punctuation %s" % rem_punctuation,
                  "Remove emoticons %s" % rem_emoticons,
                  "All Internet terms are replaced with tags"
                  "Removed %i tweets because they were shorter than threshold" % count_removed]

    print("Formatting dataset")
    data = format_dataset_char_level(train_texts, train_labels, train_metadata)
    data['x_test'], data['y_test'] = format_dataset_char_level(test_texts, test_labels, test_metadata, trained_char_index=data['char_index'])

    # TODO: REmove
    print(len(data['x_test']), data['x_test'][0])

    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    if operation == TRAIN:
        # ------- Insert models to train here -----------
        # Remember star before model getter
        # c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)
        #c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)

        # c_train(*get_char_model_2xConv_BiLSTM(num_output_nodes, num_chars), data=data, extra_info=extra_info)

        # c_train(*get_char_model_Conv_BiLSTM(num_output_nodes, num_chars), data=data, save_model=True, extra_info=extra_info)

        c_train(*get_char_model_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False,
                extra_info=extra_info)

        # c_train(*get_char_model_512lstm(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)

        # c_train(*get_char_model_2x512lstm(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)


    elif operation == TEST:
        # Evaluate model on test set
        load_and_evaluate(os.path.join(c_MODEL_DIR, trained_model_path), data=data)
        # load_and_predict(os.path.join(c_MODEL_DIR, trained_model_path), data=data, prediction_type=c_PREDICTION_TYPE, normalize=True)


def document_main():
    # Load dataset
    from document_level_classification.constants import Log_Reg
    categorical = True
    if Log_Reg:
        categorical = False

    texts, labels, metadata, labels_index = prepare_dataset(DOC_PREDICTION_TYPE)

    # Clean texts with parser
    parser = Parser()
    print("Remove Stopwords...")
    texts = parser.remove_stopwords(texts)
    print("Parsing Twitter Specific Syntax...")
    texts = parser.replace_all(texts)

    data = {}
    # Create format_dataset_tfidf
    print("Format Dataset to Document Level")
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], \
    data['meta_val'], data['x_test'], data['y_test'], data['meta_test'] = format_dataset_doc_level(texts, labels, metadata, categorical=categorical)

    input_size = data['x_train'].shape[1]
    output_size = data['y_train'].shape[1]

    # document_trainer(*get_2048_1024_512(input_size, output_size), data=data)
    # document_trainer(*get_4096_2048_1024_512(input_size, output_size), data=data)
    # document_trainer(*get_1024_512(input_size, output_size), data=data)
    document_trainer(*get_512_256_128(input_size, output_size), data=data)

    # Logistic Regression
    # output_size = data['y_train'].shape[1]
    # print("Output Size Log_Reg: ", output_size)
    # document_trainer(*get_logistic_regression(input_size, output_size), data=data)


def char_word_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    texts = text_parser.remove_stopwords(texts)
    # texts = text_parser.replace_urls(texts)

    # Format for character model
    c_data = {}
    c_data['x_train'], c_data['y_train'], c_data['meta_train'], c_data['x_val'], c_data['y_val'], c_data['meta_val'], c_data[
        'x_test'], c_data['y_test'], c_data['meta_test'], c_data['char_index'] = format_dataset_char_level(texts, labels,
                                                                                                     metadata)
    # Format for word model
    w_data = {}
    w_data['x_train'], w_data['y_train'], w_data['meta_train'], w_data['x_val'], w_data['y_val'], w_data['meta_val'], w_data[
        'x_test'], w_data['y_test'], w_data['meta_test'], w_data[
        'word_index'] = format_dataset_word_level(texts, labels,
                                                  metadata)

    embedding_layer = get_embedding_layer(w_data['word_index'])
    num_chars = len(c_data['char_index'])
    num_output_nodes = len(labels_index)

    extra_info = []

    cw_train(*get_cw_model(embedding_layer, num_output_nodes, num_chars), c_data=c_data, w_data=w_data, save_model=True, extra_info=extra_info)


if __name__ == '__main__':

    # For more conservative memory usage
    tf_config = k_tf.tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    k_tf.set_session(k_tf.tf.Session(config=tf_config))

    # Train all models in character main
    # char_main(operation=TRAIN)

    # Train all models in doc main
    """ DOCUMENT MODEL """
    # document_main()

    # Train all models in word main
    """ WORD MODEL """
    word_main(operation=TRAIN)

    # Train char-word models in char word main
    # char_word_main()


    # Load model and run test data on model


    # char_main(operation=TEST, trained_model_path="Conv_BiLSTM/27.04.2017_21:07:34_Conv_BiLSTM_adam_31_0.70.h5")
    # word_main(operation=TEST, trained_model_path="Conv_BiLSTM/28.04.2017_18:59:55_Conv_BiLSTM_adam_{epoch:02d}_{val_acc:.4f}.h5")

