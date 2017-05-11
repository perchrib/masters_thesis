import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset

from character_level_classification.dataset_formatting import format_dataset_char_level, format_dataset_char_level_sentences
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
from document_level_classification.models import get_2048_1024_512, get_4096_2048_1024_512, get_50_10_
from document_level_classification.train import train as document_trainer
from document_level_classification.dataset_formatting import format_dataset_doc_level

from char_word_combined.models import get_cw_model
from char_word_combined.train import train as cw_train

TRAIN = "train"
TEST = "test"

def word_main(operation, trained_model_path=None):
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(w_PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    # texts = text_parser.remove_stopwords(texts)

    # Add extra info, e.g., about parsing here
    extra_info = ["All Internet terms are replaced with tags"]

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['x_test'], data['y_test'], data['meta_test'], data[
        'word_index'] = format_dataset_word_level(texts, labels,
                                                  metadata)

    if operation == TRAIN:
        embedding_layer = get_embedding_layer(data['word_index'])
        num_output_nodes = len(labels_index)

        # ------- Insert models to train here -----------
        # Remember star before model getter
        # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=True)
        # w_train(*get_word_model_Conv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_3xConv_BiLSTM(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        # w_train(*get_word_model_2x512_256_lstm_128_full(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        w_train(*get_word_model_3x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info,
                save_model=False)
        w_train(*get_word_model_3x512_128lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)
        w_train(*get_word_model_4x512lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info, save_model=False)

    elif operation == TEST:
        # Evaluate model on test set
        # load_and_evaluate(os.path.join(w_MODEL_DIR, trained_model_path), data=data)
        load_and_predict(os.path.join(w_MODEL_DIR, trained_model_path), data=data, prediction_type=w_PREDICTION_TYPE,
                         normalize=True)

def char_main(operation, trained_model_path=None):
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(c_PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    texts = text_parser.remove_stopwords(texts)
    # texts = text_parser.replace_urls(texts)

    # Add extra info, e.g., about parsing here
    extra_info = ["Stopwords removed", "All Internet terms are replaced with tags"]

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['x_test'], data['y_test'], data['meta_test'], data['char_index'] = format_dataset_char_level(texts, labels,
                                                                                                 metadata)
    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)


    if operation == TRAIN:
        # ------- Insert models to train here -----------
        # Remember star before model getter
        # c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data)
        # c_train(*get_char_model_BiLSTM_full(num_output_nodes, num_chars), data=data)
        #c_train(*get_char_model_3xConv(num_output_nodes), data=data)
        #c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)
        # c_train(*get_char_model_3xConv_4xBiLSTM(num_output_nodes, num_chars), data=data)

        # c_train(*get_char_model_2xConv_BiLSTM(num_output_nodes, num_chars), data=data)

        # c_train(*get_char_model_Conv_BiLSTM(num_output_nodes, num_chars), data=data, save_model=True, extra_info=extra_info)
        # c_train(*get_char_model_Conv_BiLSTM_2(num_output_nodes, num_chars), data=data, save_model=False)
        # c_train(*get_char_model_Conv_BiLSTM_3(num_output_nodes, num_chars), data=data, save_model=True)
        # c_train(*get_char_model_Conv_BiLSTM_mask(num_output_nodes, num_chars), data=data, save_model=False,
        #         extra_info=extra_info)
        # c_train(*get_char_model_Conv_BiLSTM_4(num_output_nodes, num_chars), data=data, save_model=False, extra_info=extra_info)

        c_train(*get_char_model_BiLSTM(num_output_nodes, num_chars), data=data, save_model=False,
                extra_info=extra_info)

        c_train(*get_word_model_4x512lstm(num_output_nodes, num_chars), data=data, save_model=False,
                extra_info=extra_info)



        # Dummy model for fast train and save model --- DELETE
        # c_train(*get_dummy_model(num_output_nodes, num_chars), data=data, save_model=True, log_sess=False)

    elif operation == TEST:
        # Evaluate model on test set
        # load_and_evaluate(os.path.join(c_MODEL_DIR, trained_model_path), data=data)
        load_and_predict(os.path.join(c_MODEL_DIR, trained_model_path), data=data, prediction_type=c_PREDICTION_TYPE, normalize=True)


def document_main():
    # Load dataset
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
    data['meta_val'], data['x_test'], data['y_test'], data['meta_test'] = format_dataset_doc_level(texts, labels, metadata)

    input_size = data['x_train'].shape[1]
    output_size = len(labels_index)

    # document_trainer(*get_2048_1024_512(input_size, output_size), data=data)
    # document_trainer(*get_4096_2048_1024_512(input_size, output_size), data=data)
    document_trainer(*get_50_10_(input_size, output_size), data=data)


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
    document_main()

    # Train all models in word main

    # word_main(operation=TRAIN)

    # Train char-word models in char word main
    # char_word_main()


    # Load model and run test data on model
    #char_main(operation=TEST, trained_model_path="Conv_BiLSTM/27.04.2017_21:07:34_Conv_BiLSTM_adam_31_0.70.h5")
    #word_main(operation=TEST, trained_model_path="Conv_BiLSTM/28.04.2017_18:59:55_Conv_BiLSTM_adam_{epoch:02d}_{val_acc:.4f}.h5")