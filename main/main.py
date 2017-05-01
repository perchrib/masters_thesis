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
from word_embedding_classification.constants import PREDICTION_TYPE as w_PREDICTION_TYPE
from word_embedding_classification.ann import train as w_train, get_embedding_layer
from word_embedding_classification.models import *

import keras.backend.tensorflow_backend as k_tf
from helpers.model_utils import load_and_evaluate

from document_level_classification.constants import PREDICTION_TYPE as DOC_PREDICTION_TYPE
from document_level_classification.models import get_2048_1024_512, get_4096_2048_1024_512
from document_level_classification.train import train as document_trainer
from document_level_classification.dataset_formatting import format_dataset_doc_level

TRAIN = "train"
TEST = "test"


def char_sent_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    # texts = text_parser.replace_urls(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['x_test'], data['y_test'], data['meta_test'], data[
        'char_index'] = format_dataset_char_level_sentences(texts, labels,
                                                  metadata)
    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    c_train(*get_char_model_3xConv_Bi_lstm_sent(num_output_nodes, num_chars), data=data)


def word_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    # text_parser = Parser()
    # texts = text_parser.replace_all(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['x_test'], data['y_test'], data['meta_test'], data[
        'word_index'] = format_dataset_word_level(texts, labels,
                                                  metadata)

    embedding_layer = get_embedding_layer(data['word_index'])

    num_output_nodes = len(labels_index)

    extra_info = []

    # ------- Insert models to train here -----------
    # Remember star before model getter
    # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info)


def char_main(operation, trained_model_path=None):
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(w_PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    # texts = text_parser.replace_urls(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['x_test'], data['y_test'], data['meta_test'], data['char_index'] = format_dataset_char_level(texts, labels,
                                                                                                 metadata)
    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    extra_info = ["Text is not reversed "]

    if operation == TRAIN:
        # ------- Insert models to train here -----------
        # Remember star before model getter
        # c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data)
        # c_train(*get_char_model_BiLSTM_full(num_output_nodes, num_chars), data=data)
        #c_train(*get_char_model_3xConv(num_output_nodes), data=data)
        #c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)
        # c_train(*get_char_model_3xConv_4xBiLSTM(num_output_nodes, num_chars), data=data)

        # c_train(*get_char_model_2xConv_BiLSTM(num_output_nodes, num_chars), data=data)

        c_train(*get_char_model_Conv_BiLSTM(num_output_nodes, num_chars), data=data, save_model=True)
        c_train(*get_char_model_Conv_BiLSTM_2(num_output_nodes, num_chars), data=data, save_model=True)
        c_train(*get_char_model_Conv_BiLSTM_3(num_output_nodes, num_chars), data=data, save_model=True)
        # c_train(*get_char_model_Conv_BiLSTM_4(num_output_nodes, num_chars), data=data, save_model=True)

        # Dummy model for fast train and save model --- DELETE
        # c_train(*get_dummy_model(num_output_nodes, num_chars), data=data, save_model=True, log_sess=False)

    elif operation == TEST:
        # Evaluate model on test set
        load_and_evaluate(os.path.join(c_MODEL_DIR, trained_model_path), data=data, batch_size=256)



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

    #document_trainer(*get_2048_1024_512(input_size, output_size), data=data)
    document_trainer(*get_4096_2048_1024_512(input_size, output_size), data=data)

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
    # word_main()

    # Load model and run test data on model
    # char_main(TEST, "Conv_BiLSTM/27.04.2017_14:51:31_Conv_BiLSTM_adam_00_0.54.h5")
