import sys
import os

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from preprocessors.parser import Parser
from preprocessors.dataset_preparation import prepare_dataset

from character_level_classification.dataset_formatting import format_dataset_char_level, format_dataset_char_level_sentences
from character_level_classification.constants import PREDICTION_TYPE
from character_level_classification.ann import train as c_train
from character_level_classification.models import *
from character_level_classification.model_sent import get_char_model_3xConv_Bi_lstm_sent

from word_level_classification.dataset_formatting import format_dataset_word_level
from word_level_classification.constants import PREDICTION_TYPE
from word_level_classification.ann import train as w_train, get_embedding_layer
from word_level_classification.models import *


def char_sent_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    # texts = text_parser.replace_urls(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data[
        'char_index'] = format_dataset_char_level_sentences(texts, labels,
                                                  metadata)
    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    c_train(*get_char_model_3xConv_Bi_lstm_sent(num_output_nodes, num_chars), data=data)


def char_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    text_parser = Parser()
    texts = text_parser.replace_all(texts)
    # texts = text_parser.replace_urls(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data['char_index'] = format_dataset_char_level(texts, labels,
                                                                                                 metadata)
    num_chars = len(data['char_index'])
    num_output_nodes = len(labels_index)

    extra_info = []

    # ------- Insert models to train here -----------
    # Remember star before model getter
    c_train(*get_char_model_3xConv_2xBiLSTM(num_output_nodes, num_chars), data=data)
    # c_train(*get_char_model_3xConv(num_output_nodes), data=data)
    # c_train(*get_char_model_BiLSTM_full(num_output_nodes, num_chars), data=data)
    # c_train(*get_char_model_3xConv_LSTM(num_output_nodes, num_chars), data=data)
    # c_train(*get_char_model_3xConv_4xBiLSTM(num_output_nodes, num_chars), data=data)


def word_main():
    # Load dataset
    texts, labels, metadata, labels_index = prepare_dataset(PREDICTION_TYPE)

    # Clean texts
    # text_parser = Parser()
    # texts = text_parser.replace_all(texts)

    data = {}
    data['x_train'], data['y_train'], data['meta_train'], data['x_val'], data['y_val'], data['meta_val'], data[
        'word_index'] = format_dataset_word_level(texts, labels,
                                                  metadata)

    embedding_layer = get_embedding_layer(data['word_index'])

    num_output_nodes = len(labels_index)

    extra_info = []

    # ------- Insert models to train here -----------
    # Remember star before model getter
    # w_train(*get_word_model_2x512_256_lstm(embedding_layer, num_output_nodes), data=data, extra_info=extra_info)


if __name__ == '__main__':
    # Train sent char
    # char_sent_main()

    # Train all models in character main
    char_main()

    # Train all models in word main
    # word_main()

