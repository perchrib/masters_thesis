import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Convolution1D, MaxPooling1D, merge
from char_classification.constants import MAX_SEQUENCE_LENGTH

# TODO: Automate
nb_chars = 1149


def get_char_model(num_output_nodes):
        tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
        embedding = Lambda(binarize, output_shape=binarize_outshape)(tweet_input)

        filter_length = [5, 3, 3]
        nb_filter = [196, 196, 256]
        pool_length = 2

        for i in range(len(nb_filter)):
            embedding = Convolution1D(nb_filter=nb_filter[i],
                                      filter_length=filter_length[i],
                                      border_mode='valid',
                                      activation='relu',
                                      init='glorot_normal',
                                      subsample_length=1)(embedding)

            embedding = Dropout(0.1)(embedding)
            embedding = MaxPooling1D(pool_length=pool_length)(embedding)

        forward = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu')(embedding)
        backward = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2, consume_less='gpu',
                                 go_backwards=True)(embedding)

        encoding = merge([forward, backward], mode='concat', concat_axis=-1)
        output = Dropout(0.3)(encoding)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.3)(output)
        output = Dense(num_output_nodes, activation='softmax')(output)
        model = Model(input=tweet_input, output=output)

        return model

def get_char_model_2(num_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(binarize, output_shape=binarize_outshape)(tweet_input)

    embedding = LSTM(128, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(embedding)

    output = Dropout(0.3)(embedding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output)

    return model


def binarize(x, chars=nb_chars):
    return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], nb_chars