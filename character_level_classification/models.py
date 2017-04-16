import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Convolution1D, MaxPooling1D, merge
from character_level_classification.constants import MAX_SEQUENCE_LENGTH

# TODO: Automate
nb_chars = 77

# TODO: Try Conv2D layers?

# Model name: 3xConv_2xLSTMmerge_model
def get_char_model_3xConv_2xBiLSTM(num_output_nodes):
        tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
        embedding = Lambda(binarize, output_shape=binarize_outshape)(tweet_input)

        filter_length = [5, 3, 3]
        nb_filter = [196, 196, 256]
        pool_length = 2

        # len of nb_filter = num conv layers
        for i in range(len(nb_filter)):
            embedding = Convolution1D(nb_filter=nb_filter[i],
                                      filter_length=filter_length[i],
                                      border_mode='valid',
                                      activation='relu',
                                      init='glorot_uniform',
                                      subsample_length=1)(embedding)

            embedding = Dropout(0.1)(embedding)
            embedding = MaxPooling1D(pool_length=pool_length)(embedding)

        forward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(embedding)
        backward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, go_backwards=True)(embedding)

        encoding = merge([forward, backward], mode='concat', concat_axis=-1)
        output = Dropout(0.5)(encoding)
        output = Dense(128, activation='relu')(output)
        output = Dropout(0.5)(output)
        output = Dense(num_output_nodes, activation='softmax')(output)
        model = Model(input=tweet_input, output=output, name='3xConv_2xBiLSTM')

        return model


def get_model_2x512_256_lstm(num_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(binarize, output_shape=binarize_outshape)(tweet_input)

    encoding = LSTM(512, return_sequences=True)(embedding)
    encoding = LSTM(512, return_sequences=True)(encoding)
    encoding = LSTM(256, return_sequences=True)(encoding)

    output = Dense(num_output_nodes, activation='softmax')(encoding)

    model = Model(inputs=tweet_input, outputs=output, name="2x512_256LSTM")

    return model


def get_char_model_BiLSTM_full(num_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(binarize, output_shape=binarize_outshape)(tweet_input)

    forward = LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(embedding)
    backward = LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(encoding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='BiLSTM_full')

    return model


def binarize(x, chars=nb_chars):
    return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0, axis=-1))
    # return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], nb_chars