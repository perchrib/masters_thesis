import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Convolution1D, MaxPooling1D, merge, Flatten
from character_level_classification.constants import MAX_SEQUENCE_LENGTH

# TODO: Try Conv2D layers?

# Not ideal, but functioning fix for setting and accessing nb_chars in models

nb_chars = None


def get_char_model_3xConv_2xBiLSTM(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    # filter_length = [7, 5, 3]
    # nb_filter = [196, 196, 256]
    pool_length = 2

    # len of nb_filter = num conv layers
    for i in range(len(nb_filter)):
        embedding = Convolution1D(nb_filter=nb_filter[i],
                                  filter_length=filter_length[i],
                                  activation='relu',
                                  init='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    # recurrent_dropout=0.2
    forward = LSTM(256, return_sequences=False, dropout=0.5, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.5, consume_less='gpu',
                    go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(encoding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv_2xBiLSTM')

    model_info = ["LSTM dropout = 0.5, 0.2", "No recurrent dropout"]
    return model, model_info


def get_char_model_2x512_256_lstm(num_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    encoding = LSTM(512, return_sequences=True)(embedding)
    encoding = LSTM(512, return_sequences=True)(encoding)
    encoding = LSTM(256, return_sequences=True)(encoding)

    output = Dense(num_output_nodes, activation='softmax')(encoding)

    model = Model(inputs=tweet_input, outputs=output, name="2x512_256LSTM")

    return model


def get_char_model_BiLSTM_full(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    # dropout = 0.5, recurrent_dropout = 0.5
    forward = LSTM(512, return_sequences=False, consume_less='gpu')(embedding)
    backward = LSTM(512, return_sequences=False, consume_less='gpu',
                    go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(encoding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='BiLSTM_full')

    extra_info = ["LSTM dropout = 0.5, 0.5"]
    return model, extra_info


def get_char_model_3xConv(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    pool_length = 2

    # len of nb_filter = num conv layers
    for i in range(len(nb_filter)):
        embedding = Convolution1D(nb_filter=nb_filter[i],
                                  filter_length=filter_length[i],
                                  activation='relu',
                                  init='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.1)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    output = Flatten()(embedding)
    # output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv')

    extra_info = []
    return model, extra_info


def get_char_model_3xConv_LSTM(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    # filter_length = [7, 5, 3]
    # nb_filter = [196, 196, 256]
    pool_length = 2

    # len of nb_filter = num conv layers
    for i in range(len(nb_filter)):
        embedding = Convolution1D(nb_filter=nb_filter[i],
                                  filter_length=filter_length[i],
                                  activation='relu',
                                  init='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    embedding = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(embedding)

    # embedding = Dropout(0.5)(embedding)
    embedding = Dense(128, activation='relu')(embedding)
    # embedding = Dropout(0.5)(embedding)
    output = Dense(num_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output, name='3xConv_LSTM')

    model_info = ["LSTM dropout = 0.5, 0.2"]
    return model, model_info


def get_char_model_3xConv_4xBiLSTM(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    # filter_length = [7, 5, 3]
    # nb_filter = [196, 196, 256]
    pool_length = 2

    # len of nb_filter = num conv layers
    for i in range(len(nb_filter)):
        embedding = Convolution1D(nb_filter=nb_filter[i],
                                  filter_length=filter_length[i],
                                  activation='relu',
                                  init='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    embedding = merge([forward, backward], mode='concat', concat_axis=-1)

    forward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    embedding = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(embedding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv_4xBiLSTM')

    model_info = ["LSTM dropout = 0.5, 0.2"]
    return model, model_info

def one_hot(x):
    return tf.to_float(tf.one_hot(x, nb_chars, on_value=1, off_value=0, axis=-1))
    # return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0))


def one_hot_out(in_shape):
    return in_shape[0], in_shape[1], nb_chars
