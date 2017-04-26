import tensorflow as tf
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Conv1D, MaxPooling1D, merge, Flatten
from character_level_classification.constants import MAX_SEQUENCE_LENGTH



def get_char_model_3xConv_2xBiLSTM(num_output_nodes, char_num):

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]

    # kernel_size = [7, 5, 3]
    # filters = [256, 256]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv_2xBiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2"]
    return model, model_info



def get_char_model_2xConv_BiLSTM(num_output_nodes, char_num):

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]

    # kernel_size = [7, 5, 3]
    filters = [256, 256]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='2xConv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2", "No dense dropout", "filters = [256, 256]"]
    return model, model_info

def get_char_model_2x256_lstm_full(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    encoding = LSTM(256, return_sequences=True)(embedding)
    encoding = LSTM(256, return_sequences=False)(encoding)

    output = Dense(200, activation='relu')(encoding)
    output = Dense(num_output_nodes, activation='softmax')(output)

    model = Model(inputs=tweet_input, outputs=output, name="2x256LSTM_full")

    model_info = []
    return model, model_info


def get_char_model_BiLSTM_full(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    # dropout = 0.5, recurrent_dropout = 0.5
    forward = LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(0.5)(encoding)
    output = Dense(256, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='BiLSTM_full')

    extra_info = ["LSTM dropout = 0.2, 0.2"]
    return model, extra_info


def get_char_model_3xConv(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]
    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
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

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]
    # kernel_size = [7, 5, 3]
    # filters = [196, 196, 256]
    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    embedding = LSTM(512, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(embedding)

    # embedding = Dropout(0.5)(embedding)
    embedding = Dense(256, activation='relu')(embedding)
    # embedding = Dropout(0.5)(embedding)
    output = Dense(num_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output, name='3xConv_LSTM')

    model_info = ["LSTM dropout = 0.5, 0.2", "No dense dropout"]
    return model, model_info


def get_char_model_3xConv_4xBiLSTM(num_output_nodes, char_num):

    """
    im(x)))
ValueError: Input 0 is incompatible with layer lstm_5: expected ndim=3, found nd
im=2

    :param num_output_nodes:
    :param char_num:
    :return:
    """
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = Lambda(one_hot, output_shape=one_hot_out)(tweet_input)

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]
    # kernel_size = [7, 5, 3]
    # filters = [196, 196, 256]
    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
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

    output = Dropout(0.5)(embedding)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv_4xBiLSTM')

    model_info = ["LSTM dropout = 0.5, 0.2"]
    return model, model_info


def get_char_model_Conv_BiLSTM(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]
    filters = [1024]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2", "No dense dropout", "filters = [1024]"]
    return model, model_info


def get_char_model_Conv_BiLSTM_2(num_output_nodes, char_num):


    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]
    filters = [1024]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(256, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2", "No dense dropout", "filters = [1024]", "LSTM layers 512, Dense 256"]
    return model, model_info

def get_char_model_Conv_BiLSTM_3(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 2]
    filters = [1024]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                                  kernel_size=kernel_size[i],
                                  activation='relu',
                                  kernel_initializer='glorot_uniform',
                                  subsample_length=1)(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2", "No dense dropout", "filters = [1024]", "kernel_size = [5, 3, 2]"]
    return model, model_info

def get_one_hot_layer(input_layer, nb_chars):
    return Lambda(K.one_hot, arguments={'num_classes': nb_chars}, output_shape=(input_layer.shape[1], nb_chars))(input_layer)

def one_hot(x):
    return tf.to_float(tf.one_hot(x, nb_chars, on_value=1, off_value=0, axis=-1))
    # return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0))


def one_hot_out(in_shape):
    return in_shape[0], in_shape[1], nb_chars
