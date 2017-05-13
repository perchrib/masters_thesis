# import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Conv1D, MaxPooling1D, merge, Flatten, Masking
from character_level_classification.constants import MAX_SEQUENCE_LENGTH



def get_char_model_4x512lstm(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    dropout = 0.5

    embedding = LSTM(512, return_sequences=True)(embedding)
    # embedding = Dropout(dropout)(embedding)
    embedding = LSTM(512, return_sequences=True)(embedding)
    # embedding = Dropout(dropout)(embedding)
    embedding = LSTM(512, return_sequences=True)(embedding)
    embedding = LSTM(512, return_sequences=False)(embedding)

    output = Dense(num_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output, name="4x512LSTM")
    extra_info = ["No dropout"]

    return model, extra_info


def get_char_model_512lstm(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    dropout = 0.5

    embedding = LSTM(512, return_sequences=False)(embedding)

    output = Dense(num_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output, name="512LSTM")
    extra_info = ["No dropout"]

    return model, extra_info


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
                           kernel_initializer='glorot_uniform')(embedding)

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
                           kernel_initializer='glorot_uniform')(embedding)

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
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    encoding = LSTM(256, return_sequences=True)(embedding)
    encoding = LSTM(256, return_sequences=False)(encoding)

    output = Dense(200, activation='relu')(encoding)
    output = Dense(num_output_nodes, activation='softmax')(output)

    model = Model(inputs=tweet_input, outputs=output, name="2x256LSTM_full")

    model_info = []
    return model, model_info


def get_char_model_BiLSTM(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    lstm_drop = 0.2
    lstm_drop_rec = 0.2

    forward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu')(
        embedding)
    backward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu',
                    go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    # encoding = Dropout(0.5)(encoding)
    output = Dense(num_output_nodes, activation='softmax')(encoding)
    model = Model(input=tweet_input, output=output, name='BiLSTM')

    model_info = ["LSTM dropout: %f, LSTM recurrent dropout %f" % (lstm_drop, lstm_drop_rec), "No merge dropout"]
    return model, model_info


def get_char_model_BiLSTM_full(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    lstm_drop = 0.2
    lstm_drop_rec = 0.2

    forward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu')(
        embedding)
    backward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu',
                    go_backwards=True)(embedding)

    encoding = merge([forward, backward], mode='concat', concat_axis=-1)
    # encoding = Dropout(0.5)(encoding)
    encoding = Dense(128, activation='relu')(encoding)
    # encoding = Dropout(0.5)(encoding)
    output = Dense(num_output_nodes, activation='softmax')(encoding)
    model = Model(input=tweet_input, output=output, name='BiLSTM_full')

    model_info = ["LSTM dropout: %f, LSTM recurrent dropout %f" % (lstm_drop, lstm_drop_rec), "No dense dropout"]
    return model, model_info


def get_char_model_3xConv(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]
    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                           kernel_size=kernel_size[i],
                           activation='relu',
                           kernel_initializer='glorot_uniform')(embedding)

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
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

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
                           kernel_initializer='glorot_uniform')(embedding)

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


def get_char_model_Conv_BiLSTM(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = 5
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5
    lstm_drop = 0.2
    lstm_drop_rec = 0.2
    dense_drop1 = 0.5
    dense_drop2 = 0.2

    embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(embedding)

    embedding = Dropout(conv_dropout)(embedding)
    embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu')(
        embedding)
    backward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    output = Dropout(dense_drop1)(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(dense_drop2)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "LSTM dropout: %f, LSTM recurrent dropout %f" % (lstm_drop, lstm_drop_rec),
                  "Conv dropout: %f" % conv_dropout, "Dense drop1 %f" % dense_drop1, "Dense drop2 %f" % dense_drop2]

    return model, model_info


def get_char_model_Conv_BiLSTM_2(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = 5
    filters = 512
    pool_length = 2
    conv_dropout = 0.5

    embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(embedding)

    embedding = Dropout(conv_dropout)(embedding)
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

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "Conv dropout: %f" % conv_dropout,
                  "LSTM dropout = 0.2, 0.2", "No dense dropout"]
    return model, model_info


def get_char_model_Conv_BiLSTM_3(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = 4
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5

    embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(embedding)

    embedding = Dropout(conv_dropout)(embedding)
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

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "Conv dropout: %f" % conv_dropout,
                  "LSTM dropout = 0.2, 0.2", "No dense dropout"]
    return model, model_info


def get_char_model_Conv_BiLSTM_4(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = 7
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5

    embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(embedding)

    embedding = Dropout(conv_dropout)(embedding)
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

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "Conv dropout: %f" % conv_dropout,
                  "LSTM dropout = 0.2, 0.2", "No dense dropout"]
    return model, model_info


def get_dummy_model(num_output_nodes, char_num):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = get_one_hot_layer(tweet_input, char_num)

    kernel_size = [5, 3, 3]
    filters = [1]

    pool_length = 2

    # len of filters = num conv layers
    for i in range(len(filters)):
        embedding = Conv1D(filters=filters[i],
                           kernel_size=kernel_size[i],
                           activation='relu',
                           kernel_initializer='glorot_uniform')(embedding)

        embedding = Dropout(0.5)(embedding)
        embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(1, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu')(embedding)
    backward = LSTM(1, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(1, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(num_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2", "No dense dropout", "filters = [1024]"]
    return model, model_info


def get_one_hot_layer(input_layer, nb_chars):
    """
    Create a layer that creates one hot vectors on the fly for memory efficiency
    :param input_layer: Input layer of network
    :param nb_chars: Number of characters in vocabulary
    :return:
    """
    return Lambda(one_hot, arguments={'num_classes': nb_chars}, output_shape=(MAX_SEQUENCE_LENGTH, nb_chars))(
        input_layer)


# Same as K.one_hot. Workaround because of global name tf error
def one_hot(indices, num_classes):
    import tensorflow as tf
    return tf.one_hot(indices, depth=num_classes, axis=-1)

## To be removed
# def one_hot_back(x):
#     return tf.to_float(tf.one_hot(x, nb_chars, on_value=1, off_value=0, axis=-1))
#     # return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0))
#
#
# def one_hot_out(in_shape):
#     return in_shape[0], in_shape[1], nb_chars
