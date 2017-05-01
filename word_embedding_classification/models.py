from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, GRU, MaxPooling1D, Conv1D, merge, Input
from word_embedding_classification.constants import PREDICTION_TYPE, MAX_SEQUENCE_LENGTH


def get_word_model_3xsimple_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="3xSimpleLSTM")

    model.add(embedding_layer)
    model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_2x512_256_lstm(embedding_layer, nb_output_nodes):
    # model = Sequential(name="2x512_256LSTM")
    #
    # model.add(embedding_layer)
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(256))
    # model.add(Dense(nb_output_nodes, activation='softmax'))

    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = embedding_layer(tweet_input)
    embedding = LSTM(512, return_sequences=True)(embedding)
    embedding = Dropout(0.5)(embedding)
    embedding = LSTM(512, return_sequences=True)(embedding)
    embedding = Dropout(0.5)(embedding)
    embedding = LSTM(256, return_sequences=False)(embedding)

    output = Dense(nb_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output, name="2x512_256LSTM")
    extra_info = ["Dropout: 0.5"]

    return model, extra_info


def get_word_test(embedding_layer, nb_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = embedding_layer(tweet_input)
    embedding = LSTM(512, return_sequences=True)(embedding)
    embedding = Dropout(0.5)(embedding)
    embedding = LSTM(512, return_sequences=True)(embedding)
    embedding = Dropout(0.5)(embedding)
    embedding = LSTM(256, return_sequences=False)(embedding)

    output = Dense(nb_output_nodes, activation='softmax')(embedding)
    model = Model(input=tweet_input, output=output) # TODO: Name
    extra_info = ["Dropout: 0.5"]
    return model, extra_info


def get_word_model_2x512_256_gru(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x512_256GRU")

    model.add(embedding_layer)
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(256))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_2x512_256_lstm_128_full(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x512_256LSTM_128full")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_3x512_256_lstm_128_full(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x512_256LSTM_128full")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_3x512_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="3x512_LSTM")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_3x512_rec_dropout_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="3x512_recDropoutLSTM")

    model.add(embedding_layer)
    model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_2x1024_512_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x1024_512_LSTM")

    model.add(embedding_layer)
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    model_info = []
    return model, model_info


def get_word_model_Conv_BiLSTM(embedding_layer, nb_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = embedding_layer(tweet_input)

    kernel_size = 5
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5
    lstm_drop = 0.2
    lstm_drop_rec = 0.2
    dense_drop = 0.5

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
    output = Dropout(dense_drop)(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(dense_drop)(output)
    output = Dense(nb_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "LSTM dropout: %f, LSTM recurrent dropout %f" % (lstm_drop, lstm_drop_rec),
                  "Conv dropout: %f" % conv_dropout, "Dense dropout: %f" % dense_drop]
    return model, model_info


def get_word_model_3xConv_BiLSTM(embedding_layer, nb_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = embedding_layer(tweet_input)

    kernel_size = [5, 3, 3]
    filters = [196, 196, 256]

    dense_drop = 0.5

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
    output = Dropout(dense_drop)(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(dense_drop)(output)
    output = Dense(nb_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='3xConv_BiLSTM')

    model_info = ["LSTM dropout = 0.2, 0.2"]

    return model, model_info
