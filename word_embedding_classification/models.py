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

    return model


def get_word_model_2x512_256_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x512_256LSTM")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dense(nb_output_nodes, activation='softmax'))

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

    return model


def get_word_model_2x512_256_lstm_128_full(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x512_256LSTM_128full")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    return model


def get_word_model_3x512_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="3x512_LSTM")

    model.add(embedding_layer)
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    return model


def get_word_model_3x512_rec_dropout_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="3x512_recDropoutLSTM")

    model.add(embedding_layer)
    model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(512, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    return model


def get_word_model_2x1024_512_lstm(embedding_layer, nb_output_nodes):
    model = Sequential(name="2x1024_512_LSTM")

    model.add(embedding_layer)
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(nb_output_nodes, activation='softmax'))

    return model



def get_word_model_Conv_BiLSTM(embedding_layer, nb_output_nodes):
    tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
    embedding = embedding_layer(tweet_input)

    kernel_size = 5
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5
    lstm_drop = 0.2
    lstm_drop_rec = 0.2

    embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(embedding)

    embedding = Dropout(conv_dropout)(embedding)
    embedding = MaxPooling1D(pool_length=pool_length)(embedding)

    forward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu')(embedding)
    backward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu',
                    go_backwards=True)(embedding)

    output = merge([forward, backward], mode='concat', concat_axis=-1)
    # output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(nb_output_nodes, activation='softmax')(output)
    model = Model(input=tweet_input, output=output, name='Conv_BiLSTM')

    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length, "LSTM dropout: %f, LSTM recurrent dropout %f" % (lstm_drop, lstm_drop_rec), "Conv dropout: %f" % conv_dropout, "No dense dropout"]
    return model, model_info





def get_num_output_nodes():
    if PREDICTION_TYPE == 'gender':
        return 2
