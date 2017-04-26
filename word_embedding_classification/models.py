from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from word_embedding_classification.constants import PREDICTION_TYPE


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




def get_num_output_nodes():
    if PREDICTION_TYPE == 'gender':
        return 2
