from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def get_word_model(num_output_nodes, embedding_layer):
    model = Sequential(name="3xSimpleLSTM")

    model.add(embedding_layer)
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dense(num_output_nodes, activation='softmax'))

    return model