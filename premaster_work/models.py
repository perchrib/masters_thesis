from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


class BaseSequentialModel(Sequential):

    def __init__(self, embedding_layer):
        super().__init__()
        self.add(embedding_layer)


class SeqLSTM(BaseSequentialModel):

    def __init__(self, embedding_layer, num_output_nodes):
        super().__init__(embedding_layer)
        self.add(LSTM(64, return_sequences=True))
        self.add(Dropout(0.5))
        self.add(LSTM(32, return_sequences=True))
        self.add(Dropout(0.5))
        self.add(LSTM(32))
        self.add(Dense(num_output_nodes, activation='softmax'))