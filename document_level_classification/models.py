# keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model


def get_2048_1024_512(input_length, output_length):
    dropout = 0.5

    inputs = Input(shape=(input_length,))
    x = Dense(2048, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network"]
    return model, model_info
