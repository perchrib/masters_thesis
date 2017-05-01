# keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model


def get_2048_1024_512(input_length, output_length):
    dropout = 0.5

    inputs = Input(shape=(input_length,))
    #x = BatchNormalization()(inputs)
    x = Dense(2048, activation='sigmoid')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation='sigmoid')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='sigmoid')(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network"]
    return model, model_info


def get_4096_2048_1024_512(input_length, output_length):
    dropout = 0.5

    inputs = Input(shape=(input_length,))
    x = Dense(4096, activation='relu')(inputs)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network"]
    return model, model_info
