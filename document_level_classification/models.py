# keras
from keras.layers import Input, Dense
from keras.models import Model


def get_2048_1024_512(input_length, output_length):
    inputs = Input(shape=(input_length,))
    x = Dense(2048, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    model_info = ["Feed Forward Netwotk"]
    return model, model_info
