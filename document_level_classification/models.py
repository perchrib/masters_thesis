# keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from constants import N_GRAM, FEATURE_MODEL, DIM_REDUCTION_SIZE, DIM_REDUCTION, ACTIVATION, \
    OUTPUT_ACTIVATION, DROPOUT, LAYERS, L1, L2, MODEL_TYPE

from keras import regularizers
def generate_model(input_shape, output_layer, hidden_layers):

    input_layers = input_shape
    l1_reg = regularizers.l1(L1)
    l2_reg = regularizers.l2(L2)
    if l1_reg == 0:
        l1_reg = None
    if l2_reg == 0:
        l2_reg = None
    for i, layer in enumerate(hidden_layers):
        input_layers = Dense(layer, activation=ACTIVATION,
                             kernel_regularizer=l2_reg,
                             activity_regularizer=l1_reg)(input_layers)
        if i > 0 and DROPOUT > 0:
            input_layers = Dropout(DROPOUT)(input_layers)

    input_layers = Dense(output_layer, activation=OUTPUT_ACTIVATION)(input_layers)

    return input_layers


def generate_name(model_name):
    constant_names = ["dropout", "l1_reg", "l2_reg"]
    model_constants = [DROPOUT, L1, L2]
    name = model_name
    for l in LAYERS:
        name += "_" + str(l)

    for i in range(len(model_constants)):
        c_value = model_constants[i]
        if c_value > 0:
            c_name = constant_names[i]
            name += "_" + c_name + "_" + c_value

    return name


def check_if_zero(integer):
    if integer == 0:
        return None
    return integer


def get_ann_model(input_length, output_length):
    # parameters
    dropout = check_if_zero(DROPOUT)
    l1 = check_if_zero(L1)
    l2 = check_if_zero(L2)

    # model information
    model_name = generate_name(MODEL_TYPE)
    model_info = ["\n--- Regularisation ---\n",
                  "\t-Dropout: %s" % dropout,
                  "\t-L1: %s" % l1,
                  "\t-L2: %s" % l2,
                  "\n--- Feature Info ---\n",
                  "\t-Embedding: %s" % FEATURE_MODEL,
                  "\t-Ngram: %s" % (N_GRAM, ),
                  "\t-Autoencoder: %s" % DIM_REDUCTION,
                  "\t-Reduction size: %s" % DIM_REDUCTION_SIZE]
    # create model
    inputs = Input(shape=(input_length,))
    outputs = generate_model(inputs, output_length, LAYERS)
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, model_info


def get_512_256_128(input_length, output_length):
    dropout = 0.5
    activation = "relu"
    inputs = Input(shape=(input_length,))
    x = Dense(512, activation=activation)(inputs)
    x = Dense(256, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation=activation)(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions, name="512_256_128 WIth encoder Reduction size: %i" % input_length)

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network with encoder Reduction"]
    return model, model_info

def get_2048_1024_512(input_length, output_length):
    dropout = 0.5
    activation = "relu"
    inputs = Input(shape=(input_length,))
    #x = BatchNormalization()(inputs)
    x = Dense(2048, activation=activation)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation=activation)(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions, name="2048_1024_512")

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network With Encoder Reduction"]
    return model, model_info


def get_4096_2048_1024_512(input_length, output_length):
    dropout = 0.5
    activation = "relu"
    inputs = Input(shape=(input_length,))
    x = Dense(4096, activation=activation)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(2048, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation=activation)(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions, name="4096_2048_1024_512")

    model_info = ["Dropout: %f" % dropout, "Feed Forward Network"]
    return model, model_info

def get_1024_512(input_length, output_length):
    dropout = 0.5
    activation = "relu"
    inputs = Input(shape=(input_length,))
    x = Dense(1024, activation=activation)(inputs)
    x = Dropout(dropout)(x)
    x = Dense(512, activation=activation)(x)
    predictions = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions, name="1024_512")

    model_info = ["Dropout: %f" %dropout, "Feed Forward Network with encoder Reduction"]
    return model, model_info


def get_logistic_regression(input_length, output_length):
    activation = "sigmoid"
    inputs = Input(shape=(input_length,))
    predictions = Dense(output_length, activation=activation)(inputs)
    model = Model(inputs=inputs, outputs=predictions, name="Logistic Regression")
    model_info = ["Logistic Regression"]
    return model, model_info
