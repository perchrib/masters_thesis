# keras
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from constants import N_GRAM, FEATURE_MODEL, DIM_REDUCTION_SIZE, DIM_REDUCTION, ACTIVATION, \
    OUTPUT_ACTIVATION, DROPOUT, L1, L2, MODEL_TYPE, LAYER_PENALTY, DROPOUT_FIRST_LAYER, BATCH_NORM

from keras import regularizers



def generate_model(input_shape, output_layer, hidden_layers):
    if DROPOUT_FIRST_LAYER:
        print("Dropout added on input layer")
        input_layers = Dropout(DROPOUT)(input_shape)
    elif BATCH_NORM:
        print("Batch normalization added")
        input_layers = BatchNormalization()(input_shape)
    else:
        input_layers = input_shape

    for i, layer in enumerate(hidden_layers):
        if i == LAYER_PENALTY:
            if L1 != 0 or L2 != 0:
                l1_reg = regularizers.l1(L1)
                l2_reg = regularizers.l2(L2)
                if L1 == 0:
                    l1_reg = None
                if L2 == 0:
                    l2_reg = None
                print("Regularization added on layer: ", layer)
                input_layers = Dense(layer, activation=ACTIVATION,
                                     kernel_regularizer=l2_reg,
                                     activity_regularizer=l1_reg)(input_layers)

        print("dens layer added, layer: ", layer)
        input_layers = Dense(layer, activation=ACTIVATION)(input_layers)

        if DROPOUT > 0:
            print("Dropout added on Layer: ", layer)
            input_layers = Dropout(DROPOUT)(input_layers)

    input_layers = Dense(output_layer, activation=OUTPUT_ACTIVATION)(input_layers)

    return input_layers


def generate_name(model_name, layers):
    constant_names = ["dropout", "l1_reg", "l2_reg"]
    model_constants = [DROPOUT, L1, L2]
    name = model_name
    for l in layers:
        name += "_" + str(l)

    for i in range(len(model_constants)):
        c_value = model_constants[i]
        if c_value > 0:
            c_name = constant_names[i]
            name += "_" + c_name + "_" + str(c_value)

    return name


def get_ann_model(input_length, output_length, layers):
    # model information

    model_name = generate_name(MODEL_TYPE, layers)
    model_info = ""

    # create model
    inputs = Input(shape=(input_length,))
    outputs = generate_model(inputs, output_length, layers)
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, model_info


def get_logistic_regression(input_length, output_length):
    activation = "sigmoid"
    inputs = Input(shape=(input_length,))
    predictions = Dense(output_length, activation=activation)(inputs)
    model = Model(inputs=inputs, outputs=predictions, name="logistic_regression")
    model_info = ""  # get_model_info()
    return model, model_info
