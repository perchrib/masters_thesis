from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
from document_level_classification.constants import MODEL_DIR
import os


class Autoencoder:
    def __init__(self, reduction_dim, input_dim):
        self.name = str(input_dim)[:2] + "k_" + str(reduction_dim)
        self.reduction_dim = reduction_dim
        self.input_dim = input_dim
        self.model, self.encoder, self.decoder = self.build_model(self.input_dim)

    def build_model(self, input_dim):

        activation = 'tanh'
        activation_last_layer = 'softmax'

        #activation_last_layer = 'softmax'
        loss = 'categorical_crossentropy'

        #loss = 'mean_squared_error'
        optimizer = "adam"


        input_vector = Input(shape=(input_dim,))
        #
        # self.name += "_autoencoder_simple" + "_" + activation + "_" + activation_last_layer + "_" + loss
        # encoded = Dense(self.reduction_dim, activation=activation, activity_regularizer=regularizers.l1(10e-5))(
        #     input_vector)
        #
        # decoded = Dense(input_dim, activation=activation_last_layer)(encoded)
        #
        # # build autoencoder
        # autoencoder = Model(input_vector, decoded)
        # encoder = Model(input_vector, encoded)
        #
        # # build decoder
        # encoded_input = Input(shape=(self.reduction_dim,))
        # decoder_layer_1 = autoencoder.layers[-1](encoded_input)
        # decoder = Model(encoded_input, decoder_layer_1)

        ##########################################################################################################
        self.name += "_autoencoder_deep" + "_" + activation + "_" + activation_last_layer + "_" + loss
        encoded = Dense(1000, activation=activation, activity_regularizer=regularizers.l1(10e-5))(input_vector)
        encoded = Dense(self.reduction_dim, activation=activation)(encoded)

        decoded = Dense(1000, activation=activation, activity_regularizer=regularizers.l1(10e-5))(encoded)
        decoded = Dense(input_dim, activation=activation_last_layer)(decoded)

        # build autoencoder
        autoencoder = Model(input_vector, decoded)

        # build encoder
        encoder = Model(input_vector, encoded)

        # build decoder
        encoded_input = Input(shape=(self.reduction_dim,))
        decoder_layer_1 = autoencoder.layers[-2](encoded_input)
        decoder_layer_2 = autoencoder.layers[-1](decoder_layer_1)
        decoder = Model(encoded_input, decoder_layer_2)

        ############################################################################################################

        autoencoder.compile(optimizer=optimizer, loss=loss)

        return autoencoder, encoder, decoder

    def train_model(self, x_samples, x_test_samples):
        early_stopping = EarlyStopping(monitor='val_loss', patience=0)
        self.model.fit(x_samples, x_samples,
                       epochs=200,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(x_test_samples, x_test_samples),
                       callbacks=[early_stopping])

    def save_encoder(self):
        save_dir = "autoencoders"
        if not os.path.exists(os.path.join(MODEL_DIR, save_dir)):
            os.makedirs(os.path.join(MODEL_DIR, save_dir))
        self.encoder.save(os.path.join(MODEL_DIR, save_dir, self.name + ".h5"))
        print(self.name, " saved")


class DimReduction:
    def __init__(self, reduction_dim, train=True, encoder=None):
        self.reduction_dim = reduction_dim
        self.train = train
        self.autoencoder = None
        self.encoder = encoder

    def fit_transform(self, x_samples, x_test_samples=None):
        if self.train:
            self.autoencoder = Autoencoder(self.reduction_dim, x_samples.shape[1])
            self.autoencoder.train_model(x_samples, x_test_samples)
            self.train = False
            # save encoder model
            self.autoencoder.save_encoder()

        if self.encoder:
            reduced_x_samples = self.encoder.predict(x_samples)

        else:
            reduced_x_samples = self.autoencoder.encoder.predict(x_samples)

        return reduced_x_samples
