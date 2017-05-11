from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.decomposition import PCA

class Autoencoder:

    def __init__(self, reduction_dim, input_dim):

        self.reduction_dim = reduction_dim
        self.input_dim = input_dim
        self.model, self.encoder, self.decoder = self.build_model(self.input_dim)

    def build_model(self, input_dim):
        input_vector = Input(shape=(input_dim,))

        encoded = Dense(self.reduction_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_vector)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        # build autoencoder
        autoencoder = Model(input_vector, decoded)

        # build encoder
        encoder = Model(input_vector, encoded)

        # build decoder
        encoded_input = Input(shape=(self.reduction_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, encoder, decoder

    def train_model(self, x_samples, x_test_samples):
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        self.model.fit(x_samples, x_samples,
                        epochs=100,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test_samples, x_test_samples),
                        callbacks=[early_stopping])


class DimReduction:
    def __init__(self, reduction_dim, train=True):
        self.reduction_dim = reduction_dim
        self.train = train
        self.autoencoder = None

    def fit_transform(self, x_samples, x_test_samples=None):
        if self.train:
            self.autoencoder = Autoencoder(self.reduction_dim, x_samples.shape[1])
            self.autoencoder.train_model(x_samples, x_test_samples)
            self.train = False
        reduced_x_samples = self.autoencoder.encoder.predict(x_samples)
        return reduced_x_samples
















def pca(reduction_dim):
    pass


