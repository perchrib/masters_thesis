import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Convolution1D, MaxPooling1D, merge, Flatten, TimeDistributed
from character_level_classification.constants import MAX_SEQUENCE_LENGTH, MAX_CHAR_SENT_LENGTH, MAX_SENTENCE_LENGTH


nb_chars = None


def get_char_model_3xConv_Bi_lstm_sent(num_output_nodes, char_num):
    # Set number of chars for use in one hot encoder
    global nb_chars
    nb_chars = char_num

    tweet_input = Input(shape=(MAX_SENTENCE_LENGTH, MAX_CHAR_SENT_LENGTH), dtype='int64')
    sentence_input = Input(shape=(MAX_CHAR_SENT_LENGTH,), dtype='int64')
    sent_embedding = Lambda(one_hot, output_shape=one_hot_out)(sentence_input)

    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 256]
    # filter_length = [7, 5, 3]
    # nb_filter = [256, 256, 256]
    pool_length = 2

    # len of nb_filter = num conv layers
    for i in range(len(nb_filter)):
        sent_embedding = Convolution1D(nb_filter=nb_filter[i],
                                  filter_length=filter_length[i],
                                  activation='relu',
                                  init='glorot_uniform',
                                  subsample_length=1)(sent_embedding)

        sent_embedding = Dropout(0.5)(sent_embedding)
        sent_embedding = MaxPooling1D(pool_length=pool_length)(sent_embedding)

    forward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(sent_embedding)
    backward = LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(sent_embedding)

    sent_embedding = merge([forward, backward], mode='concat', concat_axis=-1)
    sent_embedding = Dropout(0.2)(sent_embedding)

    sentence_embedder = Model(input=sentence_input, output=sent_embedding)
    tweet_embedding = TimeDistributed(sentence_embedder)(tweet_input)
    # name='3xConv_2xBiLSTM'

    forward = LSTM(80, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu')(tweet_embedding)
    backward = LSTM(80, return_sequences=False, dropout=0.5, recurrent_dropout=0.2, consume_less='gpu',
                    go_backwards=True)(tweet_embedding)

    tweet_embedding = merge([forward, backward], mode='concat', concat_axis=-1)
    tweet_embedding = Dropout(0.3)(tweet_embedding)
    tweet_embedding = Dense(128, activation='relu')(tweet_embedding)
    output = Dense(num_output_nodes, activation='softmax')(tweet_embedding)

    model = Model(input=tweet_input, output=output, name='3xConv_BiLSTM_sent')
    model_info = ["LSTM dropout = 0.5, 0.2", "Dense dropout: 0.3"]

    return model, model_info

def one_hot(x):
    return tf.to_float(tf.one_hot(x, nb_chars, on_value=1, off_value=0, axis=-1))
    # return tf.to_float(tf.one_hot(x, chars, on_value=1, off_value=0))


def one_hot_out(in_shape):
    return in_shape[0], in_shape[1], nb_chars
