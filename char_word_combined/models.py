from character_level_classification.models import get_one_hot_layer
from char_word_combined.constants import MAX_CHAR_SEQUENCE_LENGTH as c_MAX_SEQUENCE_LENGTH, MAX_WORD_SEQUENCE_LENGTH as w_MAX_SEQUENCE_LENGTH

from keras.regularizers import l2
from keras.layers import Input, Dense, LSTM, Dropout, Lambda, Conv1D, MaxPooling1D, concatenate, Flatten, Masking
from keras.models import Model

def get_cw_model(embedding_layer, num_output_nodes, char_num):

    ## Char level --
    c_tweet_input = Input(shape=(c_MAX_SEQUENCE_LENGTH,), dtype='int64', name='c_input')
    c_embedding = get_one_hot_layer(c_tweet_input, char_num)

    kernel_size = 5
    filters = 1024
    pool_length = 2
    conv_dropout = 0.5
    lstm_drop = 0.2
    lstm_drop_rec = 0.2
    dense_drop = 0.5

    c_embedding = Conv1D(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_initializer='glorot_uniform')(c_embedding)

    c_embedding = Dropout(conv_dropout)(c_embedding)
    c_embedding = MaxPooling1D(pool_length=pool_length)(c_embedding)

    forward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu')(
        c_embedding)
    backward = LSTM(256, return_sequences=False, dropout=lstm_drop, recurrent_dropout=lstm_drop_rec, consume_less='gpu',
                    go_backwards=True)(c_embedding)

    char_output = concatenate([forward, backward])
    # output = Dropout(0.5)(output)
    char_output = Dense(256, activation='relu')(char_output)
    # output = Dropout(0.5)(output)

    ## Word Level
    w_tweet_input = Input(shape=(w_MAX_SEQUENCE_LENGTH,), dtype='int64', name='w_input')
    w_embedding = embedding_layer(w_tweet_input)
    w_embedding = LSTM(512, return_sequences=True)(w_embedding)
    w_embedding = Dropout(dense_drop)(w_embedding)
    w_embedding = LSTM(512, return_sequences=True)(w_embedding)
    w_embedding = Dropout(dense_drop)(w_embedding)
    word_output = LSTM(256, return_sequences=False)(w_embedding)

    ## Merge
    encoding = concatenate([char_output, word_output])
    encoding = Dense(512, kernel_regularizer=l2(), activation='relu')(encoding)
    encoding = Dropout(dense_drop)(encoding)
    encoding = Dense(256, kernel_regularizer=l2(), activation='relu')(encoding)
    output = Dense(num_output_nodes, activation='softmax')(encoding)
    model = Model(inputs=[c_tweet_input, w_tweet_input], output=output, name="Conv_BiLSTM_3xLSTM")


    model_info = ["Kernel_size: %i" % kernel_size, "Filters: %i" % filters, "Pool length: %i" % pool_length,
                  "LSTM char dropout: %f, LSTM char recurrent dropout %f" % (lstm_drop, lstm_drop_rec),
                  "Conv dropout: %f" % conv_dropout, "No dense drop on char", "Dense drop %f" % dense_drop]

    return model, model_info