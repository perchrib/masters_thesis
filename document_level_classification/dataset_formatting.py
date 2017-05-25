from __future__ import print_function
from keras.utils import to_categorical

from document_level_classification.features import TF_IDF, BOW, SentimentAnalyzer, feature_adder
from constants import MAX_FEATURE_LENGTH, N_GRAM, DIM_REDUCTION, \
    DIM_REDUCTION_SIZE, CATEGORICAL, FEATURE_MODEL, C_BAG_OF_WORDS, C_TF_IDF, C_TF_IDF_DISSIMILARITY, \
    C_BAG_OF_WORDS_DISSIMILARITY, SENTIMENT_FEATURE
from preprocessors.dataset_preparation import split_dataset
import time
from helpers.helper_functions import get_time_format

from helpers.dimension_reduction import DimReduction
import numpy as np


def format_dataset_doc_level(texts, labels, metadata, is_test=False, feature_model_type=FEATURE_MODEL, n_gram=N_GRAM,
                             max_feature_length=MAX_FEATURE_LENGTH, feature_model=None, reduction_model=None):
    if not is_test:

        x_train, y_train, meta_train, x_val, y_val, meta_val = split_dataset(texts, labels, metadata, data_type_is_string=True)

        x_train_texts = x_train
        x_val_texts = x_val

        if feature_model_type == C_TF_IDF:
            print("USING: ", C_TF_IDF)
            feature_model = TF_IDF(x_train, y_train, max_feature_length, n_gram)

        elif feature_model_type == C_BAG_OF_WORDS:
            print("USING: ", C_BAG_OF_WORDS)
            feature_model = BOW(x_train, n_gram, max_features=max_feature_length)

        elif feature_model_type == C_BAG_OF_WORDS_DISSIMILARITY:
            print("USING: ", C_BAG_OF_WORDS_DISSIMILARITY)
            vocabulary = TF_IDF(x_train, y_train, max_feature_length, n_gram, dissimilarity_vocabulary=True).train_vocabulary
            feature_model = BOW(x_train, n_gram, vocabulary=vocabulary, max_features=max_feature_length)

        elif feature_model_type == C_TF_IDF_DISSIMILARITY:
            print("USING: ", C_TF_IDF_DISSIMILARITY)
            feature_model = TF_IDF(x_train, y_train, max_feature_length, n_gram, dissimilarity_vocabulary=True)


        x_train = feature_model.fit_to_training_data()
        x_val = feature_model.fit_to_new_data(x_val)

        if SENTIMENT_FEATURE:
            SA = SentimentAnalyzer()
            print("Validate Training Set Sentiments")

            x_train_sentiments = SA.analyze(x_train_texts)
            print("SHAPE X_TRAIN: ", x_train.shape)
            del x_train_texts

            x_train = feature_adder(x_train, x_train_sentiments, feature_model)

            print("Validate Validation Set Sentiments")

            x_val_sentiments = SA.analyze(x_val_texts)
            print("SHAPE X_VAL: ", x_val.shape)
            del x_val_texts

            x_val = feature_adder(x_val, x_val_sentiments, feature_model)

        if DIM_REDUCTION:
            start = time.time()
            print("Starting With Dimensionality Reduction From Size %i to %i..." % (x_train.shape[1], DIM_REDUCTION_SIZE))

            if not reduction_model:
                reduction_model = DimReduction(DIM_REDUCTION_SIZE, train=True)

            elif reduction_model:
                print("REDUCTION MODEL::: ", reduction_model)
                print("Load pretrained encoder...")
                reduction_model = DimReduction(DIM_REDUCTION_SIZE, train=False, encoder=reduction_model)

            x_train = reduction_model.fit_transform(x_train, x_val)
            x_val = reduction_model.fit_transform(x_val)

            print("Reduction Time: ", get_time_format(time.time() - start))

        if CATEGORICAL:
            y_train = to_categorical(y_train)
            y_val = to_categorical(y_val)

        else:
            y_train = np.asarray([[i] for i in y_train])
            y_val = np.asarray([[i] for i in y_val])

        return x_train, y_train, meta_train, x_val, y_val, meta_val, feature_model, reduction_model

    elif is_test:
        x_test = feature_model.fit_to_new_data(texts)

        if SENTIMENT_FEATURE:
            SA = SentimentAnalyzer()
            print("Validate Test Set Sentiments")
            x_test_sentiments = SA.analyze(texts)
            x_test = feature_adder(x_test, x_test_sentiments, feature_model)

        if DIM_REDUCTION:
            x_test = reduction_model.fit_transform(x_test)

        if CATEGORICAL:
            y_test = to_categorical(labels)

        else:
            y_test = np.asarray([[i] for i in labels])

        return x_test, y_test, metadata,


