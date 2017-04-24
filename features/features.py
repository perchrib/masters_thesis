from __future__ import absolute_import
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
import sys
sys.path.insert(0, "/Users/per/Documents/NTNU_Courses/5th_year/2-semester/master/github/master")

from text_mining.helpers import word_tokenize
from text_mining.dataset_characteristics import most_common

from collections import Counter

class TF_IDF():
    def __init__(self, documents, max_len_features, ngram_range=(1, 1)):
        # All tweets containing a list
        self.train_docs = np.asarray(documents)
        self.train_max_len_features = max_len_features
        self.train_unique_tokens = self.n_frequent_words_in_texts()
        self.train_vocabulary = self.train_unique_tokens.keys()

        # contains the vocabulary, count word frequencies
        self.train_cv = CountVectorizer(vocabulary=self.train_vocabulary, ngram_range=ngram_range)
        # Construct the bag of word model
        self.train_bow = self.train_cv.fit_transform(self.train_docs)

        self.tfidf_transformer = TfidfTransformer()

    def fit_to_training_data(self):
        tfidf_features = self.tfidf_transformer.fit_transform(self.train_bow)
        return tfidf_features.toarray()

    def fit_to_new_data(self, new_texts):
        new_cv = self.train_cv.transform(np.asarray(new_texts))
        new_tfidf_features = self.tfidf_transformer.transform(new_cv)
        return new_tfidf_features.toarray()

    def n_frequent_words_in_texts(self):
        return most_common(Counter(word_tokenize(self.train_docs)), self.train_max_len_features)


    # def tf_idf_1(documents):
    #     vocabulary = n_frequent_words_in_texts(documents, 2000)
    #     cv = CountVectorizer(vocabulary=vocabulary)
    #     counts = cv.fit_transform(np.asarray(documents))
    #     # print "Count " , counts.__dict__,cv.vocabulary_, "\n"
    #     tfidf_transformer = TfidfTransformer()
    #     tf_idfs = tfidf_transformer.fit_transform(counts)
    #     print tf_idfs.shape
    #     return tf_idfs




if __name__ == "__main__":
    from preprocessors.parser import Parser
    from helpers.global_constants import GENDER
    from preprocessors.dataset_preparation import prepare_dataset, prepare_dataset_women, prepare_dataset_men
    from keras.utils.np_utils import to_categorical

    parser = Parser()
    texts, labels, metadata, labels_index = prepare_dataset(GENDER)
    feature_length = 5000
    #texts = ["This is a test for a given input", "we are the world, we are the people", "this is it"]
    print "Removing Stopwords..."
    parsed_texts = parser.remove_stopwords(texts)

    np.random.seed(1337)
    indices = np.arange(len(texts))
    print texts[0]
    texts_and_indices = list(zip(texts, indices))
    print texts_and_indices[0]
    np.random.shuffle(texts_and_indices)
    # shuffled texts and indices
    print texts_and_indices[0]
    texts, indices = zip(*texts_and_indices)
    texts, indices = np.asarray(texts), np.asarray(list(indices))
    print "Indices", len(labels)
    # shuffled labels
    labels = to_categorical(np.asarray(labels))
    labels = labels[indices]
    nb_validation_samples = int(0.15 * len(texts))

    tfidf = TF_IDF(texts[:-nb_validation_samples], feature_length)
    x_train = tfidf.fit_to_training_data()
    y_train = labels[:-nb_validation_samples]
    x_val = tfidf.fit_to_new_data(texts[-nb_validation_samples:])
    y_val = labels[-nb_validation_samples:]

    # keras
    from keras.layers import Input, Dense
    from keras.models import Model



    inputs = Input(shape=(feature_length,))
    x = Dense(2048, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=[x_val, y_val])


    #
    #
    # def tf_idf(documents):
    #     fe = FeatureExtraction(documents)
    #     np.set_printoptions(precision=2)
    #     features = fe.tfidf.fit_transform(fe.count.fit_transform(fe.docs)).toarray()
    #     print("SUM: ", sum(features[0]), " length: ", len(features[0]))
    #     print features
    #     return features
