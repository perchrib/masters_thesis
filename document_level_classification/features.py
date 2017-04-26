from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
import sys
sys.path.insert(0, "/Users/per/Documents/NTNU_Courses/5th_year/2-semester/master/github/master")

from text_mining.helpers import word_tokenize
from text_mining.dataset_characteristics import most_common
from keras.utils.np_utils import to_categorical

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


def shuffle(x_input, y_label):
    if len(x_input) != len(y_label):
        raise TypeError("Not Same Length")
    else:
        x_input, y_label = np.asarray(x_input), np.asarray(y_label)
        np.random.seed(1337)
        indices = np.arange(len(x_input))

        texts_and_indices = list(zip(x_input, indices))
        np.random.shuffle(texts_and_indices)
        x_input, indices = zip(*texts_and_indices)
        x_input, indices = np.asarray(x_input), np.asarray(indices)
        y_label = to_categorical(y_label)
        y_label = y_label[indices]
        return x_input, y_label

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

    # shuffle text and labels
    texts, labels = shuffle(texts, labels)

    nb_validation_samples = int(0.15 * len(texts))

    tfidf = TF_IDF(texts[:-nb_validation_samples], feature_length)
    x_train = tfidf.fit_to_training_data()
    y_train = labels[:-nb_validation_samples]
    x_val = tfidf.fit_to_new_data(texts[-nb_validation_samples:])
    y_val = labels[-nb_validation_samples:]




    #
    #
    # def tf_idf(documents):
    #     fe = FeatureExtraction(documents)
    #     np.set_printoptions(precision=2)
    #     document_level_classification = fe.tfidf.fit_transform(fe.count.fit_transform(fe.docs)).toarray()
    #     print("SUM: ", sum(document_level_classification[0]), " length: ", len(document_level_classification[0]))
    #     print document_level_classification
    #     return document_level_classification
