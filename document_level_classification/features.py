from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
import sys
sys.path.insert(0, "/Users/per/Documents/NTNU_Courses/5th_year/2-semester/master/github/master")

from text_mining.helpers import word_tokenize
from text_mining.dataset_characteristics import most_common
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def feature_adder(f1, f2):
    sum_ = np.zeros((f1.shape[0], f1.shape[1] + 1))
    print("F1 SHAPE: ", f1.shape)
    print("F2 SHAPE: ", f2.shape)

    for index_row in range(f1.shape[0]):
        sum_[index_row] = np.hstack([f1[index_row], f2[index_row]])
    print("SUMMED Feature Vector shape: ", sum_.shape)
    del f1
    del f2
    return sum_


class SentimentAnalyzer():

    def analyze_and_merge(self, documents, merge_documents):
        analyzer = SIA()
        combined_features = []
        print("Running Foor Loop...")
        number_of_documents = len(documents)
        for index in range(number_of_documents):
            t = documents[index]
            result = analyzer.polarity_scores(t)
            result['compound'] = -1
            sentiment = max(result, key=result.get)
            if sentiment == 'pos':
                combined_features.append(np.hstack([merge_documents[index], 1]))
            elif sentiment == 'neg':
                combined_features.append(np.hstack([merge_documents[index], 0]))
            else:
                combined_features.append(np.hstack([merge_documents[index], 0.5]))
            if index % 10000 == 0:
                print(index)
            #documents = np.delete(documents, 0, axis=0)
            #merge_documents = np.delete(merge_documents, 0, axis=0)

        del documents
        del merge_documents
        return np.asarray(combined_features)


class BOW():
    def __init__(self, documents, ngram_range=(1, 1), vocabulary=None, max_features=None, dissimilarity_vocabulary=False):
        self.documents = documents
        self.vocabulary = vocabulary
        self.ngram_range = ngram_range
        self.cv = CountVectorizer(vocabulary=self.vocabulary, max_features=max_features, ngram_range=self.ngram_range)
        self.bag = self.cv.fit_transform(self.documents)

    def get_count(self):
        return self.cv

    def get_bag(self):
        return self.bag.toarray()

    def fit_to_training_data(self):
        return self.get_bag()

    def fit_to_new_data(self, texts):
        return self.cv.transform(texts).toarray()


class TF_IDF:
    def __init__(self, documents, labels, max_len_features, ngram_range=(1, 1), dissimilarity_vocabulary=False):
        # All tweets containing a list of strings ["hello #yolo", "yoyo @google"]
        self.labels = labels
        self.train_docs = np.asarray(documents)
        self.train_max_len_features = max_len_features
        self.ngram_range = ngram_range

        if not dissimilarity_vocabulary:
            self.train_unique_tokens = self.n_frequent_words_in_texts()
            self.train_vocabulary = self.train_unique_tokens.keys()

        else:
            self.train_vocabulary, self.train_vocabulary_counts = self.n_frequent_ngram_token_dissimilarity_vocabulary()

        """New"""
        bow = BOW(self.train_docs, self.ngram_range, self.train_vocabulary)
        self.train_counts = bow.get_count()
        self.train_bow = bow.get_bag()

        self.tfidf_transformer = TfidfTransformer()

    def fit_to_training_data(self):
        tfidf_features = self.tfidf_transformer.fit_transform(self.train_bow)
        return tfidf_features.toarray()

    def fit_to_new_data(self, new_texts):
        new_cv = self.train_counts.transform(np.asarray(new_texts))
        new_tfidf_features = self.tfidf_transformer.transform(new_cv)
        return new_tfidf_features.toarray()

    def get_count(self):
        self.train_counts

    def n_frequent_words_in_texts(self):
        """
        :return: A Counter with n frequent words in the text ie Counter({"#yoloy":500, "hey":300}) 
        """
        return most_common(Counter(word_tokenize(self.train_docs)), self.train_max_len_features)

    def n_frequent_ngram_token_dissimilarity_vocabulary(self):
        male_texts = [text for i, text in enumerate(self.train_docs) if self.labels[i] == 0]
        female_texts = [text for i, text in enumerate(self.train_docs) if self.labels[i] == 1]

        print("Create ngram-Token (n = %s) Vocabulary for Training Set" %(self.ngram_range,))
        all_token_vocabulary_counts = self.get_ngram_occurrences_counter(self.train_docs, self.ngram_range, self.train_max_len_features, "training")
        print("Create ngram-Token (n = %s) Vocabulary for Male in Training Set" %(self.ngram_range,))
        male_token_vocabulary_counts = self.get_ngram_occurrences_counter(male_texts, self.ngram_range, self.train_max_len_features, "male")
        print("Create ngram-Token (n = %s) Vocabulary for Female in Training Set" %(self.ngram_range,))
        female_token_vocabulary_counts = self.get_ngram_occurrences_counter(female_texts, self.ngram_range, self.train_max_len_features, "female")
        total_men_tokens = sum(male_token_vocabulary_counts.values())
        total_female_tokens = sum(female_token_vocabulary_counts.values())

        different_value_vocabulary_counts = dict()
        print("Create Vocabulary for Training Set (%i Most Distinct ngram-Token (n = %s) Between Gender in Training Set)" %(self.train_max_len_features, self.ngram_range,))
        for token in all_token_vocabulary_counts.keys():
            number_of_men_token = male_token_vocabulary_counts[token]
            number_of_female_token = female_token_vocabulary_counts[token]
            diff_value = abs((number_of_men_token/total_men_tokens) - (number_of_female_token/total_female_tokens))
            different_value_vocabulary_counts[token] = diff_value

        #tokens = sorted(different_value_vocabulary_counts, key=different_value_vocabulary_counts.get, reverse=True)

        #for key in tokens:
        #    print key, " : ", different_value_vocabulary_counts[key]

        return different_value_vocabulary_counts.keys(), different_value_vocabulary_counts


    def get_ngram_occurrences_counter(self, texts, ngram_range, max_features, type):
        print("CountVectorize for %s set create..." %type)
        CV = CountVectorizer(ngram_range=ngram_range, lowercase=False, token_pattern="[^ ]+", max_features=max_features)
        print("CountVectorize for %s set created" %type)
        print("BoW for %s set create..." %type)
        bag_of_words = CV.fit_transform(texts)
        print("CountVectorize for %s set created" %type)
        # represent vocabulary for arbitrary ngrams
        vocabulary = CV.vocabulary_

        # sums occurrences of tokens in the bag of words model

        #n_occurrences = np.sum(bag_of_words.toarray(), axis=0)
        bag_of_words_array = bag_of_words.toarray()
        print(bag_of_words_array.shape)
        #n_occurrences = sum_col(bag_of_words_array)
        print("Summing %s BoW..." %type)
        n_occurrences = np.sum(bag_of_words_array, axis=0)
        print("Summed")
        n_counts = Counter({k: n_occurrences[v] for k, v in vocabulary.iteritems()})
        return n_counts
