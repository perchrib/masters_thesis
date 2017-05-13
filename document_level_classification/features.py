from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
import sys
sys.path.insert(0, "/Users/per/Documents/NTNU_Courses/5th_year/2-semester/master/github/master")

from text_mining.helpers import word_tokenize
from text_mining.dataset_characteristics import most_common
from collections import Counter
from helpers.helper_functions import print_progress


class TF_IDF():
    def __init__(self, documents, labels, max_len_features, ngram_range=(1, 1)):
        # All tweets containing a list of strings ["hello #yolo", "yoyo @google"]
        self.labels = labels
        self.train_docs = np.asarray(documents)
        self.train_max_len_features = max_len_features
        #self.train_unique_tokens = self.n_frequent_words_in_texts()

        self.ngram_range = ngram_range
        self.train_vocabulary = self.n_frequent_ngram_token_dissimilarity_vocabulary() # self.train_unique_tokens.keys()

        # contains the vocabulary, count word frequencies
        self.train_counts = CountVectorizer(vocabulary=self.train_vocabulary, ngram_range=self.ngram_range)

        # Construct the bag of word model and transform documnets into sparse features vectors
        self.train_bow = self.train_counts.fit_transform(self.train_docs)

        self.tfidf_transformer = TfidfTransformer()


    def fit_to_training_data(self):
        tfidf_features = self.tfidf_transformer.fit_transform(self.train_bow)
        return tfidf_features.toarray()

    def fit_to_new_data(self, new_texts):
        new_cv = self.train_counts.transform(np.asarray(new_texts))
        new_tfidf_features = self.tfidf_transformer.transform(new_cv)
        return new_tfidf_features.toarray()

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

        return different_value_vocabulary_counts.keys()



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
