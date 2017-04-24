from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class FeatureExtraction:
    def __init__(self, documents):
        # All tweets containing a list
        self.docs = np.asarray(documents)
        self.unique_tokens = self.unique_tokens()
        self.vocabulary = self.unique_tokens.keys()
        # contains the vocabulary, count word frequencies
        self.count = CountVectorizer(vocabulary=self.vocabulary)
        # Construct the bag of word model
        self.bow = self.count.fit_transform(self.docs)

        self.tfidf = TfidfTransformer()

    def unique_tokens(self):
        unique_tokens = dict()
        for doc in self.docs:
            for token in doc.split():
                if token in unique_tokens:
                    unique_tokens[token] += 1
                else:
                    unique_tokens[token] = 1
        return unique_tokens


def tf_idf(documents):
    fe = FeatureExtraction(documents)
    np.set_printoptions(precision=2)
    features = fe.tfidf.fit_transform(fe.count.fit_transform(fe.docs)).toarray()
    print("SUM: ", sum(features[0]), " length: ", len(features[0]))
    return features

if __name__ == "__main__":
    from preprocessors.parser import Parser
    from preprocessors.dataset_preparation import prepare_dataset, prepare_dataset_women, prepare_dataset_men
    parser = Parser()
    texts, labels, metadata, labels_index = prepare_dataset()
    texts = ["This is a test for a given input", "we are the world, we are the people", "this is it"]
    print "Removing Stopwords..."
    parsed_texts = parser.remove_stopwords(texts)
    tf_idfs = tf_idf(parsed_texts)
