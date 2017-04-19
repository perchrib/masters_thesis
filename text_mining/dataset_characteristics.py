import nltk
from helpers import flatten
from collections import Counter
import re

class Characteristics():
    """
    :param male_authors, list containing texts with gender male
    :param female_authors, , list containing texts with gender female
    :param authors, list containing all the data as the object Authors
    """
    def __init__(self, texts):
        self.tokens = flatten(map(lambda x: x.split(), texts))
        self.word_count = Counter(self.tokens)
        self.emoticon_count = emoticon_counter(self.tokens)
    # def __init__(self, male_authors, female_authors, authors):
    #     self.male = male_authors
    #     self.female = female_authors
    #     self.authors = authors

    def most_common_tokens(self, n_frequent_tokens=None):
        """
        :param n_frequent_tokens, number of most frequent tokens 
        :returns list of tuples with the token and the frequency
        """
        return self.word_count.most_common(n_frequent_tokens)


def emoticon_counter(tokens):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:<3)', " ".join(tokens))
    return Counter(emoticons)


def equal_token_count(dist_1, dist_2, n_frequent_tokens=None):
    if n_frequent_tokens:
        common_tokens = map(lambda x: x[0], (dist_1 & dist_2).most_common(n_frequent_tokens))
    else:
        common_tokens = (dist_1 & dist_2).keys()
    dist_1 = Counter({k: v for k, v in dist_1.iteritems() if k in common_tokens})
    dist_2 = Counter({k: v for k, v in dist_2.iteritems() if k in common_tokens})

    return dist_1, dist_2




def display_dataset_statistics(texts):
    """
    Given a dataset as a list of texts, display statistics: Number of tweets, avg length of characters and tokens.
    :param texts: List of string texts
    """

    # Number of tokens per tweet
    tokens_all_texts = list(map(lambda tweet: tweet.split(" "), texts))
    avg_token_len = reduce(lambda total_len, tweet_tokens: total_len + len(tweet_tokens), tokens_all_texts,
                           0) / len(
        tokens_all_texts)

    # Number of characters per tweet
    char_length_all_texts = list(map(lambda tweet: len(tweet), texts))
    avg_char_len = reduce(lambda total_len, tweet_len: total_len + tweet_len, char_length_all_texts) / len(
        texts)

    print("Number of tweets: %i" % len(texts))
    print("Average number of tokens per tweet: %f" % avg_token_len)
    print("Average number of characters per tweet: %f" % avg_char_len)
