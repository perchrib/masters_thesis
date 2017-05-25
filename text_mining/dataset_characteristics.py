from __future__ import print_function
from helpers import word_tokenize
from collections import Counter
import re
from preprocessors.parser import Parser
from nltk.corpus import stopwords
import nltk
from nltk.tag import map_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


class Characteristics():
    """
    :param texts, a list of texts, each text is an element in the list
    """

    def __init__(self, texts):
        self.tokens = word_tokenize(texts)
        self.token_count = Counter(self.tokens)
        self.clean_token_count = Counter(clean_text(texts))

        self.emoticon_count = emoticon_counter(self.tokens)
        self.hashtag_count = tag_counter(self.tokens, "#")
        self.mention_count = tag_counter(self.tokens, "@")
        self.twitter_syntax_token_count = twitter_syntax_token_counter(texts, self.emoticon_count)
        self.length_of_text_char_count, self.length_of_text_word_count = length_of_texts_counter(texts)
        self.stopwords_count = stopwords_counter(texts)

    def most_common(self, n_freq):
        """
        :param counter, Counter object
        :param n_freq, number of most frequent tokens 
        :returns most_common_counter, a Counter object with the most common tokens (and the frequency)
        """

        return most_common(self.clean_token_count, n_freq)

    def least_common(self, n_freq):
        """
        :param counter, Counter object
        :param n_freq, number of least frequent tokens 
        :returns most_common_counter, a Counter object with the most common tokens (and the frequency)
        """
        return least_common(self.clean_token_count, n_freq)


def least_common(counter, n_freq):
    least_common = counter.most_common()[-n_freq:]
    least_common_counter = Counter({k: v for k, v in least_common})
    return least_common_counter


def most_common(counter, n_freq):
    common_tokens = map(lambda x: x[0], counter.most_common(n_freq))
    most_common_counter = Counter({k: v for k, v in counter.iteritems() if k in common_tokens})
    return most_common_counter


def emoticon_counter(tokens):
    """
    :param tokens: list of tokens 
    :return: Counter object with number of each emoticons
    """
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:<3)', " ".join(tokens))
    return Counter(emoticons)


def tag_counter(tokens, tag):
    """
    :param tokens: list of tokens 
    :param tag: type if tag token ie. "#" or "@"
    :return: Counter object with all tags starts with the given tag type
    """
    tags = [t for t in tokens if t.startswith(tag) and len(t) > 1]
    return Counter(tags)


def twitter_syntax_token_counter(texts, emoticon_count=None):
    parser = Parser()
    parsed_texts = parser.replace(texts, url="URLs", pic="PICTURES", mention="MENTIONS", hashtag="HASHTAGS")
    parsed_tokens = word_tokenize(parsed_texts)
    twitter_syntax_tokens = re.findall(r'(?:URLs|HASHTAGS|MENTIONS|PICTURES)', " ".join(parsed_tokens))

    if emoticon_counter:
        num_of_emoticons = sum(emoticon_count.values())
    else:
        num_of_emoticons = sum(emoticon_counter(word_tokenize(texts)).values())

    emoticons = ["EMOTICONS" for _ in range(num_of_emoticons)]
    twitter_syntax_tokens.extend(emoticons)

    return Counter(twitter_syntax_tokens)


def clean_text(texts):
    parser = Parser()
    clean_texts = parser.remove_stopwords(texts)
    clean_texts = parser.lowercase(clean_texts)
    clean_texts = parser.replace_all_twitter_syntax_tokens(clean_texts)
    clean_tokens = word_tokenize(clean_texts)
    return clean_tokens


def length_of_texts_counter(texts):
    parser = Parser()
    parsed_texts = parser.replace(texts, pic="", url="")
    length_of_texts_chars = [len(text) for text in parsed_texts]
    length_of_texts_words = [len(text.split()) for text in parsed_texts]
    return Counter(length_of_texts_chars), Counter(length_of_texts_words)


def unequal_token_count(dist_1, dist_2, n_frequent_tokens=None):
    """
    :param dist_1: Counter object from distribution_1  
    :param dist_2: Counter object from distribution_2 
    :param n_frequent_tokens: number of frequent tokens
    :return: n 
    """
    common_tokens = (dist_1 & dist_2).keys()
    dist_1 = Counter({k: v for k, v in dist_1.iteritems() if k not in common_tokens})
    dist_2 = Counter({k: v for k, v in dist_2.iteritems() if k not in common_tokens})
    if n_frequent_tokens:
        dist_1 = most_common(dist_1, n_frequent_tokens)
        dist_2 = most_common(dist_2, n_frequent_tokens)

    return dist_1, dist_2


def equal_token_count(dist_1, dist_2, n_frequent_tokens=None):
    """
    :param dist_1: Counter object from distribution_1  
    :param dist_2: Counter object from distribution_2 
    :param n_frequent_tokens: number of frequent tokens
    :return: n common words in both distributions
    """
    if n_frequent_tokens:
        common_tokens = map(lambda x: x[0], (dist_1 & dist_2).most_common(n_frequent_tokens))
    else:
        common_tokens = (dist_1 & dist_2).keys()
    dist_1 = Counter({k: v for k, v in dist_1.iteritems() if k in common_tokens})
    dist_2 = Counter({k: v for k, v in dist_2.iteritems() if k in common_tokens})

    return dist_1, dist_2


def lower(counter):
    new_counter = Counter({k.lower(): v for k, v in counter.iteritems()})
    return new_counter


def stopwords_counter(texts):
    stop = set(stopwords.words('english'))
    all_stopwords = [word for text in texts for word in text.lower().split() if word in stop]
    return Counter(all_stopwords)


def pos_tag_counter(tokens, simple_pos_tags=True):
    postags = nltk.pos_tag(tokens)
    if simple_pos_tags:
        simple_tags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in postags]
        pos_tag_counts = Counter([tag for word, tag in simple_tags])
    else:
        ordinary_tags = [(word, tag) for word, tag in postags]
        pos_tag_counts = Counter([tag for word, tag in ordinary_tags])
    return pos_tag_counts


def sentiment_tweet_counter(texts):
    analyzer = SIA()
    sentiment_counts = []
    for t in texts:
        result = analyzer.polarity_scores(t)
        if is_equals(result):
            print("NEUTRAL")
            sentiment_counts.append('neu')
        else:
            result['compound'] = -1
            sentiment_counts.append(max(result, key=result.get))
    return Counter(sentiment_counts)


def is_equals(results):
    del results['compound']
    if set(results.values()) == 0:
        return True
    return False


def sentiment_word_counter(texts):
    pos_word_count = []
    neg_word_count = []
    neu_word_count = []
    all_words = word_tokenize(texts)
    analyzer = SIA()
    for w in all_words:
        if len(w) < 2:
            continue
        result = analyzer.polarity_scores(w)
        result['compound'] = -1
        sentiment_result = max(result, key=result.get)

        if sentiment_result == "pos":
            pos_word_count.append(w)
        elif sentiment_result == "neg":
            neg_word_count.append(w)
        elif sentiment_result == "neu":
            neu_word_count.append(w)
    return Counter(pos_word_count), Counter(neg_word_count), Counter(neu_word_count)









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
