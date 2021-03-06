from __future__ import print_function
import os
import sys

# Append path to use modules outside pycharm environment, e.g. remote server
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from functools import reduce
from preprocessors.parser import Parser
from nltk import sent_tokenize
import numpy as np

from preprocessors.language_detection import detect_language
from helpers.global_constants import TRAIN_DATA_DIR, GENDER, AGE, VALIDATION_SPLIT, TEST_SPLIT, TEST_DATA_DIR, TRAIN, \
    TEST, REM_STOPWORDS, REM_EMOTICONS, REM_PUNCTUATION, LEMMATIZE, REM_INTERNET_TERMS, LOWERCASE
from helpers.helper_functions import shuffle
from helpers.model_utils import get_argmax_classes

SEED = 1337


def prepare_dataset(prediction_type=GENDER, folder_path=TRAIN_DATA_DIR, gender=None):
    """
    --Used in both word_level and character_level--
    Iterate over dataset folder and create sequences of word indices
    Expecting a directory of text files, one for each author. Each line in files corresponds to a tweet
    :return: texts, labels, metadata, labels_index
    """

    texts = []  # list of text samples
    labels_index = construct_labels_index(prediction_type)  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    metadata = []  # list of dictionaries with author information (age, gender)

    print("\n------Parsing %s files.." % folder_path)
    for sub_folder_name in sorted(filter(lambda x: ".DS" not in x, list(os.listdir(folder_path)))):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        tweet_count = 0
        for file_name in sorted(os.listdir(sub_folder_path)):
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(sub_folder_path, file_name)

                with open(file_path, 'r') as txt_file:
                    data_samples = [line.strip() for line in txt_file]

                author_data = data_samples.pop(0).split(':::')  # ID, gender and age of author

                # Remaining lines correspond to the tweets by the author
                # TODO: If anything other than gender needs to be classified, this part must be refactored
                gender_author = author_data[1].upper()
                if not gender:
                    gender_author = gender
                if gender == gender_author:
                    for tweet in data_samples:
                        texts.append(tweet)
                        metadata.append({GENDER: author_data[1].upper(), AGE: author_data[2]})
                        labels.append(labels_index[metadata[-1][prediction_type]])
                        tweet_count += 1
        print("%i tweets in %s" % (tweet_count, sub_folder_name))

    print('\nFound %i texts.' % len(texts))
    return texts, labels, metadata, labels_index


def split_dataset(data, labels, metadata, data_type_is_string=False):
    """
    Given correctly formatted dataset, split into training, validation and test
    :param data: formatted dataset, i.e., sequences of char/word indices
    :return: training set, validation set, test set and metadata
    """
    np.random.seed(SEED)
    # shuffle and split the data into a training set and a validation set
    if data_type_is_string:
        data, labels, indices = shuffle(data, labels)
    else:
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

    metadata = [metadata[i] for i in indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    meta_train = metadata[:-nb_validation_samples]

    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    meta_val = metadata[-nb_validation_samples:]

    return x_train, y_train, meta_train, x_val, y_val, meta_val


def construct_labels_index(prediction_type):
    """
    Constuct appropriate dictionary mappings class labels to IDs
    :param prediction_type: constants.PREDICT_GENDER or constants.PREDICT_AGE
    :return: dictionary mapping label name to numeric ID
    """
    if prediction_type == GENDER:
        return {'MALE': 0, 'FEMALE': 1}
    elif prediction_type == AGE:
        return {'18-24': 0, '25-34': 1, '35-49': 2, '50-64': 3, '65-xx': 4}


def filter_gender(x_train, y_train, meta_train, labels_index, gender):
    """
    Given training samples and labels, filter the set leaving only the specified gender
    :param x_train: formatted training samples
    :param y_train: categorical labels
    :param labels_index: dictionary containing label integer indices for each gender
    :param gender: the gender to keep. Constants MALE or FEMALE
    :return: filtered training samples and labels, containing only the specified gender
    """

    print("Keeping only %s in the training set" % gender)
    modified_y_train = []
    modified_x_train = []
    modified_metadata = []

    # Convert from categorical form
    y_labels = get_argmax_classes(y_train)

    for i in range(len(x_train)):
        if y_labels[i] == labels_index[gender]:
            modified_x_train.append(x_train[i])
            modified_y_train.append(y_train[i])
            modified_metadata.append(meta_train[i])

    return np.asarray(modified_x_train), np.asarray(modified_y_train), modified_metadata


def display_dataset_statistics(texts):
    """
    Given a dataset, display statistics: Number of tweets, avg length of characters and tokens.
    :param texts: list of string texts
    """

    print("\n---Dataset statistics---")
    # Number of tokens/words per tweet
    tokens_length_all_texts = list(map(lambda twt: len(twt.split(" ")), texts))
    avg_token_len = reduce(lambda total_len, twt_len: total_len + twt_len, tokens_length_all_texts, 0) / float(len(
        tokens_length_all_texts))
    max_word_length = max(tokens_length_all_texts)
    min_word_length = min(tokens_length_all_texts)
    median_word_length = np.median(tokens_length_all_texts)

    # Number of characters per tweet
    char_length_all_texts = list(map(lambda twt: len(twt), texts))
    max_char_length = max(char_length_all_texts)
    min_char_length = min(char_length_all_texts)
    median_char_length = np.median(char_length_all_texts)
    avg_char_len = reduce(lambda total_len, tweet_len: total_len + tweet_len, char_length_all_texts) / float(len(texts))

    # Number of empty tweets
    num_empty_tweets = reduce(lambda a, twt: a + 1 if len(twt) == 0 else a, texts, 0)

    print("Number of tweets: %i" % len(texts))
    # print("Number of empty tweets (given pre-processing; removal of stopwords etc...): %i" % num_empty_tweets)
    print("Average number of tokens/words per tweet: %f" % avg_token_len)
    print("Median of number of words in a tweet: %f" % median_word_length)
    print("Max number of words in a tweet: %f" % max_word_length)
    print("Min number of words in a tweet: %f" % min_word_length)

    print("\nAverage number of characters per tweet: %f" % avg_char_len)
    print("Median of number of characters in a tweet: %f" % median_char_length)
    print("Max number of characters in a tweet: %f" % max_char_length)
    print("Min number of characters in a tweet: %f" % min_char_length)


    # Split list of texts into lists of sentences
    txt_sents = list(map(lambda tweet: sent_tokenize(tweet), texts))

    # Number of sentences per tweet
    avg_sents_per_tweet = reduce(lambda total_len, sents: total_len + len(sents), txt_sents, 0) / float(len(texts))

    # Number of characters per sentence - len in chars
    sent_len_tweets = [list(map(lambda s: len(s), tweet)) for tweet in txt_sents]  # Lists of sentence lengths
    avg_sent_len_tweets = [reduce(lambda total_len, s_l: total_len + s_l, tweet, 0) / float(len(tweet)) for tweet in sent_len_tweets]  # len(tweet) here is number of sentences
    avg_char_per_sent = reduce(lambda total_len, avg_chars: total_len + avg_chars, avg_sent_len_tweets, 0) / float(len(texts))

    print("Average number of sentences per tweet: %f" % avg_sents_per_tweet)
    print("Average number of characters per sentences: %f" % avg_char_per_sent)


def filter_dataset(texts, labels, metadata, filters, train_or_test):
    """
    Filter dataset based on what's specified in filters dict
    :param texts:
    :param labels:
    :param metadata:
    :param filters: dict of values specifying what should be removed
    :param train_or_test: Values= TRAIN or TEST. If TEST, short tweets are not removed
    :return:
    """

    print("-->Specified filters: ")
    print(filters)

    modified_texts = texts
    modified_labels = labels
    modified_metadata = metadata
    count_removed = 0  # Stays zero unless removed for training set

    # Extra parsing info for use in log
    extra_info = ["Remove stopwords %s" % filters[REM_STOPWORDS],
                  "Lemmatize %s" % filters[LEMMATIZE],
                  "Remove punctuation %s" % filters[REM_PUNCTUATION],
                  "Remove emoticons %s" % filters[REM_EMOTICONS]]

    # Text pre-processor
    text_parser = Parser()

    # Lowercase
    if LOWERCASE in filters and not filters[LOWERCASE]:
        extra_info.append("Text is not lowercased")
    else:
        modified_texts = text_parser.lowercase(modified_texts)
        extra_info.append("Text is lowercased")

    # Either remove Internet specific tokens or replace with tags
    if REM_INTERNET_TERMS in filters and filters[REM_INTERNET_TERMS]:
        modified_texts = text_parser.remove_all_twitter_syntax_tokens(modified_texts)
        extra_info.append("Internet terms have been REMOVED")
    else:
        modified_texts = text_parser.replace_all_twitter_syntax_tokens(modified_texts)
        extra_info.append("Internet terms have been replaced with placeholders")

    # Other filtering
    if filters[REM_STOPWORDS]:
        modified_texts = text_parser.remove_stopwords(modified_texts)

    if filters[REM_PUNCTUATION]:
        modified_texts = text_parser.remove_punctuation(modified_texts)

    if filters[LEMMATIZE]:
        modified_texts = text_parser.lemmatize(modified_texts)

    if filters[REM_EMOTICONS]:
        modified_texts = text_parser.remove_emoticons(modified_texts)

    # Remove short texts from training
    if train_or_test == TRAIN:
        modified_texts, modified_labels, modified_metadata, count_removed = text_parser.remove_texts_shorter_than_threshold(
            modified_texts, modified_labels, modified_metadata)

    # Extra parsing info for use in log
    extra_info.append("Removed %i tweet because they were shorter than threshold" % count_removed)

    return modified_texts, modified_labels, modified_metadata, extra_info


def display_gender_distribution(metadata):
    num_total = len(metadata)
    num_males = reduce(lambda total, x: total + 1 if x['gender'] == 'MALE' else total, metadata, 0)
    num_females = reduce(lambda total, x: total + 1 if x['gender'] == 'FEMALE' else total, metadata, 0)

    print("\nTotal number of texts %i" % num_total)
    print("Number of male texts: %i. Fraction of total: %f" % (num_males, float(num_males) / num_total))
    print("Number of female texts: %i Fraction of total: %f" % (num_females, float(num_females) / num_total))


if __name__ == '__main__':
    prepare_dataset(folder_path=TEST_DATA_DIR)
    # txts, labels, metadata, labels_index = prepare_dataset(GENDER)
    # display_gender_distribution(metadata)
#     parser = Parser()
#     # txts = parser.replace_all_twitter_syntax_tokens(txts)  # Replace Internet terms and lowercase
#
#     # txts = parser.remove_stopwords(txts)
#     # txts, labels, metadata = parser.remove_texts_shorter_than_threshold(txts, labels, metadata)
#
#     # txts = parser.remove_emoticons(txts)
#     # txts = parser.remove_punctuation(txts)
#     # txts = parser.lemmatize(txts)
#     display_dataset_statistics(txts)
