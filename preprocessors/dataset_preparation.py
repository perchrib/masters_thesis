from __future__ import print_function
import os
from functools import reduce
from preprocessors.parser import Parser
from nltk import sent_tokenize
import numpy as np

from helpers.global_constants import TEXT_DATA_DIR, GENDER, AGE, VALIDATION_SPLIT, TEST_SPLIT

np.random.seed(1337)


def prepare_dataset(prediction_type, folder_path=TEXT_DATA_DIR, gender=None):
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

    print("------Parsing txt files..")
    for sub_folder_name in sorted(list(filter(lambda x: 'pan' in x, os.listdir(folder_path)))):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        tweet_count = 0
        for file_name in sorted(os.listdir(sub_folder_path)):
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(sub_folder_path, file_name)

                with open(file_path, 'r') as txt_file:
                    data_samples = [line.strip() for line in txt_file]

                author_data = data_samples.pop(0).split(':::')  # ID, gender and age of author

                # Remaining lines correspond to the tweets by the author
                # TODO: If anything other than gender needs to be classified, this needs to be moved
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

    print('\nFound %s texts.' % len(texts))
    return texts, labels, metadata, labels_index


def split_dataset(data, labels, metadata):
    """
    Given correctly formatted dataset, split into training, validation and test
    :param data: formatted dataset, i.e., sequences of char/word indices
    :return: training set, validation set, test set and metadata
    """

    # shuffle and split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    metadata = [metadata[i] for i in indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    nb_test_samples = int(TEST_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples-nb_test_samples]
    y_train = labels[:-nb_validation_samples-nb_test_samples]
    meta_train = metadata[:-nb_validation_samples-nb_test_samples]

    x_val = data[-nb_validation_samples-nb_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_samples-nb_test_samples:-nb_test_samples]
    meta_val = metadata[-nb_validation_samples-nb_test_samples:-nb_test_samples]

    x_test = data[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]
    meta_test = metadata[-nb_test_samples:]

    return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test

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


def prepare_dataset_men():
    return prepare_dataset(prediction_type=GENDER, gender='MALE')


def prepare_dataset_women():
    return prepare_dataset(prediction_type=GENDER, gender='FEMALE')


def display_dataset_statistics(texts):
    """
    Given a dataset, display statistics: Number of tweets, avg length of characters and tokens.
    :param texts: list of string texts
    :param labels: list of classification labels
    :param metadata: list of metadata dictionaries
    :param pred_type: classification_type
    """

    # Number of tokens/words per tweet
    tokens_all_texts = list(map(lambda tweet: tweet.split(" "), texts))
    avg_token_len = reduce(lambda total_len, tweet_tokens: total_len + len(tweet_tokens), tokens_all_texts, 0) / float(len(
        tokens_all_texts))

    # Number of characters per tweet
    char_length_all_texts = list(map(lambda tweet: len(tweet), texts))
    avg_char_len = reduce(lambda total_len, tweet_len: total_len + tweet_len, char_length_all_texts) / float(len(texts))

    print("Number of tweets: %i" % len(texts))
    print("Average number of tokens/words per tweet: %f" % avg_token_len)
    print("Average number of characters per tweet: %f" % avg_char_len)

    # Split list of texts into lists of sentences
    txt_sents = list(map(lambda tweet: sent_tokenize(tweet), texts))

    # Number of sentences per tweet
    avg_sents_per_tweet = reduce(lambda total_len, sents: total_len + len(sents), txt_sents, 0) / float(len(texts))

    # Number of characters per sentence - len in chars
    sent_len_tweets = [list(map(lambda s: len(s), tweet)) for tweet in txt_sents]
    avg_sent_len_tweets = [reduce(lambda total_len, s_l: total_len + s_l, tweet, 0) / float(len(tweet)) for tweet in sent_len_tweets]
    avg_char_per_sent = reduce(lambda total_len, avg_chars: total_len + avg_chars, avg_sent_len_tweets, 0) / float(len(texts))

    print("Average number of sentences per tweet: %f" % avg_sents_per_tweet)
    print("Average number of characters per sentences: %f" % avg_char_per_sent)


def display_gender_distribution(metadata):

    num_total = len(metadata)
    num_males = reduce(lambda total, x: total + 1 if x['gender'] == 'MALE' else total, metadata, 0)
    num_females = reduce(lambda total, x: total + 1 if x['gender'] == 'FEMALE' else total, metadata, 0)

    print("Total number of texts %i" % num_total)
    print("Number of male texts: %i. Fraction of total: %f" % (num_males, float(num_males) / num_total))
    print("Number of female texts: %i Fraction of total: %f" % (num_females, float(num_females) / num_total))


if __name__ == '__main__':
    txts, labels, metadata, labels_index = prepare_dataset(GENDER)
    # txts = ["LJ_Barca all UK is a bigger grain harvest and banking ! We all still get taken for mugs by the Gov ! UK will selfdestruct eventually", "Hei. Duasdfsadfasdf. sadf. asd. Jeg"]
    parser = Parser()
    txts = parser.replace_all(txts)
    display_dataset_statistics(txts)
