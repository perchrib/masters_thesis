import os
from functools import reduce

import numpy as np

from character_level_classification.constants import *

np.random.seed(1337)


def prepare_dataset(folder_path=TEXT_DATA_DIR, gender=None):
    """
    --Used in both word_level and character_level--
    Iterate over dataset folder and create sequences of word indices
    Expecting a directory of text files, one for each author. Each line in files corresponds to a tweet
    :return: texts, labels, metadata, labels_index
    """

    texts = []  # list of text samples
    labels_index = construct_labels_index(PREDICTION_TYPE)  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    metadata = []  # list of dictionaries with author information (age, gender)

    print("------Parsing txt files..")
    for sub_folder_name in sorted(list(filter(lambda x: 'pan' in x, os.listdir(folder_path)))):
        print(sub_folder_name)
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        for file_name in sorted(os.listdir(sub_folder_path)):
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(sub_folder_path, file_name)

                with open(file_path, 'r') as txt_file:
                    data_samples = [line.strip() for line in txt_file]

                author_data = data_samples.pop(0).split(':::')  # ID, gender and age of author

                # Remaining lines correspond to the tweets by the author
                gender_author = author_data[1].upper()
                if not gender:
                    gender_author = gender
                if gender == gender_author:
                    for tweet in data_samples:
                        texts.append(tweet)
                        metadata.append({GENDER: author_data[1].upper(), AGE: author_data[2]})
                        labels.append(labels_index[metadata[-1][PREDICTION_TYPE]])

    print('Found %s texts.' % len(texts))
    return texts, labels, metadata, labels_index


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
    return prepare_dataset(gender='MALE')


def prepare_dataset_women():
    return prepare_dataset(gender='FEMALE')