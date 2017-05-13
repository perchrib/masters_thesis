from __future__ import print_function
import os
import pandas as pd
from helpers.global_constants import CROWDFLOWER_CSV_PATH, TEST_DATA_DIR, MALE, FEMALE
from preprocessors.parser import Parser


"""
Crowdflower Gender-Annotated Tweet Dataset Parser

"""

def parse_crowdflower(file_path=CROWDFLOWER_CSV_PATH, save_dir_path=TEST_DATA_DIR, save_to_file=False):

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    df = pd.read_csv(file_path)  # Load dataset to pandas DataFrame

    print(df['gender'].value_counts())

    male_tweets = []
    female_tweets = []

    duplicates_males = 0
    duplicates_females = 0

    for index, row in df.iterrows():
        tweet = row['text']
        if row['gender'] == 'male':
            if tweet not in male_tweets:
                male_tweets.append(tweet)
            else:
                duplicates_males += 1
        elif row['gender'] == 'female':
            if tweet not in female_tweets:
                female_tweets.append(tweet)
            else:
                duplicates_females += 1
        else:
            continue  # Ignore classes non-gender classes; brand and unknown

    print("Number of duplicate tweets in male set %i" % duplicates_males)
    print("Number of duplicate tweets in female set %i" % duplicates_females)

    _write_tweets_to_file(MALE, 'male_tweets', male_tweets)
    _write_tweets_to_file(FEMALE, 'female_tweets', female_tweets)


def _write_tweets_to_file(class_name, file_name, tweets):
    """
    Write list of tweets to file
    :param class_name: name of class. Use constant values specified in global constants. E.g. MALE
    :param file_name: name, not path
    :param tweets: list of tweets to write to file
    """

    parser = Parser()

    # Placeholders to replicate PAN training set file patterns
    ID_PLACEHOLDER = "_"
    SECONDARY_ATTR_PLACEHOLDER = "_"

    # Number of tweets which are successfully parsed and pass quality control
    num_accepted_tweets = 0

    with open(os.path.join(TEST_DATA_DIR, file_name + '.txt'), 'wb') as dataset_file:
        dataset_file.write("%s:::%s:::%s" % (ID_PLACEHOLDER, class_name, SECONDARY_ATTR_PLACEHOLDER))

        for twt in tweets:
            twt = parser.clean_html(twt)  # Remove non-english characters and correct spacing
            if len(twt) > 1:
                dataset_file.write("\n%s" % twt)
                num_accepted_tweets += 1

        print("\n%s" % class_name)
        print("Number of tweets written to file: %i" % num_accepted_tweets)
        print("Number of tweets declined: %i" % (len(tweets) - num_accepted_tweets))


if __name__ == '__main__':
    parse_crowdflower(save_to_file=True)
